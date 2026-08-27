# KServe Demo: Tile + Embed WSIs with Virchow

A local, single-node stand-in for a [KServe](https://kserve.github.io/website/) `InferenceService`,
run with Docker Compose. It takes a whole-slide image (WSI) and returns per-tile
[Virchow](https://huggingface.co/paige-ai/Virchow) embeddings written to HDF5.

The two containers mirror the two halves of a KServe `InferenceService`:

```
                 (transformer)                    (predictor)
  request  -->  preprocessor:8080  --gRPC-->  triton:8001  --> GPU
  {slide_uri}   tile w/ slideflow   uint8      Virchow          |
                    ^                          TorchScript      |
                    |                                           |
                    +---- (B, 2560) float32 embeddings <--------+
                    |
                    v
              /out/<slide>.h5
```

* **preprocessor** ([`preprocessor/tiling/main.py`](preprocessor/tiling/main.py)) — a KServe
  `Model` that opens the slide with [slideflow](https://github.com/jamesdolezal/slideflow), applies
  Otsu tissue detection, extracts 224 px tiles at 20x (0.5 MPP), and streams batches of tiles to the
  predictor over gRPC. Tile reading happens in a `forkserver` process pool; batches move through an
  `asyncio.Queue` to several concurrent predictor calls, so reading and inference overlap.
* **triton** — [Triton Inference Server](https://github.com/triton-inference-server/server) serving
  Virchow as a TorchScript model ([`model-repo/virchow/config.pbtxt`](model-repo/virchow/config.pbtxt)).

Normalization and Virchow's embedding step (concat of the class token with the mean of the patch
tokens) are baked into the serialized model, so the wire format between the two containers is
`uint8` tiles in, `(B, 2560)` `float32` embeddings out. See
[`virchow-torchscript.ipynb`](virchow-torchscript.ipynb) for why.

## Prerequisites

* Docker with Compose and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (Triton needs a GPU).
* The `wsi` conda environment from the repo root ([`env.yaml`](../env.yaml)) to run the notebook.
* At least one slide to embed — see the [repo README](../README.md) for downloading WSIs from the GDC.
* A Hugging Face account with access to [paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow)
  (gated) and `huggingface-cli login`, so the notebook can pull the weights.

## Setup

1. **Serialize Virchow.** Run [`virchow-torchscript.ipynb`](virchow-torchscript.ipynb) to trace the
   model and write it to `model-repo/virchow/1/model.pt`, where Triton expects it:

    ```
    model-repo/
    └── virchow/
        ├── config.pbtxt
        └── 1/
            └── model.pt   # written by the notebook, gitignored
    ```

2. **Stage a slide.** Put the `.svs` files you want to embed in `kserve-demo/sample/`, which is
   mounted into the preprocessor at `/sample`. Embeddings land in `kserve-demo/out/` (`/out` in the
   container). Both directories are gitignored.

    ```bash
    mkdir -p sample out
    cp /path/to/TCGA-XX-XXXX-...-DX1.<uuid>.svs sample/
    ```

3. **Start the services.** `CUDA_VISIBLE_DEVICES` selects which GPU Triton reserves and has no
   default, so set it explicitly:

    ```bash
    CUDA_VISIBLE_DEVICES=0 docker compose up --build
    ```

    Triton is ready once it lists `virchow` as `READY` in its model table; the preprocessor is ready
    once it reports its HTTP server listening on port 8080.

## Run an inference

The preprocessor speaks the KServe v1 REST protocol. `slide_uri` is a path *inside* the container,
i.e. under `/sample`:

```bash
curl -X POST http://localhost:8080/v1/models/virchow:predict -H 'Content-Type: application/json' -d '{"slide_uri": "/sample/TCGA-E2-A14P-01Z-00-DX1.663B02FF-C64B-41A6-8685-FD61CD76F9C6.svs"}'
```

The response is metadata about the run; the embeddings themselves are written to disk:

```json
{
  "slide_uri": "/sample/TCGA-E2-A14P-....svs",
  "features_uri": "/out/TCGA-E2-A14P-....h5",
  "n_tiles": 12345,
  "embed_dim": 2560,
  "dtype": "float32",
  "tile_px": 224,
  "tile_um": 112,
  "encoder": "virchow"
}
```

The HDF5 file holds two datasets, row-aligned so that `features[i]` is the embedding of the tile at
`coords[i]`:

| dataset    | shape        | dtype     | notes                          |
| ---------- | ------------ | --------- | ------------------------------ |
| `features` | `(N, 2560)`  | `float32` | gzip compressed                |
| `coords`   | `(N, 2)`     | `int32`   | tile grid indices, not pixels  |

`N` is the number of tiles that passed Otsu tissue filtering, so it is smaller than the full tile
grid and varies by slide.

Triton's own endpoints are also exposed for debugging — HTTP on 8000, gRPC on 8001, Prometheus
metrics on 8002:

```bash
curl http://localhost:8000/v2/models/virchow/config
```

## Tuning

The throughput knobs are passed as `command` args to the preprocessor in
[`docker-compose.yaml`](docker-compose.yaml):

| flag                             | default in compose | what it does                                              |
| -------------------------------- | ------------------ | --------------------------------------------------------- |
| `--tile_batch`                   | 512                | tiles per inference request (≤ `max_batch_size` of 2048)   |
| `--concurrent_calls_to_predictor`| 8                  | in-flight gRPC requests to Triton                          |
| `--num_tile_readers`             | 4                  | worker processes decoding tiles from the slide             |
| `--queue_depth`                  | 16                 | batches buffered between readers and predictor calls       |

These have not been tuned; reading tiles off large slides is usually the bottleneck before the GPU is.

## Known gaps

This is a demo, not a deployment. Notable shortcuts:

* Input is a container-local path. A real pipeline would trigger off an S3 notification / CloudEvent
  and read the slide from object storage.
* Output is written to a mounted host directory rather than uploaded to object storage, so
  `features_uri` is a local path.
* Compose stands in for an actual KServe `InferenceService`; there are no manifests here yet, and
  nothing autoscales.
