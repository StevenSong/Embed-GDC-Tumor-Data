# Virchow tile embeddings (no KServe)

One container, one command: point it at a directory of whole-slide images and it writes one HDF5
of [Virchow](https://huggingface.co/paige-ai/Virchow) tile embeddings per slide, plus a thumbnail
showing what tissue detection kept.

```
  /slides/*.svs  -->  slideflow tiling (Otsu, 224 px @ 20x)  -->  Virchow  -->  /out/<slide>.h5
                              ^ forkserver pool                     ^ local GPU     /out/thumbnails/<slide>.png
                              +---- tile batches in memory ---------+
```

Same tiling and same embeddings as [`../../kserve-demo`](../../kserve-demo), minus the serving
machinery: the model runs in-process, so there is no transformer/predictor split, no gRPC hop, and
no TorchScript export. Tiles are never written to disk — no tile directories, no tfrecords — they
go from the slide reader into a bounded queue and straight onto the GPU.

## Build

Virchow is gated, so the build needs a Hugging Face account with access to
[paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow). The weights are pulled **at build
time** into `$HF_HOME` inside the image; at runtime the container sets `HF_HUB_OFFLINE=1` and never
talks to huggingface.co.

The token is passed as a BuildKit secret, not a `--build-arg` — an ARG would stay readable in the
image history:

```bash
HF_TOKEN=hf_xxx docker build --secret id=hf_token,env=HF_TOKEN -t virchow-embed embed/virchow
```

The pinned stack is torch 2.13.0 / torchvision 0.28.0 on cu130 wheels; override with
`--build-arg TORCH_VERSION=... --build-arg TORCHVISION_VERSION=... --build-arg TORCH_INDEX_URL=...`
if the cluster needs a different CUDA build.

slideflow and timm are installed with `--no-deps` and their runtime imports are listed explicitly.
slideflow's own dependency list is ~35 packages, most of which tiling never touches — a
hyperparameter search stack (`smac` -> `pyrfr`, which needs swig to build on py>3.10), a GUI
(`imgui`/`glfw`/`pyopengl`), umap/numba, rasterio, tensorboard — and timm would otherwise pull a
second copy of torch from PyPI. The cost of `--no-deps` is that a transitive import slideflow makes
lazily could be missing; the build runs `import slideflow; qc.Otsu()` to catch that at build time
rather than on the cluster, so if something is missing, add it to the list in the
[`Dockerfile`](Dockerfile) and rebuild.

## Run

```bash
docker run --rm --gpus '"device=0"' \
  -v /path/to/slides:/slides:ro \
  -v /path/to/out:/out \
  virchow-embed --slide-dir /slides --out-dir /out
```

`--slide-dir` is searched recursively. Slides that already have an `.h5` in `--out-dir` are skipped
unless `--overwrite` is passed, so a killed job can be restarted; output is written to a hidden temp
file and renamed, so a partial file never looks finished. A slide that fails is logged and the run
moves on, with a non-zero exit at the end.

There is no multi-GPU support inside the container by design. Run one container per GPU over
disjoint input directories:

```bash
docker run --rm --gpus '"device=0"' -v /slides/part0:/slides:ro -v /out:/out virchow-embed --slide-dir /slides --out-dir /out &
docker run --rm --gpus '"device=1"' -v /slides/part1:/slides:ro -v /out:/out virchow-embed --slide-dir /slides --out-dir /out &
```

For a quick smoke test on the cluster, `--max-tiles 256` caps work per slide.

### Flags

| flag                 | default            | what it does                                          |
| -------------------- | ------------------ | ----------------------------------------------------- |
| `--slide-dir`        | required           | directory of WSIs, searched recursively                |
| `--out-dir`          | required           | where the per-slide `.h5` files go                     |
| `--thumbnail-dir`    | `<out-dir>/thumbnails` | where the masked thumbnails go                    |
| `--ext`              | `.svs .tif .tiff .ndpi .scn .mrxs .bif .svslide` | slide extensions to look for |
| `--batch-size`       | 64                 | tiles per forward pass                                 |
| `--num-tile-readers` | 8                  | worker processes decoding tiles                        |
| `--queue-depth`      | 8                  | tile batches buffered ahead of the GPU                 |
| `--precision`        | `fp16`             | autocast dtype (`fp16`/`bf16`/`fp32`)                  |
| `--gzip-level`       | 4                  | `features` compression; 0 disables                     |
| `--thumbnail-width`  | 2048               | masked thumbnail width in px                           |
| `--max-tiles`        | 0 (all)            | cap tiles per slide, for smoke tests                   |
| `--overwrite`        | off                | re-embed slides that already have an `.h5`             |

Nothing here is tuned. Reading tiles is usually the bottleneck before the GPU is, so
`--num-tile-readers` and `--batch-size` are the first knobs to reach for.

fp16 autocast is what the Virchow model card prescribes; embeddings still come out fp32 because the
model's final op is a LayerNorm that runs in mixed precision.

## Output

`<out-dir>/<slide>.h5`, row-aligned so that `features[i]` belongs to the tile at `grid[i]`/`loc[i]`:

| dataset    | shape       | dtype     | notes                                        |
| ---------- | ----------- | --------- | -------------------------------------------- |
| `features` | `(N, 2560)` | `float32` | gzip compressed, written incrementally       |
| `grid`     | `(N, 2)`    | `int32`   | tile grid indices                            |
| `loc`      | `(N, 2)`    | `int32`   | base-level pixel coords (`-1` if unavailable)|

`N` is the number of tiles that passed Otsu tissue filtering, so it is smaller than the full tile
grid and varies by slide.

Root attributes carry the provenance needed to interpret the file later: `slide`, `slide_path`,
`encoder`, `encoder_source`, `embed_dim`, `n_tiles`, `tile_px`, `tile_um`, `stride_div`, `qc`,
`precision`, `mpp`, `slide_dimensions`, `grid_shape`, `thumbnail`, `created_utc`, `torch_version`,
`slideflow_version`.

`<out-dir>/thumbnails/<slide>.png` is the slide thumbnail with everything Otsu rejected dimmed and
tinted, for eyeballing tissue detection across a cohort.

## Files

* [`Dockerfile`](Dockerfile) — image, including the build-time weight pull
* [`virchow.py`](virchow.py) — the model wrapper (normalization + cls/patch-mean concat baked in)
* [`download_model.py`](download_model.py) — build-time fetch and CPU smoke test
* [`embed_wsi.py`](embed_wsi.py) — the CLI that walks a slide directory

## Notes for the first cluster run

* Untested end to end — this machine has no GPU, no slides, and no room to build the image.
* If slideflow's tile dicts turn out not to carry `loc`, the run logs a warning per slide and `loc`
  is filled with `-1`; the grid indices are unaffected.
* Under Apptainer, `$HOME` is often read-only or remapped; `HF_HOME`, `MPLCONFIGDIR` and
  `NUMBA_CACHE_DIR` are all set to writable paths in the image for that reason:

    ```bash
    apptainer build virchow-embed.sif docker-daemon://virchow-embed:latest
    apptainer run --nv -B /path/to/slides:/slides,/path/to/out:/out virchow-embed.sif \
      --slide-dir /slides --out-dir /out
    ```
