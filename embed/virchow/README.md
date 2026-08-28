# Virchow Tile Embeddings

Embeds whole-slide images with [Virchow](https://huggingface.co/paige-ai/Virchow). Point the
container at a directory of WSIs and it writes, per slide:

* `<out-dir>/<slide>.h5` — one embedding per tile, with tile coordinates
* `<out-dir>/thumbnails/<slide>.png` — the slide thumbnail with non-tissue dimmed out

Slides are tiled with [slideflow](https://github.com/jamesdolezal/slideflow) into 224 px tiles at
20x (0.5 MPP) under Otsu tissue detection, and each tile is embedded on the local GPU. Tiles are
held in memory and streamed to the GPU; they are never written to disk.

## Build

Virchow is gated, so the build needs a Hugging Face account with access to
[paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow). The weights are pulled into the image
at build time, and the token is passed as a BuildKit secret:

```bash
HF_TOKEN=hf_xxx docker build --secret id=hf_token,env=HF_TOKEN -t virchow-embed embed/virchow
```

The running container never contacts huggingface.co.

## Run

```bash
docker run --rm --gpus '"device=0"' \
  -v /path/to/slides:/slides:ro \
  -v /path/to/out:/out \
  virchow-embed --slide-dir /slides --out-dir /out
```

`--slide-dir` is searched recursively. Slides that already have an `.h5` in `--out-dir` are skipped
unless `--overwrite` is passed, so an interrupted run can simply be restarted. A slide that fails is
logged and the run moves on, exiting non-zero at the end.

`--max-tiles 256` caps the work per slide, for a quick check that the whole path works.

### Flags

| flag                 | default                | what it does                             |
| -------------------- | ---------------------- | ---------------------------------------- |
| `--slide-dir`        | required               | directory of WSIs, searched recursively  |
| `--out-dir`          | required               | where the per-slide `.h5` files go       |
| `--thumbnail-dir`    | `<out-dir>/thumbnails` | where the masked thumbnails go           |
| `--ext`              | `.svs .tif .tiff .ndpi .scn .mrxs .bif .svslide` | slide extensions to look for |
| `--batch-size`       | 64                     | tiles per forward pass                   |
| `--num-tile-readers` | 8                      | worker processes decoding tiles          |
| `--queue-depth`      | 8                      | tile batches buffered ahead of the GPU   |
| `--device`           | `cuda` if available    | torch device to embed on                 |
| `--gzip-level`       | 4                      | `features` compression; 0 disables       |
| `--thumbnail-width`  | 2048                   | masked thumbnail width in px             |
| `--max-tiles`        | 0 (all)                | cap tiles per slide                      |
| `--overwrite`        | off                    | re-embed slides that already have an `.h5` |
| `--log-every`        | 50                     | log progress every N batches             |

Reading tiles is usually the bottleneck before the GPU is, so `--num-tile-readers` and
`--batch-size` are the first knobs to reach for.

## Output

`<out-dir>/<slide>.h5` holds three row-aligned datasets, so that `features[i]` is the embedding of
the tile at `grid[i]` / `loc[i]`:

| dataset    | shape       | dtype     | notes                                  |
| ---------- | ----------- | --------- | -------------------------------------- |
| `features` | `(N, 2560)` | `float32` | gzip compressed                        |
| `grid`     | `(N, 2)`    | `int32`   | tile grid indices                      |
| `loc`      | `(N, 2)`    | `int32`   | base-level pixel coordinates           |

`N` is the number of tiles that passed tissue filtering, so it is smaller than the full tile grid
and varies by slide. Embeddings are fp32: inference runs under fp16 autocast, but Virchow's final
op is a LayerNorm that runs in mixed precision.

Root attributes record how the file was produced: `slide`, `slide_path`, `encoder`,
`encoder_source`, `embed_dim`, `n_tiles`, `tile_px`, `tile_um`, `stride_div`, `qc`, `precision`,
`mpp`, `slide_dimensions`, `grid_shape`, `thumbnail`, `created_utc`, `torch_version`,
`slideflow_version`.
