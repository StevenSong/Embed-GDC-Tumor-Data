# Virchow Tile Embeddings

Embeds whole-slide images with [Virchow](https://huggingface.co/paige-ai/Virchow). Point the
container at a directory of WSIs and it writes, per slide:

* `<out-dir>/embeddings/<slide>.h5` — one embedding per tile, with tile coordinates
* `<out-dir>/thumbnails/<slide>.png` — the slide thumbnail with everything that was not
  embedded dimmed out

Slides are read directly off the Aperio pyramid with [tifffile](https://github.com/cgohlke/tifffile)
(see `src/embedder/_reader.py`) and cut into 224 px tiles at 20x (0.5 MPP). Tissue detection follows
CLAM (Lu et al., *Nat Biomed Eng* 2021) — HSV saturation, median blur, Otsu, morphological closing,
then a contour filter with hole filling — and a tile is kept when its centre lands on tissue. Each
tile is embedded on the local GPU; tiles are held in memory and streamed to the GPU, never written
to disk.

**Only Aperio-style tiled SVS pyramids are supported.** Striped TIFFs and other vendor formats
(`.mrxs`, `.ndpi`, `.bif`) are rejected with `UnsupportedSlideError` rather than half-supported.

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
docker run -it --rm --gpus '"device=0"' \
  -v /path/to/slides:/slides:ro \
  -v /path/to/out:/out \
  virchow-embed --slide-dir /slides --out-dir /out
```

`--slide-dir` is searched recursively. Slides that already have an `.h5` in `<out-dir>/embeddings`
are skipped unless `--overwrite` is passed, so an interrupted run can simply be restarted. A slide
that fails is logged and the run moves on, exiting non-zero at the end.

Tiles are read by a `DataLoader` whose workers are separate processes, so the container needs
`--shm-size` raised well above Docker's 64 MB default — roughly
`batch-size × 224 × 224 × 3 B × prefetch-factor × num-tile-readers`, about 1.2 GB at the defaults.

### Flags

| flag                 | default                | what it does                             |
| -------------------- | ---------------------- | ---------------------------------------- |
| `--slide-dir`        | `/slides`              | directory of WSIs, searched recursively  |
| `--out-dir`          | `/out`                 | parent of `embeddings/` and `thumbnails/` |
| `--exts`             | `.svs`                 | slide extensions to look for             |
| `--batch-size`       | 128                    | tiles per forward pass                   |
| `--num-tile-readers` | 16                     | worker processes reading tiles; 0 reads inline |
| `--prefetch-factor`  | 4                      | tile batches each reader buffers ahead of the GPU |
| `--device`           | `cuda` if available    | torch device to embed on                 |
| `--thumbnail-width`  | 2048                   | masked thumbnail width in px             |
| `--overwrite`        | off                    | re-embed slides that already have an `.h5` |

`--num-tile-readers` and `--prefetch-factor` are the first knobs to reach for: the GPU should stay
saturated, and if it does not, the readers are not keeping up.

## Output

`<out-dir>/embeddings/<slide>.h5` holds three row-aligned datasets, so that `features[i]` is the
embedding of the tile at `grid[i]` / `loc[i]`, plus a slide-level summary:

| dataset    | shape       | dtype     | notes                                        |
| ---------- | ----------- | --------- | -------------------------------------------- |
| `features` | `(N, 2560)` | `float32` | uncompressed                                 |
| `grid`     | `(N, 2)`    | `int64`   | `(col, row)` in the full tile grid           |
| `loc`      | `(N, 2)`    | `int64`   | base-level `(x, y)` of the tile's top-left   |
| `mean`     | `(2560,)`   | `float32` | column average of `features`, the slide embedding |

The two coordinate datasets are redundant by construction — `loc == grid * extract_px`, where
`extract_px = round(tile_px * 0.5 / mpp)` is how many base-level pixels a tile covers — which makes
a cheap integrity check on a finished file.

`N` is the number of tiles that passed tissue filtering, so it is smaller than the full tile grid
and varies by slide. Embeddings are fp32: inference runs under fp16 autocast, but Virchow's final
op is a LayerNorm that runs in mixed precision.

The file is written to `.<slide>.h5.tmp` and renamed only once `mean` and the attributes are in
place, so an interrupted run never leaves a plausible-looking `.h5` behind.

Root attributes record how the file was produced: `slide`, `slide_path`, `encoder`,
`encoder_source`, `embed_dim`, `n_tiles`, `tile_px`, `tile_um`, `qc`, `precision`, `mpp`,
`slide_dimensions`, `grid_shape`, `thumbnail`, `created_utc`, `torch_version`.
