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
**A CUDA device is required**; there is no CPU path.

### Throughput

The pipeline is built to embed a TCGA-sized corpus — ~155 M tiles across ~11.7 k slides — in a
single pass on one GPU, and sustains **~1200 tiles/s on an H200** at the defaults. Three things get
it there, and all three are load-bearing:

* **Whole batches come out of the dataset**, not out of the `DataLoader`'s collation. One dataset
  item is one already-stacked batch (`batch_size=None` on the loader), and workers split *batch*
  indices in contiguous runs. Letting the loader collate per worker makes every worker emit its own
  short trailing batch, and that ragged shape stream costs more than the workers buy.
* **Virchow is always `torch.compile`d** with `dynamic=False`, which is only possible because of the
  above. The slide's one genuine short batch is padded up to `batch_size` and the result sliced back,
  so the compiled graph only ever sees one shape.
* **Two reader pools alternate.** While slide N is on the GPU, a background thread runs slide N+1's
  whole setup — open, Otsu, tile grid, thumbnail — and fills the other pool, so the GPU never waits
  for a slide to start.

There is no eager mode and no flag to disable compilation: an uncompiled run does the same work in
about half again the time, so a compile that fails is fatal rather than something to fall back from
— silently taking 54 h over a corpus that should take 35 h is the worse outcome. Triton needs a C
compiler on `PATH`, which is why the image installs `gcc`.

> **Triton here means the kernel language**, [triton-lang/triton](https://github.com/triton-lang/triton),
> the compiler backend TorchInductor emits GPU kernels in — *not*
> [Triton Inference Server](https://github.com/triton-inference-server/server), which is the
> NVIDIA model server used over in [kserve-demo/](../../kserve-demo). Two unrelated projects, same
> name. Nothing is served here; the model is compiled in-process.

### Slides that need special handling

Two classes of file in TCGA break a naive reader, and both are handled in `_reader.py`:

* **No `MPP` in the Aperio header.** The header value is authoritative; where it is missing, `AppMag`
  is used instead (20x → 0.5 MPP), but only for files large enough to be diagnostic slides. Macro
  images, label images and reduced exports are rejected rather than embedded at a fabricated
  magnification. `mpp_source` in the output records which path was taken.
* **Too few pyramid levels.** Tissue detection wants a level near 64x downsample; a slide with only
  one level would have to decode its entire base image to get there — up to 173 GB of RGB on the
  largest TCGA file. Above a byte budget, that level is instead streamed segment by segment into a
  reduced canvas, holding one decoded segment at a time. Ordinary slides keep the direct path.

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
docker run -it --rm --gpus '"device=0"' --shm-size=12g \
  -e TORCHINDUCTOR_CACHE_DIR=/inductor -v /path/to/inductor-cache:/inductor \
  -v /path/to/slides:/slides:ro \
  -v /path/to/out:/out \
  virchow-embed --slide-dir /slides --out-dir /out
```

`--slide-dir` is searched recursively. Slides that already have an `.h5` in `<out-dir>/embeddings`
are skipped unless `--overwrite` is passed, so an interrupted run can simply be restarted. A slide
that fails — to open, to parse, or to embed — is logged and the run moves on, exiting non-zero at
the end.

`--shm-size` is not optional. Tiles cross from the reader processes to the parent through shared
memory, and there are two pools of readers, so budget roughly

```
2 × num-tile-readers × prefetch-factor × batch-size × 224 × 224 × 3 B
```

which is about **10 GB at the defaults**. Docker's 64 MB default gets you a bus error.

`TORCHINDUCTOR_CACHE_DIR` on a persistent volume is worth setting: it amortises the ~60 s
compilation across runs. Without it, every run pays it again.

### Flags

| flag                 | default                | what it does                             |
| -------------------- | ---------------------- | ---------------------------------------- |
| `--slide-dir`        | `/slides`              | directory of WSIs, searched recursively  |
| `--out-dir`          | `/out`                 | parent of `embeddings/` and `thumbnails/` |
| `--exts`             | `.svs`                 | slide extensions to look for             |
| `--batch-size`       | 512                    | tiles per forward pass                   |
| `--num-tile-readers` | 32                     | worker processes reading tiles, **per pool**; there are two |
| `--prefetch-factor`  | 2                      | tile batches each reader buffers ahead of the GPU |
| `--thumbnail-width`  | 2048                   | masked thumbnail width in px             |
| `--overwrite`        | off                    | re-embed slides that already have an `.h5` |

The defaults are a set, not independent knobs. More readers cost shared memory linearly and stop
helping once the GPU is saturated — and past a point they have nothing to do, since a typical slide
is only ~27 batches at `--batch-size 512` and each reader takes a contiguous run of them. What sets
how many you need is the codec: JPEG 2000 slides (about 30% of TCGA) decode 2–3x slower than
baseline JPEG, and 16 readers per pool is not enough to keep the GPU fed on those.

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

Rows are **not** in slide raster order. Tiles are read concurrently and arrive in whichever order
the readers finish, so anything comparing two files should sort on `loc` first (`np.lexsort`) rather
than assume row-major.

Root attributes record how the file was produced: `slide`, `slide_path`, `encoder`,
`encoder_source`, `embed_dim`, `n_tiles`, `tile_px`, `tile_um`, `qc`, `precision`, `mpp`,
`mpp_source`, `slide_dimensions`, `grid_shape`, `thumbnail`, `created_utc`, `torch_version`.

`mpp_source` is `header` when the Aperio header carried an `MPP` field, and otherwise names the
`AppMag` it was derived from — the slides worth knowing about when reviewing a finished run.
