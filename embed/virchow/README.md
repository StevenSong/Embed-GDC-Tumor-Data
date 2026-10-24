# Virchow Tile Embeddings

Embeds whole-slide images with [Virchow](https://huggingface.co/paige-ai/Virchow). Point the
container at a directory of WSIs — Aperio `.svs` (TCGA) or DICOM whole-slide series (CPTAC) — and
it writes, per slide:

* `<out-dir>/embeddings/<slide>.h5` — one embedding per tile, with tile coordinates
* `<out-dir>/thumbnails/<slide>.png` — the slide thumbnail with everything that was not
  embedded dimmed out

A second step, `python -m embedder.aggregate`, then collects every slide's mean embedding into one
`<out-dir>/hist.h5` — see [Slide-level `hist.h5`](#slide-level-histh5).

Slides are read directly off the Aperio pyramid with [tifffile](https://github.com/cgohlke/tifffile)
(see `src/embedder/_reader.py`) and cut into 224 px tiles at 20x (0.5 MPP). Tissue detection
thresholds **darkness** — inverted grayscale — with Otsu, closes the result, and drops connected
components smaller than a physical area floor; a tile is kept when at least 10% of its footprint is
tissue. Each tile is embedded on the local GPU; tiles are held in memory and streamed to the GPU,
never written to disk.

Two choices there are load-bearing and both are counterintuitive, so they are argued from
measurements in the `_reader.py` docstrings rather than restated here: **darkness rather than HSV
saturation**, because faintly stained slides have no second saturation mode for Otsu to find and it
splits inside the tissue instead (8.0 mm² kept of a 92.7 mm² specimen on the worst corpus slide);
and **occupancy rather than the centre pixel**, because a tile is only ~15 mask pixels across at
the segmentation level, so a centre test point-samples 1 pixel in ~236.

**Only Aperio-style tiled SVS pyramids, and the DICOM-TIFF series converted from them, are
supported.** Striped TIFFs and other vendor formats (`.mrxs`, `.ndpi`, `.bif`) are rejected with
`UnsupportedSlideError` rather than half-supported. **A CUDA device is required**; there is no CPU
path.

### DICOM slides (CPTAC)

[IDC](https://portal.imaging.datacommons.cancer.gov/) distributes the CPTAC pathology as DICOM, one series directory per slide holding one `.dcm`
per pyramid level plus a label, overview and thumbnail. Every one of those files is a
*dual-personality* DICOM-TIFF — PixelMed converted them from the original `.svs`, so each is also
a valid tiled TIFF whose tiles are the DICOM frames, still in the Aperio codecs. The reader
therefore decodes them on exactly the SVS path; only the metadata comes from the DICOM header
(via pydicom): which files are pyramid levels (`ImageType` `VOLUME`), the pixel size
(`PixelSpacing`, which agrees with the Aperio `MPP` the header still carries), and the slide's name.

A DICOM slide is named by its `ContainerIdentifier` — `C3L-04408-26`, exactly the stem of the
`.svs` it was converted from — since the series directory's own name is a UID. Discovery reads one
header per directory, for that name. A series that is not a whole-slide image is rejected when it is
opened, so the slide directory should hold slides only — the download this repo builds is tumor
H&E slides and nothing else, see [Download data](../../README.md#cptac-slides).

### Throughput

The pipeline is built to embed a TCGA-sized corpus — ~184 M tiles across ~11.6 k slides — and
sustains **~1200 tiles/s on an H200** at the defaults, so one GPU is ~41 h and four are ~10 h.
Three things get it there, and all three are load-bearing:

* **Whole batches come out of the dataset**, not out of the `DataLoader`'s collation. One dataset
  item is one already-stacked batch (`batch_size=None` on the loader), and workers split *batch*
  indices in contiguous runs. Letting the loader collate per worker makes every worker emit its own
  short trailing batch, and that ragged shape stream costs more than the workers buy.
* **Virchow is always `torch.compile`d** with `dynamic=False`, which is only possible because of the
  above. The slide's one genuine short batch is padded up to `batch_size` and the result sliced back,
  so the compiled graph only ever sees one shape.
* **Two reader pools alternate.** While slide N is on the GPU, a background thread runs slide N+1's
  whole setup — open, tissue detection, tile grid, thumbnail — and fills the other pool, so the GPU
  never waits for a slide to start.

A fourth thing is load-bearing and is *not* a code property: **`--shm-size` must be generous.** Tiles
cross from the readers to the parent through `/dev/shm`, and a tmpfs that merely runs tight does not
fail loudly — it collapses throughput 3–6x on any slide needing more batches than the pools can
buffer. On the last corpus run that was 16% of slides consuming 57% of the wall clock, at 130–400
tiles/s against a 1240 tiles/s ceiling. Give it `64g`; the measurable working set is ~10 GB and the
memory is free.

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

Two files are excluded outright, by name, in `__main__.py`: `TCGA-5P-A9KA` and `TCGA-5P-A9KC` are
single-level pyramids of ~210k × 270k px, so the streaming path above would decode ~800 k JPEG 2000
segments serially per slide — hours each — and they exceed the reader's tile cap regardless. Two
files out of 11,684 do not justify the machinery to handle them.

## Build

Virchow is gated, so the build needs a Hugging Face account with access to
[paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow). The weights are pulled into the image
at build time, and the token is passed as a BuildKit secret:

```bash
HF_TOKEN=hf_xxx docker build --secret id=hf_token,env=HF_TOKEN \
  -t virchow-embedder:latest embed/virchow
```

The running container never contacts huggingface.co.

**The source is baked into the image** (`COPY src/ /app/src/`), so any change under `src/` needs a
rebuild before it takes effect — a container built from older source will fail on new flags with an
`unrecognized arguments` error from argparse rather than anything more helpful. Rebuilds after the
first are cheap: the weight download and `pip install` layers are cached, and only the source copy
is invalidated. `virchow-embedder:latest` is the tag `embed-corpus.sbatch` runs by default; keep
them in step or pass `IMAGE=` to the launcher.

## Run

```bash
docker run -it --rm --gpus '"device=0"' --shm-size=64g \
  -e TORCHINDUCTOR_CACHE_DIR=/inductor -v /path/to/inductor-cache:/inductor \
  -v /path/to/slides:/slides:ro \
  -v /path/to/out:/out \
  virchow-embedder:latest --slide-dir /slides --out-dir /out
```

`--slide-dir` is searched recursively. Slides that already have an `.h5` in `<out-dir>/embeddings`
are skipped unless `--overwrite` is passed, so an interrupted run can simply be restarted — but that
same behaviour means **a re-run with changed QC must start from an empty `embeddings/`**, or the
stale files are kept and you get a corpus that is half one config and half the other. Move the old
directory aside rather than deleting it until the new run is verified. A slide that fails — to open,
to parse, or to embed — is logged and the run moves on, exiting non-zero at the end.

On a shared cluster, book the GPU through the scheduler and use the index it assigns; picking an
idle-looking device out of `nvidia-smi` collides with whatever gets scheduled onto it next.

`--shm-size` is not optional, and sizing it to the working set is not enough. Tiles cross from the
reader processes to the parent through shared memory, and there are two pools of readers, so the
resident working set is roughly

```
2 × num-tile-readers × prefetch-factor × batch-size × 224 × 224 × 3 B
```

which is about **10 GB at the defaults**. Docker's 64 MB default gets you a bus error, but sizing it
at 16 GB gets you something worse — see the throughput note above. Use **64g**.

To use several GPUs, run one container per GPU against the *same* `--slide-dir` and `--out-dir`,
varying only the device and the shard:

```bash
for i in 0 1 2 3; do
  docker run -d --rm --gpus "\"device=$i\"" --shm-size=64g \
    -v /path/to/slides:/slides:ro -v /path/to/out:/out \
    virchow-embedder:latest --slide-dir /slides --out-dir /out --shard "$i" --shards 4
done
```

Sharding the slide list rather than splitting the input directory is what keeps resume working: every
container skips any slide that already has an `.h5`, so a shard that dies can be restarted, or the
whole run re-submitted with a different `--shards`, without reshuffling files on disk. A plain stride
balances to within 1% over a corpus this size, so there is nothing to tune.

### On Slurm

`run-corpus.sh` submits a whole run: `SHARDS` jobs of `embed-corpus.sbatch`, one GPU each, then one
`aggregate.sbatch` that depends on all of them (`afterok`) and writes `hist.h5`:

```bash
SLIDE_DIR=/path/to/slides OUT_DIR=/path/to/out SHARDS=4 embed/virchow/run-corpus.sh
```

The aggregate job only runs if every shard exits 0. A shard that fails a slide or hits the time
limit cancels it (`--kill-on-invalid-dep=yes`) rather than leaving it pending; fix the cause and run
the script again. The embed jobs skip slides already embedded, so a rerun only redoes what is
missing, and the aggregate job always rebuilds `hist.h5` from scratch. `IMAGE` and `SHM_SIZE` in
the environment pass through to the jobs.

`TORCHINDUCTOR_CACHE_DIR` on a persistent volume is worth setting: it amortises the ~60 s
compilation across runs. Without it, every run pays it again.

### Flags

| flag                 | default                | what it does                             |
| -------------------- | ---------------------- | ---------------------------------------- |
| `--slide-dir`        | `/slides`              | directory of WSIs, searched recursively  |
| `--out-dir`          | `/out`                 | parent of `embeddings/` and `thumbnails/` |
| `--exts`             | `.svs .dcm`            | slide extensions to look for; `.dcm` means DICOM series directories |
| `--batch-size`       | 512                    | tiles per forward pass                   |
| `--num-tile-readers` | 32                     | worker processes reading tiles, **per pool**; there are two |
| `--prefetch-factor`  | 2                      | tile batches each reader buffers ahead of the GPU |
| `--thumbnail-width`  | 2048                   | masked thumbnail width in px             |
| `--shards`           | 1                      | split the slide list across this many processes |
| `--shard`            | 0                      | which shard this process takes, 0-based  |
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
| `features` | `(N, 2560)` | `float16` | uncompressed                                 |
| `grid`     | `(N, 2)`    | `int32`   | `(col, row)` in the full tile grid           |
| `loc`      | `(N, 2)`    | `int32`   | base-level `(x, y)` of the tile's top-left   |
| `mean`     | `(2560,)`   | `float32` | column average of `features`, the slide embedding |

The two coordinate datasets are redundant by construction — `loc == grid * extract_px`, where
`extract_px = round(tile_px * 0.5 / mpp)` is how many base-level pixels a tile covers — which makes
a cheap integrity check on a finished file.

`N` is the number of tiles that passed tissue filtering, so it is smaller than the full tile grid
and varies by slide. Embeddings are stored fp16, which is the precision inference actually produces
— Virchow runs under fp16 autocast and only its final LayerNorm is mixed-precision. The slide-level
`mean` is accumulated in float64 from the full-precision batch before that cast, so it does not
inherit the storage precision. `feature_dtype` in the root attributes records this; files without
it are the older fp32 ones.

The file is written to `.<slide>.h5.tmp` and renamed only once `mean` and the attributes are in
place, so an interrupted run never leaves a plausible-looking `.h5` behind.

Rows are **not** in slide raster order. Tiles are read concurrently and arrive in whichever order
the readers finish, so anything comparing two files should sort on `loc` first (`np.lexsort`) rather
than assume row-major.

Root attributes record how the file was produced: `slide`, `slide_path`, `encoder`,
`encoder_source`, `embed_dim`, `n_tiles`, `tile_px`, `tile_um`, `qc`, `qc_min_tissue_mm2`,
`qc_min_occupancy`, `precision`, `feature_dtype`, `mpp`, `mpp_source`, `slide_dimensions`,
`grid_shape`, `thumbnail`, `created_utc`, `torch_version`. DICOM slides add `case_id` (the
header's `PatientID`); `slide` is the `ContainerIdentifier` and `slide_path` the series directory.
Everything else about a CPTAC slide is in the metadata table the download manifest was built with.

`qc` names the tissue-detection config the tile set came from — `gray-otsu-occupancy` is the
current one, `otsu` the older saturation-and-centre-pixel scheme — alongside `qc_min_tissue_mm2`
and `qc_min_occupancy`. A tile's *embedding* never depends on QC, only which tiles exist.

`mpp_source` is `header` when the Aperio header carried an `MPP` field, `PixelSpacing` for a DICOM
slide, and otherwise names the `AppMag` it was derived from — the slides worth knowing about when
reviewing a finished run.

## Slide-level `hist.h5`

`python -m embedder.aggregate --emb-dir <out-dir>/embeddings --output-h5 <out-dir>/hist.h5` (run
by `aggregate.sbatch`, no GPU, a few minutes) collects every slide's `mean` into one file in the
layout of `embed/embed_hist_uni2.py` from
[multimodal-cancer-modeling](https://github.com/StevenSong/multimodal-cancer-modeling), so its
survival notebooks read it unchanged:

```
hist.h5
  /<case>/          one group per case, by TCGA barcode, e.g. TCGA-OR-A5J1
     <slide>        float32 (2560,), the slide's `mean`; named by file stem
```

A DICOM slide is filed under its `case_id` attribute instead (`C3L-04408`), since a CPTAC slide
barcode does not say which part of it is the case.

A case with several slides keeps them all, and averaging across a case is left to the reader (the
notebooks take an unweighted mean). The vector is copied from `mean` rather than recomputed from
`features`, which is both ~10 KB a slide instead of the whole corpus and more exact, since `mean`
was accumulated before the fp16 cast. Each slide dataset carries `n_tiles`, `mpp` and
`mpp_source`. The root carries the corpus-wide config — `encoder`, `encoder_source`, `embed_dim`,
`precision`, `tile_px`, `tile_um`, `qc`, `qc_min_tissue_mm2`, `qc_min_occupancy` — plus
`aggregation`, `emb_dir`, `n_slides`, `n_cases` and `created_utc`.

Every `.h5` in `--emb-dir` goes in; there is no cohort filtering. The input is checked as a whole
first, and any bad file fails the run without writing anything: no `mean`, a non-finite `mean`, no
case to file it under (no `case_id`, and a name that is not a TCGA barcode), or a corpus-wide
attribute that differs from the other files (a stale `qc` from an older run, say). The file is written to `.hist.h5.tmp` and renamed into place,
and an existing `hist.h5` is kept unless `--overwrite` is passed.
