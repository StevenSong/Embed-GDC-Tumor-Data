"""Tile SVS or DICOM slides and embed every tile with Virchow, one HDF5 per slide.

Slides are tiled by `_reader.Slide` -- tifffile straight onto the Aperio pyramid, with
darkness-Otsu tissue detection and occupancy-based tiling -- at 224 px / 20x, and
Virchow runs in this process on the local GPU. Tiles live in memory and stream straight
to the GPU -- the only things written to disk are the per-slide HDF5 and its thumbnail.

    python -m embedder --slide-dir /data/slides --out-dir /data/embeddings

Two reader pools alternate: while slide N is on the GPU, a background thread runs
slide N+1's setup (open, tissue detection, grid, thumbnail) and fills the other pool, so
the GPU never waits for a slide to start. That costs shared memory --
`2 * num_tile_readers * prefetch_factor * batch_size * 150 KB`, about 10 GB at the
defaults -- so run with `--shm-size=64g`. Too small is not just a bus error: a tmpfs
that merely runs *tight* collapses throughput 3-6x on slides with more batches than the
pools can buffer, which is where most of the corpus wall time went on the last run.

Virchow is always `torch.compile`d, with no eager path and no flag to turn it off
-- an uncompiled run does the same work in 54h instead of 35h, so a compile that
fails is fatal rather than something to shrug off. Triton needs a C compiler on
PATH, which is why the image installs gcc.

Two things are written next to the embeddings for diagnosis rather than for use:
`stats.jsonl` gets one record per slide -- the wait/GPU/h5 split of its wall
time, the setup timings, and what the readers did -- interleaved with `kind:
"sys"` samples of node CPU, our own CPU, and disk throughput. Read that, not the
log, when asking where the time went. The log itself is plain unwrapped text and
progress bars turn themselves off when stderr is not a terminal.

Placement across GPUs is the caller's job, but the slide list is not: run one container
per GPU against the *same* --slide-dir and --out-dir, with CUDA_VISIBLE_DEVICES and
`--shard i --shards n` varying between them. Sharding here rather than by directory is
what keeps the resume-by-skipping-existing-.h5 behaviour working across all of them.
"""

import sys
import time
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ._dataset import TileDataset
from ._logging import StatsLog, progress_disabled, setup_logging
from ._model import TARGET_MPP, TILE_PX, WrappedVirchow
from ._pipeline import compile_model, embed_slide, prepare_slide
from ._reader import NoTissueError, SlideRef, UnsupportedSlideError, find_slides
from ._sysmon import SysMonitor

# .dcm finds DICOM whole-slide series, one slide per series directory
SLIDE_EXTS = [".svs", ".dcm"]
# one pool feeds the GPU while the other fills for the next slide
N_READER_POOLS = 2

# Single-level pyramids of 212906x271605 and 190464x251630 with no downsampled
# level to segment on, so tissue detection would have to decode ~800k JPEG 2000
# segments serially to build one reduced canvas -- hours per slide, and they
# exceed the reader's tile cap anyway. Two files out of 11,684 do not justify
# the machinery to handle them; they are skipped by name.
EXCLUDED_SLIDES = ("TCGA-5P-A9KA", "TCGA-5P-A9KC")


def make_reader_pool(
    *,  # enforce kwargs
    slide_paths: list[Path],
    batch_size: int,
    prefetch_factor: int,
    num_tile_readers: int,
) -> tuple[TileDataset, DataLoader]:
    """A (dataset, loader) pair whose workers read one slide at a time.

    The dataset yields whole pre-stacked batches, so the loader does no batching
    of its own -- see `TileDataset` for why the shape stream has to be static.
    Tile geometry comes from the model so the parent's Slide in `_pipeline` and
    the workers' Slides can never disagree about the grid.

    The workers are forked here, on an empty slide, rather than lazily on the
    first real one. `main` calls this before the model exists, and a reader that
    forks before CUDA is initialised inherits no CUDA state -- in particular none
    of the CUDA graphs max-autotune captures, which a forked child cannot destroy
    and warns about all the way out. It also keeps Virchow's weights out of every
    worker's page tables.
    """
    dataset = TileDataset(
        slide_paths=slide_paths,
        tile_px=TILE_PX,
        target_mpp=TARGET_MPP,
        batch_size=batch_size,
        num_workers=num_tile_readers,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,  # the dataset batches; see TileDataset
        drop_last=False,
        num_workers=num_tile_readers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    # get the persistent dataloader workers spun up by faking a slide with no tiles
    dataset.set_next_slide_tiles(
        slide_path=slide_paths[0],
        masked_coords=np.empty((0, 2), dtype=np.int32),
    )
    for _ in loader:
        # should not yield anything, error if it does
        raise AssertionError("empty slide yielded a batch")

    return dataset, loader


def main(args):
    logger = setup_logging()

    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.exts}
    slides = find_slides(args.slide_dir, exts)
    slides = [s for s in slides if not any(x in s.name for x in EXCLUDED_SLIDES)]
    # two slides with one name would write one .h5, and the second would then be
    # skipped as already embedded; refuse before either happens
    dupes = sorted(n for n, c in Counter(s.name for s in slides).items() if c > 1)
    if dupes:
        logger.error("%d slide names are not unique: %s", len(dupes), ", ".join(dupes))
        return 1
    # Shard before anything else, so each GPU's process sees only its own slides
    # and the two of them never race to write the same .h5. A plain stride is
    # enough: over 11.6k slides it lands within 1% of a tile-count-balanced
    # split, so there is nothing to gain from predicting per-slide cost.
    if args.shards > 1:
        slides = slides[args.shard :: args.shards]
    n_slides = len(slides)

    if not slides:
        logger.error("no slides matching %s under %s", args.exts, args.slide_dir)
        return 1
    embed_dir = args.out_dir / "embeddings"
    thumb_dir = args.out_dir / "thumbnails"
    embed_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    if args.shards > 1:
        logger.info(
            "shard %d/%d: %d slides under %s",
            args.shard,
            args.shards,
            n_slides,
            args.slide_dir,
        )
    else:
        logger.info("found %d slides under %s", n_slides, args.slide_dir)

    # resolve skips up front: the prefetch below runs one slide ahead, and it can
    # only do that against a list of slides it is actually going to embed
    todo: list[SlideRef] = []
    for slide in slides:
        out_path = embed_dir / f"{slide.name}.h5"
        if out_path.exists() and not args.overwrite:
            logger.info("%s exists, skipping", out_path.name)
            continue
        todo.append(slide)
    if not todo:
        logger.info("nothing to do: all %d slides already embedded", n_slides)
        return 0
    logger.info("embedding %d/%d slides", len(todo), n_slides)

    # NOTE: the reader pools are built first, and deliberately so -- they fork
    # their workers, which must happen before the model initialises CUDA
    if args.num_tile_readers <= 0:
        raise ValueError(
            f"Must specify non-zero positive number of tile readers, got {args.num_tile_readers}"
        )
    pools = [
        make_reader_pool(
            slide_paths=[s.path for s in todo],
            batch_size=args.batch_size,
            num_tile_readers=args.num_tile_readers,
            prefetch_factor=args.prefetch_factor,
        )
        for _ in range(N_READER_POOLS)
    ]
    logger.info(
        "%d reader pools x %d workers ready", N_READER_POOLS, args.num_tile_readers
    )

    if not torch.cuda.is_available():
        raise NotImplementedError("Non-cuda accelerator not supported")
    device = torch.device("cuda")
    model = WrappedVirchow().eval().to(device)
    logger.info("virchow ready on %s", device)
    model = compile_model(
        logger=logger,
        model=model,
        batch_size=args.batch_size,
        device=device,
    )

    # one thread, so preparations stay ordered and only ever one is in flight
    prep_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="slide-prep")

    def submit(i: int) -> Future | None:
        """Start slide `i`'s setup and reader fill on the background thread."""
        if i >= len(todo):
            return None
        # alternating pools: slide i+1 fills the pool slide i is not draining
        tile_ds, tile_dl = pools[i % N_READER_POOLS]
        return prep_pool.submit(
            prepare_slide,
            logger=logger,
            slide_path=todo[i].path,
            thumbnail_path=thumb_dir / f"{todo[i].name}.png",
            thumbnail_width=args.thumbnail_width,
            out_path=embed_dir / f"{todo[i].name}.h5",
            tile_ds=tile_ds,
            tile_dl=tile_dl,
        )

    # a slide the reader refuses (not a diagnostic pyramid, or no tissue on it)
    # is a verdict, not a crash, and is tracked apart from genuine failures so
    # the exit status of a two-day job still means something
    failed = []
    rejected = []
    stats_path = args.out_dir / "stats.jsonl"
    logger.info("per-slide stats -> %s", stats_path)
    try:
        with (
            StatsLog(stats_path) as stats,
            SysMonitor(stats=stats, interval=args.sysmon_interval),
        ):
            stats.write(
                kind="run",
                t=time.time(),
                slides=len(todo),
                # sharded runs share one stats.jsonl -- appends are atomic so the
                # file stays valid, but without this the two interleaved streams
                # of `sys` samples cannot be told apart after the fact
                shard=args.shard,
                shards=args.shards,
                batch_size=args.batch_size,
                num_tile_readers=args.num_tile_readers,
                prefetch_factor=args.prefetch_factor,
                pools=N_READER_POOLS,
            )
            pending = submit(0)
            for i, (_, slide_name) in enumerate(
                tqdm(
                    todo,
                    desc="slides",
                    unit="slide",
                    position=0,
                    disable=progress_disabled(),
                )
            ):
                logger.info("[%d/%d] %s", i + 1, len(todo), slide_name)
                # queue the next slide *before* blocking on this one, so its setup
                # and pipeline fill overlap this slide's GPU time
                this, pending = pending, submit(i + 1)
                assert this is not None

                try:
                    prep = this.result()
                except (UnsupportedSlideError, NoTissueError) as exc:
                    logger.warning("rejected %s: %s", slide_name, exc)
                    rejected.append(slide_name)
                    stats.write(
                        kind="slide",
                        t=time.time(),
                        slide=slide_name,
                        index=i,
                        stage="prepare",
                        ok=False,
                        rejected=type(exc).__name__,
                        reason=str(exc),
                    )
                    continue
                except Exception:
                    logger.exception("failed to prepare %s", slide_name)
                    failed.append(slide_name)
                    stats.write(
                        kind="slide",
                        t=time.time(),
                        slide=slide_name,
                        index=i,
                        stage="prepare",
                        ok=False,
                    )
                    continue

                try:
                    timings = embed_slide(
                        logger=logger,
                        prep=prep,
                        batch_size=args.batch_size,
                        model=model,
                        device=device,
                    )
                except Exception:
                    logger.exception("failed to embed %s", slide_name)
                    failed.append(slide_name)
                    stats.write(
                        kind="slide",
                        t=time.time(),
                        slide=slide_name,
                        index=i,
                        stage="embed",
                        ok=False,
                        prep=prep.timings,
                    )
                    continue

                # safe to read here and only here: slide i+1 is filling the
                # *other* pool, and slide i+2 -- which reuses this one -- is not
                # submitted until the top of the next iteration
                stats.write(
                    kind="slide",
                    t=time.time(),
                    slide=slide_name,
                    index=i,
                    ok=True,
                    pool=i % N_READER_POOLS,
                    mpp=prep.mpp,
                    mpp_source=prep.mpp_source,
                    width=prep.dimensions[0],
                    height=prep.dimensions[1],
                    extract_px=prep.extract_px,
                    prep=prep.timings,
                    readers=pools[i % N_READER_POOLS][0].worker_stats(),
                    **timings,
                )
    finally:
        prep_pool.shutdown(wait=True)

    if rejected:
        logger.warning(
            "%d/%d slides rejected: %s", len(rejected), len(todo), ", ".join(rejected)
        )
    if failed:
        logger.error(
            "%d/%d slides failed: %s", len(failed), len(todo), ", ".join(failed)
        )
        return 1
    logger.info(
        "done: %d slides, %d rejected", len(todo) - len(rejected), len(rejected)
    )
    return 0


if __name__ == "__main__":
    p = ArgumentParser(__doc__)
    p.add_argument(
        "--slide-dir",
        type=Path,
        default=Path("/slides"),  # default inside container
        help="directory of WSIs, searched recursively",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/out"),  # default inside container
        help="where the per-slide .h5 files and thumbnails go",
    )
    p.add_argument(
        "--exts",
        nargs="+",
        default=SLIDE_EXTS,
        help="slide extensions to look for; .dcm finds DICOM whole-slide series,"
        " one slide per series directory",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="tiles to embed per forward pass",
    )
    p.add_argument(
        "--num-tile-readers",
        type=int,
        default=20,
        help="worker processes reading tiles, per pool; there are two pools, so"
        " this many again are alive prefetching the next slide",
    )
    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="tile batches each reader buffers ahead of the GPU",
    )
    p.add_argument(
        "--thumbnail-width",
        type=int,
        default=2048,
        help="masked thumbnail width in px",
    )
    p.add_argument(
        "--shards",
        type=int,
        default=1,
        help="split the slide list across this many processes; run one container"
        " per GPU with the same --slide-dir and --out-dir, varying --shard",
    )
    p.add_argument(
        "--shard",
        type=int,
        default=0,
        help="which shard this process takes, 0-based",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="re-embed slides that already have an .h5",
    )
    p.add_argument(
        "--sysmon-interval",
        type=float,
        default=5.0,
        help="seconds between CPU/disk samples written to stats.jsonl; these are"
        " what separate another tenant loading the node from a stall of our own",
    )
    args = p.parse_args()
    if not 0 <= args.shard < args.shards:
        p.error(f"--shard must be in [0, {args.shards}), got {args.shard}")
    sys.exit(main(args))
