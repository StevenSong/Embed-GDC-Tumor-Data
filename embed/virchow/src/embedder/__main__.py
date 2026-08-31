"""Tile SVS slides and embed every tile with Virchow, one HDF5 per slide.

Slides are tiled by `_reader.Slide` -- tifffile straight onto the Aperio pyramid, with
CLAM-style tissue detection -- at 224 px / 20x, and Virchow runs in this process on the
local GPU. Tiles live in memory and stream straight to the GPU -- the only things
written to disk are the per-slide HDF5 and its thumbnail.

    python -m embedder --slide-dir /data/slides --out-dir /data/embeddings

Two reader pools alternate: while slide N is on the GPU, a background thread runs
slide N+1's setup (open, Otsu, grid, thumbnail) and fills the other pool, so the
GPU never waits for a slide to start. That costs shared memory --
`2 * num_tile_readers * prefetch_factor * batch_size * 150 KB`, about 10 GB at the
defaults -- so run with `--shm-size=12g` or the readers die with a bus error.

Virchow is always `torch.compile`d, with no eager path and no flag to turn it off
-- an uncompiled run does the same work in 54h instead of 35h, so a compile that
fails is fatal rather than something to shrug off. Triton needs a C compiler on
PATH, which is why the image installs gcc.

Placement across GPUs is the caller's job: run one container per GPU with
CUDA_VISIBLE_DEVICES pointed at a disjoint --slide-dir.
"""

import sys
from argparse import ArgumentParser
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ._dataset import TileDataset
from ._logging import setup_logging
from ._model import TARGET_MPP, TILE_PX, WrappedVirchow
from ._pipeline import compile_model, embed_slide, prepare_slide

SLIDE_EXTS = [".svs"]
# one pool feeds the GPU while the other fills for the next slide
N_READER_POOLS = 2


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
    slide_paths = sorted(
        p for p in args.slide_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )
    n_slides = len(slide_paths)

    if not slide_paths:
        logger.error("no slides matching %s under %s", args.exts, args.slide_dir)
        return 1
    embed_dir = args.out_dir / "embeddings"
    thumb_dir = args.out_dir / "thumbnails"
    embed_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    logger.info("found %d slides under %s", n_slides, args.slide_dir)

    # resolve skips up front: the prefetch below runs one slide ahead, and it can
    # only do that against a list of slides it is actually going to embed
    todo = []
    for slide_path in slide_paths:
        out_path = embed_dir / f"{slide_path.stem}.h5"
        if out_path.exists() and not args.overwrite:
            logger.info("%s exists, skipping", out_path.name)
            continue
        todo.append(slide_path)
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
            slide_paths=todo,
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
            slide_path=todo[i],
            thumbnail_path=thumb_dir / f"{todo[i].stem}.png",
            thumbnail_width=args.thumbnail_width,
            out_path=embed_dir / f"{todo[i].stem}.h5",
            tile_ds=tile_ds,
            tile_dl=tile_dl,
        )

    failed = []
    try:
        pending = submit(0)
        for i, slide_path in enumerate(
            tqdm(todo, desc="slides", unit="slide", position=0)
        ):
            logger.info("[%d/%d] %s", i + 1, len(todo), slide_path.name)
            # queue the next slide *before* blocking on this one, so its setup
            # and pipeline fill overlap this slide's GPU time
            this, pending = pending, submit(i + 1)
            assert this is not None

            try:
                prep = this.result()
            except Exception:
                logger.exception("failed to prepare %s", slide_path.name)
                failed.append(slide_path.name)
                continue

            try:
                embed_slide(
                    logger=logger,
                    prep=prep,
                    batch_size=args.batch_size,
                    model=model,
                    device=device,
                )
            except Exception:
                logger.exception("failed to embed %s", slide_path.name)
                failed.append(slide_path.name)
    finally:
        prep_pool.shutdown(wait=True)

    if failed:
        logger.error(
            "%d/%d slides failed: %s", len(failed), len(todo), ", ".join(failed)
        )
        return 1
    logger.info("done: %d slides", len(todo))
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
        help="slide extensions to look for",
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
        default=32,
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
        "--overwrite",
        action="store_true",
        help="re-embed slides that already have an .h5",
    )
    args = p.parse_args()
    sys.exit(main(args))
