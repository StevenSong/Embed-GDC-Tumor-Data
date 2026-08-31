"""Tile SVS slides and embed every tile with Virchow, one HDF5 per slide.

Slides are tiled by `_reader.Slide` -- tifffile straight onto the Aperio pyramid, with
CLAM-style tissue detection -- at 224 px / 20x, and Virchow runs in this process on the
local GPU. Tiles live in memory and stream straight to the GPU -- the only things
written to disk are the per-slide HDF5 and its thumbnail.

    python embed_wsi.py --slide-dir /data/slides --out-dir /data/embeddings

Placement across GPUs is the caller's job: run one container per GPU with
CUDA_VISIBLE_DEVICES pointed at a disjoint --slide-dir.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ._dataset import TileDataset
from ._logging import setup_logging
from ._model import TARGET_MPP, TILE_PX, WrappedVirchow
from ._pipeline import embed_slide

SLIDE_EXTS = [".svs"]


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

    device = torch.device(args.device)
    model = WrappedVirchow().eval().to(device)
    logger.info("virchow ready on %s", device)

    # special dataset with shared attributes to worker instances in dataloader.
    # we treat each slide as an epoch and iterate over batches of tiles, so we end up with n_slides "epochs".
    # tile geometry comes from the model so the parent's Slide in _pipeline and the
    # workers' Slides can never disagree about the grid
    tile_ds = TileDataset(
        slide_paths=slide_paths,
        tile_px=TILE_PX,
        target_mpp=TARGET_MPP,
    )
    # persistent_workers/prefetch_factor are only meaningful with real workers, and
    # DataLoader rejects them outright at num_workers=0
    worker_opts = (
        {"persistent_workers": True, "prefetch_factor": args.prefetch_factor}
        if args.num_tile_readers > 0
        else {}
    )
    tile_dl = DataLoader(
        tile_ds,
        batch_size=args.batch_size,
        num_workers=args.num_tile_readers,
        drop_last=False,  # NOTE: do not set this to True, will drop alot of tiles
        pin_memory=args.device != "cpu",
        **worker_opts,
    )
    failed = []
    for i, slide_path in enumerate(
        tqdm(slide_paths, desc="slides", unit="slide", position=0)
    ):
        # check if we should skip slide or not
        out_path = embed_dir / f"{slide_path.stem}.h5"
        if out_path.exists() and not args.overwrite:
            logger.info("[%d/%d] %s exists, skipping", i, n_slides, out_path.name)
            continue
        logger.info("[%d/%d] %s", i, n_slides, slide_path.name)

        try:
            embed_slide(
                logger=logger,
                slide_path=slide_path,
                thumbnail_path=thumb_dir / f"{slide_path.stem}.png",
                thumbnail_width=args.thumbnail_width,
                out_path=out_path,
                tile_ds=tile_ds,
                tile_dl=tile_dl,
                model=model,
                device=device,
            )
        except Exception:
            logger.exception("failed to embed %s", slide_path.name)
            failed.append(slide_path.name)

    if failed:
        logger.error(
            "%d/%d slides failed: %s", len(failed), n_slides, ", ".join(failed)
        )
        return 1
    logger.info("done: %d slides", n_slides)
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
        default=128,
        help="tiles to embed per forward pass",
    )
    p.add_argument(
        "--num-tile-readers",
        type=int,
        default=16,
        help="worker processes reading tiles; set to 0 to instead read on the main process",
    )
    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="tile batches each reader buffers ahead of the GPU",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
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
