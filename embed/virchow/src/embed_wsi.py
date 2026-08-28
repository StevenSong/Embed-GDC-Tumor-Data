#!/usr/bin/env python
"""Tile WSIs with slideflow and embed every tile with Virchow, one HDF5 per slide.

Slides are tiled with Otsu tissue detection at 224 px / 20x, and Virchow runs in this
process on the local GPU. Tiles live in memory and stream straight to the GPU -- the
only things written to disk are the per-slide HDF5 and its thumbnail.

    python embed_wsi.py --slide-dir /data/slides --out-dir /data/embeddings

Placement across GPUs is the caller's job: run one container per GPU with
CUDA_VISIBLE_DEVICES pointed at a disjoint --slide-dir.
"""

import argparse
import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Iterable

# our pinned slideflow version still uses pkg_resources, we have setuptools pinned
# so just suppress the warning before slideflow is imported
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
)

import h5py
import numpy as np
import slideflow as sf
import slideflow.slide.qc as sf_qc
import slideflow.util as sf_util
import torch
from PIL import Image
from tqdm import tqdm

from virchow import (  # isort: skip
    EMBED_DIM,
    HF_HUB_ID,
    TILE_PX,
    TILE_UM,
    WrappedVirchow,
)

logger = logging.getLogger("embed_wsi")

SLIDE_EXTS = (".svs", ".tif", ".tiff", ".ndpi", ".scn", ".mrxs", ".bif", ".svslide")
STRIDE_DIV = 1  # no overlap between tiles
H5_CHUNK_ROWS = 256

# tile readers are forked before this process ever touches CUDA
_CTX = mp.get_context("forkserver")

# queue flags between the reader thread and the inference loop
_BATCH, _DONE, _ERROR = "batch", "done", "error"

# how long to keep unblocking the reader thread before giving up on it
READER_DRAIN_TIMEOUT_S = 30


class TqdmLoggingHandler(logging.Handler):
    """Emit through tqdm.write so log lines never smear the progress bars."""

    def emit(self, record: logging.LogRecord):
        try:
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:
            self.handleError(record)


def open_slide(slide_path: Path) -> sf.WSI:
    """slideflow WSI at Virchow's magnification, with Otsu tissue detection applied."""
    wsi = sf.WSI(
        str(slide_path),
        tile_px=TILE_PX,
        tile_um=TILE_UM,
        stride_div=STRIDE_DIV,
        enable_downsample=True,
        use_edge_tiles=False,
        roi_method="ignore",  # no ROIs; extract across the whole slide
    )
    wsi.qc(sf_qc.Otsu())
    return wsi


def _stack(
    *,  # enforce kwargs
    imgs: list[np.ndarray],
    grids: list[list],
    locs: list[list],
) -> tuple[
    np.ndarray,  # stacked imgs
    np.ndarray,  # stacked grids
    np.ndarray,  # stacked locs
]:
    return (
        np.stack(imgs),  # (B, C, H, W) uint8
        np.asarray(grids, dtype=np.int32),  # (B, 2) grid indices
        np.asarray(locs, dtype=np.int32),  # (B, 2) base-level pixel coords
    )


def read_tiles(
    *,  # enforce kwargs
    wsi: sf.WSI,
    pool: Pool,
    batch_size: int,
    q: queue.Queue,
    max_tiles: int | None = None,
):
    """Fill `q` with tile batches. Runs on a worker thread so reads overlap inference."""
    try:
        gen: Callable[[], Iterable[dict]] = wsi.build_generator(  # type: ignore
            shuffle=False,
            deterministic=True,
            whitespace_fraction=1,  # disable, just use the Otsu grid
            grayspace_fraction=1,  # disable, just use the Otsu grid
            img_format="numpy",
            lazy_iter=True,  # keep memory usage down
            pool=pool,  # prebuilt pool, shared across slides
            show_progress=False,
            max_tiles=max_tiles,
        )
        if gen is None:  # no tiles survived QC
            q.put((_DONE, None))
            return

        imgs, grids, locs = [], [], []
        for tile in gen():
            imgs.append(tile["image"])
            grids.append(tile["grid"])
            locs.append(tile["loc"])
            if len(imgs) == batch_size:
                q.put((_BATCH, _stack(imgs=imgs, grids=grids, locs=locs)))
                imgs, grids, locs = [], [], []
        if imgs:  # stragglers
            q.put((_BATCH, _stack(imgs=imgs, grids=grids, locs=locs)))
        q.put((_DONE, None))
    except BaseException as exc:  # re-raised on the main thread
        q.put((_ERROR, exc))


class SlideH5:
    """Streaming writer: embeddings are appended per batch, never held in full."""

    def __init__(
        self,
        *,  # enforce kwargs
        path: Path,
        gzip_level: int,
    ):
        compression = (
            {"compression": "gzip", "compression_opts": gzip_level}
            if gzip_level
            else {}
        )
        self.f = h5py.File(path, "w")
        self.features = self.f.create_dataset(
            "features",
            shape=(0, EMBED_DIM),
            maxshape=(None, EMBED_DIM),
            chunks=(H5_CHUNK_ROWS, EMBED_DIM),
            dtype=np.float32,  # virchow embedding type
            **compression,
        )
        self.grid = self._coords("grid")
        self.loc = self._coords("loc")
        self.n = 0

    def _coords(self, name: str) -> h5py.Dataset:
        return self.f.create_dataset(
            name,
            shape=(0, 2),
            maxshape=(None, 2),
            chunks=(4096, 2),
            dtype=int,
        )

    def append(
        self,
        *,  # enforce kwargs
        features: np.ndarray,
        grid: np.ndarray,
        loc: np.ndarray,
    ):
        n = len(features)
        for ds, data in (
            (self.features, features),
            (self.grid, grid),
            (self.loc, loc),
        ):
            ds.resize(self.n + n, axis=0)
            ds[self.n : self.n + n] = data
        self.n += n

    def write_metadata(self, meta: dict):
        for key, value in meta.items():
            self.f.attrs[key] = value

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def save_masked_thumbnail(
    *,  # enforce kwargs
    wsi: sf.WSI,
    path: Path,
    width: int,
):
    """Slide thumbnail with everything Otsu rejected dimmed out."""
    thumb = wsi.thumb(width=width, low_res=True)
    if thumb is None:
        raise ValueError(f"no thumbnail for {path.stem}")
    thumb = thumb.convert("RGB")
    rgb = np.asarray(thumb).astype(np.float32)

    # slideflow QC masks are True where tissue was *rejected*
    mask = wsi.qc_mask
    if mask is None:
        raise ValueError(f"no qc mask for {path.stem}")

    # the QC mask is at the QC downsample level, so resample it to the thumbnail's
    # size; PIL is the shortest route to a nearest-neighbour resize of a bool array
    mask = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8))
    rejected = np.asarray(mask.resize(thumb.size, Image.Resampling.NEAREST)) > 0
    tint = np.array([20.0, 26.0, 56.0], dtype=np.float32)
    rgb[rejected] = 0.35 * rgb[rejected] + 0.65 * tint

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def slide_metadata(
    *,  # enforce kwargs
    wsi: sf.WSI,
    slide_path: Path,
    n_tiles: int,
    thumbnail_path: Path,
) -> dict[str, str | int | float | np.ndarray]:
    grid = getattr(wsi, "grid", None)
    grid_shape = np.asarray(grid).shape[:2] if grid is not None else (0, 0)
    return {
        "slide": slide_path.name,
        "slide_path": str(slide_path),
        "encoder": "virchow",
        "encoder_source": HF_HUB_ID,
        "embed_dim": EMBED_DIM,
        "n_tiles": n_tiles,
        "tile_px": TILE_PX,
        "tile_um": TILE_UM,
        "stride_div": STRIDE_DIV,
        "qc": "otsu",
        "precision": "fp16",
        "mpp": float(wsi.mpp) if wsi.mpp else float("nan"),
        "slide_dimensions": np.asarray(wsi.dimensions, dtype=np.int64),
        "grid_shape": np.asarray(grid_shape, dtype=np.int64),
        "thumbnail": thumbnail_path.name if thumbnail_path else "",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_version": str(torch.__version__),
        "slideflow_version": str(sf.__version__),
    }


def embed_batch(
    *,  # enforce kwargs
    model: WrappedVirchow,
    device: torch.device,
    imgs: np.ndarray,
) -> np.ndarray:
    # imgs come in as (B, H, W, C)
    imgs = imgs.transpose(0, 3, 1, 2)  # (B, C, H, W)
    x = torch.from_numpy(imgs).to(device, non_blocking=True)
    with torch.inference_mode():
        # fp16 autocast is what the Virchow model card prescribes
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            embs = model(x)
        # Virchow returns fp32 even under autocast (final op is a mixed-precision
        # LayerNorm); the cast is belt and braces
        return embs.float().cpu().numpy()


def embed_slide(
    *,  # enforce kwargs
    slide_path: Path,
    out_path: Path,
    thumbnail_path: Path,
    thumbnail_width: int,
    pool: Pool,
    queue_depth: int,
    model: WrappedVirchow,
    device: torch.device,
    batch_size: int,
    gzip_level: int,
    max_tiles: int | None = None,
):
    start = time.time()

    wsi = open_slide(slide_path)
    logger.info(
        "%s: %s px, %.4f mpp, ~%d tiles after QC",
        slide_path.name,
        "x".join(str(d) for d in wsi.dimensions),
        wsi.mpp,
        wsi.estimated_num_tiles,
    )
    save_masked_thumbnail(wsi=wsi, path=thumbnail_path, width=thumbnail_width)

    # only an estimate: the Otsu grid is counted before tiles are actually read
    n_expected = wsi.estimated_num_tiles
    if max_tiles is not None:
        n_expected = min(n_expected, max_tiles)

    # queue contains tuples of (flag, item)
    q = queue.Queue(maxsize=queue_depth)

    # start reader thread for asynchronously producing tiles,
    # reader uses separate multiprocessing pool for actual read
    reader = threading.Thread(
        target=read_tiles,
        kwargs={
            "wsi": wsi,
            "pool": pool,
            "batch_size": batch_size,
            "max_tiles": max_tiles,
            "q": q,
        },
        name=f"tiles:{slide_path.stem}",
        daemon=True,
    )
    reader.start()

    # write to a temp file so a crashed run never leaves a plausible-looking .h5
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    try:
        with (
            SlideH5(path=tmp_path, gzip_level=gzip_level) as h5,
            tqdm(
                total=n_expected,
                desc=slide_path.name,
                unit="tile",
                leave=False,
                position=1,
            ) as tile_bar,
        ):
            while True:
                flag, item = q.get()
                if flag == _BATCH:
                    imgs, grid, loc = item
                    embs = embed_batch(model=model, device=device, imgs=imgs)
                    h5.append(features=embs, grid=grid, loc=loc)
                    tile_bar.update(len(embs))
                elif flag == _DONE:
                    break
                elif flag == _ERROR:
                    raise RuntimeError(
                        f"tile reader failed for {slide_path.name}"
                    ) from item
                else:
                    raise ValueError(f"Unknown queue flag: {flag}")
            n_tiles = h5.n
            h5.write_metadata(
                slide_metadata(
                    wsi=wsi,
                    slide_path=slide_path,
                    n_tiles=n_tiles,
                    thumbnail_path=thumbnail_path,
                )
            )
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        # bailing early leaves the reader parked in q.put() on the bounded queue, so
        # it never reaches its return: drain to unblock it rather than join a wedge
        deadline = time.time() + READER_DRAIN_TIMEOUT_S
        while reader.is_alive() and time.time() < deadline:
            try:
                q.get(timeout=0.1)
            except queue.Empty:
                pass
        if reader.is_alive():  # still holding pool workers; flag it, don't hang
            logger.warning("tile reader for %s did not exit", slide_path.name)

    elapsed = time.time() - start
    logger.info(
        "%s -> %s: %d tiles in %.1fs (%.0f tiles/s)",
        slide_path.name,
        out_path.name,
        n_tiles,
        elapsed,
        n_tiles / max(elapsed, 1e-6),
    )


def find_slides(
    *,  # enforce kwargs
    slide_dir: Path,
    exts: Iterable[str],
) -> list[Path]:
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    return sorted(
        p for p in slide_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        default=list(SLIDE_EXTS),
        help="slide extensions to look for",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="tiles to embed per forward pass",
    )
    p.add_argument(
        "--num-tile-readers",
        type=int,
        default=8,
        help="worker processes reading tiles",
    )
    p.add_argument(
        "--queue-depth",
        type=int,
        default=8,
        help="tile batches buffered ahead of the GPU",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--gzip-level",
        type=int,
        default=4,
        help="0 disables compression of features",
    )
    p.add_argument(
        "--thumbnail-width",
        type=int,
        default=2048,
        help="masked thumbnail width in px",
    )
    p.add_argument(
        "--max-tiles",
        type=int,
        help="optionally cap tiles per slide for testing",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="re-embed slides that already have an .h5",
    )
    return p.parse_args()


def main(args):
    handler = TqdmLoggingHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.basicConfig(level=logging.WARNING, handlers=[handler])
    logger.setLevel(logging.INFO)

    slides = find_slides(slide_dir=args.slide_dir, exts=args.exts)
    if not slides:
        logger.error("no slides matching %s under %s", args.exts, args.slide_dir)
        return 1
    embed_dir = args.out_dir / "embeddings"
    thumb_dir = args.out_dir / "thumbnails"
    embed_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    logger.info("found %d slides under %s", len(slides), args.slide_dir)

    # build the reader pool before the CUDA context exists
    pool = _CTX.Pool(
        processes=args.num_tile_readers,
        initializer=sf_util.set_ignore_sigint,
    )

    device = torch.device(args.device)
    model = WrappedVirchow().eval().to(device)
    logger.info("virchow ready on %s", device)

    failed = []
    try:
        with tqdm(slides, desc="slides", unit="slide", position=0) as slide_bar:
            for i, slide_path in enumerate(slide_bar, start=1):
                out_path = embed_dir / f"{slide_path.stem}.h5"
                if out_path.exists() and not args.overwrite:
                    logger.info(
                        "[%d/%d] %s exists, skipping", i, len(slides), out_path.name
                    )
                    continue
                logger.info("[%d/%d] %s", i, len(slides), slide_path.name)
                try:
                    embed_slide(
                        slide_path=slide_path,
                        out_path=out_path,
                        thumbnail_path=thumb_dir / f"{slide_path.stem}.png",
                        thumbnail_width=args.thumbnail_width,
                        pool=pool,
                        queue_depth=args.queue_depth,
                        model=model,
                        device=device,
                        batch_size=args.batch_size,
                        gzip_level=args.gzip_level,
                        max_tiles=args.max_tiles,
                    )
                except Exception:
                    logger.exception("failed to embed %s", slide_path.name)
                    failed.append(slide_path.name)
    finally:
        pool.close()
        pool.join()

    if failed:
        logger.error(
            "%d/%d slides failed: %s", len(failed), len(slides), ", ".join(failed)
        )
        return 1
    logger.info("done: %d slides", len(slides))
    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))
