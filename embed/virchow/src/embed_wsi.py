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
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import slideflow as sf
import slideflow.slide.qc as sf_qc
import slideflow.util as sf_util
import torch
from PIL import Image

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

# queue protocol between the reader thread and the inference loop
_BATCH, _DONE, _ERROR = "batch", "done", "error"


# --------------------------------------------------------------------------- #
# tiling
# --------------------------------------------------------------------------- #
def open_slide(slide_path):
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
    qc_img = wsi.qc(sf_qc.Otsu())
    return wsi, qc_img


def _stack(imgs, grids, locs):
    return (
        np.stack(imgs),  # (B, C, H, W) uint8
        np.asarray(grids, dtype=np.int32),  # (B, 2) grid indices
        np.asarray(locs, dtype=np.int32),  # (B, 2) base-level pixel coords
    )


def read_tiles(wsi, pool, batch_size, max_tiles, q):
    """Fill `q` with tile batches. Runs on a worker thread so reads overlap inference."""
    try:
        gen = wsi.build_generator(
            shuffle=False,
            deterministic=True,
            whitespace_fraction=1,  # disable, just use the Otsu grid
            grayspace_fraction=1,  # disable, just use the Otsu grid
            img_format="numpy",
            lazy_iter=True,  # keep memory usage down
            pool=pool,  # prebuilt pool, shared across slides
            show_progress=False,
        )
        if gen is None:  # no tiles survived QC
            q.put((_DONE, None))
            return

        imgs, grids, locs = [], [], []
        n_read = 0
        for tile in gen():
            if n_read == 0 and "loc" not in tile:
                raise KeyError(
                    f"slideflow tiles carry no 'loc'; got keys {sorted(tile)}"
                )
            imgs.append(tile["image"].transpose(2, 0, 1))  # (C, H, W)
            grids.append(tile["grid"])
            locs.append(tile["loc"])
            n_read += 1
            if len(imgs) == batch_size:
                q.put((_BATCH, _stack(imgs, grids, locs)))
                imgs, grids, locs = [], [], []
            if max_tiles and n_read >= max_tiles:
                break
        if imgs:  # stragglers
            q.put((_BATCH, _stack(imgs, grids, locs)))
        q.put((_DONE, None))
    except BaseException as exc:  # re-raised on the main thread
        q.put((_ERROR, exc))


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
class SlideH5:
    """Streaming writer: embeddings are appended per batch, never held in full."""

    def __init__(self, path, gzip_level):
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
            dtype=np.float32,
            **compression,
        )
        self.grid = self._coords("grid")
        self.loc = self._coords("loc")
        self.n = 0

    def _coords(self, name):
        return self.f.create_dataset(
            name, shape=(0, 2), maxshape=(None, 2), chunks=(4096, 2), dtype=np.int32
        )

    def append(self, features, grid, loc):
        n = len(features)
        for ds, data in ((self.features, features), (self.grid, grid), (self.loc, loc)):
            ds.resize(self.n + n, axis=0)
            ds[self.n : self.n + n] = data
        self.n += n

    def write_metadata(self, meta):
        for key, value in meta.items():
            self.f.attrs[key] = value

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def save_masked_thumbnail(wsi, path, width, qc_img=None):
    """Slide thumbnail with everything Otsu rejected dimmed out."""
    thumb = wsi.thumb(width=width, low_res=True)
    if thumb is None:
        logger.warning("no thumbnail available for %s", path.stem)
        return None
    thumb = thumb.convert("RGB")
    rgb = np.asarray(thumb).astype(np.float32)

    # slideflow QC masks are True where tissue was *rejected*
    mask = getattr(wsi, "qc_mask", None)
    if mask is None and qc_img is not None:
        mask = np.asarray(qc_img.convert("L")) > 127
    if mask is None:
        logger.warning("no QC mask to overlay on the thumbnail for %s", path.stem)
    else:
        mask = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
        rejected = np.asarray(mask.resize(thumb.size, Image.NEAREST)) > 127
        tint = np.array([20.0, 26.0, 56.0], dtype=np.float32)
        rgb[rejected] = 0.35 * rgb[rejected] + 0.65 * tint

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(path)
    return path


def slide_metadata(wsi, slide_path, n_tiles, thumb_path):
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
        "thumbnail": thumb_path.name if thumb_path else "",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "slideflow_version": sf.__version__,
    }


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #
def embed_batch(model, device, imgs):
    x = torch.from_numpy(imgs).to(device, non_blocking=True)
    with torch.inference_mode():
        # fp16 autocast is what the Virchow model card prescribes
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            emb = model(x)
        # Virchow returns fp32 even under autocast (final op is a mixed-precision
        # LayerNorm); the cast is belt and braces
        return emb.float().cpu().numpy()


def embed_slide(slide_path, out_path, thumb_path, model, device, pool, args):
    start = time.time()

    wsi, qc_img = open_slide(slide_path)
    estimated = int(getattr(wsi, "estimated_num_tiles", 0) or 0)
    logger.info(
        "%s: %s px, %.4f mpp, ~%d tiles after QC",
        slide_path.name,
        "x".join(str(d) for d in wsi.dimensions),
        wsi.mpp or float("nan"),
        estimated,
    )
    save_masked_thumbnail(wsi, thumb_path, args.thumbnail_width, qc_img=qc_img)

    q = queue.Queue(maxsize=args.queue_depth)
    reader = threading.Thread(
        target=read_tiles,
        args=(wsi, pool, args.batch_size, args.max_tiles, q),
        name=f"tiles:{slide_path.stem}",
        daemon=True,
    )
    reader.start()

    # write to a temp file so a crashed run never leaves a plausible-looking .h5
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    try:
        with SlideH5(tmp_path, args.gzip_level) as h5:
            n_batches = 0
            while True:
                kind, item = q.get()
                if kind == _ERROR:
                    raise RuntimeError(
                        f"tile reader failed for {slide_path.name}"
                    ) from item
                if kind == _DONE:
                    break
                imgs, grid, loc = item
                h5.append(embed_batch(model, device, imgs), grid, loc)
                n_batches += 1
                if n_batches % args.log_every == 0:
                    rate = h5.n / max(time.time() - start, 1e-6)
                    logger.info(
                        "%s: %d/%d tiles (%.0f tiles/s)",
                        slide_path.name,
                        h5.n,
                        estimated,
                        rate,
                    )
            n_tiles = h5.n
            h5.write_metadata(slide_metadata(wsi, slide_path, n_tiles, thumb_path))
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        # if we bailed early the reader may be blocked on a full queue
        while reader.is_alive():
            try:
                q.get(timeout=0.1)
            except queue.Empty:
                pass
        reader.join(timeout=5)

    elapsed = time.time() - start
    logger.info(
        "%s -> %s: %d tiles in %.1fs (%.0f tiles/s)",
        slide_path.name,
        out_path.name,
        n_tiles,
        elapsed,
        n_tiles / max(elapsed, 1e-6),
    )
    return n_tiles


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def find_slides(slide_dir, exts):
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    return sorted(
        p for p in slide_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--slide-dir",
        type=Path,
        required=True,
        help="directory of WSIs, searched recursively",
    )
    p.add_argument(
        "--out-dir", type=Path, required=True, help="where the per-slide .h5 files go"
    )
    p.add_argument(
        "--thumbnail-dir", type=Path, default=None, help="default: <out-dir>/thumbnails"
    )
    p.add_argument(
        "--ext",
        nargs="+",
        default=list(SLIDE_EXTS),
        help="slide extensions to look for",
    )
    p.add_argument("--batch-size", type=int, default=64, help="tiles per forward pass")
    p.add_argument(
        "--num-tile-readers",
        type=int,
        default=8,
        help="worker processes decoding tiles",
    )
    p.add_argument(
        "--queue-depth",
        type=int,
        default=8,
        help="tile batches buffered ahead of the GPU",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--gzip-level", type=int, default=4, help="0 disables compression of features"
    )
    p.add_argument(
        "--thumbnail-width", type=int, default=2048, help="masked thumbnail width in px"
    )
    p.add_argument(
        "--max-tiles",
        type=int,
        default=0,
        help="cap tiles per slide (0 = all); for smoke tests",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="re-embed slides that already have an .h5",
    )
    p.add_argument(
        "--log-every", type=int, default=50, help="log progress every N batches"
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    slides = find_slides(args.slide_dir, args.ext)
    if not slides:
        logger.error("no slides matching %s under %s", args.ext, args.slide_dir)
        return 1
    out_dir = args.out_dir
    thumb_dir = args.thumbnail_dir or out_dir / "thumbnails"
    out_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    logger.info("found %d slides under %s", len(slides), args.slide_dir)

    # build the reader pool before the CUDA context exists: forking a process that
    # already initialized CUDA is a well-known way to get stuck children
    pool = _CTX.Pool(
        processes=args.num_tile_readers, initializer=sf_util.set_ignore_sigint
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = WrappedVirchow()
    model.eval()
    model.to(device)
    logger.info("virchow ready on %s", device)

    failed = []
    try:
        for i, slide_path in enumerate(slides, start=1):
            out_path = out_dir / f"{slide_path.stem}.h5"
            if out_path.exists() and not args.overwrite:
                logger.info(
                    "[%d/%d] %s exists, skipping", i, len(slides), out_path.name
                )
                continue
            logger.info("[%d/%d] %s", i, len(slides), slide_path.name)
            try:
                embed_slide(
                    slide_path,
                    out_path,
                    thumb_dir / f"{slide_path.stem}.png",
                    model,
                    device,
                    pool,
                    args,
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
    sys.exit(main())
