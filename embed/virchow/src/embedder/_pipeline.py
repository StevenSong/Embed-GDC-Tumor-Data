import os
import time
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ._dataset import TileDataset
from ._model import EMBED_DIM, HF_HUB_ID, TARGET_MPP, TILE_PX, TILE_UM, WrappedVirchow
from ._reader import Slide
from ._thumbnail import save_masked_thumbnail
from ._writer import SlideH5

PRECISION = torch.float16


def slide_metadata(
    *,  # enforce kwargs
    slide: Slide,
    slide_path: Path,
    thumbnail_path: Path,
) -> dict[str, str | int | float | np.ndarray]:
    return {
        "slide": slide_path.name,
        "slide_path": str(slide_path),
        "encoder": "virchow",
        "encoder_source": HF_HUB_ID,
        "embed_dim": EMBED_DIM,
        "n_tiles": len(slide.coords),
        "tile_px": TILE_PX,
        "tile_um": TILE_UM,
        "qc": "otsu",
        "precision": str(PRECISION),
        "mpp": float(slide.mpp) if slide.mpp else float("nan"),
        "slide_dimensions": np.asarray(slide.dimensions, dtype=np.int64),
        # the full (n_cols, n_rows) tile grid, before tissue filtering
        "grid_shape": np.asarray(slide.grid_shape, dtype=np.int64),
        "thumbnail": thumbnail_path.name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_version": str(torch.__version__),
    }


def embed_slide(
    *,  # enforce kwargs
    logger: Logger,
    slide_path: Path,
    thumbnail_path: Path,
    thumbnail_width: int,
    out_path: Path,
    tile_ds: TileDataset,
    tile_dl: DataLoader,
    model: WrappedVirchow,
    device: torch.device,
):
    start = time.time()
    # this parent Slide only reads pyramid levels (Otsu, thumbnail) and hands the
    # grid to the workers; closing it releases its file handles and tissue mask
    with Slide(slide_path, tile_px=TILE_PX, target_mpp=TARGET_MPP) as slide:
        masked_coords = slide.coords  # computes Otsu mask
        n_tiles = len(masked_coords)
        logger.info(
            "%s: %s px, %.4f mpp, %d tiles after QC",
            slide_path.name,
            "x".join(str(d) for d in slide.dimensions),
            slide.mpp,
            n_tiles,
        )
        save_masked_thumbnail(slide=slide, path=thumbnail_path, width=thumbnail_width)

        # update the base dataset so that worker instances work on the target slide
        tile_ds.set_next_slide_tiles(slide_path=slide_path, masked_coords=masked_coords)

        # write to a temp file so a crashed run never leaves a plausible-looking .h5
        tmp_path = out_path.with_name(f".{out_path.name}.tmp")
        try:
            with (
                SlideH5(
                    path=tmp_path,
                    gzip_level=0,
                    embed_dim=EMBED_DIM,
                ) as h5,
                tqdm(
                    total=n_tiles,
                    desc=slide_path.name,
                    unit="tile",
                    leave=False,
                    position=1,
                ) as pbar,
            ):
                for tile_batch, coord_batch in tile_dl:
                    # default collation stacks the dataset's numpy tiles into a
                    # (B, H, W, C) uint8 tensor; permute on device, where it is a
                    # free view and the host->device copy stays contiguous
                    x = tile_batch.to(device, non_blocking=True)
                    x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

                    with torch.inference_mode():
                        # fp16 autocast is what the Virchow model card prescribes
                        with torch.autocast(device_type=device.type, dtype=PRECISION):
                            emb_batch = model(x)
                        # Virchow returns fp32 even under autocast (final op is a mixed-precision LayerNorm)
                        emb_batch = emb_batch.float().cpu().numpy()

                    # `loc` is the tile's top-left corner in base-level pixels --
                    # exactly what the dataset yields -- and `grid` is its (col, row)
                    # in the full tile grid, so loc == grid * extract_px
                    loc = coord_batch.numpy()
                    h5.append(
                        features=emb_batch,
                        # NOTE: do not use `slide.grid` as the concurrent extraction
                        # order is not the same as the slide rasterization order
                        grid=loc // slide.extract_px,
                        loc=loc,
                    )

                    # NOTE: update with actual extracted batches which may be jagged
                    # in shape as each worker may emit a shortened trailing batch
                    pbar.update(len(emb_batch))

                if h5.n != n_tiles:
                    raise ValueError(
                        f"Embedded number of tiles ({h5.n}) disagrees with "
                        f"expected number of tiles ({n_tiles})!"
                    )
                h5.write_mean()
                h5.write_metadata(
                    slide_metadata(
                        slide=slide,
                        slide_path=slide_path,
                        thumbnail_path=thumbnail_path,
                    )
                )
            os.replace(tmp_path, out_path)
        except:
            tmp_path.unlink(missing_ok=True)
            raise

    elapsed = time.time() - start
    logger.info(
        "%s -> %s: %d tiles in %.1fs (%.0f tiles/s)",
        slide_path.name,
        out_path.name,
        n_tiles,
        elapsed,
        n_tiles / max(elapsed, 1e-6),
    )
