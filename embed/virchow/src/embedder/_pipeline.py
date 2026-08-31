import contextlib
import io
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Iterator

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
# max-autotune measured 1247 tiles/s on the GPU leg against `default`'s 1203, for
# ~40 s more compile once per run
COMPILE_MODE = "max-autotune"


def slide_metadata(prep: "PreparedSlide") -> dict[str, str | int | float | np.ndarray]:
    return {
        "slide": prep.slide_path.name,
        "slide_path": str(prep.slide_path),
        "encoder": "virchow",
        "encoder_source": HF_HUB_ID,
        "embed_dim": EMBED_DIM,
        "n_tiles": prep.n_tiles,
        "tile_px": TILE_PX,
        "tile_um": TILE_UM,
        "qc": "otsu",
        "precision": str(PRECISION),
        "mpp": float(prep.mpp) if prep.mpp else float("nan"),
        # "header" for the Aperio MPP field, otherwise the AppMag it was inferred
        # from -- the handful of slides with a truncated header are worth being
        # able to find again in the output
        "mpp_source": prep.mpp_source,
        "slide_dimensions": np.asarray(prep.dimensions, dtype=np.int64),
        # the full (n_cols, n_rows) tile grid, before tissue filtering
        "grid_shape": np.asarray(prep.grid_shape, dtype=np.int64),
        "thumbnail": prep.thumbnail_path.name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_version": str(torch.__version__),
    }


def compile_model(
    *,  # enforce kwargs
    logger: Logger,
    model: WrappedVirchow,
    batch_size: int,
    device: torch.device,
) -> WrappedVirchow:
    """`torch.compile` the transformer, warmed up and ready for the first slide.

    Worth ~30% on the GPU leg (943 -> 1203 tiles/s at batch 512, more with
    max-autotune), but only with static shapes -- hence `dynamic=False` here and
    the padded tail batch in `embed_slide`.

    There is deliberately no eager fallback. Triton needs a C compiler on PATH
    and fails at warmup rather than at the `torch.compile` call, so a fallback
    would turn a missing `gcc` into a run that quietly does 35 h of work in 54 h.
    Failing here costs a restart; falling back costs the corpus.

    Set TORCHINDUCTOR_CACHE_DIR to a persistent path to amortise the ~60 s
    compile across runs.
    """
    logger.info("compiling virchow (mode=%s, batch size=%d)", COMPILE_MODE, batch_size)
    start = time.time()
    model.virchow = torch.compile(model.virchow, dynamic=False, mode=COMPILE_MODE)  # type: ignore
    # warm with the memory layout the pipeline really feeds: a (B, H, W, C) uint8
    # batch permuted to (B, C, H, W), which is channels_last. A plain
    # torch.zeros(B, 3, 224, 224) is NCHW -- dynamo guards on strides, so that
    # would compile a throwaway graph and pay the entire cost a second time
    warm = torch.zeros(
        (batch_size, TILE_PX, TILE_PX, 3), dtype=torch.uint8, device=device
    ).permute(0, 3, 1, 2)
    # inductor writes its autotune tables and choice stats straight to
    # sys.stderr rather than through logging, so there is no logger to quiet --
    # capture the stream instead. On a cold TORCHINDUCTOR_CACHE_DIR that is ~50
    # lines of kernel timings nobody reads. Held rather than dropped, and
    # replayed if the warmup fails, when it is the only context there is
    captured = io.StringIO()
    try:
        with (
            contextlib.redirect_stderr(captured),
            torch.inference_mode(),
            torch.autocast(device_type=device.type, dtype=PRECISION),
        ):
            model(warm)
    except BaseException:
        sys.stderr.write(captured.getvalue())
        raise

    logger.info("compiled in %.1fs", time.time() - start)
    return model


@dataclass
class PreparedSlide:
    """A slide whose readers are already running, ready for the GPU.

    Deliberately holds no `Slide`. Everything downstream of `prepare_slide`
    wants is a handful of scalars -- the geometry that turns a tile's `loc` into
    a `grid` index, plus what goes in the .h5 attributes -- so the file is closed
    before this is handed on, and nothing owns an open handle across the thread
    boundary. The tile coordinates themselves already crossed to the readers
    through the dataset's shared buffer, and the tissue mask has done its job by
    the time the mask-derived coords exist.
    """

    slide_path: Path
    thumbnail_path: Path
    out_path: Path
    n_tiles: int
    mpp: float
    mpp_source: str
    dimensions: tuple[int, int]
    extract_px: int
    batches: Iterator  # live DataLoader iterator, already filling

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Full tile grid (n_cols, n_rows) before tissue filtering."""
        return (
            self.dimensions[0] // self.extract_px,
            self.dimensions[1] // self.extract_px,
        )


def prepare_slide(
    *,  # enforce kwargs
    logger: Logger,
    slide_path: Path,
    thumbnail_path: Path,
    thumbnail_width: int,
    out_path: Path,
    tile_ds: TileDataset,
    tile_dl: DataLoader,
) -> PreparedSlide:
    """Everything a slide needs before the first forward pass.

    Opening the file, Otsu, the tile grid and the thumbnail cost ~0.87 s, and
    filling the reader pipeline another 1.2 s (JPEG) to 3.5 s (JPEG 2000), all of
    it with the GPU idle. `__main__` runs this on a background thread for slide
    N+1 while slide N is still embedding, so the whole of it lands inside the
    previous slide's compute -- which is why the setup belongs here and not in
    `embed_slide`.

    The Slide is opened and closed entirely inside this function. It only ever
    reads the slide at low resolution -- a pyramid level for Otsu and the
    thumbnail, or, on the handful of slides whose coarsest level is still too big
    to hold, that level streamed segment by segment into a reduced canvas. The
    base-level tile reads all happen in the workers, off the grid this hands
    them through the dataset's shared buffer, so by the time this returns there
    is nothing left worth keeping open: what `embed_slide` needs is geometry, and
    that is a few scalars on the returned `PreparedSlide`.
    """
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
        if slide.mpp_source != "header":
            logger.warning(
                "%s: no MPP in the Aperio header, inferred %.4f from %s",
                slide_path.name,
                slide.mpp,
                slide.mpp_source,
            )
        save_masked_thumbnail(slide=slide, path=thumbnail_path, width=thumbnail_width)

        # update the base dataset so that worker instances work on the target slide
        tile_ds.set_next_slide_tiles(slide_path=slide_path, masked_coords=masked_coords)

        prep = PreparedSlide(
            slide_path=slide_path,
            thumbnail_path=thumbnail_path,
            out_path=out_path,
            n_tiles=n_tiles,
            mpp=slide.mpp,
            mpp_source=slide.mpp_source,
            dimensions=slide.dimensions,
            extract_px=slide.extract_px,
            # last, so nothing above can fail with readers already in flight
            batches=iter(tile_dl),
        )
    return prep


def embed_slide(
    *,  # enforce kwargs
    logger: Logger,
    prep: PreparedSlide,
    batch_size: int,
    model: WrappedVirchow,
    device: torch.device,
):
    """Drain a prepared slide's readers into its .h5.

    Consumes the `PreparedSlide` -- its `batches` iterator is exhausted, so it is
    not reusable -- but owns nothing: the slide file was already closed by
    `prepare_slide`.
    """
    start = time.time()
    slide_path, out_path = prep.slide_path, prep.out_path
    n_tiles = prep.n_tiles

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
            for tile_batch, coord_batch in prep.batches:
                x = tile_batch.to(device, non_blocking=True)
                n = len(x)
                if n < batch_size:
                    # the slide's genuine last batch, padded up so the compiled
                    # graph only ever sees one shape. Pad in (B, H, W, C), before
                    # the permute: padding after it would hand the model an NCHW
                    # tensor, which dynamo guards -- and compiles -- separately
                    x = torch.cat([x, x.new_zeros((batch_size - n, *x.shape[1:]))])
                # permute on device, where it is a free view and the host->device
                # copy above stays contiguous
                x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

                with torch.inference_mode():
                    # fp16 autocast is what the Virchow model card prescribes
                    with torch.autocast(device_type=device.type, dtype=PRECISION):
                        emb_batch = model(x)
                    # Virchow returns fp32 even under autocast (final op is a mixed-precision LayerNorm)
                    emb_batch = emb_batch[:n].float().cpu().numpy()

                # `loc` is the tile's top-left corner in base-level pixels --
                # exactly what the dataset yields -- and `grid` is its (col, row)
                # in the full tile grid, so loc == grid * extract_px
                loc = coord_batch.numpy()
                h5.append(
                    features=emb_batch,
                    # NOTE: derive the grid from this batch's own `loc`; the
                    # concurrent extraction order is not the slide's raster order
                    grid=loc // prep.extract_px,
                    loc=loc,
                )

                pbar.update(n)

            if h5.n != n_tiles:
                raise ValueError(
                    f"Embedded number of tiles ({h5.n}) disagrees with "
                    f"expected number of tiles ({n_tiles})!"
                )
            h5.write_mean()
            h5.write_metadata(slide_metadata(prep))
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
