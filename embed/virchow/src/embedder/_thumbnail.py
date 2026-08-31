from pathlib import Path

import cv2
import numpy as np

from ._reader import Slide

# RGB the rejected regions are pulled towards, and how far
TINT = (20.0, 26.0, 56.0)
TINT_STRENGTH = 0.65


def save_masked_thumbnail(
    *,  # enforce kwargs
    slide: Slide,
    path: Path,
    width: int,
):
    """Slide thumbnail with everything that was *not* embedded dimmed out.

    The kept set is `slide.coords`, not the raw tissue mask, so a single image
    shows every reason a region is missing from the .h5: the Otsu rejection, the
    centre-on-tissue rule, and the partial tiles dropped at the right/bottom edge.
    """
    # thumbnail() refuses to upsample, and a slide narrower than `width` is not
    # worth failing a run over
    width = min(width, slide.dimensions[0])
    rgb = slide.thumbnail(width=width).astype(np.float32)
    height = rgb.shape[0]

    n_cols, n_rows = slide.grid_shape
    kept = np.zeros((n_rows, n_cols), dtype=np.uint8)
    cols, rows = slide.grid.T  # (N, 2) -> two (N,) index arrays
    kept[rows, cols] = 1

    # the grid spans only n_cols/n_rows whole tiles; the remainder on the right
    # and bottom never became a tile, so scale the grid onto that sub-rectangle
    # of the thumbnail and leave the leftover strip dimmed
    scale = width / slide.dimensions[0]
    kept_w = min(width, max(1, round(n_cols * slide.extract_px * scale)))
    kept_h = min(height, max(1, round(n_rows * slide.extract_px * scale)))
    kept = cv2.resize(kept, (kept_w, kept_h), interpolation=cv2.INTER_NEAREST)

    embedded = np.zeros((height, width), dtype=bool)
    embedded[:kept_h, :kept_w] = kept.astype(bool)

    tint = np.array(TINT, dtype=np.float32)
    rejected = ~embedded
    rgb[rejected] = (1 - TINT_STRENGTH) * rgb[rejected] + TINT_STRENGTH * tint

    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    # imwrite reports failure by returning False rather than raising
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"could not write thumbnail to {path}")
