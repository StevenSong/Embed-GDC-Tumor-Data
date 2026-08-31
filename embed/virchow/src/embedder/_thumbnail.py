from pathlib import Path

from ._reader import Slide


# TODO: @Claude
def save_masked_thumbnail(
    *,  # enforce kwargs
    slide: Slide,
    path: Path,
    width: int,
):
    """Slide thumbnail with everything Otsu rejected dimmed out."""
    thumb = slide.thumbnail(width=width)
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
