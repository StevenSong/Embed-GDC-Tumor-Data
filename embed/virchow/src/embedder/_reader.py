"""Tile reader for Aperio SVS slides, built directly on tifffile.

Scope is deliberately narrow: GDC/TCGA diagnostic slides, which are pyramidal
tiled TIFFs whose base level is either baseline JPEG (compression 7, abbreviated
streams that need the page's shared `jpegtables`) or Aperio JPEG 2000
(compression 33003/33005, self-contained codestreams). Anything else -- striped
TIFFs, .mrxs, .ndpi, .bif -- is rejected rather than half-supported.

Tissue detection follows CLAM (Lu et al., Nat Biomed Eng 2021): HSV saturation,
median blur, Otsu, morphological closing, then a contour filter with hole
filling. A tile is kept when its centre falls on tissue.

`Slide` is a sequence of tiles::

    slide = Slide("TCGA-A2-A25A-01Z-00-DX1.svs")
    len(slide)          # tiles surviving tissue detection
    slide[0]            # (224, 224, 3) uint8 RGB
    slide.coords[0]     # base-level (x, y) of that tile's top-left corner

Indexing decodes only the TIFF segments the tile actually overlaps. Adjacent
tiles share segments, so reads stay near the sequential-scan cost as long as
indices arrive in roughly ascending order -- see `Slide.cache_stats`.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np
import tifffile

# every Slide lives in its own worker process so prevent OpenCV from doing it's own parallelism
cv2.setNumThreads(0)

# Aperio's own compression codes, alongside baseline JPEG (7)
JPEG = 7
APERIO_JP2000_YCBC = 33003
APERIO_JP2000_RGB = 33005
SUPPORTED_COMPRESSION = (JPEG, APERIO_JP2000_YCBC, APERIO_JP2000_RGB)

# CLAM segmentation defaults (CLAM/wsi_core/WholeSlideImage.py segmentTissue)
SEG_DOWNSAMPLE = 64  # target downsample to segment at; nearest available is used
MEDIAN_BLUR_PX = 7
CLOSE_KERNEL_PX = 4
REF_PATCH_PX = 512  # area unit the contour thresholds below are expressed in
MIN_TISSUE_AREA = 100  # in ref-patch areas
MIN_HOLE_AREA = 16  # in ref-patch areas
MAX_HOLES_PER_CONTOUR = 8

# decoded segments held per slide; ~2 segment rows of a 90k-wide slide, which is
# all a row-major scan ever needs live at once
SEGMENT_CACHE_MB = 96


class UnsupportedSlideError(ValueError):
    """Slide is not a tiled Aperio-style pyramid we can read."""


def _parse_mpp(description: str) -> float:
    """Microns per pixel from the Aperio header in the base page's description."""
    match = re.search(r"\|\s*MPP\s*=\s*([0-9.]+)", description)
    if match is None:
        raise UnsupportedSlideError("no MPP in the Aperio image description")
    return float(match.group(1))


class _SegmentCache:
    """Byte-budgeted LRU over decoded TIFF segments."""

    def __init__(self, max_bytes: int):
        self._entries: OrderedDict[int, np.ndarray] = OrderedDict()
        self._max_bytes = max_bytes
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: int) -> np.ndarray | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry

    def put(self, key: int, segment: np.ndarray):
        if key in self._entries:
            return
        self._entries[key] = segment
        self._bytes += segment.nbytes
        # keep at least one entry so a cache smaller than a single segment still
        # serves the tile being assembled right now
        while self._bytes > self._max_bytes and len(self._entries) > 1:
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= evicted.nbytes

    def clear(self):
        self._entries.clear()
        self._bytes = 0


class Slide:
    """An SVS slide as an indexable sequence of tissue tiles.

    Tiles are read from the base level and resized to `tile_px`. Most of TCGA is
    scanned at 40x, where a 0.5 MPP tile is a ~2x downsample of ~440-490 base
    pixels; a minority are 20x (MPP up to ~0.504), where `extract_px` lands at
    or just under `tile_px`. Either way the base level is the only one worth
    reading -- the next level down is a 4x downsample, i.e. ~1.0 MPP.

    Pass `coords` to skip tissue detection entirely -- reader processes take the
    grid computed once by the parent instead of re-running Otsu per worker.
    """

    def __init__(
        self,
        path: Path,
        *,  # enforce kwargs
        tile_px: int,
        target_mpp: float,
        coords: np.ndarray | None = None,
        segment_cache_mb: int = SEGMENT_CACHE_MB,
    ):
        self.path = str(path)
        self.tile_px = tile_px
        self.target_mpp = target_mpp

        self._tif = tifffile.TiffFile(self.path)
        series = self._tif.series[0]  # "Baseline"; series[1] is Aperio's thumbnail
        self._levels = series.levels
        self._page = self._levels[0].keyframe

        page = self._page
        if not page.is_tiled:
            raise UnsupportedSlideError(f"{self.path}: base level is not tiled")
        if page.compression not in SUPPORTED_COMPRESSION:
            raise UnsupportedSlideError(
                f"{self.path}: unsupported compression {page.compression!r}"
            )
        # asserted once here so every read below can assume (h, w, 3) uint8 and
        # index rather than reshape defensively; a CMYK or 16-bit page would
        # otherwise sail through and come out as plausible-looking garbage
        if page.photometric != tifffile.PHOTOMETRIC.RGB:
            raise UnsupportedSlideError(
                f"{self.path}: photometric {page.photometric!r}, expected RGB"
            )
        if page.samplesperpixel != 3 or page.extrasamples:
            raise UnsupportedSlideError(
                f"{self.path}: {page.samplesperpixel} samples/px"
                f"{' + extrasamples' if page.extrasamples else ''}, expected 3"
            )
        if page.bitspersample != 8:
            raise UnsupportedSlideError(
                f"{self.path}: {page.bitspersample} bits/sample, expected 8"
            )

        self.mpp = _parse_mpp(page.description)
        self.dimensions = (int(page.imagewidth), int(page.imagelength))  # (x, y)

        # base-level pixels per tile: ~440-490 on the 40x majority, but down to
        # tile_px itself on the 20x tail, so read_region resizes in either
        # direction and skips the resize entirely when they happen to match
        self.extract_px = int(round(tile_px * target_mpp / self.mpp))

        self._tile_w = int(page.tilewidth)
        self._tile_h = int(page.tilelength)
        self._cols = -(-self.dimensions[0] // self._tile_w)  # segments per row
        self._jpegtables = page.jpegtables  # None for JPEG 2000

        # own handle: tifffile's is guarded by a lock we would only contend on
        self._fh = open(self.path, "rb")
        self._cache = _SegmentCache(segment_cache_mb * 1024 * 1024)

        self._seg_level = self._pick_seg_level()
        self._tissue_mask: np.ndarray | None = None
        self._coords: np.ndarray | None = None
        if coords is not None:
            self._coords = np.ascontiguousarray(coords, dtype=np.int32)

    # ------------------------------------------------------------------ levels

    def _pick_seg_level(self) -> int:
        """Pyramid level nearest SEG_DOWNSAMPLE, for tissue detection."""
        downsamples = [self.dimensions[0] / level.shape[1] for level in self._levels]
        return int(np.argmin([abs(d - SEG_DOWNSAMPLE) for d in downsamples]))

    @property
    def seg_downsample(self) -> float:
        return self.dimensions[0] / self._levels[self._seg_level].shape[1]

    def level_image(self, level: int) -> np.ndarray:
        """Whole pyramid level as (H, W, 3) uint8 RGB."""
        return np.asarray(self._levels[level].asarray())

    def thumbnail(self, width: int = 2048) -> np.ndarray:
        """Downsampled whole-slide image, `width` px across."""
        # levels run large -> small, so the highest qualifying index is the
        # smallest level still at least `width` across. The fallback is level 0,
        # the full base image, reachable only if `width` exceeds the slide itself
        # -- decoding that would materialise the whole slide, so refuse instead
        if width > self.dimensions[0]:
            raise ValueError(
                f"thumbnail width {width} exceeds slide width {self.dimensions[0]}"
            )
        level = max(
            (i for i, lv in enumerate(self._levels) if lv.shape[1] >= width),
            default=0,
        )
        img = self.level_image(level)
        height = max(1, round(img.shape[0] * width / img.shape[1]))
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------- tissue detection

    @property
    def tissue_mask(self) -> np.ndarray:
        """Bool mask at the segmentation level, True on tissue."""
        if self._tissue_mask is None:
            self._tissue_mask = self._segment_tissue()
        return self._tissue_mask

    def _segment_tissue(self) -> np.ndarray:
        img = self.level_image(self._seg_level)

        # saturation separates stained tissue from the white/grey background far
        # better than luminance; blur first so Otsu is not led by JPEG speckle
        saturation = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]
        saturation = cv2.medianBlur(saturation, MEDIAN_BLUR_PX)
        _, binary = cv2.threshold(
            saturation, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY
        )
        kernel = np.ones((CLOSE_KERNEL_PX, CLOSE_KERNEL_PX), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        mask = np.zeros(binary.shape, dtype=np.uint8)
        if hierarchy is None:
            return mask.astype(bool)

        # thresholds are given in units of a 512 px reference patch at level 0
        scale = self.seg_downsample
        ref_area = REF_PATCH_PX**2 / (scale * scale)
        min_tissue = MIN_TISSUE_AREA * ref_area
        min_hole = MIN_HOLE_AREA * ref_area

        hierarchy = hierarchy[0]  # (n, 4): next, prev, first_child, parent
        for i, contour in enumerate(contours):
            if hierarchy[i][3] != -1:  # not a top-level (foreground) contour
                continue
            holes = [j for j in range(len(contours)) if hierarchy[j][3] == i]
            hole_areas = [cv2.contourArea(contours[j]) for j in holes]
            # a ring of tissue is worth its ring, not its bounding blob
            area = cv2.contourArea(contour) - sum(hole_areas)
            if area <= min_tissue:
                continue
            cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
            # only the largest few holes are real lumen; the rest is speckle
            biggest = sorted(zip(hole_areas, holes), reverse=True)
            for hole_area, j in biggest[:MAX_HOLES_PER_CONTOUR]:
                if hole_area > min_hole:
                    cv2.drawContours(mask, [contours[j]], -1, 0, cv2.FILLED)

        return mask.astype(bool)

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Full tile grid (n_cols, n_rows) before tissue filtering."""
        return (
            self.dimensions[0] // self.extract_px,
            self.dimensions[1] // self.extract_px,
        )

    @property
    def coords(self) -> np.ndarray:
        """(N, 2) int32 base-level (x, y) top-left corner of each kept tile."""
        if self._coords is None:
            self._coords = self._tissue_coords()
        return self._coords

    @property
    def grid(self) -> np.ndarray:
        """(N, 2) int32 (col, row) index of each kept tile in the full grid."""
        return self.coords // self.extract_px

    def _tissue_coords(self) -> np.ndarray:
        """Row-major grid of tiles whose centre lands on tissue."""
        n_cols, n_rows = self.grid_shape
        # partial tiles at the right/bottom edge are dropped rather than padded
        xs = np.arange(n_cols, dtype=np.int32) * self.extract_px
        ys = np.arange(n_rows, dtype=np.int32) * self.extract_px

        mask = self.tissue_mask
        scale = self.seg_downsample
        centre = self.extract_px / 2
        mx = np.clip(((xs + centre) / scale).astype(int), 0, mask.shape[1] - 1)
        my = np.clip(((ys + centre) / scale).astype(int), 0, mask.shape[0] - 1)

        keep = mask[np.ix_(my, mx)]  # (n_rows, n_cols)
        rows, cols = np.nonzero(keep)  # nonzero is row-major, so coords are too
        return np.stack([xs[cols], ys[rows]], axis=1).astype(np.int32)

    # ------------------------------------------------------------ tile reading

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: int) -> np.ndarray:
        """Tile `index` as (tile_px, tile_px, 3) uint8 RGB."""
        x, y = (int(v) for v in self.coords[index])
        return self.read_region(x, y)

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self)):
            yield self[i]

    def read_batch(self, indices: Sequence[int]) -> np.ndarray:
        """(B, tile_px, tile_px, 3) uint8. Ascending indices read fastest."""
        out = np.empty((len(indices), self.tile_px, self.tile_px, 3), dtype=np.uint8)
        for i, index in enumerate(indices):
            out[i] = self[index]
        return out

    def read_region(self, x: int, y: int) -> np.ndarray:
        """`extract_px` square at base-level (x, y), resized to `tile_px`."""
        size = self.extract_px
        tw, th = self._tile_w, self._tile_h

        # the segments this region straddles -- typically 3x3 for a 444 px region
        # over 240 px segments, and never the whole page
        col0, col1 = x // tw, (x + size - 1) // tw
        row0, row1 = y // th, (y + size - 1) // th

        canvas = np.empty(
            ((row1 - row0 + 1) * th, (col1 - col0 + 1) * tw, 3), dtype=np.uint8
        )
        for row in range(row0, row1 + 1):
            for col in range(col0, col1 + 1):
                segment = self._segment(row * self._cols + col)
                top, left = (row - row0) * th, (col - col0) * tw
                canvas[top : top + th, left : left + tw] = segment

        top, left = y - row0 * th, x - col0 * tw
        region = canvas[top : top + size, left : left + size]
        if size == self.tile_px:  # 20x scan at exactly the target MPP
            return np.ascontiguousarray(region)
        # INTER_AREA is the right filter for the ~2x downsample a 40x scan needs,
        # but degenerates to nearest when upsampling, which some 20x scans do
        interpolation = cv2.INTER_AREA if size > self.tile_px else cv2.INTER_LINEAR
        return cv2.resize(
            region, (self.tile_px, self.tile_px), interpolation=interpolation
        )

    def _segment(self, key: int) -> np.ndarray:
        """Decoded (tile_h, tile_w, 3) segment `key`, from cache when possible."""
        segment = self._cache.get(key)
        if segment is not None:
            return segment

        count = self._page.databytecounts[key]
        if count == 0:  # sparse segment: Aperio writes nothing for blank tiles
            segment = np.zeros((self._tile_h, self._tile_w, 3), dtype=np.uint8)
        else:
            self._fh.seek(self._page.dataoffsets[key])
            data = self._fh.read(count)
            # tifffile owns the Aperio quirks: abbreviated JPEG streams need the
            # page's shared tables, and 33003 carries an undeclared YCbCr
            # transform that a raw jpeg2k decode gets wrong
            decoded, _, _ = self._page.decode(data, key, jpegtables=self._jpegtables)
            # decode returns a leading sample axis; the shape is guaranteed by the
            # photometric/samplesperpixel checks in __init__, so reshape strictly
            segment = np.asarray(decoded).reshape(self._tile_h, self._tile_w, 3)

        self._cache.put(key, segment)
        return segment

    @property
    def cache_stats(self) -> tuple[int, int]:
        """(hits, misses) on the segment cache -- the read-order health check."""
        return self._cache.hits, self._cache.misses

    # ----------------------------------------------------------------- cleanup

    def close(self):
        self._cache.clear()
        self._fh.close()
        self._tif.close()

    def __enter__(self) -> "Slide":
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        return (
            f"Slide({self.path!r}, {self.dimensions[0]}x{self.dimensions[1]}, "
            f"{self.mpp:.4f} mpp, extract_px={self.extract_px})"
        )
