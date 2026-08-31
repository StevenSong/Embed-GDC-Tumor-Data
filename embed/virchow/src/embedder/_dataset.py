from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from ._reader import Slide

MAX_TILES_PER_WSI = 200000


class TileDataset(IterableDataset):
    """Tiles of the current slide, yielded one whole batch at a time.

    One item is one already-stacked ``(tiles, coords)`` batch, so this must be
    used with ``DataLoader(batch_size=None)`` -- the loader does no collation of
    its own. That is what keeps the shape stream static: letting the loader
    collate per worker makes every one of the N workers emit its own short
    trailing batch, and a ragged shape stream forces dynamo onto a dynamic-shape
    graph that gives up essentially all of `torch.compile`'s speedup. Here the
    only short batch in a slide is its genuine last one, which `_pipeline` pads
    up to `batch_size` and slices back.

    Workers split *batch* indices, not tile indices, and take a contiguous run of
    them: a tile spans 3x3 TIFF segments, so scattering a worker's tile indices
    costs 1.7x (JPEG) / 2.2x (JPEG 2000) on the segment cache.
    """

    def __init__(
        self,
        *,  # enforce kwargs
        slide_paths: list[Path],
        tile_px: int,
        target_mpp: float,
        batch_size: int,
    ):
        super().__init__()
        self.slide_paths = slide_paths
        self._slide_to_index = {p: i for i, p in enumerate(slide_paths)}
        self.tile_px = tile_px
        self.target_mpp = target_mpp
        self.batch_size = batch_size

        # information set on parent instance that dataloder worker instances need
        self.coords_buffer = torch.zeros(
            (MAX_TILES_PER_WSI, 2),
            dtype=torch.int32,
        ).share_memory_()
        self._curr_slide = torch.tensor(-1, dtype=torch.int32).share_memory_()
        self._num_tiles = torch.tensor(-1, dtype=torch.int32).share_memory_()

    def set_next_slide_tiles(
        self,
        *,
        slide_path: Path,
        masked_coords: np.ndarray,
    ):
        num_tiles = len(masked_coords)
        if num_tiles > MAX_TILES_PER_WSI:
            raise ValueError(
                f"Slide has more tiles ({num_tiles}) than maximum allowed ({MAX_TILES_PER_WSI})"
            )
        self.coords_buffer[:num_tiles] = torch.as_tensor(masked_coords)
        self._num_tiles.fill_(num_tiles)
        # NOTE: the index, rather than the path itself, is what crosses to the workers.
        # This is a small amount of indirection that simplifies the rest of the code base.
        self._curr_slide.fill_(self._slide_to_index[slide_path])

    def __len__(self) -> int:
        """Batches in the current slide."""
        return -(-int(self._num_tiles) // self.batch_size)

    def __iter__(self):
        curr_slide, num_tiles = int(self._curr_slide), int(self._num_tiles)
        if curr_slide < 0:
            raise ValueError(
                "looks like curr_slide was not set in parent dataset, did you call `set_next_slide_tiles`?`"
            )
        if num_tiles < 0:
            raise ValueError(
                "looks like num_coords was not set in parent dataset, did you call `set_next_slide_tiles`?"
            )

        n_batches = -(-num_tiles // self.batch_size)
        lo, hi = 0, n_batches
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # contiguous blocks so we get good slide segment cache hit rate.
            # `round` on a float stride rather than a ceiling division: the ranges
            # still tile [0, n_batches) exactly, but the remainder is spread over
            # the workers instead of starving the last one
            per_worker = n_batches / worker_info.num_workers
            lo = round(worker_info.id * per_worker)
            hi = round((worker_info.id + 1) * per_worker)
        else:
            raise RuntimeError(
                "TileDataset should only be iterated inside a torch dataloader worker process!"
            )
        if lo >= hi:  # fewer batches than workers
            return

        coords = self.coords_buffer[:num_tiles]
        with Slide(
            path=self.slide_paths[curr_slide],
            tile_px=self.tile_px,
            target_mpp=self.target_mpp,
            coords=coords.numpy(),
        ) as slide:
            for batch in range(lo, hi):
                start = batch * self.batch_size
                end = min(start + self.batch_size, num_tiles)
                tiles = slide.read_batch(range(start, end))
                # clone: `coords` is a view on the shared buffer the parent
                # overwrites for the next slide, and only an owned tensor is safe
                # to hand across the queue
                yield torch.from_numpy(tiles), coords[start:end].clone()
