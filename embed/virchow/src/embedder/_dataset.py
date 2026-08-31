from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from ._reader import Slide

MAX_TILES_PER_WSI = 200000


class TileDataset(IterableDataset):
    # NOTE: each dataloader worker collates its own batch so each worker will
    # emit a partial batch. two consequences follow:
    # * simple length/batch_size estimate underestimates (cosmetic error)
    # * DataLoader `drop_last` drops each worker's last batch (correctness error)
    # def __len__(self):

    def __init__(self, *, slide_paths: list[Path], tile_px: int, target_mpp: float):
        super().__init__()
        self.slide_paths = slide_paths
        self._slide_to_index = {p: i for i, p in enumerate(slide_paths)}
        self.tile_px = tile_px
        self.target_mpp = target_mpp

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
        slide_index = self._slide_to_index[
            slide_path
        ]  # @Claude, this is a little bit of indirection, is it easier to fill a str tensor?
        self._curr_slide.fill_(slide_index)

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

        start, end = 0, num_tiles
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # contiguous blocks so we get good slide segment cache hit rate
            per_worker = -(-num_tiles // worker_info.num_workers)
            start = worker_info.id * per_worker
            end = min(start + per_worker, num_tiles)
        if start >= end:  # fewer tiles than workers
            return

        coords = self.coords_buffer[:num_tiles]
        with Slide(
            path=self.slide_paths[curr_slide],
            tile_px=self.tile_px,
            target_mpp=self.target_mpp,
            coords=coords.numpy(),
        ) as slide:
            for i in range(start, end):
                yield slide[i], coords[i]
