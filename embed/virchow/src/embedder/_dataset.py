import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from ._reader import Slide

MAX_TILES_PER_WSI = 200000


def _shared_uint8(*shape: int) -> torch.Tensor:
    """A uint8 tensor whose storage is allocated directly in shared memory.

    The point is what it avoids. A plain `torch.empty(...)` lives in ordinary
    anonymous memory, and handing it to the parent makes `reduce_tensor` call
    `share_memory_()`, which allocates a *second* buffer in /dev/shm and copies
    into it -- so a 77 MB batch costs two fresh-page allocations and a 77 MB
    memcpy. Allocating shared up front and decoding the tiles straight into it
    makes the handover an fd pass: one allocation, no copy.

    That matters far more than the memcpy. Fresh pages have to come from
    somewhere, and by late in a corpus run the page cache has taken all of
    memory, so every allocation lands in synchronous direct reclaim. Halving the
    allocations halves the exposure. This is the same idiom torch's own default
    collate uses to build a batch in a worker.
    """
    numel = 1
    for dim in shape:
        numel *= int(dim)
    proto = torch.empty(0, dtype=torch.uint8)
    storage = proto._typed_storage()._new_shared(numel, device=proto.device)
    return proto.new(storage).resize_(*shape)

# per-worker counters, one row per worker so no two processes ever write the same
# cell and the whole thing needs no locking
STAT_HITS = 0  # segment cache hits
STAT_MISSES = 1  # segment cache misses -- (hits+misses)/tiles is segments per tile
STAT_BATCHES = 2
STAT_TILES = 3
STAT_OPEN_US = 4  # Slide.__init__, paid once per worker per slide
STAT_READ_US = 5  # inside Slide.read_batch
STAT_FIRST_US = 6  # __iter__ entry to first yield returning
STAT_WALL_US = 7  # whole __iter__, so wall-read-open is time blocked on the queue
N_STATS = 8


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
        num_workers: int,
    ):
        super().__init__()
        self.slide_paths = slide_paths
        self._slide_to_index = {p: i for i, p in enumerate(slide_paths)}
        self.tile_px = tile_px
        self.target_mpp = target_mpp
        self.batch_size = batch_size
        self.num_workers = num_workers

        # information set on parent instance that dataloder worker instances need
        self.coords_buffer = torch.zeros(
            (MAX_TILES_PER_WSI, 2),
            dtype=torch.int32,
        ).share_memory_()
        self._curr_slide = torch.tensor(-1, dtype=torch.int32).share_memory_()
        self._num_tiles = torch.tensor(-1, dtype=torch.int32).share_memory_()
        # written back by the workers, one row each; see the STAT_* columns
        self._stats = torch.zeros(
            (max(num_workers, 1), N_STATS), dtype=torch.int64
        ).share_memory_()

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

    def worker_stats(self) -> dict:
        """Summary of what the readers did on the epoch that just finished.

        Only meaningful once the loader has raised StopIteration: each worker
        writes its row as its `__iter__` unwinds, which happens before the
        end-of-iteration signal the parent is waiting on.
        """
        stats = self._stats.numpy()
        active = stats[stats[:, STAT_TILES] > 0]
        tiles = int(stats[:, STAT_TILES].sum())
        segments = int(stats[:, STAT_HITS].sum() + stats[:, STAT_MISSES].sum())
        read_us = stats[:, STAT_READ_US]
        return {
            "workers_active": int(len(active)),
            "tiles": tiles,
            "seg_per_tile": round(segments / tiles, 2) if tiles else None,
            "cache_hit_rate": (
                round(float(stats[:, STAT_HITS].sum()) / segments, 3)
                if segments
                else None
            ),
            "open_s_max": round(float(stats[:, STAT_OPEN_US].max()) / 1e6, 3),
            # a worker's own read time vs its wall time: the gap is time it spent
            # blocked pushing batches onto the queue, i.e. waiting for the GPU
            "read_s_max": round(float(read_us.max()) / 1e6, 2),
            "read_s_mean_active": (
                round(float(read_us[stats[:, STAT_TILES] > 0].mean()) / 1e6, 2)
                if len(active)
                else None
            ),
            "wall_s_max": round(float(stats[:, STAT_WALL_US].max()) / 1e6, 2),
            "first_yield_s_max": round(float(stats[:, STAT_FIRST_US].max()) / 1e6, 2),
            # straggler ratio: 1.0 means every active worker took the same wall
            # time, higher means the slide waited on one of them
            "straggler": (
                round(
                    float(active[:, STAT_WALL_US].max())
                    / float(active[:, STAT_WALL_US].mean()),
                    2,
                )
                if len(active)
                else None
            ),
            "batches_per_worker": [int(v) for v in stats[:, STAT_BATCHES]],
            "read_s_per_worker": [round(float(v) / 1e6, 1) for v in read_us],
        }

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
            # zero this worker's row up front so a worker that ends up with no
            # batches reports nothing rather than last slide's numbers
            stats = self._stats[worker_info.id]
            stats.zero_()
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
        t_enter = time.perf_counter_ns()
        read_ns = 0
        slide = None
        try:
            t = time.perf_counter_ns()
            with Slide(
                path=self.slide_paths[curr_slide],
                tile_px=self.tile_px,
                target_mpp=self.target_mpp,
                coords=coords.numpy(),
            ) as slide:
                stats[STAT_OPEN_US] = (time.perf_counter_ns() - t) // 1000
                for batch in range(lo, hi):
                    start = batch * self.batch_size
                    end = min(start + self.batch_size, num_tiles)
                    t = time.perf_counter_ns()
                    tiles = _shared_uint8(end - start, self.tile_px, self.tile_px, 3)
                    slide.read_batch_into(tiles.numpy(), range(start, end))
                    read_ns += time.perf_counter_ns() - t
                    if batch == lo:  # this worker's share of the pipeline fill
                        stats[STAT_FIRST_US] = (time.perf_counter_ns() - t_enter) // 1000
                    stats[STAT_BATCHES] += 1
                    stats[STAT_TILES] += end - start
                    # clone: `coords` is a view on the shared buffer the parent
                    # overwrites for the next slide, and only an owned tensor is
                    # safe to hand across the queue
                    yield tiles, coords[start:end].clone()
        finally:
            # in a finally so a worker shut down mid-epoch still reports what it
            # did; this runs before the end-of-iteration signal the parent waits
            # on. Slide.close leaves the cache counters intact, so reading them
            # after the `with` has unwound is still correct
            if slide is not None:
                stats[STAT_HITS], stats[STAT_MISSES] = slide.cache_stats
            stats[STAT_READ_US] = read_ns // 1000
            stats[STAT_WALL_US] = (time.perf_counter_ns() - t_enter) // 1000
