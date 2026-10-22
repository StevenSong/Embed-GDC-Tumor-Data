from pathlib import Path

import h5py
import numpy as np

H5_CHUNK_ROWS = 256

# Storage dtype for `features`. Virchow runs under fp16 autocast and only its
# final LayerNorm is mixed-precision, so fp32 storage was recording precision
# the model never produced -- at 2x the bytes. The corpus is ~184M tiles of 2560
# floats, which is 1.9 TB at fp32 against 0.94 TB here. `mean` is accumulated in
# float64 from the full-precision batch before this cast, so the slide-level
# vector does not inherit the storage precision.
FEATURE_DTYPE = np.float16


class SlideH5:
    """Streaming writer: embeddings are appended per batch, never held in full."""

    def __init__(self, *, path: Path, gzip_level: int, embed_dim):  # enforce kwargs
        compression = (
            {"compression": "gzip", "compression_opts": gzip_level}
            if gzip_level
            else {}
        )
        self.f = h5py.File(path, "w")
        self.features = self.f.create_dataset(
            "features",
            shape=(0, embed_dim),
            maxshape=(None, embed_dim),
            chunks=(H5_CHUNK_ROWS, embed_dim),
            dtype=FEATURE_DTYPE,
            **compression,
        )
        self.grid = self._coords("grid")
        self.loc = self._coords("loc")
        self.n = 0
        # running sum for the slide-level mean embedding; float64 so a slide with
        # hundreds of thousands of tiles doesn't lose precision to accumulation
        self.total = np.zeros(embed_dim, dtype=np.float64)

    def _coords(self, name: str) -> h5py.Dataset:
        return self.f.create_dataset(
            name,
            shape=(0, 2),
            maxshape=(None, 2),
            chunks=(4096, 2),
            # int32 is ample: base-level coordinates top out around 3e5
            dtype=np.int32,
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
        self.total += features.sum(axis=0, dtype=np.float64)

    def write_mean(self):
        """Global average embedding over every tile in the slide.

        A tile-less slide used to get a NaN mean here. It now raises: the caller
        is expected to have failed the slide long before this point (see
        `NoTissueError`), and silently emitting a vector that looks valid is the
        one outcome worth ruling out twice.
        """
        if not self.n:
            raise ValueError("no tiles were appended, so there is no mean to write")
        self.f.create_dataset("mean", data=(self.total / self.n).astype(np.float32))

    def write_metadata(self, meta: dict):
        for key, value in meta.items():
            self.f.attrs[key] = value

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
