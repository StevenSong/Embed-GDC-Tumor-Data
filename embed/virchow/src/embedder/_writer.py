from pathlib import Path

import h5py
import numpy as np

H5_CHUNK_ROWS = 256


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
            dtype=np.float32,  # virchow embedding type
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
            dtype=int,
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
        """Global average embedding over every tile in the slide."""
        mean = self.total / self.n if self.n else np.full_like(self.total, np.nan)
        self.f.create_dataset("mean", data=mean.astype(np.float32))

    def write_metadata(self, meta: dict):
        for key, value in meta.items():
            self.f.attrs[key] = value

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
