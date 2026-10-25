"""Embed GDC bulk RNA-seq with ctds-rnaseq, one 256-d vector per sample, into one HDF5.

    python -m embedder --rna-dir /data/rna-seq --metadata meta.csv \\
        [--manifest manifest.txt] --output-h5 /out/expr.h5

--rna-dir is a `gdc-client download` directory as it is: one `<file id>/`
directory per file, each holding a STAR-counts TSV (plus `logs/` and sometimes
`annotations.txt`, which are ignored). Every TSV in it is embedded unless
--manifest, a GDC manifest (`id`/`filename` columns), narrows that to the files it
lists; a listed file that is not on disk fails the run, so a partial download
cannot pass for a complete one. --metadata maps each file's name to its case
(`file_name`/`case_id` columns, as the tables in metadata/ have), because nothing
in a GDC TSV says which case it came from.

The layout is that of `embed/embed_expr_bulkrnabert.py` in multimodal-cancer-modeling
and of m3ai-demos' `nsclc-eval-data/expr/`: one group per case, named by submitter
id (`TCGA-05-4244`, `C3L-00081`), and one float32 `(256,)` dataset per file under
it, named by the UUID the TSV's file name starts with -- not the GDC file id, which
is the directory name and goes in the dataset's attributes. A case with several
files keeps them all; averaging across a case is the reader's call.

Every input is parsed before the model loads, and one unreadable file, or one
without a case, fails the run without writing anything. The whole of TCGA is ~10k
samples, minutes of GPU, so there is no resume: the file is written to a dotfile
and renamed into place at the end.

A CUDA device is required. Attention runs over all ~20k genes of a sample, and only
the fused kernels do that without materialising a G x G score matrix per head.
"""

import datetime
import hashlib
import logging
import os
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from . import _model
from ._logging import progress_disabled, setup_logging
from ._sample_io import TPM_COLUMN, load_genes, read_tpm

TSV_SUFFIX = ".rna_seq.augmented_star_gene_counts.tsv"
N_PROGRESS_LINES = 20  # per phase, in place of a progress bar when there is none


class Progress:
    """Log `done/total` about N_PROGRESS_LINES times, only when tqdm is disabled.

    A slurm log gets no progress bar (see _logging), and without this it is silent
    from model load until the file is written.
    """

    def __init__(self, logger: logging.Logger, what: str, total: int):
        self.logger, self.what, self.total = logger, what, total
        self.every = max(1, total // N_PROGRESS_LINES)
        self.next = self.every  # a threshold, not a modulus: batches step past it
        self.enabled = progress_disabled()
        self.start = time.monotonic()

    def update(self, done: int):
        if not self.enabled or (done < self.next and done != self.total):
            return
        self.next = (done // self.every + 1) * self.every
        elapsed = time.monotonic() - self.start
        eta = elapsed / done * (self.total - done)
        self.logger.info(
            "%s %d/%d (%.0fs elapsed, ~%.0fs left)",
            self.what,
            done,
            self.total,
            elapsed,
            eta,
        )


@dataclass
class Sample:
    path: Path
    case_id: str
    name: str  # the dataset name: the UUID the file name starts with

    @property
    def file_id(self) -> str:
        # gdc-client writes each file into a directory named by its GDC file id
        return self.path.parent.name


def read_manifest(path: Path) -> set[tuple[str, str]]:
    df = pd.read_csv(path, sep="\t", dtype=str)
    missing = {"id", "filename"} - set(df.columns)
    if missing:
        raise ValueError(f"no {', '.join(sorted(missing))} column in {path}")
    return set(zip(df["id"], df["filename"]))


def read_metadata(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, dtype=str)
    missing = {"file_name", "case_id"} - set(df.columns)
    if missing:
        raise ValueError(f"no {', '.join(sorted(missing))} column in {path}")
    df = df[["file_name", "case_id"]].drop_duplicates()
    dup = df.file_name[df.file_name.duplicated()]
    if len(dup):
        raise ValueError(f"{path} files {dup.iloc[0]} under more than one case")
    return dict(zip(df.file_name, df.case_id))


def select_samples(
    logger: logging.Logger, rna_dir: Path, metadata: Path, manifest: Path | None
) -> list[Sample] | None:
    """The TSVs to embed, each with its case; None (with the reasons logged) if any is unusable."""
    paths = sorted(rna_dir.glob(f"*/*{TSV_SUFFIX}"))
    logger.info("%d RNA-seq TSVs in %s", len(paths), rna_dir)
    if manifest is not None:
        wanted = read_manifest(manifest)
        found = {(p.parent.name, p.name) for p in paths}
        absent = sorted(wanted - found)
        if absent:
            logger.error(
                "%d/%d files in %s are not in %s, e.g. %s",
                len(absent),
                len(wanted),
                manifest.name,
                rna_dir,
                ", ".join("/".join(a) for a in absent[:5]),
            )
            return None
        paths = [p for p in paths if (p.parent.name, p.name) in wanted]
        logger.info("%d selected by %s", len(paths), manifest.name)
    if not paths:
        logger.error("nothing to embed")
        return None

    cases = read_metadata(metadata)
    samples, bad = [], []
    for p in paths:
        if p.name not in cases:
            bad.append(f"{p.parent.name}/{p.name}: no case in {metadata.name}")
            continue
        samples.append(Sample(p, cases[p.name], p.name.split(".", 1)[0]))
    keys = pd.Series([(s.case_id, s.name) for s in samples])
    for case, name in keys[keys.duplicated()]:
        bad.append(f"{case}/{name}: more than one file would be written here")
    if bad:
        for b in bad:
            logger.error("%s", b)
        logger.error("%d/%d files unusable; nothing written", len(bad), len(paths))
        return None
    return samples


_genes: pd.Index | None = None


def _init_worker(genes: pd.Index):
    global _genes
    _genes = genes


def _read(path: Path) -> tuple[np.ndarray | None, dict | str]:
    try:
        return read_tpm(path, _genes)
    except (OSError, ValueError, KeyError) as exc:  # KeyError: a column is missing
        return None, f"{type(exc).__name__}: {exc}"


def read_samples(
    logger: logging.Logger, samples: list[Sample], genes: pd.Index, num_workers: int
) -> tuple[np.ndarray, list[dict]] | None:
    """(N, G) float32 TPM, NaN = absent, and per-sample stats; None if any file is bad."""
    tpm = np.empty((len(samples), len(genes)), dtype=np.float32)
    stats, bad = [], 0
    progress = Progress(logger, "read", len(samples))
    with ProcessPoolExecutor(
        num_workers, initializer=_init_worker, initargs=(genes,)
    ) as pool:
        results = pool.map(_read, [s.path for s in samples], chunksize=8)
        for i, (s, (x, st)) in enumerate(
            tqdm(
                zip(samples, results),
                total=len(samples),
                desc="reading TSVs",
                disable=progress_disabled(),
            )
        ):
            progress.update(i + 1)
            if x is None:
                logger.error("%s/%s: %s", s.file_id, s.path.name, st)
                bad += 1
                continue
            tpm[i] = x
            stats.append(st)
    if bad:
        logger.error("%d/%d files unreadable; nothing written", bad, len(samples))
        return None
    return tpm, stats


@torch.inference_mode()
def embed_tpm(
    logger: logging.Logger,
    encoder: _model.Encoder,
    tpm: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    # Only the fused kernels attend over ~20k positions in bounded memory; pinning
    # them turns a silent fallback to the math kernel (G x G scores per head, OOM at
    # any useful batch size) into an error. The CPU path is for tests.
    backends = [SDPBackend.EFFICIENT_ATTENTION]
    if device.type == "cpu":
        backends += [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
    out = []
    progress = Progress(logger, "embedded", len(tpm))
    with sdpa_kernel(backends):
        for start in tqdm(
            range(0, len(tpm), batch_size),
            desc="embedding",
            unit="batch",
            disable=progress_disabled(),
        ):
            x = torch.from_numpy(tpm[start : start + batch_size]).to(device)
            out.append(encoder(x).cpu().numpy())
            progress.update(min(start + batch_size, len(tpm)))
    return np.concatenate(out)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def embed(
    *,  # enforce kwargs
    logger: logging.Logger,
    rna_dir: Path,
    metadata: Path,
    manifest: Path | None,
    output_h5: Path,
    weights_dir: Path,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
    device: torch.device,
) -> int:
    if output_h5.exists() and not overwrite:
        logger.error("%s exists; move it aside or pass --overwrite", output_h5)
        return 1

    samples = select_samples(logger, rna_dir, metadata, manifest)
    if samples is None:
        return 1
    n_cases = len({s.case_id for s in samples})
    logger.info("%d files across %d cases", len(samples), n_cases)

    genes = load_genes(weights_dir / _model.GENES)
    read = read_samples(logger, samples, genes, num_workers)
    if read is None:
        return 1
    tpm, stats = read
    n_absent = np.array([st["n_absent"] for st in stats])
    if n_absent.any():
        # absent genes go in as MASK_ID, so a high count means the embedding
        # partly reflects missingness rather than expression
        logger.warning(
            "%d/%d samples lack some of the %d vocabulary genes (at most %d)",
            int((n_absent > 0).sum()),
            len(samples),
            len(genes),
            int(n_absent.max()),
        )

    encoder = _model.load_encoder(weights_dir, len(genes)).to(device)
    logger.info("loaded %s on %s", weights_dir, device)
    emb = embed_tpm(logger, encoder, tpm, batch_size, device)
    if not np.isfinite(emb).all():
        bad = [s.name for s, ok in zip(samples, np.isfinite(emb).all(1)) if not ok]
        logger.error("non-finite embeddings for %s; nothing written", ", ".join(bad))
        return 1

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_h5.with_name(f".{output_h5.name}.tmp")
    with h5py.File(tmp, "w") as h5:
        for s, e, st in zip(samples, emb, stats):
            ds = h5.require_group(s.case_id).create_dataset(s.name, data=e)
            ds.attrs.update(file_id=s.file_id, file_name=s.path.name, **st)
        h5.attrs.update(
            {
                "encoder": "ctds-rnaseq",
                "architecture": "BulkRNABert",
                "weights_sha256": sha256(weights_dir / _model.WEIGHTS),
                "genes_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
                "n_genes": len(genes),
                "embed_dim": emb.shape[1],
                "layer": _model.LAYER,
                "pooling": _model.POOLING,
                "binner": encoder.kind,
                "tpm_column": TPM_COLUMN,
                "dtype": "float32",
                "rna_dir": str(rna_dir),
                "manifest": manifest.name if manifest is not None else "",
                "metadata": metadata.name,
                "n_files": len(samples),
                "n_cases": n_cases,
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                # a str subclass, which h5py would store as numpy unicode and fail on
                "torch_version": str(torch.__version__),
            }
        )
    os.replace(tmp, output_h5)
    logger.info("wrote %s: %d files across %d cases", output_h5, len(samples), n_cases)
    return 0


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument(
        "--rna-dir",
        type=Path,
        default=Path("/rna"),  # default inside container
        help="gdc-client download directory of STAR-counts TSVs",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="CSV with file_name and case_id columns, covering every file embedded",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        help="GDC manifest; embed only the files it lists (default: every TSV)",
    )
    p.add_argument("--output-h5", type=Path, required=True)
    p.add_argument(
        "--weights-dir",
        type=Path,
        default=Path(_model.WEIGHTS_DIR),  # baked into the image here
        help=f"directory holding {_model.WEIGHTS}, {_model.BINNER},"
        f" {_model.RUN_CONFIG} and {_model.GENES}",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="samples per forward pass",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="processes parsing TSVs",
    )
    p.add_argument("--overwrite", action="store_true", help="replace --output-h5")
    args = p.parse_args()
    missing = [
        f
        for f in (_model.WEIGHTS, _model.BINNER, _model.RUN_CONFIG, _model.GENES)
        if not (args.weights_dir / f).is_file()
    ]
    if missing:
        p.error(f"--weights-dir {args.weights_dir} has no {', '.join(missing)}")
    if not torch.cuda.is_available():
        p.error("a CUDA device is required")
    sys.exit(
        embed(
            logger=setup_logging(),
            rna_dir=args.rna_dir,
            metadata=args.metadata,
            manifest=args.manifest,
            output_h5=args.output_h5,
            weights_dir=args.weights_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
            device=torch.device("cuda"),
        )
    )
