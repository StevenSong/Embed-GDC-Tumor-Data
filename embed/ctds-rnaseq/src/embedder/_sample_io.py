"""Read one GDC STAR-counts TSV onto the model's gene vocabulary.

float32[G] TPM in vocabulary order, NaN where the gene is absent from the file.

Copied from uc-cdis/omics-encoder `kserve-demo/preprocessor/rnaseq/sample_io.py`
(Apache-2.0, Copyright 2024 University of Chicago; see NOTICE), less the
file-object input the KServe preprocessor needs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TPM_COLUMN = "tpm_unstranded"
META_COLUMNS = ["gene_id", "gene_name", "gene_type"]


def clean_frame(df: pd.DataFrame, keep_par_y: bool = False) -> pd.DataFrame:
    """filter to protein-coding genes and reduce to one row per Ensembl id"""
    df = df[df.gene_id.notna() & df.gene_name.notna()]
    df = df[df.gene_type == "protein_coding"].copy()

    if not keep_par_y:
        df = df[~df.gene_id.str.endswith("_PAR_Y")].copy()

    # split gene id and version
    parts = df.gene_id.str.split(".", n=1, expand=True)
    df["gene_id"] = parts[0]
    df["gene_version"] = (
        pd.to_numeric(parts[1].str.replace("_PAR_Y", "", regex=False), errors="coerce")
        .fillna(-1)
        .astype(int)
    )
    df = df.sort_values(
        ["gene_id", "gene_version"], kind="mergesort", ascending=[True, False]
    )
    return df.drop_duplicates(subset="gene_id", keep="first")


def load_genes(genes_path: Path) -> pd.Index:
    """Training vocabulary. Row order is the model's positional index."""
    return pd.Index(pd.read_csv(genes_path, sep="\t").gene_id)


def read_tpm(path: Path, genes: pd.Index) -> tuple[np.ndarray, dict]:
    """One TSV -> (float32[G] with NaN for absent genes, stats)."""
    df = clean_frame(
        pd.read_csv(path, sep="\t", comment="#", usecols=META_COLUMNS + [TPM_COLUMN])
    )
    series = df.set_index("gene_id")[TPM_COLUMN]

    # NaN *values* in the file are a corrupt source, not an absent gene.
    if series.isna().any():
        raise ValueError(f"{int(series.isna().sum())} NaN TPM values")

    tpm = series.reindex(genes).to_numpy(dtype=np.float32)
    absent = np.isnan(tpm)
    present = tpm[~absent]
    if (~np.isfinite(present)).any() or (present < 0).any():
        raise ValueError("negative or infinite TPM values")

    stats = {
        "n_matched": int((~absent).sum()),
        "n_absent": int(absent.sum()),
        # genes in the file the model has no position for; dropped by reindex
        "n_unknown": int((~series.index.isin(genes)).sum()),
    }
    return tpm, stats
