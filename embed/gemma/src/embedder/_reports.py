"""The report CSV both steps read, and the one `summarize` writes.

Every CSV in this pipeline has the shape of `TCGA_Reports.csv` (Kefeli and
Tatonetti 2024): a `patient_filename` of the form `<case_id>.<file uuid>` and a
`text` column. Summaries are written in that same shape, so `embed` takes the
raw reports and the summaries interchangeably.
"""

import logging
import os
from pathlib import Path

import pandas as pd

REPORT_ID = "patient_filename"
TEXT = "text"


def case_id(report_id: str) -> str:
    """`TCGA-BP-5195.25c0b433-...` -> `TCGA-BP-5195`, the key the H5 groups by."""
    return report_id.split(".")[0]


def read_reports(logger: logging.Logger, path: Path) -> pd.DataFrame:
    """Load a report CSV, refusing anything a later step would silently mangle.

    Rows with no text are dropped with a warning rather than embedded or
    summarized: pandas reads an empty cell as NaN, and NaN goes to the model as
    the string "nan".
    """
    df = pd.read_csv(path)
    missing = {REPORT_ID, TEXT} - set(df.columns)
    if missing:
        raise ValueError(f"{path} has no {sorted(missing)} column(s)")
    dups = df[REPORT_ID][df[REPORT_ID].duplicated()]
    if len(dups):
        raise ValueError(f"{path} repeats {REPORT_ID}s: {', '.join(dups.unique())}")

    empty = df[TEXT].isna() | (df[TEXT].astype(str).str.strip() == "")
    if empty.any():
        logger.warning(
            "dropping %d reports with no text: %s",
            empty.sum(),
            ", ".join(df.loc[empty, REPORT_ID]),
        )
        df = df[~empty]
    return df.reset_index(drop=True)


def write_csv_atomic(df: pd.DataFrame, path: Path):
    """Write to a dotfile and rename, so a killed job never leaves half a CSV."""
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
