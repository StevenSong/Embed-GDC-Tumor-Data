"""Collect every slide's `mean` into one slide-level HDF5, `hist.h5`.

The layout is that of `embed/embed_hist_uni2.py` in multimodal-cancer-modeling,
so the survival notebooks there read this file unchanged: one group per case,
named by TCGA submitter barcode (`TCGA-OR-A5J1`), and one float32 `(2560,)`
dataset per slide under it, named by the slide's file stem. A case with several
slides keeps them all; averaging across a case is the reader's call.

A DICOM slide (CPTAC) is grouped by the `case_id` its file carries -- the
header's PatientID, `C3L-04408` -- rather than by parsing its name, since a
CPTAC slide barcode does not encode which part is the case. A file without
`case_id` is a TCGA .svs and must be named by a TCGA barcode.

    python -m embedder.aggregate --emb-dir /out/embeddings --output-h5 /out/hist.h5

The slide vector is the per-slide file's own `mean`, copied rather than
recomputed from `features`. That is both cheaper -- ~10 KB read per slide
against the corpus's ~0.9 TB of tiles -- and more exact: `mean` was accumulated
in float64 from the batches before they were cast to fp16 for storage, so
averaging the stored fp16 `features` would be the lossier of the two.

Every `.h5` in --emb-dir goes in; there is no cohort filtering here. The input
is checked as a whole before anything is written, and one bad file -- no
`mean`, a non-finite one, no case to file it under, or a tiling or
encoder config that differs from the rest -- fails the run. A slide-level file
quietly missing slides, or mixing two QC configs, is the outcome worth ruling
out.
"""

import datetime
import logging
import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from ._logging import progress_disabled, setup_logging

# `TCGA-OR-A5J1-01Z-00-DX1.<uuid>` -> `TCGA-OR-A5J1`, the key the H5 groups by
CASE_ID = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?=-)")

# Per-slide attributes that must agree across the corpus, because they decide
# what a slide vector means. They are copied to the root of the output. Not
# `feature_dtype` or `torch_version`: neither changes `mean`.
CORPUS_ATTRS = (
    "encoder",
    "encoder_source",
    "embed_dim",
    "precision",
    "tile_px",
    "tile_um",
    "qc",
    "qc_min_tissue_mm2",
    "qc_min_occupancy",
)
# per-slide attributes carried onto each slide's dataset
SLIDE_ATTRS = ("n_tiles", "mpp", "mpp_source")


def case_id(slide: str) -> str:
    m = CASE_ID.match(slide)
    if m is None:
        raise ValueError(f"{slide} does not start with a TCGA case barcode")
    return m.group()


def read_slide(path: Path) -> tuple[np.ndarray, dict]:
    with h5py.File(path, "r") as f:
        if "mean" not in f:
            raise ValueError("no `mean` dataset")
        mean = f["mean"][:]
        attrs = dict(f.attrs)
    missing = [k for k in CORPUS_ATTRS + SLIDE_ATTRS if k not in attrs]
    if missing:
        raise ValueError(f"missing root attributes {', '.join(missing)}")
    if mean.shape != (attrs["embed_dim"],):
        raise ValueError(f"`mean` is {mean.shape}, expected ({attrs['embed_dim']},)")
    if not np.isfinite(mean).all():
        raise ValueError("`mean` is not finite")
    return mean, attrs


def aggregate(
    *,  # enforce kwargs
    logger: logging.Logger,
    emb_dir: Path,
    output_h5: Path,
    overwrite: bool,
) -> int:
    # The whole corpus reads in minutes, so there is no resume: the file is
    # written once, to a dotfile, and renamed into place at the end.
    if output_h5.exists() and not overwrite:
        logger.error("%s exists; move it aside or pass --overwrite", output_h5)
        return 1

    # in-flight `.<slide>.h5.tmp` files from the embed step end in .tmp, not .h5
    paths = sorted(emb_dir.glob("*.h5"))
    if not paths:
        logger.error("no .h5 files in %s", emb_dir)
        return 1
    logger.info("reading %d slides from %s", len(paths), emb_dir)

    slides = []
    bad = 0
    config = None
    for path in tqdm(paths, desc="slides", disable=progress_disabled()):
        try:
            mean, attrs = read_slide(path)
            case = str(attrs["case_id"]) if "case_id" in attrs else case_id(path.stem)
            if not case:
                raise ValueError("empty `case_id`")
        except (OSError, ValueError) as exc:
            logger.error("%s: %s", path.name, exc)
            bad += 1
            continue
        slide_config = {k: attrs[k] for k in CORPUS_ATTRS}
        if config is None:
            config, config_from = slide_config, path.name
        diff = [
            k for k in CORPUS_ATTRS if not np.array_equal(slide_config[k], config[k])
        ]
        if diff:
            logger.error(
                "%s: %s differ from %s (%s vs %s)",
                path.name,
                ", ".join(diff),
                config_from,
                [slide_config[k] for k in diff],
                [config[k] for k in diff],
            )
            bad += 1
            continue
        slides.append((case, path.stem, mean, {k: attrs[k] for k in SLIDE_ATTRS}))
    if bad:
        logger.error(
            "%d/%d slides unusable; not writing %s", bad, len(paths), output_h5
        )
        return 1

    tmp = output_h5.with_name(f".{output_h5.name}.tmp")
    with h5py.File(tmp, "w") as h5:
        for case, slide, mean, slide_attrs in slides:
            ds = h5.require_group(case).create_dataset(slide, data=mean)
            ds.attrs.update(slide_attrs)
        n_cases = len(h5)
        h5.attrs.update(
            {
                **config,
                "aggregation": "mean",
                "emb_dir": str(emb_dir),
                "n_slides": len(slides),
                "n_cases": n_cases,
                "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            }
        )
    os.replace(tmp, output_h5)
    logger.info("wrote %s: %d slides across %d cases", output_h5, len(slides), n_cases)
    return 0


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--emb-dir",
        type=Path,
        default=Path("/out/embeddings"),
        help="the embed step's <out-dir>/embeddings",
    )
    p.add_argument("--output-h5", type=Path, default=Path("/out/hist.h5"))
    p.add_argument("--overwrite", action="store_true", help="replace --output-h5")
    args = p.parse_args()
    sys.exit(
        aggregate(
            logger=setup_logging(logging.getLogger("aggregate")),
            emb_dir=args.emb_dir,
            output_h5=args.output_h5,
            overwrite=args.overwrite,
        )
    )
