"""Split CPTAC `hist.h5` and/or `expr.h5` into the per-project files m3ai-demos reads.

The lung cancer classification notebook loads its CPTAC eval set from
`nsclc-eval-data/{hist,expr}/{CPTAC-LUAD,CPTAC-LSCC}.h5`: one file per modality
and project, one group per case, one 1-D float32 dataset per slide or RNA-seq
file under it, averaged per case by the notebook. `embed/virchow`'s aggregate and
`embed/ctds-rnaseq` already write that layout, but for both projects in one file,
so all this does is split each on the `collection_id` its metadata table gives
a case. Slides are matched to the table by dataset name, their ContainerIdentifier,
and RNA-seq files by the dataset's `file_name` attribute.

For slides, `collection_id` is the IDC collection the slide is in. For RNA-seq,
it is mapped from the case's GDC disease type by the manifest notebook, since
GDC files every CPTAC lung case under one project, CPTAC-3. The two sources
disagree on a case now and then (C3N-02282: IDC cptac_luad, GDC squamous).
The notebook intersects cases across the union of both projects, so such a
case would be kept under LUAD in one modality and LSCC in the other and fail
its per-class count asserts. It is dropped from both modalities instead,
which is why both metadata tables are always read, even to export one modality.

    python scripts/export_m3ai_demos_cptac.py \
        --hist-h5 /path/to/virchow/hist.h5 \
        --expr-h5 /path/to/ctds-rnaseq/expr.h5 \
        --out-dir external/m3ai-demos/demos/nsclc-eval-data

Every dataset must be in its metadata under the same case, or nothing is
written. Metadata entries missing from an `.h5` are only reported.
"""

import csv
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import h5py

METADATA = Path(__file__).resolve().parent.parent / "metadata"

# metadata `collection_id` -> the file name the notebook loads
PROJECTS = {"cptac_luad": "CPTAC-LUAD", "cptac_lscc": "CPTAC-LSCC"}
# modality -> the metadata column a dataset is matched on, and the output's count attribute
MODALITIES = {
    "hist": {"key": "slide", "count": "n_slides"},
    "expr": {"key": "file_name", "count": "n_files"},
}


def read_metadata(path: Path, modality: str) -> tuple[dict, dict]:
    """(key -> (case, project), case -> project); unknown or mixed labels raise."""
    by_key, by_case = {}, {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            case, collection = row["case_id"], row["collection_id"]
            if collection not in PROJECTS:
                raise ValueError(
                    f"{path.name}: {case} has unknown collection {collection!r}"
                )
            project = PROJECTS[collection]
            if by_case.setdefault(case, project) != project:
                raise ValueError(
                    f"{path.name}: {case} is under more than one collection"
                )
            by_key[row[MODALITIES[modality]["key"]]] = (case, project)
    return by_key, by_case


def dataset_key(modality: str, ds: h5py.Dataset, name: str) -> str:
    # a slide is named by its ContainerIdentifier; an RNA-seq file carries its name
    return name if modality == "hist" else ds.attrs.get("file_name")


def plan(modality: str, src: h5py.File, meta: dict, drop: set) -> tuple[dict, list]:
    """(project -> [(case, dataset)], problems), and report what is left out."""
    by_project, seen, bad = defaultdict(list), set(), []
    for case in src:
        for name in src[case]:
            key = dataset_key(modality, src[case][name], name)
            if key not in meta:
                bad.append(f"{modality} {case}/{name}: {key} not in metadata")
            elif meta[key][0] != case:
                bad.append(
                    f"{modality} {case}/{name}: metadata files it under {meta[key][0]}"
                )
            else:
                seen.add(key)
                if case not in drop:
                    by_project[meta[key][1]].append((case, name))
    missing = sorted(set(meta) - seen)
    if missing:
        print(
            f"warning: {len(missing)}/{len(meta)} {modality} metadata entries not in "
            f"{Path(src.filename).name}, e.g. {', '.join(missing[:5])}",
            file=sys.stderr,
        )
    return by_project, bad


def write(modality: str, src: h5py.File, by_project: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for project in sorted(set(PROJECTS.values())):
        pairs = by_project[project]
        output = out_dir / f"{project}.h5"
        tmp = output.with_name(f".{output.name}.tmp")
        with h5py.File(tmp, "w") as dst:
            for case, name in pairs:
                src.copy(src[case][name], dst.require_group(case), name=name)
            n_cases = len(dst)
            dst.attrs.update(src.attrs)  # encoder, ... from the embed step
            dst.attrs.update(
                {MODALITIES[modality]["count"]: len(pairs), "n_cases": n_cases}
            )
        os.replace(tmp, output)
        print(f"wrote {output}: {len(pairs)} datasets across {n_cases} cases")


def main(args) -> int:
    inputs = {m: h5 for m, h5 in (("hist", args.hist_h5), ("expr", args.expr_h5)) if h5}
    try:
        metas = {
            "hist": read_metadata(args.slide_metadata, "hist"),
            "expr": read_metadata(args.rna_metadata, "expr"),
        }
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    hist_cases, expr_cases = metas["hist"][1], metas["expr"][1]
    drop = {
        c
        for c in hist_cases.keys() & expr_cases.keys()
        if hist_cases[c] != expr_cases[c]
    }
    for case in sorted(drop):
        print(
            f"dropping {case}: histology {hist_cases[case]}, RNA-seq {expr_cases[case]}",
            file=sys.stderr,
        )

    outputs = [
        args.out_dir / m / f"{p}.h5" for m in inputs for p in set(PROJECTS.values())
    ]
    existing = [str(o) for o in outputs if o.exists()]
    if existing and not args.overwrite:
        print(f"{', '.join(existing)} exist; pass --overwrite", file=sys.stderr)
        return 1

    srcs = {m: h5py.File(h5, "r") for m, h5 in inputs.items()}
    try:
        # check every input before writing any output
        plans, bad = {}, []
        for m, src in srcs.items():
            plans[m], b = plan(m, src, metas[m][0], drop)
            bad += b
        if bad:
            print("\n".join(bad), file=sys.stderr)
            print(f"{len(bad)} datasets unusable; nothing written", file=sys.stderr)
            return 1
        for m, src in srcs.items():
            write(m, src, plans[m], args.out_dir / m)
    finally:
        for src in srcs.values():
            src.close()
    return 0


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--hist-h5", type=Path, help="embed/virchow aggregate's hist.h5")
    p.add_argument("--expr-h5", type=Path, help="embed/ctds-rnaseq's expr.h5")
    p.add_argument(
        "--slide-metadata",
        type=Path,
        default=METADATA / "cptac-nsclc-primary-tumor-slide-metadata.csv",
    )
    p.add_argument(
        "--rna-metadata",
        type=Path,
        default=METADATA / "cptac-nsclc-primary-tumor-rna-seq.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="nsclc-eval-data/; each modality goes in its own hist/ or expr/ under it",
    )
    p.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    args = p.parse_args()
    if not (args.hist_h5 or args.expr_h5):
        p.error("give --hist-h5, --expr-h5 or both")
    sys.exit(main(args))
