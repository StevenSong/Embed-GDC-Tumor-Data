# Bulk RNA-seq Embeddings with ctds-rnaseq

Embeds GDC bulk RNA-seq STAR-counts tables with **ctds-rnaseq**, a BulkRNABert-architecture
encoder trained on TCGA at UChicago CTDS in [uc-cdis/omics-encoder](https://github.com/uc-cdis/omics-encoder).
"ctds-rnaseq" is a working name. The pipeline reads a `gdc-client` download directory and writes one 256-d float32 vector per
sample into a single `expr.h5`. That file has the layout of multimodal-cancer-modeling's
`embed_expr_bulkrnabert.py` and of m3ai-demos' `nsclc-eval-data/expr/`. **A CUDA device is
required.**

omics-encoder serves this model through KServe and Triton (`kserve-demo/`). This pipeline is the
same computation run in-process and offline. The model and TSV-parsing code are copied from
`kserve-demo/` (Apache-2.0, see [NOTICE](./NOTICE)): `model.py`, `wrapped.py` and
`preprocessor/rnaseq/sample_io.py`. Those are the frozen copies of the training code the checkpoint
came from, not omics-encoder's current `src/models/`. Starting from the same five reference inputs
`kserve-demo/export.py` writes, the copy here gives exactly the same output as `wrapped.py` on CPU,
and is within 1.3e-6 of the reference GPU embeddings.

## What the model does with a sample

1. **Parse.** Keep `protein_coding` rows, drop `_PAR_Y`, strip Ensembl versions, and reindex
   `tpm_unstranded` onto the 19,944-gene training vocabulary (`genes.tsv`). A vocabulary gene the
   file lacks is *absent*, not zero.
2. **Tokenize.** `log10(1 + TPM)` is binned into 64 bins using the checkpoint's `binner.npz`. Exact zeros go to bin 0
   and absent genes to `[MASK]`.
3. **Encode.** Four transformer blocks attend over all ~20k genes. The embedding is the last block's
   output, before `final_norm`, mean-pooled over every gene. Absent genes are included in that mean.
   The layer and pooling are fixed. They match the served model and the collaborator's offline
   pipeline.

GDC's GENCODE v36 TSVs, TCGA and CPTAC alike, cover the whole vocabulary, so nothing is absent in
practice. The run warns if any sample lacks vocabulary genes, and each dataset records how many.

Two quirks of the trained model are reproduced on purpose. `embedding_norm` is applied even though
the checkpoint's `run_config.json` says `no_embedding_norm: true`, because the training code ignored
that flag. The feed-forward has no activation between `fc1` and `fc2`. Both are how the checkpoint
was trained.

## Build

The weights are **not in this repo**. They are baked into the image at build time from a named build
context, `weights`, which is an omics-encoder checkout:

```
<weights>/                    # the omics-encoder repo root
└── kserve-demo/
    ├── checkpoint/
    │   ├── model.pt          # state dict, 25 MB
    │   ├── binner.npz
    │   └── run_config.json
    └── model-repo/bulkrnabert/
        └── genes.tsv         # vocabulary; row order = position embedding index
```

The context is the repo root, not `kserve-demo/`. BuildKit applies a named context's own
`.dockerignore`, and `kserve-demo/.dockerignore`, written for the KServe preprocessor image, excludes
`checkpoint/`. `genes.tsv` travels with the weights rather than the code because it decides what
each position embedding means. compose points the context at `$CTDS_RNASEQ_WEIGHTS`, by default a
clone at `external/omics-encoder`:

```bash
git clone git@github.com:uc-cdis/omics-encoder.git external/omics-encoder
docker compose build ctds-rnaseq
# or another checkout:
CTDS_RNASEQ_WEIGHTS=/path/to/omics-encoder docker compose build ctds-rnaseq
# or without compose:
docker build --build-context weights=external/omics-encoder \
  -t ctds-rnaseq-embedder:latest embed/ctds-rnaseq
```

Inside the image the four files sit flat in `/opt/ctds-rnaseq`, which is the default
`--weights-dir`. Point `--weights-dir` at another flat directory, mounted in, to run another
checkpoint without rebuilding. The source is baked in too (`COPY src/`), so changes under `src/`
need a rebuild.

## Run

On the cluster, for the CPTAC NSCLC eval set:

```bash
RNA_DIR=/opt/gpudata/cptac-primary-tumor-rna-seq \
METADATA=metadata/cptac-nsclc-primary-tumor-rna-seq.csv \
MANIFEST=manifests/cptac-nsclc-primary-tumor-rna-seq.txt \
OUT_DIR=/path/to/out \
  sbatch embed/ctds-rnaseq/embed.sbatch
```

That download holds 2,422 CPTAC primary-tumor samples across cancer types. The manifest narrows it
to the 338 NSCLC files (337 cases). Leave `MANIFEST` unset to embed every TSV under `RNA_DIR`. For
TCGA, with every file in the download listed in its metadata table, that is:

```bash
RNA_DIR=/opt/gpudata/tcga-primary-tumor-rna-seq \
METADATA=metadata/tcga-primary-tumor-rna-seq-metadata.csv \
OUT_DIR=/path/to/out \
  sbatch embed/ctds-rnaseq/embed.sbatch
```

By hand:

```bash
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
  -v /opt/gpudata/cptac-primary-tumor-rna-seq:/rna:ro \
  -v "$PWD/metadata:/metadata:ro" -v "$PWD/manifests:/manifests:ro" \
  -v /path/to/out:/out \
  ctds-rnaseq-embedder:latest \
    --metadata /metadata/cptac-nsclc-primary-tumor-rna-seq.csv \
    --manifest /manifests/cptac-nsclc-primary-tumor-rna-seq.txt \
    --output-h5 /out/expr.h5
```

There is no resume. Every input is parsed before the model loads, and the run fails, without writing
anything, if any of these is true:

* a manifest entry is not on disk;
* a file has no case in `--metadata`;
* a TSV is unreadable or has NaN, negative or infinite TPM;
* an embedding is non-finite.

omics-encoder quotes ~11 samples/s in fp32 on one H200 through Triton, which puts all of TCGA at
about a quarter of an hour.

### Flags

| flag | default | what it does |
| --- | --- | --- |
| `--rna-dir` | `/rna` | `gdc-client` download directory: `<file id>/<name>.rna_seq.augmented_star_gene_counts.tsv` |
| `--metadata` | required | CSV with `file_name` and `case_id` columns, as in [metadata/](../../metadata) |
| `--manifest` | none | GDC manifest (`id`, `filename` columns); embed only the files it lists |
| `--output-h5` | required | refused if it exists |
| `--weights-dir` | `/opt/ctds-rnaseq` | flat directory with `model.pt`, `binner.npz`, `run_config.json`, `genes.tsv` |
| `--batch-size` | 32 | samples per forward pass |
| `--num-workers` | 8 | processes parsing TSVs |
| `--overwrite` | off | replace `--output-h5` |

## Output

One group per case, named by its submitter id, and one `(256,)` float32 dataset per file under it:

```
C3L-00081/
  fa46d4bc-7aa8-4868-938b-404fc2d30a1a   attrs: file_id, file_name, n_matched, n_absent, n_unknown
```

The dataset is named by the UUID the TSV's file name starts with, as in m3ai-demos and
multimodal-cancer-modeling. That is **not** the GDC file id. The file id is the `gdc-client`
directory name and the manifest's `id`, and it is kept in the `file_id` attribute. A case with
several files keeps them all, and the readers average them; CPTAC NSCLC has one such case,
`C3L-03961`.

Root attributes record how the file was made:

* `encoder`, `architecture`, `weights_sha256`, `genes_sha256`, `n_genes`, `embed_dim`
* `layer`, `pooling`, `binner`, `tpm_column`, `dtype`
* `rna_dir`, `manifest`, `metadata`, `n_files`, `n_cases`
* `created_utc`, `torch_version`

The file is written to `.<name>.tmp` and renamed when complete.

To split a CPTAC NSCLC `expr.h5` into the per-project files m3ai-demos loads, use
[scripts/export_m3ai_demos_cptac.py](../../scripts/export_m3ai_demos_cptac.py).
