# Download and Embed GDC Tumor Data

Pull TCGA primary-tumor data from the [GDC](https://portal.gdc.cancer.gov/) and turn it into
embeddings from pathology, transcriptomics and language foundation models. Three modalities are in scope:

* **Whole-slide images** — 11,684 diagnostic `.svs` slides (~12.9 TB), embedded per-tile with
  [Virchow](https://huggingface.co/paige-ai/Virchow).
* **Bulk RNA-seq** — 10,157 STAR gene-count tables, embedded with ctds-rnaseq, a BulkRNABert-architecture
  encoder from [omics-encoder](https://github.com/uc-cdis/omics-encoder).
* **Pathology reports** — 9,523 OCR'd reports from [TCGA-Reports](https://data.mendeley.com/datasets/hyg5xkznpx/1),
  summarized with [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it) and embedded with
  [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2).

## Repo Layout

| path | what it is |
| --- | --- |
| [notebooks/](./notebooks) | GDC API and IDC index queries that build the download manifests |
| [manifests/](./manifests) | the manifests those notebooks emit, fed to `gdc-client` or `idc` |
| [embed/virchow/](./embed/virchow) | WSI → per-tile Virchow embeddings; our own, very fast WSI tiling for TCGA slides; the production embedding pipeline |
| [embed/gemma/](./embed/gemma) | pathology reports → Gemma 4 summaries → EmbeddingGemma 2 embeddings |
| [embed/ctds-rnaseq/](./embed/ctds-rnaseq) | RNA-seq STAR-counts tables → ctds-rnaseq embeddings, optionally narrowed by a GDC manifest |
| [scripts/](./scripts) | `export_m3ai_demos_cptac.py`: splits the CPTAC embeddings into the files m3ai-demos loads |
| [kserve-demo/](./kserve-demo) | a serving-shaped take on the same WSI job (uses slideflow): a local stand-in for a KServe `InferenceService` |
| `compose.yaml` | builds all the `embed/` images (`docker compose build`) |
| `env.yaml` / `requirements.txt` | the `emb-gdc` conda environment for the notebooks and local analysis |
<!-- | `sample/` | two slides (one baseline JPEG, one JPEG 2000) used for smoke tests and benchmarks; gitignored | -->

## Download data

The WSIs are large, so downloads go through the
[GDC Data Transfer Tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool) rather than the
API directly.

1. **Build a manifest.** The notebooks in [notebooks/](./notebooks) query the GDC `/files` endpoint
   with a filter (project, primary tumor, open access) and save both a metadata table and the
   `return_type: manifest` response:

   | notebook | manifest | files |
   | --- | --- | --- |
   | [tcga-primary-tumor-dx-slide-manifest.ipynb](./notebooks/tcga-primary-tumor-dx-slide-manifest.ipynb) | [tcga-primary-tumor-dx-slide.txt](./manifests/tcga-primary-tumor-dx-slide.txt) | 11,684 diagnostic slides (`.svs`) |
   | [tcga-primary-tumor-rna-seq-manifest.ipynb](./notebooks/tcga-primary-tumor-rna-seq-manifest.ipynb) | [tcga-primary-tumor-rna-seq.txt](./manifests/tcga-primary-tumor-rna-seq.txt) | 10,157 RNA-seq gene-count tables (`.tsv`) |
   | [cptac-nsclc-primary-tumor-rna-seq-manifest.ipynb](./notebooks/cptac-nsclc-primary-tumor-rna-seq-manifest.ipynb) | [cptac-nsclc-primary-tumor-rna-seq.txt](./manifests/cptac-nsclc-primary-tumor-rna-seq.txt) | 338 CPTAC-3 lung RNA-seq gene-count tables (`.tsv`), 337 cases |

   GDC files every CPTAC lung case under one project, CPTAC-3. So instead of `project_id`, the CPTAC
   metadata table has a `collection_id`, `cptac_luad` or `cptac_lscc`, mapped from each case's GDC
   disease type. These are the same ids IDC uses for the CPTAC slide collections.

2. **Download.**

    ```bash
    gdc-client download \
      --manifest manifests/tcga-primary-tumor-dx-slide.txt \
      --log-file download.log \
      --dir /path/to/save/data  # < create this directory first
    ```

    The tool is resumable — rerun the same command against the same `--dir` to pick up whatever failed. Transient API errors are normal on a run this size, the tool with automatically retry and succeed on most during the run; the tail of the log reports the successful count.

3. **Extract and cleanup.** `gdc-client` writes one directory per file UUID. Embedding code should traverse the nested download structure.

### CPTAC slides

The CPTAC pathology is not on the GDC; it is DICOM in the
[Imaging Data Commons](https://portal.imaging.datacommons.cancer.gov/) (IDC), and downloads go
through its [`idc-index`](https://github.com/ImagingDataCommons/idc-index) client (MIT, in
`requirements.txt`) and the `idc` command it installs. The chain is the same shape as the GDC one.

1. **Build a manifest.**
   [cptac-nsclc-primary-tumor-slide-manifest.ipynb](./notebooks/cptac-nsclc-primary-tumor-slide-manifest.ipynb)
   queries the IDC index for H&E slides whose specimen is coded `Neoplasm, Primary`. Slides coded
   normal and slides with no tissue type are left out, and so are the CT/PET/MR series in the same
   collections. The notebook writes
   [cptac-nsclc-primary-tumor-slide.s5cmd](./manifests/cptac-nsclc-primary-tumor-slide.s5cmd), with one
   `cp s3://...` line per slide series, plus a metadata table that maps each slide name
   (`ContainerIdentifier`) to its case. For CPTAC-LUAD and CPTAC-LSCC that is 1,343 slides
   (681 + 662) across 432 patients, ~618 GB. The manifest's first line records the IDC data
   release it was built against.

2. **Download.**

    ```bash
    idc download-from-manifest \
      --manifest-file manifests/cptac-nsclc-primary-tumor-slide.s5cmd \
      --download-dir /path/to/save/data \
      --use-s5cmd-sync True
    ```

    The files come from IDC's public buckets, so no credentials are needed. The layout is
    `<collection>/<patient>/<study UID>/SM_<series UID>/*.dcm`, one directory per slide holding
    its pyramid levels, label, overview and thumbnail. `--use-s5cmd-sync` makes a rerun against
    the same `--download-dir` fetch only what is missing.

`embed/virchow` reads that directory as it is. It treats each DICOM series as one slide.

## Embedding

[compose.yaml](./compose.yaml) at the repo root builds every embedding image under the tag its
sbatch launcher expects. `HF_TOKEN` is needed for the gated Virchow weights:

```bash
HF_TOKEN=hf_xxx docker compose build   # or append a service: virchow, gemma, ctds-rnaseq
```

### `embed/virchow` — whole-slide images

A self-contained container that walks a directory of `.svs` slides (or CPTAC's DICOM series) and
writes, per slide, an `.h5` of per-tile Virchow embeddings plus a masked thumbnail. Slides are read straight off the Aperio
pyramid with `tifffile`, tissue is detected CLAM-style, and 224 px tiles at 20x are streamed to a
`torch.compile`d Virchow. It sustains ~1200 tiles/s on one H200 — roughly 35 h for the whole TCGA
diagnostic corpus on a single GPU. A second, GPU-free step collects each slide's mean embedding
into one `hist.h5` in [multimodal-cancer-modeling](https://github.com/StevenSong/multimodal-cancer-modeling)'s
layout, so its survival notebooks read it unchanged; `run-corpus.sh` submits both to Slurm. See
[embed/virchow/README.md](./embed/virchow/README.md) for build, run, tuning and output format.

### `embed/ctds-rnaseq` — bulk RNA-seq

A container that embeds a `gdc-client` download of STAR-counts TSVs with ctds-rnaseq (working name).
It is a BulkRNABert-architecture encoder trained at UChicago CTDS, run in-process from the
checkpoint that omics-encoder serves through KServe. It writes one 256-d vector per sample into a
single `expr.h5`, in the layout of multimodal-cancer-modeling and m3ai-demos. An optional GDC
manifest narrows the download to the files it lists, such as the CPTAC NSCLC subset of the CPTAC
primary-tumor download. A metadata table from [metadata/](./metadata) supplies each file's case. The weights are baked into the image at
build time from an omics-encoder checkout, not committed here. See
[embed/ctds-rnaseq/README.md](./embed/ctds-rnaseq/README.md).

### `embed/gemma` — pathology reports

A container that runs vLLM in-process for both steps of the report pipeline from
[multimodal-cancer-modeling](https://github.com/StevenSong/multimodal-cancer-modeling). Gemma 4 31B
writes a one-paragraph summary of each report. EmbeddingGemma 2 then embeds the raw reports and the
summaries, 768-d per report. The prompt and decoding match the original; the models replace its
Llama-3.1-8B and BioMistral. The `.h5` layout also matches, so that repo's survival notebooks read
the output unchanged. See [embed/gemma/README.md](./embed/gemma/README.md).

## Export to m3ai-demos

The [m3ai-demos](https://github.com/uc-cdis/m3ai-demos) lung cancer classification notebook
evaluates on CPTAC. It loads one file per modality and project from `nsclc-eval-data/`:
`{hist,expr}/{CPTAC-LUAD,CPTAC-LSCC}.h5`.
[export_m3ai_demos_cptac.py](./scripts/export_m3ai_demos_cptac.py) splits `embed/virchow`'s
`hist.h5`, `embed/ctds-rnaseq`'s `expr.h5`, or both into those files:

```bash
python scripts/export_m3ai_demos_cptac.py \
  --hist-h5 /path/to/virchow/hist.h5 \
  --expr-h5 /path/to/ctds-rnaseq/expr.h5 \
  --out-dir external/m3ai-demos/demos/nsclc-eval-data
```

The split uses the `collection_id` each modality's metadata table gives a case. For slides, that is
the IDC collection. For RNA-seq, it is mapped from the GDC disease type. The two occasionally
disagree (C3N-02282: IDC `cptac_luad`, GDC squamous). The notebook matches cases across both projects at once, so such a
case would count as LUAD in one modality and LSCC in the other, and fail its per-class count checks.
The script drops it from both modalities. Both metadata tables are therefore read even when only one
modality is exported.

## `kserve-demo`

A separate, earlier take on the WSI job, shaped as a service rather than a batch run: a Docker
Compose pair standing in for the transformer/predictor halves of a KServe `InferenceService`, where
a preprocessor tiles the slide and streams tiles over gRPC to Triton Inference Server running
Virchow as TorchScript. It is a demo of the serving pattern, not the pipeline used to embed the
corpus — for that, use `embed/virchow`. See [kserve-demo/README.md](./kserve-demo/README.md).

## A note on the two Tritons

Two unrelated projects named **Triton** show up in this repo, and they have nothing to do with each
other:

* **[Triton Inference Server](https://github.com/triton-inference-server/server)** (NVIDIA) — the
  model server in [kserve-demo/](./kserve-demo). It loads a TorchScript Virchow from a model
  repository and answers gRPC/HTTP inference requests. This is a *serving* system.
* **[Triton](https://github.com/triton-lang/triton)** (the GPU kernel language, originally OpenAI) —
  the compiler backend `torch.compile` / TorchInductor generates kernels in, used by
  [embed/virchow/](./embed/virchow). It never serves anything; it compiles the model in-process, and
  it needs a C compiler on `PATH` (which is why that image installs `gcc`).

`embed/virchow` uses the second and never the first. `kserve-demo` uses the first and never the
second. When something says "Triton", check which one it means before acting on it.
