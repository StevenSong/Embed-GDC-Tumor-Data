# Download and Embed GDC Tumor Data

Pull TCGA primary-tumor data from the [GDC](https://portal.gdc.cancer.gov/) and turn it into
embeddings from pathology and transcriptomics foundation models. Two modalities are in scope:

* **Whole-slide images** — 11,684 diagnostic `.svs` slides (~12.9 TB), embedded per-tile with
  [Virchow](https://huggingface.co/paige-ai/Virchow).
* **Bulk RNA-seq** — 10,157 STAR gene-count tables, embedded with [Bulkformer](https://github.com/KangBoming/BulkFormer).

## Repo Layout

| path | what it is |
| --- | --- |
| [notebooks/](./notebooks) | GDC API queries that build the download manifests |
| [manifests/](./manifests) | the manifests those notebooks emit, fed to `gdc-client` |
| [embed/virchow/](./embed/virchow) | WSI → per-tile Virchow embeddings; our own, very fast WSI tiling for TCGA slides; the production embedding pipeline |
| [embed/bulkformer/](./embed/bulkformer) | RNA-seq → Bulkformer embeddings — **placeholder, not yet written** |
| [kserve-demo/](./kserve-demo) | a serving-shaped take on the same WSI job (uses slideflow): a local stand-in for a KServe `InferenceService` |
| `env.yaml` / `requirements.txt` | the `emb-gdc` conda environment for the notebooks and local analysis |
<!-- | `sample/` | two slides (one baseline JPEG, one JPEG 2000) used for smoke tests and benchmarks; gitignored | -->

## Download data

The WSIs are large, so downloads go through the
[GDC Data Transfer Tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool) rather than the
API directly.

1. **Build a manifest.** The notebooks in [notebooks/](./notebooks) query the GDC `/files` endpoint
   with a filter (TCGA, primary tumor, open access) and save both a metadata table and the
   `return_type: manifest` response:

   | notebook | manifest | files |
   | --- | --- | --- |
   | [tcga-primary-tumor-dx-slide-manifest.ipynb](./notebooks/tcga-primary-tumor-dx-slide-manifest.ipynb) | [tcga-primary-tumor-dx-slide.txt](./manifests/tcga-primary-tumor-dx-slide.txt) | 11,684 diagnostic slides (`.svs`) |
   | [tcga-primary-tumor-rna-seq-manifest.ipynb](./notebooks/tcga-primary-tumor-rna-seq-manifest.ipynb) | [tcga-primary-tumor-rna-seq.txt](./manifests/tcga-primary-tumor-rna-seq.txt) | 10,157 RNA-seq gene-count tables (`.tsv`) |

2. **Download.**

    ```bash
    gdc-client download \
      --manifest manifests/tcga-primary-tumor-dx-slide.txt \
      --log-file download.log \
      --dir /path/to/save/data  # < create this directory first
    ```

    The tool is resumable — rerun the same command against the same `--dir` to pick up whatever failed. Transient API errors are normal on a run this size, the tool with automatically retry and succeed on most during the run; the tail of the log reports the successful count.

3. **Extract and cleanup.** `gdc-client` writes one directory per file UUID. Embedding code should traverse the nested download structure.

## Embedding

### `embed/virchow` — whole-slide images

A self-contained container that walks a directory of `.svs` slides and writes, per slide, an `.h5`
of per-tile Virchow embeddings plus a masked thumbnail. Slides are read straight off the Aperio
pyramid with `tifffile`, tissue is detected CLAM-style, and 224 px tiles at 20x are streamed to a
`torch.compile`d Virchow. It sustains ~1200 tiles/s on one H200 — roughly 35 h for the whole TCGA
diagnostic corpus on a single GPU. See [embed/virchow/README.md](./embed/virchow/README.md) for
build, run, tuning and output format.

### `embed/bulkformer` — bulk RNA-seq

**Not written yet.** Intended as a sibling of `embed/virchow`: same shape — a container pointed at a
directory of GDC RNA-seq gene-count `.tsv` files, writing one embedding per sample to HDF5 — using
Bulkformer as the encoder instead of Virchow. Nothing about the interface is settled beyond that.

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
