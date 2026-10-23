# Pathology Report Summaries and Embeddings

Summarizes TCGA pathology reports with [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it)
and embeds them with [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2). This is
the report pipeline from [multimodal-cancer-modeling](https://github.com/StevenSong/multimodal-cancer-modeling)
(`embed/generate_summaries.py` and `embed/embed_text_biomistral.py`) with Gemma models in place of
Llama-3.1-8B-Instruct and BioMistral-7B. Both models are Apache-2.0 and ungated.

Input is `TCGA_Reports.csv` from [Kefeli and Tatonetti 2024](https://data.mendeley.com/datasets/hyg5xkznpx/1):
9,523 reports, keyed by a `patient_filename` of the form `<case_id>.<file uuid>`. The pipeline
writes the same three files the original did:

| file | step | what it is |
| --- | --- | --- |
| `summarized_reports.csv` | `summarize` raw reports | one Gemma 4 summary per report |
| `text.h5` | `embed` raw reports | EmbeddingGemma 2 of each raw report |
| `summ.h5` | `embed` summaries | EmbeddingGemma 2 of each summary |

The `.h5` layout matches the original's, so the survival notebooks in multimodal-cancer-modeling read
these files without changes. **A CUDA device is required.**

## What carried over, and what did not

**Summarization** keeps the original's instructions word for word, plus its greedy decoding
(`temperature=0`), its 1024-token cap and its seed, so the two sets of summaries differ only in the
model. Gemma's model card recommends sampling at `temperature=1.0`; that is for chat, and this step
is extraction. Two changes are forced by Gemma 4's chat template:

* **The two system messages are merged into one.** Gemma 4's template folds only the first message
  into its system block. A second system message would be rendered as a separate turn partway
  through the conversation.
* **Thinking is off.** The template then pre-fills an empty thought block in the generation prompt,
  so the output contains only the summary.

**Embedding** changes more, because EmbeddingGemma 2 is a trained embedding model and BioMistral was
a causal LM pooled after the fact:

* **Text gets a task prefix.** The model is trained with short instruction prefixes, and quality
  drops without one. The default is `Classification` (`task: classification | query: `), the
  symmetric prefix the model card gives for training a classifier on the vectors. That is the
  closest match to fitting Cox models on them. `--prompt-name` picks any other prompt from the
  model's `config_sentence_transformers.json`. The prefix used is recorded in the `.h5`.
* **bfloat16 is pinned.** In float16 the model returns NaNs or silently degraded vectors instead of
  raising an error. vLLM's `auto` dtype falls back to float16 on GPUs without bfloat16, so the dtype
  is set explicitly.
* **Vectors are 768-d and unit-norm**: mean pooling, a 512→768 projection and L2 normalization, all
  inside the model. They are stored at full width. The model supports Matryoshka truncation to
  512/256/128, which is a slice and a re-normalize, but the downstream models apply PCA anyway.

**Inputs that are too long are refused, not truncated.** The longest TCGA report is 6,743 Gemma
tokens. That fits both EmbeddingGemma 2's 8,192-token context and the default summarization budget
of 16,384, so on this corpus nothing is refused. On other input, a report over the limit is logged by
name and skipped, and the step exits non-zero. Neither step lets vLLM truncate a report unnoticed or
fail a whole batch because of one.

Both steps load only the text half of each model (`language_model_only`), skipping the vision and
audio encoders.

## Build

```bash
docker build -t gemma-embedder:latest embed/gemma
```

The image is `vllm/vllm-openai:v0.32.0` with this package on top. `transformers` is pinned to
`>=5.19,<5.20`: vLLM 0.32.0 accepts anything from 5.16.1, but `embedding_gemma2` first ships in 5.19.0.
An older `transformers` installs cleanly and then fails when the embedding model loads.

**The weights are not baked into the image**, unlike `embed/virchow`'s. Gemma 4 31B is 62 GB of
bf16 safetensors. The weights are read instead from plain model directories under
`/opt/gpudata/models`, one per model at its full Hub id. Neither model is gated, so no token is
needed. Download them once, outside any GPU job:

```bash
hf download google/gemma-4-31B-it --local-dir /opt/gpudata/models/google/gemma-4-31B-it
hf download google/embeddinggemma-2 --local-dir /opt/gpudata/models/google/embeddinggemma-2
```

These are `--local-dir` downloads, not a Hugging Face cache (`HF_HOME`), and the code expects that
layout. `--model` is a directory path, and EmbeddingGemma's task prefixes are read from the
`config_sentence_transformers.json` inside it. The image runs with `HF_HUB_OFFLINE=1`, so a missing
model fails at once instead of starting a 62 GB download. The sbatch mounts the directory read-only
at the same path inside the container, so the defaults are identical inside and outside Docker.

The source is baked in (`COPY src/`), so changes under `src/` need a rebuild.

## Run

On the cluster, one job runs all three steps:

```bash
REPORTS_CSV=/opt/gpudata/tcga-reports/TCGA_Reports.csv \
OUT_DIR=/path/to/out \
  sbatch embed/gemma/summarize-and-embed.sbatch
```

The job is resumable. `summarize` rewrites its CSV atomically after every `--chunk-size` reports and
skips reports already in it, so a resubmitted job continues where the last one stopped. Each `embed`
takes minutes and is skipped when its `.h5` exists. Delete the `.h5` to redo it. If the job
reserves several GPUs, the summarizer is split across all of them with tensor parallelism. One H200
holds the 31B model in bf16. Set `MODEL_DIR` if the models live somewhere other than
`/opt/gpudata/models`. The job checks both model directories exist before it starts a container.

To run a single step by hand:

```bash
docker run -it --rm --gpus '"device=0"' --shm-size=16g \
  -v /opt/gpudata/models:/opt/gpudata/models:ro \
  -v /opt/gpudata/tcga-reports:/reports:ro \
  -v /path/to/out:/out \
  gemma-embedder:latest summarize \
    --input-csv /reports/TCGA_Reports.csv --output-csv /out/summarized_reports.csv

docker run ... gemma-embedder:latest embed \
    --input-csv /out/summarized_reports.csv --output-h5 /out/summ.h5
```

### Flags

`summarize`:

| flag | default | what it does |
| --- | --- | --- |
| `--input-csv` | required | report CSV with `patient_filename` and `text` columns |
| `--output-csv` | required | summaries CSV; resumed from if it exists |
| `--model` | `/opt/gpudata/models/google/gemma-4-31B-it` | model directory |
| `--max-tokens` | 1024 | cap on summary length, as in the original |
| `--max-model-len` | 16384 | prompt + summary token budget |
| `--chunk-size` | 512 | reports per checkpoint of the output CSV |
| `--tensor-parallel-size` | 1 | GPUs to split the model across; the sbatch sets it to the GPU count |
| `--gpu-memory-utilization` | 0.9 | vLLM's share of each GPU |
| `--overwrite` | off | start over instead of resuming |

`embed`:

| flag | default | what it does |
| --- | --- | --- |
| `--input-csv` | required | raw reports or `summarize` output; both have the same shape |
| `--output-h5` | required | refused if it exists |
| `--model` | `/opt/gpudata/models/google/embeddinggemma-2` | model directory |
| `--prompt-name` | `Classification` | task prefix, by its name in the model's prompt config |
| `--gpu-memory-utilization` | 0.5 | |
| `--overwrite` | off | replace `--output-h5` |

## Output

`summarized_reports.csv` has the input's `patient_filename` and `text` columns, plus `finish_reason`.
A `finish_reason` of `length` means the summary hit `--max-tokens` and was cut off. Those summaries are
kept, as the original kept them, and the run logs how many there were. Rows follow input order.

Each `.h5` has one group per case and one dataset per report under it, named by
`patient_filename`. This is the same layout as the original pipeline:

```
TCGA-BP-5195/
  TCGA-BP-5195.25c0b433-5557-4165-922e-2c1eac9c26f0   (768,) float32, attrs: n_tokens
```

A case with several reports has several datasets; the survival notebooks average them. Root attributes record how the file was made: `encoder` (the model directory),
`prompt_name`, `prompt`, `embed_dim`, `normalized`, `dtype`, `input_csv`, `n_reports`, `created_utc`,
`vllm_version`. The file is written to `.<name>.tmp` and renamed when complete.

Reports with no text are dropped with a warning in both steps. Note that TCGA has at least one
report whose entire text is `1.`. It is not empty, so it is summarized and embedded like any other.
