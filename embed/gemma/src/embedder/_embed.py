"""Embed report text with EmbeddingGemma 2, one 768-d vector per report.

A port of `embed/embed_text_biomistral.py` in multimodal-cancer-modeling, with
the same output layout, so the survival notebooks there read this file
unchanged: one group per case, one float32 dataset per report under it, named
by `patient_filename`.

EmbeddingGemma 2 differs from BioMistral in three ways that matter here:

* It is trained with task prefixes and degrades without one, so the text is
  prefixed with the prompt named by --prompt-name. The prefixes are read from
  the `config_sentence_transformers.json` in the model directory rather than
  copied here.
* It must run in bfloat16 or float32. Its activations overflow float16 and it
  then returns NaNs or quietly degraded vectors rather than an error, so the
  dtype is pinned rather than left to vLLM's `auto`, which falls back to
  float16 on GPUs without bfloat16.
* Its output is mean-pooled, projected to 768-d and L2-normalized by the model
  itself. The vectors are stored at full width; it supports Matryoshka
  truncation to 512/256/128, but that is a slice and re-normalize away and
  the survival models PCA the embeddings anyway.
"""

import datetime
import json
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import vllm
from transformers import AutoTokenizer
from vllm import LLM

from ._logging import progress_disabled
from ._reports import REPORT_ID, TEXT, case_id, read_reports

# `hf download google/embeddinggemma-2 --local-dir /opt/gpudata/models/google/embeddinggemma-2`
MODEL = "/opt/gpudata/models/google/embeddinggemma-2"
# Symmetric task prefix, `task: classification | query: `. The embeddings feed
# a supervised (Cox) model, which is the setting MTEB's classification tasks
# use this prefix for: a linear probe trained on the vectors.
PROMPT_NAME = "Classification"
# the model's context; TCGA's longest raw report is ~6.7k tokens
MAX_MODEL_LEN = 8192
DTYPE = "bfloat16"


def load_prompt(model: Path, prompt_name: str) -> str:
    with open(model / "config_sentence_transformers.json") as f:
        prompts = json.load(f)["prompts"]
    if prompt_name not in prompts:
        raise ValueError(
            f"{model} has no prompt {prompt_name!r}; it has {', '.join(prompts)}"
        )
    return prompts[prompt_name]


def embed(
    *,  # enforce kwargs
    logger: logging.Logger,
    input_csv: Path,
    output_h5: Path,
    model: Path,
    prompt_name: str,
    gpu_memory_utilization: float,
    overwrite: bool,
) -> int:
    # The whole corpus embeds in a minute or two, so there is no resume: the
    # file is written once, to a dotfile, and renamed into place at the end.
    if output_h5.exists() and not overwrite:
        logger.error("%s exists; move it aside or pass --overwrite", output_h5)
        return 1

    reports = read_reports(logger, input_csv)
    prompt = load_prompt(model, prompt_name)
    texts = [prompt + text for text in reports[TEXT]]
    logger.info(
        "%d reports in %s, prefixed with %r (%s)",
        len(reports),
        input_csv,
        prompt,
        prompt_name,
    )

    # Count tokens as the model will see them, BOS and EOS included, and
    # refuse anything over the context rather than let vLLM fail the whole
    # call on it or truncate it unseen.
    tokenizer = AutoTokenizer.from_pretrained(model)
    n_tokens = np.array([len(ids) for ids in tokenizer(texts)["input_ids"]])
    too_long = n_tokens > MAX_MODEL_LEN
    for report_id, n in zip(reports[REPORT_ID][too_long], n_tokens[too_long]):
        logger.error(
            "%s is %d tokens, over the %d-token context; not embedding it",
            report_id,
            n,
            MAX_MODEL_LEN,
        )
    reports = reports[~too_long].reset_index(drop=True)
    texts = [t for t, long in zip(texts, too_long) if not long]
    n_tokens = n_tokens[~too_long]
    if not texts:
        logger.error("no reports left to embed")
        return 1
    logger.info(
        "embedding %d reports, %d-%d tokens (median %d)",
        len(texts),
        n_tokens.min(),
        n_tokens.max(),
        np.median(n_tokens),
    )

    llm = LLM(
        model=str(model),
        runner="pooling",
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=gpu_memory_utilization,
        language_model_only=True,  # text only: skip the vision and audio towers
    )
    outputs = llm.embed(texts, use_tqdm=not progress_disabled())
    embs = np.asarray([o.outputs.embedding for o in outputs], dtype=np.float32)
    if not np.isfinite(embs).all():
        # the float16 failure mode, should the dtype pin above ever be undone
        raise RuntimeError("non-finite embeddings; check the inference dtype")
    logger.info("embedded %d reports, %d-d", *embs.shape)

    tmp = output_h5.with_name(f".{output_h5.name}.tmp")
    with h5py.File(tmp, "w") as h5:
        for report_id, emb, n in zip(reports[REPORT_ID], embs, n_tokens):
            group = h5.require_group(case_id(report_id))
            ds = group.create_dataset(report_id, data=emb)
            ds.attrs["n_tokens"] = n
        h5.attrs.update(
            {
                "encoder": str(model),
                "prompt_name": prompt_name,
                "prompt": prompt,
                "embed_dim": embs.shape[1],
                "normalized": True,
                "dtype": DTYPE,
                "input_csv": str(input_csv),
                "n_reports": len(reports),
                "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "vllm_version": vllm.__version__,
            }
        )
    os.replace(tmp, output_h5)
    logger.info("wrote %s", output_h5)

    if too_long.any():
        logger.error("%d reports too long to embed", too_long.sum())
        return 1
    return 0
