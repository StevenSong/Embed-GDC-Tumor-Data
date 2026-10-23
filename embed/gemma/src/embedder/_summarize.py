"""Summarize pathology reports with Gemma 4 31B, one paragraph per report.

A port of `embed/generate_summaries.py` in multimodal-cancer-modeling, which ran
Llama-3.1-8B-Instruct. The instructions, greedy decoding, 1024-token cap and seed
are carried over unchanged so the two sets of summaries differ by model only.

Two departures, both forced by Gemma 4 rather than chosen:

* The original's two system messages are merged into one. Gemma 4's chat
  template folds only the *first* message into its system block; a second one
  is rendered as a separate system turn in the middle of the conversation,
  which is not a shape the model was tuned on.
* Thinking is off. The template then pre-fills an empty thought block
  (`<|channel>thought\\n<channel|>`) in the generation prompt, so what comes back
  is the answer alone, with nothing to strip.
"""

import logging
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ._logging import progress_disabled
from ._reports import REPORT_ID, TEXT, read_reports, write_csv_atomic

# `hf download google/gemma-4-31B-it --local-dir /opt/gpudata/models/google/gemma-4-31B-it`
MODEL = "/opt/gpudata/models/google/gemma-4-31B-it"
SEED = 42

SYSTEM_PROMPT = (
    "You are a helpful assistant for digital pathology.\n"
    "\n"
    "Instructions:\n"
    "Extract and repeat the results of the following pathology report in a single paragraph.\n"
    "Focus on test results, diagnoses and clinical history.\n"
    "Include results of the microscopic description. Omit the gross or macroscopic description.\n"
    "Do not acknowledge this prompt. Do not give additional comments after your final answer."
)
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

# `length` means the summary hit --max-tokens and was cut off mid-thought
FINISH_REASON = "finish_reason"


def conversation(report: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": report},
    ]


def summarize(
    *,  # enforce kwargs
    logger: logging.Logger,
    input_csv: Path,
    output_csv: Path,
    model: Path,
    max_tokens: int,
    max_model_len: int,
    chunk_size: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    overwrite: bool,
) -> int:
    reports = read_reports(logger, input_csv)
    logger.info("%d reports in %s", len(reports), input_csv)

    # Resume: keep whatever an earlier, interrupted run already wrote. The CSV
    # is rewritten whole after every chunk, so it is never half a row.
    done = pd.DataFrame(columns=[REPORT_ID, TEXT, FINISH_REASON])
    if output_csv.exists() and not overwrite:
        done = pd.read_csv(output_csv)
        if set(done.columns) != {REPORT_ID, TEXT, FINISH_REASON}:
            raise ValueError(
                f"{output_csv} exists but is not a summaries CSV from this tool"
                f" (columns {list(done.columns)}); move it aside or pass --overwrite"
            )
        logger.info("%s has %d summaries already, resuming", output_csv, len(done))
    todo = reports[~reports[REPORT_ID].isin(done[REPORT_ID])].reset_index(drop=True)
    if todo.empty:
        logger.info("nothing to do: all %d reports already summarized", len(reports))
        return 0

    # Check every prompt fits before loading 62 GB of weights: an over-long
    # prompt is a validation error that would take its whole chunk down with it.
    # TCGA's longest report is ~6.7k tokens, so at the default --max-model-len
    # this never fires on that corpus.
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt_budget = max_model_len - max_tokens
    too_long = []
    for report_id, report in zip(todo[REPORT_ID], todo[TEXT]):
        prompt = tokenizer.apply_chat_template(
            conversation(report),
            tokenize=False,
            add_generation_prompt=True,
            **CHAT_TEMPLATE_KWARGS,
        )
        n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if n_tokens > prompt_budget:
            logger.error(
                "%s is %d prompt tokens, over the %d left by --max-model-len %d"
                " minus --max-tokens %d; not summarizing it",
                report_id,
                n_tokens,
                prompt_budget,
                max_model_len,
                max_tokens,
            )
            too_long.append(report_id)
    todo = todo[~todo[REPORT_ID].isin(too_long)].reset_index(drop=True)
    logger.info("summarizing %d/%d reports", len(todo), len(reports))

    llm = LLM(
        model=str(model),
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        language_model_only=True,  # text in, text out: skip the vision tower
        seed=SEED,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        max_tokens=max_tokens,
        seed=SEED,
    )

    # output rows follow input order however many times the run was resumed
    rank = {report_id: i for i, report_id in enumerate(reports[REPORT_ID])}
    n_cut = 0
    for lo in range(0, len(todo), chunk_size):
        # vLLM batches continuously within a call, so a chunk only bounds how
        # much work a kill can lose; it does not need tuning for throughput
        chunk = todo.iloc[lo : lo + chunk_size]
        outputs = llm.chat(
            [conversation(report) for report in chunk[TEXT]],
            sampling_params,
            use_tqdm=not progress_disabled(),
            chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
        )
        rows = pd.DataFrame(
            {
                REPORT_ID: chunk[REPORT_ID].to_list(),
                TEXT: [o.outputs[0].text.strip() for o in outputs],
                FINISH_REASON: [o.outputs[0].finish_reason for o in outputs],
            }
        )
        n_cut += (rows[FINISH_REASON] == "length").sum()
        done = rows if done.empty else pd.concat([done, rows], ignore_index=True)
        done = done.sort_values(
            REPORT_ID, key=lambda s: s.map(rank), na_position="last"
        )
        write_csv_atomic(done, output_csv)
        logger.info(
            "[%d/%d] summarized -> %s",
            min(lo + chunk_size, len(todo)),
            len(todo),
            output_csv,
        )

    if n_cut:
        logger.warning(
            "%d summaries hit --max-tokens %d and are truncated; they are kept,"
            " with %s=length in the CSV",
            n_cut,
            max_tokens,
            FINISH_REASON,
        )
    if too_long:
        logger.error(
            "%d reports too long to summarize: %s", len(too_long), ", ".join(too_long)
        )
        return 1
    logger.info("done: %d summaries in %s", len(done), output_csv)
    return 0
