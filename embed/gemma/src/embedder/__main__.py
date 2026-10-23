"""Summarize TCGA pathology reports with Gemma 4, and embed them with EmbeddingGemma 2.

A port of the report pipeline in multimodal-cancer-modeling (Llama-3.1-8B
summaries, BioMistral embeddings) onto google/gemma-4-31B-it and
google/embeddinggemma-2. Two steps, each its own subcommand:

    python -m embedder summarize \\
        --input-csv TCGA_Reports.csv --output-csv summarized_reports.csv
    python -m embedder embed \\
        --input-csv summarized_reports.csv --output-h5 summ.h5

`embed` takes the raw reports just as well as the summaries; the paper's
comparison embeds both. Both steps run vLLM in-process on the local GPU(s).
"""

import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from . import _embed, _summarize
from ._logging import setup_logging


def main(args) -> int:
    logger = setup_logging()
    if args.step == "summarize":
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        return _summarize.summarize(
            logger=logger,
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            model=args.model,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            chunk_size=args.chunk_size,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            overwrite=args.overwrite,
        )
    if args.step == "embed":
        args.output_h5.parent.mkdir(parents=True, exist_ok=True)
        return _embed.embed(
            logger=logger,
            input_csv=args.input_csv,
            output_h5=args.output_h5,
            model=args.model,
            prompt_name=args.prompt_name,
            gpu_memory_utilization=args.gpu_memory_utilization,
            overwrite=args.overwrite,
        )
    raise ValueError(f"unknown step {args.step}")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    steps = p.add_subparsers(dest="step", required=True)

    s = steps.add_parser("summarize", help="one-paragraph summary per report")
    s.add_argument("--input-csv", type=Path, required=True, help="report CSV")
    s.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="summaries CSV; resumed from if it exists",
    )
    s.add_argument(
        "--model",
        type=Path,
        default=Path(_summarize.MODEL),
        help="model directory, from `hf download ... --local-dir`",
    )
    s.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="cap on summary length; summaries that hit it are kept and flagged",
    )
    s.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="prompt + summary token budget; the longest TCGA report is ~6.7k",
    )
    s.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="reports per checkpoint of the output CSV",
    )
    s.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="GPUs to split the model across; 31B in bf16 is 62 GB",
    )
    s.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    s.add_argument(
        "--overwrite",
        action="store_true",
        help="start over rather than resume from an existing --output-csv",
    )

    e = steps.add_parser("embed", help="one embedding per report")
    e.add_argument("--input-csv", type=Path, required=True, help="report CSV")
    e.add_argument("--output-h5", type=Path, required=True)
    e.add_argument(
        "--model",
        type=Path,
        default=Path(_embed.MODEL),
        help="model directory, from `hf download ... --local-dir`",
    )
    e.add_argument(
        "--prompt-name",
        default=_embed.PROMPT_NAME,
        help="task prefix, by its name in the model's config_sentence_transformers.json",
    )
    # small model, small batch: there is no reason to take the whole GPU
    e.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    e.add_argument("--overwrite", action="store_true", help="replace --output-h5")

    args = p.parse_args()
    # The image runs with HF_HUB_OFFLINE=1, and a path that does not exist is
    # otherwise taken for a Hub repo id and fails somewhere deep in vLLM.
    if not (args.model / "config.json").is_file():
        p.error(f"--model {args.model} is not a model directory (no config.json)")
    rc = main(args)
    # Skip interpreter teardown. vLLM's engine is already shut down by now, but
    # a thread of its client can still be freeing tensors when Python 3.12
    # finalizes, which aborts the process (`terminate called without an active
    # exception`) and fails the job after its output was written. Every output
    # is closed and renamed into place before main() returns, so there is
    # nothing left for teardown to do but flush the logs.
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
