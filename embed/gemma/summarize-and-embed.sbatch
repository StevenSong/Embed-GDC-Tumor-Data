#!/bin/bash
#SBATCH --job-name=gemma-embed
#SBATCH --nodelist=kg35-nvl02
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.log
#
# Reproduce the multimodal-cancer-modeling report pipeline with Gemma 4 /
# EmbeddingGemma 2, writing the same three files that pipeline did:
#
#   $OUT_DIR/summarized_reports.csv   one Gemma 4 31B summary per report
#   $OUT_DIR/text.h5                  EmbeddingGemma 2 of the raw reports
#   $OUT_DIR/summ.h5                  EmbeddingGemma 2 of the summaries
#
#   REPORTS_CSV=/opt/gpudata/tcga-reports/TCGA_Reports.csv \
#   OUT_DIR=/path/to/output \
#     sbatch embed/gemma/summarize-and-embed.sbatch
#
# Summarization resumes from a partial summarized_reports.csv, so a job that
# hits the time limit can simply be resubmitted. The embedding steps take
# minutes and are skipped when their .h5 already exists; delete one to redo it.
#
# With --gres=gpu:N the model is split across all N GPUs (tensor parallel).
# One H200 holds the 62 GB of bf16 weights with room to spare.
#
# Weights are read from MODEL_DIR, which must hold one `hf download --local-dir`
# directory per model, at its full Hub id (see README.md):
#
#   $MODEL_DIR/google/gemma-4-31B-it/
#   $MODEL_DIR/google/embeddinggemma-2/
set -euo pipefail

# Required
REPORTS_CSV=${REPORTS_CSV:?set REPORTS_CSV to TCGA_Reports.csv}
OUT_DIR=${OUT_DIR:?set OUT_DIR to the directory to write summaries and embeddings into}

# Optional
IMAGE=${IMAGE:-gemma-embedder:latest}
MODEL_DIR=${MODEL_DIR:-/opt/gpudata/models}

# The image is offline, so a missing model is fatal either way; fail here, in
# one line, rather than after the container has started.
for model in google/gemma-4-31B-it google/embeddinggemma-2; do
  [[ -f "$MODEL_DIR/$model/config.json" ]] || {
    echo "no model at $MODEL_DIR/$model; see embed/gemma/README.md to download it" >&2
    exit 1
  }
done

# Read from slurm reservation
: "${CUDA_VISIBLE_DEVICES:?must specify GPU}"
IFS=, read -ra GPUS <<<"$CUDA_VISIBLE_DEVICES"

mkdir -p "$OUT_DIR" # create first so we own it, rather than root owned from inside docker

NAME="gemma-embed-$SLURM_JOB_ID"

# The container is a child of dockerd, not of this job step, so scancel and the
# time limit would otherwise leave it running on the GPU we just gave back.
cleanup() { docker stop -t 30 "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT TERM INT

echo "job $SLURM_JOB_ID on $(hostname), GPU $CUDA_VISIBLE_DEVICES, image $IMAGE"
# The source is baked into the image; print its age so a stale build is obvious.
echo "image built $(docker image inspect -f '{{.Created}}' "$IMAGE" 2>/dev/null || echo UNKNOWN)"

run() {
  docker run --rm --name "$NAME" \
    --gpus "\"device=$CUDA_VISIBLE_DEVICES\"" \
    --user "$(id -u):$(id -g)" \
    --shm-size=16g \
    -e HOME=/tmp \
    -v "$MODEL_DIR:/opt/gpudata/models:ro" \
    -v "$(dirname "$REPORTS_CSV"):/reports:ro" \
    -v "$OUT_DIR:/out" \
    "$IMAGE" "$@" &
  wait $!
}

REPORTS=/reports/$(basename "$REPORTS_CSV")

run summarize --input-csv "$REPORTS" --output-csv /out/summarized_reports.csv \
  --tensor-parallel-size "${#GPUS[@]}"

for pair in "$REPORTS:text.h5" "/out/summarized_reports.csv:summ.h5"; do
  csv=${pair%%:*} h5=${pair##*:}
  if [[ -e "$OUT_DIR/$h5" ]]; then
    echo "$h5 exists, skipping"
    continue
  fi
  run embed --input-csv "$csv" --output-h5 "/out/$h5"
done
