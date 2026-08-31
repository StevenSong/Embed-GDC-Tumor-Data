#!/bin/bash
#SBATCH --job-name=virchow-embed
#SBATCH --nodelist=kg35-nvl02
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.log
#
# Embed a corpus of WSIs with embed/virchow. Resumable: slides that already have
# an .h5 are skipped, so a job that hits the time limit can simply be resubmitted.
#
#   SLIDE_DIR=/path/to/input/wsis \
#   OUT_DIR=/path/to/output/embeddings \
#     sbatch embed/virchow/embed-corpus.sbatch
#
set -euo pipefail

# Required
SLIDE_DIR=${SLIDE_DIR:?set SLIDE_DIR to the directory of slides to embed}
OUT_DIR=${OUT_DIR:?set OUT_DIR to the directory to write embeddings into}

# Optional
IMAGE=${IMAGE:-virchow-embedder:latest}
SHM_SIZE=${SHM_SIZE:-16g}

# Read from slurm reservation
: "${CUDA_VISIBLE_DEVICES:?must specify GPU}"

mkdir -p "$OUT_DIR" # make output directory first so we own it, rather than root owned from inside docker

NAME="virchow-embed-$SLURM_JOB_ID"

# The container is a child of dockerd, not of this job step, so scancel and the
# time limit would otherwise leave it running on the GPU we just gave back.
cleanup() { docker stop -t 30 "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT TERM INT

echo "job $SLURM_JOB_ID on $(hostname), GPU $CUDA_VISIBLE_DEVICES, image $IMAGE"

docker run --rm --name "$NAME" \
  --gpus "\"device=$CUDA_VISIBLE_DEVICES\"" \
  --shm-size="$SHM_SIZE" \
  -v "$SLIDE_DIR:/slides:ro" \
  -v "$OUT_DIR:/out" \
  "$IMAGE" --slide-dir /slides --out-dir /out &

wait $!
