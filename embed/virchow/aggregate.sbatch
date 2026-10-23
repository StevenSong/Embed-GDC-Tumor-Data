#!/bin/bash
#SBATCH --job-name=virchow-aggregate
#SBATCH --nodelist=kg35-nvl02
#SBATCH --output=%x-%j.log
#
# Collect the slide-level `mean` of every $OUT_DIR/embeddings/*.h5 into one
# $OUT_DIR/hist.h5, in multimodal-cancer-modeling's layout. No GPU: it reads
# ~10 KB per slide and takes minutes.
#
#   OUT_DIR=/path/to/output \
#     sbatch embed/virchow/aggregate.sbatch
#
# run-corpus.sh submits this after the embed shards, so it is rarely run by hand.
# It always rebuilds hist.h5: after a resumed embed run, an existing one is stale.
set -euo pipefail

# Required
OUT_DIR=${OUT_DIR:?set OUT_DIR to the directory embed-corpus.sbatch wrote into}

# Optional
IMAGE=${IMAGE:-virchow-embedder:latest}

[[ -d "$OUT_DIR/embeddings" ]] || {
  echo "no $OUT_DIR/embeddings; run the embed step first" >&2
  exit 1
}

NAME="virchow-aggregate-$SLURM_JOB_ID"

# The container is a child of dockerd, not of this job step, so scancel and the
# time limit would otherwise leave it running.
cleanup() { docker stop -t 30 "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT TERM INT

echo "job $SLURM_JOB_ID on $(hostname), image $IMAGE"
# The source is baked into the image; print its age so a stale build is obvious.
echo "image built $(docker image inspect -f '{{.Created}}' "$IMAGE" 2>/dev/null || echo UNKNOWN)"

# --user so hist.h5 is ours rather than root's; nothing here needs root
docker run --rm --name "$NAME" \
  --user "$(id -u):$(id -g)" \
  -v "$OUT_DIR:/out" \
  --entrypoint python \
  "$IMAGE" -m embedder.aggregate \
  --emb-dir /out/embeddings --output-h5 /out/hist.h5 --overwrite &

wait $!
