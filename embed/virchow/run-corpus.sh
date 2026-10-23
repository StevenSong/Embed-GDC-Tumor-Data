#!/bin/bash
#
# Submit a whole corpus run: SHARDS embed jobs, one GPU each, then one aggregate
# job that waits on all of them and writes $OUT_DIR/hist.h5.
#
#   SLIDE_DIR=/path/to/input/wsis \
#   OUT_DIR=/path/to/output \
#   SHARDS=4 \
#     embed/virchow/run-corpus.sh
#
# The aggregate job runs only if every shard exits 0. A shard that fails a
# slide or hits the time limit cancels it (--kill-on-invalid-dep), rather than
# leaving it pending forever; rerun this script once the cause is fixed. Embed
# resumes by skipping slides that already have an .h5, so a rerun only redoes
# the missing ones before aggregating again.
#
# Everything else in the environment (IMAGE, SHM_SIZE) passes through to the
# jobs, since sbatch exports the submitting environment by default.
set -euo pipefail

# Required
: "${SLIDE_DIR:?set SLIDE_DIR to the directory of slides to embed}"
: "${OUT_DIR:?set OUT_DIR to the directory to write embeddings into}"

# Optional
SHARDS=${SHARDS:-1}

[[ "$SHARDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "SHARDS must be a positive integer, got $SHARDS" >&2
  exit 1
}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SLIDE_DIR OUT_DIR SHARDS

ids=()
for ((i = 0; i < SHARDS; i++)); do
  # --parsable prints `id` or `id;cluster`
  id=$(SHARD=$i sbatch --parsable "$HERE/embed-corpus.sbatch")
  ids+=("${id%%;*}")
  echo "shard $i/$SHARDS: job ${ids[-1]}"
done

deps=$(IFS=:; echo "${ids[*]}")
id=$(sbatch --parsable --dependency="afterok:$deps" --kill-on-invalid-dep=yes \
  "$HERE/aggregate.sbatch")
echo "aggregate: job ${id%%;*}, after $deps"
