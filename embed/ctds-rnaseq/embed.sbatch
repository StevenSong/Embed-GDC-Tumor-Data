#!/bin/bash
#SBATCH --job-name=ctds-rnaseq-embed
#SBATCH --nodelist=kg35-nvl02
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.log
#
# Embed a gdc-client download of RNA-seq STAR-counts TSVs with embed/ctds-rnaseq,
# writing every sample into one $OUT_DIR/expr.h5:
#
#   RNA_DIR=/opt/gpudata/cptac-primary-tumor-rna-seq \
#   METADATA=metadata/cptac-nsclc-primary-tumor-rna-seq.csv \
#   MANIFEST=manifests/cptac-nsclc-primary-tumor-rna-seq.txt \
#   OUT_DIR=/path/to/output \
#     sbatch embed/ctds-rnaseq/embed.sbatch
#
# MANIFEST is optional: without it every TSV under RNA_DIR is embedded, and
# METADATA must then name a case for all of them. There is no resume -- all of
# TCGA is minutes of GPU -- and an existing expr.h5 is refused; delete it to redo.
set -euo pipefail

# Required
RNA_DIR=${RNA_DIR:?set RNA_DIR to the gdc-client download directory}
METADATA=${METADATA:?set METADATA to a CSV with file_name and case_id columns}
OUT_DIR=${OUT_DIR:?set OUT_DIR to the directory to write expr.h5 into}

# Optional
MANIFEST=${MANIFEST:-}
IMAGE=${IMAGE:-ctds-rnaseq-embedder:latest}

# Read from slurm reservation
: "${CUDA_VISIBLE_DEVICES:?must specify GPU}"

mkdir -p "$OUT_DIR" # create first so we own it, rather than root owned from inside docker

NAME="ctds-rnaseq-embed-$SLURM_JOB_ID"

# The container is a child of dockerd, not of this job step, so scancel and the
# time limit would otherwise leave it running on the GPU we just gave back.
cleanup() { docker stop -t 30 "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT TERM INT

echo "job $SLURM_JOB_ID on $(hostname), GPU $CUDA_VISIBLE_DEVICES, image $IMAGE"
# The source is baked into the image; print its age so a stale build is obvious.
echo "image built $(docker image inspect -f '{{.Created}}' "$IMAGE" 2>/dev/null || echo UNKNOWN)"

# the two tables are mounted file by file, so they can live anywhere
mounts=(-v "$(realpath "$METADATA"):/tables/$(basename "$METADATA"):ro")
args=(--metadata "/tables/$(basename "$METADATA")")
if [[ -n "$MANIFEST" ]]; then
  mounts+=(-v "$(realpath "$MANIFEST"):/tables/$(basename "$MANIFEST"):ro")
  args+=(--manifest "/tables/$(basename "$MANIFEST")")
fi

docker run --rm --name "$NAME" \
  --gpus "\"device=$CUDA_VISIBLE_DEVICES\"" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$RNA_DIR:/rna:ro" \
  -v "$OUT_DIR:/out" \
  "${mounts[@]}" \
  "$IMAGE" --rna-dir /rna "${args[@]}" --output-h5 /out/expr.h5 &

wait $!
