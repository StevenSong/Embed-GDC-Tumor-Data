# ctds-rnaseq bulk RNA-seq embedding:
# * the weights are baked into the image at build time, from the `weights` build
#   context -- a directory laid out like omics-encoder's kserve-demo/ -- so they
#   never have to be committed here; see README.md
# * runs in-process against the local GPU

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

ENV PYTHONUNBUFFERED=1

# The context is the omics-encoder repo root, not kserve-demo/: BuildKit applies
# a named context's own .dockerignore, and kserve-demo's (written for its
# preprocessor image) excludes checkpoint/. The vocabulary lives in the served
# model's directory, not checkpoint/; it goes with the weights because it
# decides what each position embedding means.
COPY --from=weights \
     kserve-demo/checkpoint/model.pt \
     kserve-demo/checkpoint/binner.npz \
     kserve-demo/checkpoint/run_config.json \
     kserve-demo/model-repo/bulkrnabert/genes.tsv \
     /opt/ctds-rnaseq/
RUN chmod -R a+rX /opt/ctds-rnaseq

COPY pyproject.toml /app/
COPY src/ /app/src/
RUN pip install --no-cache-dir -e /app

ENTRYPOINT ["python", "-m", "embedder"]
