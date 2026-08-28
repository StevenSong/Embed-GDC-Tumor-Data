# syntax=docker/dockerfile:1.7
#
# Virchow tile embedding without KServe/Triton: the model is baked into the image
# at build time and runs in-process against the local GPU. See README.md.

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# libvips is slideflow's slide reader; libexpat1 is its XML dep
RUN apt-get update \
 && apt-get install -y --no-install-recommends libexpat1 libvips \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# slideflow's pins lag behind, so install it (and whatever it drags in) first and
# let the pinned torch below clobber what it chose
RUN pip install --no-cache-dir slideflow h5py timm

# cu130 wheels are the current default build; pinned so a rebuild later still gets
# the stack this was written against
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
RUN pip install --no-cache-dir \
      torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
      --index-url ${TORCH_INDEX_URL}

ENV HF_HOME=/opt/hf

WORKDIR /app
COPY virchow.py download_model.py embed_wsi.py /app/

# paige-ai/Virchow is gated, so the build needs a HF token with access to the repo.
# It comes in as a BuildKit secret rather than an ARG, which would be readable in
# the image history forever. The weights land in $HF_HOME and ship in the image.
RUN --mount=type=secret,id=hf_token,required=true \
    HF_TOKEN="$(cat /run/secrets/hf_token)" python download_model.py \
 && chmod -R a+rX ${HF_HOME}

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    SF_SLIDE_BACKEND=libvips \
    SF_BACKEND=torch \
    MPLCONFIGDIR=/tmp/mpl \
    NUMBA_CACHE_DIR=/tmp/numba

ENTRYPOINT ["python", "/app/embed_wsi.py"]
