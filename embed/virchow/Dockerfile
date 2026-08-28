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

WORKDIR /app

# torch, torchvision and timm, with torch pinned -- see requirements.txt. Installed
# on its own layer and before everything else: it is by far the biggest download and
# the least likely to change, so edits to the slideflow list below don't refetch it.
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# slideflow's declared dependency list is ~35 packages and mostly irrelevant to
# tiling: a hyperparameter search stack (smac -> pyrfr, which needs swig to build on
# py>3.10), a GUI (imgui/glfw/pyopengl), umap/numba, tensorboard. Install it --no-deps
# and name its real imports in requirements-slideflow.txt instead. That list is read
# off slideflow's source, so the version is pinned here: bump the two together.
ARG SLIDEFLOW_VERSION=3.0.2
COPY requirements-slideflow.txt /app/
RUN pip install --no-cache-dir --no-deps slideflow==${SLIDEFLOW_VERSION} \
 && pip install --no-cache-dir -r requirements-slideflow.txt

ENV PYTHONUNBUFFERED=1 \
    SF_SLIDE_BACKEND=libvips \
    SF_BACKEND=torch \
    MPLCONFIGDIR=/tmp/mpl \
    NUMBA_CACHE_DIR=/tmp/numba

ENV HF_HOME=/opt/hf

# --no-deps above means a missing import shows up here, at build time, rather than on
# the cluster: fail now and add the package to the list. The vips backend is imported
# explicitly because slideflow only reaches it lazily, on the first slide read.
RUN python -c "import slideflow as sf; import slideflow.util, slideflow.slide.backends.vips; import slideflow.slide.qc as qc; qc.Otsu(); print('slideflow', sf.__version__, 'ok')"

COPY virchow.py download_model.py embed_wsi.py /app/

# paige-ai/Virchow is gated, so the build needs a HF token with access to the repo.
# It comes in as a BuildKit secret rather than an ARG, which would be readable in
# the image history forever. The weights land in $HF_HOME and ship in the image.
RUN --mount=type=secret,id=hf_token,required=true \
    HF_TOKEN="$(cat /run/secrets/hf_token)" python download_model.py \
 && chmod -R a+rX ${HF_HOME}

# weights are in the image now; never reach for the network at runtime
ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "/app/embed_wsi.py"]
