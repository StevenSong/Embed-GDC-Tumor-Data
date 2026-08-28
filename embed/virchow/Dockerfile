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

# cu130 wheels are the current default build; pinned so a rebuild later still gets
# the stack this was written against. Installed first: it is the biggest layer and
# the one least likely to change, and nothing below is allowed to move it.
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
RUN pip install --no-cache-dir \
      torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
      --index-url ${TORCH_INDEX_URL}

# slideflow's dependency list is ~35 packages and mostly irrelevant to tiling: a
# hyperparameter search stack (smac -> pyrfr, which needs swig to build on py>3.10),
# a GUI (imgui/glfw/pyopengl), umap/numba, rasterio, tensorboard. Install it with
# --no-deps and name only what the tiling path imports. timm gets the same treatment
# so it cannot drag in a second copy of torch.
RUN pip install --no-cache-dir --no-deps slideflow timm \
 && pip install --no-cache-dir \
      click \
      crc32c \
      fpdf2 \
      h5py \
      huggingface_hub \
      imageio \
      matplotlib \
      opencv-python-headless \
      pandas \
      pillow \
      pyarrow \
      pyvips \
      pyyaml \
      rich \
      safetensors \
      scikit-image \
      scikit-learn \
      scipy \
      seaborn \
      shapely \
      tabulate \
      tqdm \
      zarr

ENV PYTHONUNBUFFERED=1 \
    SF_SLIDE_BACKEND=libvips \
    SF_BACKEND=torch \
    MPLCONFIGDIR=/tmp/mpl \
    NUMBA_CACHE_DIR=/tmp/numba

ENV HF_HOME=/opt/hf

# --no-deps above means a missing transitive import shows up here, at build time,
# rather than on the cluster: fail now and add the package to the list.
RUN python -c "import slideflow as sf; import slideflow.util; import slideflow.slide.qc as qc; qc.Otsu(); print('slideflow', sf.__version__, 'ok')"

WORKDIR /app
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
