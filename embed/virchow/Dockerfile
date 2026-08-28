# Virchow tile embedding:
# * the model is baked into the image at build time
# * runs in-process against the local GPU
# * tile images are never written to disk

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# libvips is slideflow's slide reader; libexpat1 is its XML dep
RUN apt-get update \
 && apt-get install -y --no-install-recommends libexpat1 libvips \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-slideflow.txt /app/

# our main requirements for doing embedding
RUN pip install --no-cache-dir -r /app/requirements.txt

# slideflow's slim tiling dependencies, verified against v3.0.2
ARG SLIDEFLOW_VERSION=3.0.2
RUN pip install --no-cache-dir --no-deps slideflow==${SLIDEFLOW_VERSION} \
 && pip install --no-cache-dir -r /app/requirements-slideflow.txt

ENV PYTHONUNBUFFERED=1 \
    SF_SLIDE_BACKEND=libvips \
    SF_BACKEND=torch \
    HF_HOME=/opt/hf

# paige-ai/Virchow is gated, so the build needs a HF token with access to the repo.
# It comes in as a BuildKit secret rather than an ARG, which would be readable in
# the image history forever. The weights land in $HF_HOME and ship in the image.
WORKDIR /app/src/
RUN --mount=type=secret,id=hf_token,required=true \
    HF_TOKEN="$(cat /run/secrets/hf_token)" \
    python -c 'import timm, torch; timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)' \
 && chmod -R a+rX "$HF_HOME"

# weights are in the image now; never reach for the network at runtime
ENV HF_HUB_OFFLINE=1

COPY src/ /app/src/

ENTRYPOINT ["python", "/app/src/embed_wsi.py"]
