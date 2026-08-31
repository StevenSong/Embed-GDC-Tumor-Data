# Virchow tile embedding:
# * the model is baked into the image at build time
# * runs in-process against the local GPU
# * tile images are never written to disk

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

ENV PYTHONUNBUFFERED=1 HF_HOME=/opt/hf

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

COPY pyproject.toml /app/
COPY src/ /app/src/
RUN pip install --no-cache-dir -e /app

ENTRYPOINT ["python", "-m", "embedder"]
