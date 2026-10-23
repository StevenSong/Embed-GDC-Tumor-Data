# Pathology report summarization (Gemma 4 31B) and embedding (EmbeddingGemma 2):
# * both models run in-process under vLLM against the local GPU(s)
# * weights are NOT baked into the image -- Gemma 4 31B is 62 GB -- but read from
#   model directories (`hf download --local-dir`) mounted at /opt/gpudata/models;
#   see README.md

ARG BASE_IMAGE=vllm/vllm-openai:v0.32.0
FROM ${BASE_IMAGE}

COPY requirements.txt /app/
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# weights come from the mounted model directories; never reach for the network
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1

COPY pyproject.toml /app/
COPY src/ /app/src/
RUN python3 -m pip install --no-cache-dir -e /app

# the base image's entrypoint is the OpenAI-compatible server
ENTRYPOINT ["python3", "-m", "embedder"]
