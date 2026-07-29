FROM python:3.10-slim

RUN pip install --no-cache-dir slideflow kserve
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN apt-get update \
 && apt-get install -y --no-install-recommends libexpat1 libvips \
 && apt-get clean

COPY tiling/tile.py /tile.py
