FROM python:3.10-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libexpat1 libvips \
 && apt-get clean
 
RUN pip install --no-cache-dir slideflow h5py
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# make sure we have the latest kserve (probably most important thing), ok if it clobbers other deps
RUN pip install --no-cache-dir kserve

COPY tiling/ /tiling
ENTRYPOINT ["python", "tiling/main.py"]
