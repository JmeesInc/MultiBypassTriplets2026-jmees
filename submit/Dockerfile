# MultiBypassTriplets2026 submission — ensemble-distilled 8-model ensemble
# (4x SPIRIT students + 4x Swin students, 4-fold CV videowise ivt mAP = 0.5181)
# CUDA 11.8 locked for eval-server compatibility. Fully offline (weights bundled).
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# offline: never try to reach HF hub / timm / torchvision downloads at runtime
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV TORCH_HOME=/app/.torch

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ca-certificates libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "numpy<2" pillow tqdm opencv-python-headless \
    && pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir "transformers==4.56.2" "timm==1.0.27" safetensors scipy \
    # dinov3 hub deps (its hubconf imports segmentors -> torchmetrics, and text utils)
    && pip install --no-cache-dir torchmetrics ftfy omegaconf regex termcolor einops pyyaml \
    # src/train_verb_fusion.py imports sklearn at module scope (metrics helper)
    && pip install --no-cache-dir scikit-learn

COPY . /app
WORKDIR /app
ENTRYPOINT ["python", "/app/main.py"]
