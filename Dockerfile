FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /code

COPY requirements.txt /code/requirements.txt
RUN python3 -m pip install --upgrade pip==25.1.1 \
    && python3 -m pip install -r /code/requirements.txt

# Network access is allowed during image construction. Runtime is fully offline.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c \
    "from huggingface_hub import snapshot_download; snapshot_download('unsloth/Qwen3-1.7B', local_dir='/models/qwen3_1_7b_base')"

COPY assets/checkpoints/qwen3_1_7b_epoch3 /code/assets/checkpoints/qwen3_1_7b_epoch3
COPY src/submission /code/src/submission
COPY predict.py inference.sh README.md LICENSE /code/

RUN chmod +x /code/inference.sh

CMD ["bash", "/code/inference.sh"]
