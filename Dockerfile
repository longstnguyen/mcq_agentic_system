FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda install -y python=3.12 pip=25.1.1 \
    && /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:${PATH}

WORKDIR /code

COPY requirements.txt /code/requirements.txt
RUN python -m pip install -r /code/requirements.txt

# Network access is allowed during image construction. Runtime is fully offline.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c \
    "from huggingface_hub import snapshot_download; snapshot_download('unsloth/Qwen3-1.7B', local_dir='/models/qwen3_1_7b_base')"

COPY assets/checkpoints/qwen3_1_7b_epoch3 /code/assets/checkpoints/qwen3_1_7b_epoch3
COPY src/submission /code/src/submission
COPY predict.py inference.sh README.md LICENSE /code/

RUN chmod +x /code/inference.sh

CMD ["bash", "/code/inference.sh"]
