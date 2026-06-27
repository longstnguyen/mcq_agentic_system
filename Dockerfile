FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/0.11.25/install.sh \
        | env UV_INSTALL_DIR=/usr/local/bin sh \
    && UV_PYTHON_INSTALL_DIR=/opt/uv-python uv python install 3.12 \
    && UV_PYTHON_INSTALL_DIR=/opt/uv-python uv venv --python 3.12 /opt/venv

ENV PATH=/opt/venv/bin:/usr/local/bin:${PATH} \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python

WORKDIR /code

COPY requirements.txt /code/requirements.txt
RUN uv pip sync --python /opt/venv/bin/python \
        --torch-backend=cu128 \
        --no-cache \
        /code/requirements.txt

# Network access is allowed during image construction. Runtime is fully offline.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c \
    "from huggingface_hub import snapshot_download; snapshot_download('unsloth/Qwen3-1.7B', local_dir='/models/qwen3_1_7b_base')"

COPY assets/checkpoints/qwen3_1_7b_epoch3 /code/assets/checkpoints/qwen3_1_7b_epoch3
COPY src/submission /code/src/submission
COPY predict.py inference.sh README.md LICENSE /code/

RUN chmod +x /code/inference.sh

# Triton compiles a small CUDA driver shim on first startup.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

CMD ["bash", "/code/inference.sh"]
