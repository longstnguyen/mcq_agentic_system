#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODEL="${BASE_MODEL_PATH:-/models/qwen3_1_7b_base}"
ADAPTER="${ADAPTER_PATH:-$ROOT/assets/checkpoints/qwen3_1_7b_epoch3}"
SERVED_MODEL="${SERVED_MODEL:-vitutor-qwen3-1.7b-epoch3}"
PORT="${LLM_PORT:-8893}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python3 -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype "${MODEL_DTYPE:-bfloat16}" \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
  --max-num-seqs 1 \
  --enforce-eager \
  --disable-log-stats \
  --generation-config vllm \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules "$SERVED_MODEL=$ADAPTER" \
  > "${VLLM_LOG:-/tmp/vllm.log}" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true' EXIT

python3 - "$PORT" <<'PY'
import sys
import time
import urllib.request

url = f"http://127.0.0.1:{sys.argv[1]}/v1/models"
for _ in range(300):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status == 200:
                raise SystemExit(0)
    except Exception:
        time.sleep(1)
raise SystemExit("vLLM did not become ready; inspect /tmp/vllm.log")
PY

export INFERENCE_BACKEND=vllm
export LLM_ENDPOINT="http://127.0.0.1:$PORT/v1/chat/completions"
export SERVED_MODEL
python3 "$ROOT/predict.py" "$@"
