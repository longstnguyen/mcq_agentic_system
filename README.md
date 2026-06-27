# ViTutor MCQ Submission System

This repository contains the final offline submission runtime for the
Vietnamese Student HackAIthon 2026 Innovator track. Research experiments are
kept outside the submission branch, and Docker copies only the files required
by the compliant runtime.

The final system uses exactly one learned model:

- Base model: `unsloth/Qwen3-1.7B`
- Fine-tuned adapter: `danny2507/vitutorqwen3-1.7b-sft-best-grpo-dpo`
- Selected checkpoint: `adapters/epoch_3.00_step_897`
- Local adapter path: `assets/checkpoints/qwen3_1_7b_epoch3`

It does not use RAG, BM25, embedding models, rerankers, web search, remote
endpoints, or closed model APIs. This follows the organizer's 27 June 2026
clarification that RAG is not allowed for the final submission.

## Pipeline Flow

The runtime is intentionally small:

```text
/code/private_test.json
        |
        v
validate and normalize question plus choices
        |
        v
ViTutor Qwen3-1.7B epoch 3 via vLLM, thinking enabled
        |
        +---- optional request to a restricted numeric calculator
        |                  |
        |                  v
        +---- same ViTutor checkpoint verifies the calculator result
        |
        v
parse and validate one allowed answer label
        |
        v
submission.csv and submission_time.csv
```

The calculator is deterministic code, not another model. It evaluates only a
small AST whitelist of numeric operators and mathematical functions. Imports,
attribute access, file access, shell commands, arbitrary Python, and expensive
expressions are rejected. Set `ENABLE_CALCULATOR=0` if the organizer does not
approve deterministic calculation tools.

The model is called once for ordinary questions. The calculator runs only when
the first response has no parseable answer, contains a syntactically valid tool
request, and grounds at least half of its numeric constants in the question. A
second call to the same checkpoint then verifies the result. Tool requests are
ignored when an answer is already available, so the tool cannot overwrite a
completed solve. If the answer still cannot be parsed, the same checkpoint
receives one no-thinking format-repair request capped at 512 tokens.

## Data Processing

The final runtime does not ingest an external corpus. Input processing is
limited to operations required by the submission contract:

1. Read a top-level list or a `data`, `test`, `items`, or `questions` list.
2. Normalize `qid`, question text, and list/dictionary choice schemas.
3. Preserve input order and every available choice, including items with more
   than four choices.
4. Remove an existing `A.`, `B)`, or similar prefix before rendering choices.
5. Validate unique qids, non-empty text, allowed answer labels, row counts, and
   non-negative per-item inference times.

No public labels, GPT-generated labels, retrieval corpus, or cached web data are
copied into the Docker image.

## Runtime Files

- `Dockerfile`: CUDA 12.2 image build and offline model initialization.
- `requirements.txt`: pinned inference engine version.
- `inference.sh`: starts local vLLM, waits for readiness, and runs prediction.
- `predict.py`: official end-to-end entry point.
- `src/submission/client.py`: local OpenAI-compatible vLLM client. A direct
  Transformers client remains available for development smoke tests.
- `src/submission/solver.py`: single-model solving and tool-call orchestration.
- `src/submission/calculator.py`: restricted deterministic calculator.
- `src/submission/schema.py`: input normalization and output validation.

## Resource Initialization

Docker downloads the open Qwen3-1.7B base weights during image construction and
stores them at `/models/qwen3_1_7b_base`. The fine-tuned LoRA adapter is copied
from the repository. Runtime environment variables force Hugging Face and
Transformers into offline mode.

The image contains no retrieval index. The `.dockerignore` file excludes the
research corpus, dense index, experiment outputs, caches, alternate checkpoints,
and test labels from the build context.

## Build

```bash
docker build -t vitutor-mcq:final .
```

The first build requires internet access to install pinned dependencies and
download the base model. Final inference requires no internet access.

## Offline Smoke Test

```bash
mkdir -p output
docker run --rm --gpus all --network none \
  -v "$PWD/private_test.json:/code/private_test.json:ro" \
  -v "$PWD/output:/output" \
  -e OUTPUT_DIR=/output \
  vitutor-mcq:final
```

Expected files:

```text
output/submission.csv
output/submission_time.csv
output/responses.jsonl
```

`submission.csv` has columns `qid,answer`. `submission_time.csv` has columns
`qid,answer,time`, where `time` measures the complete per-question solve path.
Model startup is not included in an individual question's time.

## Runtime Controls

- `GPU_MEMORY_UTILIZATION`: defaults to `0.80`, leaving roughly 3.2GB of a
  16GB GPU outside vLLM's memory budget for CUDA/runtime overhead.
- `MAX_MODEL_LEN`: defaults to `16384` tokens.
- `MODEL_DTYPE`: defaults to `bfloat16`, matching training. Use `half` only if
  the evaluation GPU does not support native bfloat16.
- `MAX_TOKENS`: defaults to `2048`. On the pinned vLLM 0.8.5 stack, some
  reasoning questions need nearly 2,000 tokens before emitting the final JSON.
- `TOOL_MAX_TOKENS`: defaults to `512` generated tokens.
- `ENABLE_CALCULATOR`: defaults to `1`; set to `0` to disable the tool.
- `OUTPUT_DIR`: defaults to `/code` as required by the organizer's template.

Generation uses greedy decoding. Qwen thinking remains enabled for the main
solve and tool-verification calls. vLLM runs one sequence at a time so the
per-item timing contract stays meaningful and peak memory remains predictable.

## Verification

Run the local unit tests:

```bash
python3 -m pytest -q tests/test_submission.py
```

Before submission, build the exact Docker image and verify all of the following:

- run with `--network none`;
- peak GPU memory stays below 16GB;
- every input qid occurs exactly once and in input order;
- every answer belongs to that item's allowed labels;
- both required CSV files are written successfully;
- the public repository and Docker image remain unchanged after submission.

The official rule document and the local compliance audit are under
`docs/regulations/`.
