# Vietnamese Student HackAIthon 2026: Innovator Submission Checklist

Official source: [Submission Guideline](https://docs.google.com/document/d/1T9ckeTxuRetd6Lu-cynNkaWn1mDG8BM7/edit?usp=sharing)

Downloaded original:
`docs/regulations/vietnamese-student-hackaithon-innovator-submission-rules.docx`

SHA-256:
`615d6ed4e6548c0c5331fb31110e140c560bc2a08d663577139571d7363a898b`

Reviewed on 27 June 2026. This checklist also includes the organizer's
supplemental Innovator notice supplied by the team.

## Hard Requirements

- Push the Docker image before 23:59 UTC+7 on 27 June 2026.
- Submit a public GitHub repository and do not modify it after submission.
- Use a CUDA 12.2 base image when GPU inference is required.
- Read `/code/private_test.json` at runtime.
- Write `submission.csv` with columns `qid,answer`.
- Write `submission_time.csv` with columns `qid,answer,time`.
- Measure inference time separately around each test item.
- Include `Dockerfile`, pinned `requirements.txt`, `predict.py`, and
  `inference.sh`.
- Use one trained or fine-tuned model only during final inference.
- Keep the model at or below 5B parameters.
- Use an open model with a license permitting the intended research or
  commercial use.
- Do not call closed model APIs, external model endpoints, search APIs, or data
  APIs during inference.
- Expect the evaluation container to run without internet access.
- Keep peak GPU memory below the organizer's 16GB VRAM limit.
- Use only open, appropriately licensed training and retrieval data.

## Organizer Clarification: 27 June 2026

The organizer confirmed in direct chat that the final pipeline may use only one
LLM and may not use an embedding model or reranker. When asked whether RAG
without embedding/reranking was allowed, the organizer answered that it was
not. The conservative and submission-safe interpretation is therefore:

- no RAG of any kind, including BM25 or SQLite FTS retrieval;
- no packaged retrieval corpus or retrieval index;
- no embedding model or reranker;
- no web search, online or from a local snapshot;
- one ViTutor Qwen3-1.7B checkpoint for all model calls.

Deterministic arithmetic is not addressed by that clarification. The final
runtime implements a restricted calculator and provides `ENABLE_CALCULATOR=0`
so it can be disabled immediately if the organizer does not approve tools.

## Compliant Final Architecture

The competition pipeline should use:

```text
private_test.json
  -> normalize question and choices
  -> ViTutor Qwen3-1.7B epoch 3.0
  -> optional restricted numeric calculator
  -> the same Qwen3-1.7B model verifies the tool result when used
  -> validate answer and record per-item time
  -> submission.csv + submission_time.csv
```

The selected model is:

- Adapter: `danny2507/vitutorqwen3-1.7b-sft-best-grpo-dpo`
- Checkpoint: `adapters/epoch_3.00_step_897`
- Recorded epoch and global step: 3.0 and 897
- Base model: `unsloth/Qwen3-1.7B`
- Local adapter: `assets/checkpoints/qwen3_1_7b_epoch3`

Multiple calls to the same Qwen3-1.7B checkpoint still use one model. The
calculator and output parser are deterministic code. No retrieval algorithm is
used in the final path.

## Components Excluded From Final Inference

- Qwen3.5-9B: exceeds the 5B parameter limit.
- BGE-M3 dense retrieval: introduces a second learned model.
- Qwen3-Reranker-4B: introduces another learned model.
- Live web search and page fetching: fail in an offline container and violate
  the prohibition on search/data APIs.
- BM25 and SQLite FTS: excluded because the organizer explicitly rejected RAG
  even without learned embedding and reranking models.
- Closed model APIs and GPT labels: benchmark artifacts only, never runtime
  dependencies.
- The meta live-web submission: it was useful for experimentation but is not a
  reproducible offline submission method.

## Current Repository Audit

| Requirement | Status | Required action |
| --- | --- | --- |
| Model at most 5B | Implemented | Qwen3-1.7B epoch 3 is selected; 9B is excluded from Docker |
| One learned model | Implemented | Final runtime calls only the selected Qwen3-1.7B checkpoint |
| 16GB VRAM | Budget smoke passed | End-to-end vLLM ran with an 11.5GB absolute budget and 16K context; exact 16GB Docker still required |
| Offline inference | Implemented, unverified in Docker | Runtime sets offline flags and loads local assets; test image with `--network none` |
| Model resources local | Implemented at build | Base weights download during build; adapter is copied locally |
| Retrieval resources | Not applicable | No corpus or index enters the final image |
| Model/data licenses | Blocking audit | Base is Apache-2.0; declare adapter and training-data licenses |
| `Dockerfile` CUDA 12.2 | Implemented, unbuilt | Build and test the exact final image |
| Pinned `requirements.txt` | Implemented | Full 153-package tree is locked; vLLM is pinned to 0.8.5 |
| `predict.py` | Implemented | Reads private test and writes both required CSV files |
| `inference.sh` | Implemented | Starts local vLLM and runs prediction end to end |
| Per-item timing | Implemented | Times each complete solve path separately |
| Output validation | Implemented | Validates qids, order, row count, labels, and times |
| README submission sections | Implemented | Documents flow, processing, resources, Docker, and verification |
| Docker Hub image | Missing | Build, offline smoke-test, then push before the deadline |
| Public frozen GitHub repo | Published, not submitted | `main` and tag `submission-2026-06-27` point to the audited source; freeze only after submitting the form |

## Submission Smoke Test

```bash
docker build -t team_submission .
docker run --rm --gpus all --network none \
  -v "$PWD/private_test.json:/code/private_test.json:ro" \
  -v "$PWD/output:/code/output" \
  team_submission
```

Verify that:

- peak VRAM remains below 16GB;
- no network request occurs;
- both CSV files contain every input qid exactly once and in input order;
- each answer is one of the choices allowed for that item;
- each time value is numeric and non-negative;
- the model, tokenizer, adapter, and retrieval index load from local paths.

## Immediate Work Order

1. Confirm whether the restricted deterministic calculator is allowed.
2. Declare the adapter license and audit all training-data licenses.
3. Build the exact CUDA 12.2 Docker image with Docker daemon access.
4. Run the image with no network and measure peak VRAM on an actual 16GB GPU.
5. Validate both CSV files against the complete public test schema.
6. Push the tested image and freeze the public repository.

The detailed license audit is in
`docs/regulations/model-and-data-license-audit.md`.

Local verification used the pinned dependency tree, vLLM 0.8.5, Qwen3 LoRA,
BF16, a 16K context window, and an 11.5GB absolute GPU-memory budget. The full
`inference.sh` path produced both required CSV files. Docker itself could not be
built in the current account because access to `/var/run/docker.sock` requires
sudo credentials.
