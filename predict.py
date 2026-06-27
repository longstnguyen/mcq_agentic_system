from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from submission.client import TransformersClient, VLLMClient  # noqa: E402
from submission.schema import Prediction, load_questions, write_outputs  # noqa: E402
from submission.solver import SingleModelAgent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline single-model MCQ submission runner")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(os.getenv("PRIVATE_TEST_PATH", "/code/private_test.json")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("OUTPUT_DIR", "/code")),
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("LLM_ENDPOINT", "http://127.0.0.1:8893/v1/chat/completions"),
    )
    parser.add_argument(
        "--backend",
        choices=("transformers", "vllm"),
        default=os.getenv("INFERENCE_BACKEND", "vllm"),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("SERVED_MODEL", "vitutor-qwen3-1.7b-epoch3"),
    )
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "2048")))
    parser.add_argument(
        "--tool-max-tokens",
        type=int,
        default=int(os.getenv("TOOL_MAX_TOKENS", "512")),
    )
    parser.add_argument(
        "--calculator",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("ENABLE_CALCULATOR", "1") == "1",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path(os.getenv("BASE_MODEL_PATH", "/models/qwen3_1_7b_base")),
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path(
            os.getenv(
                "ADAPTER_PATH",
                str(ROOT / "assets/checkpoints/qwen3_1_7b_epoch3"),
            )
        ),
    )
    args = parser.parse_args()

    questions = load_questions(args.input)
    if args.backend == "transformers":
        client = TransformersClient(args.base_model, args.adapter)
    else:
        client = VLLMClient(args.endpoint, args.model, timeout=args.timeout)
    solver = SingleModelAgent(
        client,
        calculator_enabled=args.calculator,
        max_tokens=args.max_tokens,
        tool_max_tokens=args.tool_max_tokens,
    )
    predictions: list[Prediction] = []
    print(
        f"[INFO] questions={len(questions)} backend={args.backend} "
        f"model={args.model} calculator={args.calculator}",
        flush=True,
    )
    for index, question in enumerate(questions, 1):
        started = time.perf_counter()
        answer, audit = solver.solve(question)
        elapsed = time.perf_counter() - started
        predictions.append(
            Prediction(
                qid=question.qid,
                answer=answer,
                elapsed_seconds=elapsed,
                audit=audit,
            )
        )
        print(f"[{index}/{len(questions)}] {question.qid} -> {answer} ({elapsed:.3f}s)", flush=True)

    write_outputs(args.output_dir, questions, predictions)
    print(f"[DONE] {args.output_dir / 'submission.csv'}", flush=True)
    print(f"[DONE] {args.output_dir / 'submission_time.csv'}", flush=True)


if __name__ == "__main__":
    main()
