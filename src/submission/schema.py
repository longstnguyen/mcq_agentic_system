from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    choices: tuple[str, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(chr(ord("A") + index) for index in range(len(self.choices)))

    def choices_block(self) -> str:
        return "\n".join(
            f"{label}. {strip_existing_label(choice)}"
            for label, choice in zip(self.labels, self.choices)
        )


@dataclass(frozen=True)
class Prediction:
    qid: str
    answer: str
    elapsed_seconds: float
    audit: dict[str, Any]


def strip_existing_label(choice: str) -> str:
    text = str(choice).strip()
    if len(text) >= 2 and text[0].upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        remainder = text[1:].lstrip()
        if remainder.startswith((".", ")", ":", "-")):
            return remainder[1:].lstrip()
    return text


def _extract_rows(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "test", "items", "questions"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError("private_test.json must be a list or contain a data/test/items/questions list")


def _normalize_choices(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, dict):
        keys = sorted(raw, key=lambda key: str(key))
        values = [raw[key] for key in keys]
    elif isinstance(raw, list):
        values = raw
    else:
        raise ValueError("choices/options must be a list or object")
    choices = tuple(str(value).strip() for value in values)
    if not 2 <= len(choices) <= 26 or any(not choice for choice in choices):
        raise ValueError("each item must contain 2-26 non-empty choices")
    return choices


def load_questions(path: Path) -> list[Question]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions: list[Question] = []
    seen: set[str] = set()
    for index, raw in enumerate(_extract_rows(payload)):
        if not isinstance(raw, dict):
            raise ValueError(f"item {index} is not an object")
        qid = str(raw.get("qid") or raw.get("id") or f"test_{index + 1:04d}").strip()
        question = str(raw.get("question") or raw.get("prompt") or raw.get("text") or "").strip()
        choices = _normalize_choices(raw.get("choices") or raw.get("options") or raw.get("answers"))
        if not qid or not question:
            raise ValueError(f"item {index} has an empty qid or question")
        if qid in seen:
            raise ValueError(f"duplicate qid: {qid}")
        seen.add(qid)
        questions.append(Question(qid=qid, question=question, choices=choices))
    if not questions:
        raise ValueError("private_test.json contains no questions")
    return questions


def validate_predictions(questions: list[Question], predictions: list[Prediction]) -> None:
    if len(questions) != len(predictions):
        raise ValueError(f"expected {len(questions)} predictions, received {len(predictions)}")
    for question, prediction in zip(questions, predictions):
        if prediction.qid != question.qid:
            raise ValueError(f"qid order mismatch: expected {question.qid}, received {prediction.qid}")
        if prediction.answer not in question.labels:
            raise ValueError(f"invalid answer {prediction.answer!r} for {question.qid}")
        if prediction.elapsed_seconds < 0:
            raise ValueError(f"negative inference time for {question.qid}")


def write_outputs(output_dir: Path, questions: list[Question], predictions: list[Prediction]) -> None:
    validate_predictions(questions, predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "submission.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["qid", "answer"])
        writer.writerows((item.qid, item.answer) for item in predictions)
    with (output_dir / "submission_time.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["qid", "answer", "time"])
        writer.writerows(
            (item.qid, item.answer, f"{item.elapsed_seconds:.6f}") for item in predictions
        )
    with (output_dir / "responses.jsonl").open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(
                json.dumps(
                    {
                        "qid": item.qid,
                        "answer": item.answer,
                        "time": item.elapsed_seconds,
                        **item.audit,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

