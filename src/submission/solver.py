from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from submission.calculator import CalculatorError, SafeCalculator
from submission.client import Completion, CompletionClient
from submission.schema import Question


SYSTEM_PROMPT = """Bạn là hệ thống giải trắc nghiệm tiếng Việt.
Hãy suy luận cẩn thận dựa trên đề bài và các lựa chọn. Không được giả định rằng có
tài liệu ngoài đề bài. Chỉ chọn một chữ cái hợp lệ.

Chỉ yêu cầu calculator khi đáp án phụ thuộc vào một phép tính số. Biểu thức phải
được lập trực tiếp từ các số trong đề bài, không được dùng biểu thức minh họa hoặc
số tự nghĩ ra. Calculator hỗ trợ +, -, *, /, //, %, ** và các hàm sqrt, sin,
cos, tan, log, log10, exp, factorial, comb, perm, mean, median. Góc lượng giác
mặc định là radian.

Phần trả lời cuối phải là đúng một JSON theo một trong hai dạng:
{"answer":"A","confidence":0.8,"tool":null}
{"answer":null,"confidence":0.5,"tool":{"expression":"<biểu thức lấy từ đề>","purpose":"<đại lượng cần kiểm tra>"}}
Không viết thêm văn bản sau JSON."""


FINAL_PROMPT = """Dùng kết quả calculator dưới đây để kiểm tra lại bài toán.
Nếu biểu thức không phù hợp với đề hoặc calculator báo lỗi, hãy tự giải theo đề.
Trả về đúng một JSON và không yêu cầu thêm tool:
{"answer":"A","confidence":0.8,"tool":null}"""


def _json_object(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S).strip()
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    for value in reversed(objects):
        if "answer" in value or "tool" in value:
            return value
    return objects[-1] if objects else None


def _answer_from_text(question: Question, text: str) -> str | None:
    data = _json_object(text)
    if data is not None:
        answer = str(data.get("answer") or "").strip().upper()
        if answer in question.labels:
            return answer
    patterns = (
        r"(?:đáp\s*án|answer|chọn)\s*(?:là|:)?\s*([A-Z])\b",
        r'"answer"\s*:\s*"([A-Z])"',
        r"^\s*([A-Z])\s*[.)]?\s*$",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text or "", flags=re.I | re.M)
        for answer in reversed(matches):
            answer = answer.upper()
            if answer in question.labels:
                return answer
    return None


def _tool_expression(text: str) -> str | None:
    data = _json_object(text)
    if not data or not isinstance(data.get("tool"), dict):
        return None
    expression = str(data["tool"].get("expression") or "").strip()
    return expression or None


def _expression_is_grounded(question: Question, expression: str) -> bool:
    question_numbers = set(re.findall(r"(?<![\w.])-?\d+(?:[.,]\d+)?", question.question))
    expression_numbers = set(re.findall(r"(?<![\w.])-?\d+(?:[.,]\d+)?", expression))
    normalize = lambda value: value.replace(",", ".").lstrip("+")
    question_numbers = {normalize(value) for value in question_numbers}
    expression_numbers = {normalize(value) for value in expression_numbers}
    if not question_numbers or not expression_numbers:
        return False
    overlap = question_numbers & expression_numbers
    return bool(overlap) and len(overlap) / len(expression_numbers) >= 0.5


def _completion_audit(completion: Completion) -> dict[str, Any]:
    return {
        "content": completion.content,
        "reasoning": completion.reasoning,
        "finish_reason": completion.finish_reason,
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
    }


class SingleModelAgent:
    def __init__(
        self,
        client: CompletionClient,
        *,
        calculator_enabled: bool = True,
        max_tokens: int = 2048,
        tool_max_tokens: int = 512,
    ) -> None:
        self.client = client
        self.calculator_enabled = calculator_enabled
        self.max_tokens = max_tokens
        self.tool_max_tokens = tool_max_tokens
        self.calculator = SafeCalculator()

    @staticmethod
    def _question_prompt(question: Question) -> str:
        return (
            f"Các chữ cái hợp lệ: {', '.join(question.labels)}\n\n"
            f"Câu hỏi:\n{question.question}\n\n"
            f"Các lựa chọn:\n{question.choices_block()}"
        )

    def solve(self, question: Question) -> tuple[str, dict[str, Any]]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._question_prompt(question)},
        ]
        first = self.client.complete(messages, max_tokens=self.max_tokens, enable_thinking=True)
        audit: dict[str, Any] = {"mode": "direct", "first": _completion_audit(first)}
        answer = _answer_from_text(question, first.content)
        expression = _tool_expression(first.content) if self.calculator_enabled else None

        if expression is not None:
            audit["calculator_requested"] = True
            audit["calculator_expression"] = expression
            audit["calculator_grounded"] = _expression_is_grounded(question, expression)

        if expression and answer is None and audit["calculator_grounded"]:
            try:
                tool_result = self.calculator.evaluate(expression)
            except (CalculatorError, ArithmeticError, OverflowError, ValueError) as exc:
                tool_result = f"CALCULATOR_ERROR: {exc}"
            if not tool_result.startswith("CALCULATOR_ERROR"):
                tool_prompt = (
                    f"{FINAL_PROMPT}\n\n{self._question_prompt(question)}\n\n"
                    f"Biểu thức: {expression}\nKết quả calculator: {tool_result}"
                )
                second = self.client.complete(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": tool_prompt}],
                    max_tokens=self.tool_max_tokens,
                    enable_thinking=True,
                )
                tool_answer = _answer_from_text(question, second.content)
                if tool_answer is not None:
                    answer = tool_answer
                audit.update(
                    {
                        "mode": "calculator",
                        "calculator_result": tool_result,
                        "second": _completion_audit(second),
                    }
                )
            else:
                audit["calculator_result"] = tool_result
                audit["calculator_ignored"] = "execution_failed"
        elif expression:
            if answer is not None:
                audit["calculator_ignored"] = "answer_already_available"
            else:
                audit["calculator_ignored"] = "expression_not_grounded"

        if answer is None:
            retry = self.client.complete(
                [
                    {
                        "role": "system",
                        "content": "Chọn đáp án đúng. Chỉ trả về JSON {\"answer\":\"A\"}.",
                    },
                    {"role": "user", "content": self._question_prompt(question)},
                ],
                max_tokens=512,
                enable_thinking=False,
            )
            answer = _answer_from_text(question, retry.content)
            audit["parse_retry"] = _completion_audit(retry)

        if answer is None:
            answer = question.labels[0]
            audit["parse_fallback"] = True
        return answer, audit
