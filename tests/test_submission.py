from __future__ import annotations

import json

import pytest

from submission.calculator import CalculatorError, SafeCalculator
from submission.client import Completion
from submission.schema import Question
from submission.solver import SingleModelAgent


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    def complete(self, *args, **kwargs) -> Completion:
        return Completion(
            content=next(self.responses),
            reasoning="",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=10,
        )


def test_calculator_allows_numeric_expressions() -> None:
    calculator = SafeCalculator()
    assert calculator.evaluate("comb(10, 2)") == "45"
    assert calculator.evaluate("mean([2, 4, 6])") == "4"


@pytest.mark.parametrize(
    "expression",
    ['open("/etc/passwd")', '__import__("os")', "2**10000", "(1).__class__"],
)
def test_calculator_blocks_unsafe_or_expensive_expressions(expression: str) -> None:
    with pytest.raises(CalculatorError):
        SafeCalculator().evaluate(expression)


def test_solver_uses_nested_tool_json() -> None:
    question = Question("q1", "Tính 10 chọn 2.", ("40", "45", "50", "55"))
    client = FakeClient(
        [
            json.dumps(
                {
                    "answer": None,
                    "confidence": 0.5,
                    "tool": {"expression": "comb(10, 2)", "purpose": "tính tổ hợp"},
                }
            ),
            '{"answer":"B","confidence":1.0,"tool":null}',
        ]
    )
    answer, audit = SingleModelAgent(client).solve(question)
    assert answer == "B"
    assert audit["mode"] == "calculator"
    assert audit["calculator_result"] == "45"
    assert audit["calculator_grounded"] is True


def test_solver_rejects_ungrounded_tool_expression() -> None:
    question = Question("q1", "Tính 10 chọn 2.", ("40", "45", "50", "55"))
    client = FakeClient(
        [
            '{"answer":null,"tool":{"expression":"(7+8)*9"}}',
            '{"answer":"B","tool":null}',
        ]
    )
    answer, audit = SingleModelAgent(client).solve(question)
    assert answer == "B"
    assert audit["calculator_grounded"] is False
    assert audit["calculator_ignored"] == "expression_not_grounded"
