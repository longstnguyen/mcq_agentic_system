from __future__ import annotations

import ast
import math
import operator
import statistics
from collections.abc import Callable
from typing import Any


class CalculatorError(ValueError):
    pass


def _bounded_factorial(value: float) -> int:
    integer = int(value)
    if value != integer or not 0 <= integer <= 500:
        raise CalculatorError("factorial requires an integer between 0 and 500")
    return math.factorial(integer)


def _bounded_comb(n: float, k: float) -> int:
    n_int, k_int = int(n), int(k)
    if n != n_int or k != k_int or not 0 <= k_int <= n_int <= 10000:
        raise CalculatorError("comb requires 0 <= k <= n <= 10000")
    return math.comb(n_int, k_int)


def _bounded_perm(n: float, k: float | None = None) -> int:
    n_int = int(n)
    k_int = n_int if k is None else int(k)
    if n != n_int or (k is not None and k != k_int) or not 0 <= k_int <= n_int <= 10000:
        raise CalculatorError("perm requires 0 <= k <= n <= 10000")
    return math.perm(n_int, k_int)


FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "degrees": math.degrees,
    "radians": math.radians,
    "factorial": _bounded_factorial,
    "comb": _bounded_comb,
    "perm": _bounded_perm,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "mean": statistics.mean,
    "median": statistics.median,
}
CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class SafeCalculator:
    def __init__(self, *, max_expression_chars: int = 500, max_nodes: int = 100) -> None:
        self.max_expression_chars = max_expression_chars
        self.max_nodes = max_nodes

    def evaluate(self, expression: str) -> str:
        expression = str(expression).strip()
        if not expression or len(expression) > self.max_expression_chars:
            raise CalculatorError("calculator expression is empty or too long")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise CalculatorError(f"invalid expression: {exc.msg}") from exc
        if sum(1 for _ in ast.walk(tree)) > self.max_nodes:
            raise CalculatorError("calculator expression is too complex")
        result = self._visit(tree.body)
        text = repr(result)
        if len(text) > 2000:
            raise CalculatorError("calculator result is too large")
        return text

    def _visit(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id in CONSTANTS:
            return CONSTANTS[node.id]
        if isinstance(node, ast.List | ast.Tuple):
            return [self._visit(item) for item in node.elts]
        if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
            return UNARY_OPERATORS[type(node.op)](self._visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
            left = self._visit(node.left)
            right = self._visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(float(right)) > 1000:
                raise CalculatorError("exponent is too large")
            return BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = FUNCTIONS.get(node.func.id)
            if function is None or node.keywords:
                raise CalculatorError(f"function {node.func.id!r} is not allowed")
            return function(*(self._visit(argument) for argument in node.args))
        raise CalculatorError(f"syntax {type(node).__name__} is not allowed")

