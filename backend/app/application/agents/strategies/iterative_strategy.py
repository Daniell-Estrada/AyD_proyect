"""
Iterative algorithm complexity analysis strategy.
Analyzes loop nesting depth, iteration counts, and sequential compositions.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.application.agents.strategies.base_strategy import \
    ComplexityAnalysisStrategy
from app.domain.services.complexity_service import ComplexityAnalysisService

logger = logging.getLogger(__name__)


class IterativeAnalysisStrategy(ComplexityAnalysisStrategy):
    """
    Analyzes iterative algorithms by examining loop nesting depth,
    iteration patterns, and sequential loop compositions.
    """

    def __init__(
        self,
        complexity_service: Optional[ComplexityAnalysisService] = None,
        llm_service=None,
        enable_llm_peer_review: bool = False,
    ):
        """
        Initialize iterative analysis strategy.
        """
        self._complexity_service = complexity_service or ComplexityAnalysisService()
        self._llm_service = llm_service
        self._enable_llm_peer_review = enable_llm_peer_review

    def analyze(
        self, ast_dict: Dict[str, Any], patterns: Dict[str, Any]
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, str]]:
        """
        Analyze iterative algorithm complexity.
        """
        logger.info("Analyzing iterative loop complexity")

        loop_insights = self._analyze_loops(ast_dict)
        has_geometric_loop = loop_insights["has_geometric"]
        geometric_multiplier = loop_insights.get("geometric_multiplier") or 0

        def _as_non_negative_int(value: Any, fallback: int) -> int:
            try:
                return max(int(value), 0)
            except (TypeError, ValueError):
                return fallback

        has_loops = bool(patterns.get("has_loops"))
        max_depth_raw = patterns.get("max_loop_depth")
        if max_depth_raw is None:
            max_depth_raw = 1 if has_loops else 0
        max_depth = _as_non_negative_int(max_depth_raw, 0)

        loop_count_raw = patterns.get("loop_count")
        if loop_count_raw is None:
            loop_count_raw = 1 if has_loops else 0
        loop_count = _as_non_negative_int(loop_count_raw, 0)

        has_sequential_loops = bool(patterns.get("has_sequential_loops"))

        if not has_loops or loop_count == 0 or max_depth == 0:
            complexity = "O(1)"
            explanation = "No loops detected, constant time complexity"
            step_name = "Constant-Time Check"
        elif has_geometric_loop:
            polynomial_power = max(max_depth - 1, 0)
            if polynomial_power == 0:
                complexity = "O(log n)"
            elif polynomial_power == 1:
                complexity = "O(n log n)"
            else:
                complexity = f"O(n^{polynomial_power} log n)"
            power_phrase = "1" if polynomial_power == 0 else f"n^{polynomial_power}"
            explanation = (
                "Geometric loop growth (e.g., size doubling) introduces a log n factor; "
                f"remaining nesting contributes {power_phrase}"
            )
            step_name = "Geometric Loop Analysis"
        elif max_depth == 1:
            complexity = "O(n)"
            explanation = "Single loop iterating over input"
            step_name = "Loop Nesting Analysis"
        else:
            complexity = f"O(n^{max_depth})"
            explanation = f"Nested loops with maximum depth {max_depth}"
            step_name = "Loop Nesting Analysis"

        worst_case = complexity
        average_case = complexity
        best_case = complexity

        if has_loops and patterns.get("has_repeat_until"):
            best_case = "O(n)" if max_depth >= 1 else best_case

        complexities = {
            "worst_case": worst_case,
            "best_case": best_case,
            "average_case": average_case,
        }

        steps = [
            {
                "step": "Loop Nesting Analysis",
                "technique": "loop_analysis",
                "max_nesting": max_depth,
                "loop_count": loop_count,
                "has_sequential": has_sequential_loops,
                "geometric_growth": has_geometric_loop,
                "geometric_multiplier": geometric_multiplier,
                "result": complexity,
                "explanation": explanation,
            },
        ]

        if has_geometric_loop:
            steps.append(
                {
                    "step": step_name,
                    "technique": "geometric_loop_analysis",
                    "result": complexity,
                    "multiplier": geometric_multiplier,
                    "explanation": (
                        "Detected loop variable updated multiplicatively "
                        f"(x{geometric_multiplier}), yielding log n iterations for that loop."
                    ),
                    "geometric_growth": True,
                }
            )

        if has_loops and patterns.get("has_repeat_until"):
            steps[0][
                "best_case_hint"
            ] = "repeat-until can stop after first pass when no swaps occur (O(n) best case)"

        if has_sequential_loops and loop_count > max_depth:
            steps.append(
                {
                    "step": "Sequential Loop Detection",
                    "technique": "loop_analysis",
                    "note": (
                        f"Detected {loop_count} total loops, some sequential. "
                        "Sequential loops add, while nested loops multiply complexity."
                    ),
                }
            )

        return complexities, steps, {}

    def _analyze_loops(self, node: Any) -> Dict[str, Any]:
        """Detect geometric-growth loops from the AST dictionary."""

        insights = {"has_geometric": False, "geometric_multiplier": None}

        def visit(obj: Any):
            if not isinstance(obj, dict):
                return

            node_type = obj.get("type")

            if node_type == "WhileLoop":
                cond_var = self._extract_cond_var(obj.get("cond"))
                multiplier = self._extract_multiplier_from_body(
                    obj.get("body", []), cond_var
                )
                if multiplier and multiplier > 1:
                    insights["has_geometric"] = True
                    insights["geometric_multiplier"] = multiplier

            for value in obj.values():
                if isinstance(value, list):
                    for item in value:
                        visit(item)
                elif isinstance(value, dict):
                    visit(value)

        visit(node)
        return insights

    def _extract_cond_var(self, cond: Any) -> Optional[str]:
        """Extract loop control variable name from a simple binary condition."""
        if not isinstance(cond, dict):
            return None
        if cond.get("type") == "BinOp":
            left = cond.get("left")
            right = cond.get("right")
            if isinstance(left, dict) and left.get("type") == "Var":
                return left.get("name")
            if isinstance(right, dict) and right.get("type") == "Var":
                return right.get("name")
        return None

    def _extract_multiplier_from_body(
        self, body: List[Any], var_name: Optional[str]
    ) -> Optional[float]:
        if not var_name:
            return None
        for stmt in body:
            if not isinstance(stmt, dict):
                continue
            if stmt.get("type") != "Assignment":
                continue
            target = stmt.get("target", {})
            if (
                isinstance(target, dict)
                and target.get("type") == "VarTarget"
                and target.get("name") == var_name
            ):
                multiplier = self._extract_multiplier_expr(stmt.get("value"), var_name)
                if multiplier:
                    return multiplier
        return None

    def _is_var(self, node: Any, var_name: str) -> bool:
        return (
            isinstance(node, dict)
            and node.get("type") in {"Var", "VarTarget"}
            and node.get("name") == var_name
        )

    def _is_number(self, node: Any) -> bool:
        return (
            isinstance(node, dict)
            and node.get("type") == "Number"
            and isinstance(node.get("value"), (int, float))
        )

    def _is_mul_of_var(self, node: Any, var_name: str) -> Optional[float]:
        if (
            not isinstance(node, dict)
            or node.get("type") != "BinOp"
            or node.get("op") != "*"
        ):
            return None
        left, right = node.get("left"), node.get("right")
        if self._is_var(left, var_name) and self._is_number(right):
            return float(right.get("value"))
        if self._is_var(right, var_name) and self._is_number(left):
            return float(left.get("value"))
        return None

    def _extract_multiplier_expr(self, expr: Any, var_name: str) -> Optional[float]:
        if not isinstance(expr, dict):
            return None

        if expr.get("type") != "BinOp":
            return None

        op = expr.get("op")
        left = expr.get("left")
        right = expr.get("right")

        if op == "*":
            return self._is_mul_of_var(expr, var_name)

        if op == "+":
            if self._is_var(left, var_name) and self._is_var(right, var_name):
                return 2.0
            if self._is_var(left, var_name):
                mul = self._is_mul_of_var(right, var_name)
                if mul:
                    return 1.0 + mul
            if self._is_var(right, var_name):
                mul = self._is_mul_of_var(left, var_name)
                if mul:
                    return 1.0 + mul

        return None
