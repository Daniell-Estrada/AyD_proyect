"""
Strategy for analyzing branching recursion patterns in code to estimate time complexity.
"""


import logging
from statistics import mean
from typing import Any, Dict, List, Tuple

from app.application.agents.strategies.base_strategy import (
    ComplexityAnalysisStrategy,
)
from app.domain.services.complexity_service import ComplexityAnalysisService

logger = logging.getLogger(__name__)


class BranchingRecursionAnalysisStrategy(ComplexityAnalysisStrategy):
    """Estimate exponential growth for branching constant-decrement recurrences."""

    def __init__(
        self,
        complexity_service: ComplexityAnalysisService | None = None,
        llm_service: Any = None,
        enable_llm_peer_review: bool = False,
    ) -> None:
        del llm_service  
        del enable_llm_peer_review
        self._complexity_service = complexity_service or ComplexityAnalysisService()

    def analyze(
        self, ast_dict: Dict[str, Any], patterns: Dict[str, Any]
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, str]]:
        """Simulate branching recursion to approximate its growth factor."""

        branching_factor = max(
            2,
            int(patterns.get("estimated_branching_factor") or 2),
        )
        decrements = self._resolve_decrements(patterns)

        logger.info(
            "Branching recursion detected (factor=%s, decrements=%s)",
            branching_factor,
            decrements,
        )

        growth_factor = self._estimate_growth_factor(branching_factor, decrements)
        growth_label = self._format_growth_label(growth_factor)

        worst_case = f"O({growth_label}^n)"
        complexities = {
            "worst_case": worst_case,
            "best_case": worst_case,
            "average_case": worst_case,
        }

        func_name = self._extract_function_name(ast_dict) or "T"
        _, recursion_tree = self._complexity_service.build_recursion_tree(
            func_name=func_name,
            initial_input="n",
            branching_factor=branching_factor,
            depth=min(6, 2 + len(decrements)),
            size_reduction=f"n-{decrements[0]}",
        )

        steps: List[Dict[str, Any]] = [
            {
                "step": "Branching Recursion Detection",
                "technique": "branching_recursion",
                "branching_factor": branching_factor,
                "decrements": decrements,
                "growth_label": growth_label,
                "message": (
                    "Multiple same-function calls subtract constant offsets; "
                    "treating recurrence as exponential search tree."
                ),
            }
        ]

        simulation = self._simulate_branching_sequence(decrements, limit=12)
        steps.append(
            {
                "step": "Memoized Simulation",
                "technique": "recurrence_simulation",
                "sequence": simulation,
                "estimated_growth_factor": growth_factor,
                "reasoning": (
                    "Successive value ratios converge to the dominant root, which "
                    "we report as the exponential base."
                ),
            }
        )

        diagrams = {"recursion_tree": recursion_tree}

        return complexities, steps, diagrams

    @staticmethod
    def _resolve_decrements(patterns: Dict[str, Any]) -> List[int]:
        """Return a sanitized, non-empty list of constant decrements."""

        decrements = patterns.get("recursive_constant_decrements") or [1]
        cleaned = sorted({max(1, int(value)) for value in decrements if value})
        return cleaned or [1]

    def _estimate_growth_factor(
        self, branching_factor: int, decrements: List[int]
    ) -> float:
        """Approximate the exponential base using simulated recurrence ratios."""

        sequence = self._simulate_branching_sequence(decrements, limit=16)
        ratios = [
            sequence[idx + 1] / max(sequence[idx], 1)
            for idx in range(len(sequence) - 1)
        ]

        if ratios:
            tail = ratios[-3:]
            avg_ratio = mean(tail)
            return max(avg_ratio, 1.1)

        return float(branching_factor)

    def _simulate_branching_sequence(self, decrements: List[int], limit: int) -> List[int]:
        """Generate a prefix of the recurrence assuming unit work per call."""

        memo: Dict[int, int] = {}
        max_decrement = max(decrements)
        sequence: List[int] = []

        for n in range(max_decrement + limit):
            if n < max_decrement:
                memo[n] = 1
            else:
                memo[n] = sum(memo.get(n - dec, 1) for dec in decrements)

            if n >= max_decrement:
                sequence.append(memo[n])

        return sequence

    @staticmethod
    def _format_growth_label(value: float) -> str:
        """Pad or trim the growth base so it is readable in Big-O notation."""

        if value <= 1.5:
            return "2"

        rounded = f"{value:.2f}".rstrip("0").rstrip(".")
        return rounded or "2"

    @staticmethod
    def _extract_function_name(ast_dict: Dict[str, Any]) -> str | None:
        """Reuse ParserAgent metadata to keep diagrams labeled."""

        if ast_dict.get("type") != "Program":
            return None

        for stmt in ast_dict.get("statements", []):
            if stmt.get("type") == "SubroutineDef":
                return stmt.get("name") or "T"

        return None
