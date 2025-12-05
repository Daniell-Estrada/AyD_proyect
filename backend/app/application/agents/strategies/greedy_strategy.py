"""
Greedy algorithm complexity analysis strategy.
Analyzes greedy choice patterns, sorting requirements, and approximation ratios.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.application.agents.strategies.base_strategy import ComplexityAnalysisStrategy
from app.domain.services.complexity_service import ComplexityAnalysisService

logger = logging.getLogger(__name__)


class GreedyAnalysisStrategy(ComplexityAnalysisStrategy):
    """
    Analyzes greedy algorithms by identifying sorting steps,
    greedy choice selection iterations, and approximation characteristics.
    """
    
    def __init__(
        self, 
        complexity_service: Optional[ComplexityAnalysisService] = None,
        llm_service=None,
        enable_llm_peer_review: bool = False,
    ):
        """
        Initialize greedy analysis strategy.
        """
        self._complexity_service = complexity_service or ComplexityAnalysisService()
        self._llm_service = llm_service
        self._enable_llm_peer_review = enable_llm_peer_review

    def analyze(
        self, ast_dict: Dict[str, Any], patterns: Dict[str, Any]
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, str]]:
        """
        Analyze greedy algorithm complexity.
        """
        logger.info("Analyzing greedy algorithm complexity")

        has_sorting, sorting_calls = self._detect_sorting(ast_dict, patterns)
        has_priority_queue = bool(patterns.get("has_priority_queue"))
        loop_depth = self._int_or_default(patterns.get("max_loop_depth"), 0)
        loop_count = self._int_or_default(patterns.get("loop_count"), 0)
        has_loops = bool(patterns.get("has_loops"))
        has_recursion = bool(patterns.get("has_recursion"))
        branching_factor = self._int_or_default(
            patterns.get("max_recursive_calls_per_function"), 0
        )

        structural_step = {
            "step": "Greedy Structural Scan",
            "technique": "greedy_analysis",
            "has_sorting": has_sorting,
            "sorting_calls": sorting_calls,
            "has_priority_queue": has_priority_queue,
            "loop_count": loop_count,
            "max_loop_depth": loop_depth,
            "has_recursion": has_recursion,
            "recursive_branching": branching_factor,
        }

        steps: List[Dict[str, Any]] = [structural_step]

        if has_sorting or has_priority_queue:
            worst_case = "O(n log n)"
            explanation = (
                "Sorting step dominates complexity"
                if has_sorting
                else "Priority queue operations introduce logarithmic factor"
            )
            complexities = {
                "worst_case": worst_case,
                "best_case": "Ω(n)",
                "average_case": worst_case,
            }

            structural_step["explanation"] = explanation
            if has_sorting:
                steps.append(
                    {
                        "step": "Sorting Phase",
                        "technique": "sorting_analysis",
                        "complexity": "O(n log n)",
                        "note": "Detected explicit sorting call",
                    }
                )
            else:
                steps.append(
                    {
                        "step": "Priority Queue Operations",
                        "technique": "priority_queue_analysis",
                        "complexity": "O(n log n)",
                        "note": "Priority queue maintenance incurs log factor",
                    }
                )

            return complexities, steps, {}

        structural_step["explanation"] = (
            "No sorting or priority queue detected; deriving complexity from loops"
            if has_loops or has_recursion
            else "No structural evidence of logarithmic behaviour"
        )

        if has_recursion and branching_factor > 1:
            neutral_complexities = {
                "worst_case": "analysis unavailable",
                "best_case": "analysis unavailable",
                "average_case": "analysis unavailable",
            }
            steps.append(
                {
                    "step": "Recursion Branching Guard",
                    "technique": "heuristic_safeguard",
                    "branching_factor": branching_factor,
                    "note": (
                        "Multiple recursive branches detected; insufficient data to "
                        "derive safe polynomial bounds without sorting evidence"
                    ),
                }
            )
            return neutral_complexities, steps, {}

        loop_term = self._loop_term(loop_depth)
        terms: List[str] = []
        if loop_term != "1":
            terms.append(loop_term)
        if has_recursion:
            terms.append("n")

        combined = self._combine_terms(terms)
        worst_case = f"O({combined})" if combined != "1" else "O(1)"

        best_case = "Ω(n)" if (has_recursion or has_loops) else "Ω(1)"
        complexities = {
            "worst_case": worst_case,
            "best_case": best_case,
            "average_case": worst_case,
        }

        steps.append(
            {
                "step": "Loop Contribution",
                "technique": "loop_contribution",
                "max_nesting": loop_depth,
                "loop_count": loop_count,
                "result": f"O({loop_term})" if loop_term != "1" else "O(1)",
                "explanation": "Loop depth determines per-call work",
            }
        )

        if has_recursion:
            steps.append(
                {
                    "step": "Recursion Contribution",
                    "technique": "recursion_analysis",
                    "branching_factor": branching_factor or 1,
                    "result": "O(n)",
                    "note": "Linear recursion multiplies loop cost across frames",
                }
            )

        if not terms:
            steps.append(
                {
                    "step": "Greedy Selection Phase",
                    "technique": "iteration_analysis",
                    "complexity": "O(1)",
                    "note": "No loops or recursion detected",
                }
            )

        return complexities, steps, {}

    def _detect_sorting(
        self, ast_dict: Dict[str, Any], patterns: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        if "has_sorting" in patterns and "sorting_calls" in patterns:
            return bool(patterns.get("has_sorting")), list(patterns.get("sorting_calls", []))

        calls: List[str] = []

        def traverse(node: Any) -> None:
            if isinstance(node, dict):
                node_type = node.get("type")
                if node_type in {"CallStmt", "FuncCallExpr"}:
                    name = node.get("name")
                    if isinstance(name, str):
                        calls.append(name)
                for value in node.values():
                    traverse(value)
            elif isinstance(node, list):
                for item in node:
                    traverse(item)

        traverse(ast_dict)

        sorting_calls = [name for name in calls if "sort" in name.lower()]
        return bool(sorting_calls), sorting_calls

    def _loop_term(self, depth: int) -> str:
        if depth <= 0:
            return "1"
        if depth == 1:
            return "n"
        return f"n^{depth}"

    def _combine_terms(self, terms: List[str]) -> str:
        if not terms:
            return "1"

        exponent = 0
        extras: List[str] = []

        for term in terms:
            normalized = term.strip()
            if normalized in {"1", "O(1)"}:
                continue
            if normalized == "n":
                exponent += 1
                continue
            if normalized.startswith("n^"):
                try:
                    exponent += int(normalized[2:])
                    continue
                except ValueError:
                    pass
            extras.append(normalized)

        parts: List[str] = []
        if exponent > 0:
            parts.append("n" if exponent == 1 else f"n^{exponent}")
        parts.extend(extras)
        return " * ".join(parts) if parts else "1"

    def _int_or_default(self, value: Any, fallback: int) -> int:
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            return fallback
