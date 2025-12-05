"""
Dynamic Programming complexity analysis strategy.
Analyzes subproblem count, memoization patterns, and state space dimensions.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.application.agents.strategies.base_strategy import \
    ComplexityAnalysisStrategy
from app.domain.services.complexity_service import ComplexityAnalysisService

logger = logging.getLogger(__name__)


class DynamicProgrammingAnalysisStrategy(ComplexityAnalysisStrategy):
    """
    Analyzes dynamic programming algorithms by counting subproblems
    and determining time per subproblem computation.
    """

    def __init__(
        self,
        complexity_service: Optional[ComplexityAnalysisService] = None,
        llm_service=None,
        enable_llm_peer_review: bool = False,
    ):
        self._complexity_service = complexity_service or ComplexityAnalysisService()
        self._llm_service = llm_service
        self._enable_llm_peer_review = enable_llm_peer_review

    def analyze(
        self, ast_dict: Dict[str, Any], patterns: Dict[str, Any]
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, str]]:
        """
        Analyze dynamic programming complexity.
        """
        logger.info("Analyzing dynamic programming algorithm")

        raw_state_dimensions = patterns.get("state_dimensions")
        if raw_state_dimensions in (None, 0):
            raw_state_dimensions = patterns.get("memo_dimensions")
        state_dimensions = self._int_or_default(raw_state_dimensions, 1)
        state_dimensions = max(1, state_dimensions)
        has_memoization = patterns.get("has_memoization", True)
        memo_table_size = patterns.get("memo_table_size")
        state_labels = patterns.get("state_labels") or []
        memo_variables = patterns.get("memo_variables") or []
        subproblem_time = patterns.get("subproblem_time", "O(1)")
        loop_depth = self._int_or_default(patterns.get("max_loop_depth"), 0)
        loop_depth_hint = loop_depth if loop_depth and loop_depth >= 2 else 0

        if memo_table_size:
            time_term = memo_table_size
        elif state_labels:
            time_term = self._format_state_volume(state_labels)
        else:
            time_term = self._dimension_to_term(state_dimensions)

        state_space_term = time_term

        def _extract_exponent(term: str) -> Optional[float]:
            cleaned = term.replace("O(", "").replace("Θ(", "").replace(")", "")
            cleaned = cleaned.strip()
            if not cleaned:
                return None

            if "*" in cleaned or " " in cleaned:
                return None

            if "^" in cleaned:
                base, exponent = cleaned.split("^", 1)
                try:
                    return float(exponent.replace("²", "2"))
                except ValueError:
                    return None

            if cleaned.lower() in {"n", "m", "k"} or cleaned.isalpha():
                return 1.0

            return None

        base_exp = _extract_exponent(time_term)
        time_exp = _extract_exponent(subproblem_time)
        target_dimension = max(state_dimensions, loop_depth_hint) if loop_depth_hint else state_dimensions
        if (
            target_dimension
            and base_exp is not None
            and target_dimension > base_exp
        ):
            time_term = self._dimension_to_term(int(target_dimension))
            base_exp = float(target_dimension)

        if base_exp is not None and time_exp is not None:
            combined_exp = base_exp + time_exp
            if combined_exp == 2:
                worst_case = "O(n^2)"
            elif combined_exp == 3:
                worst_case = "O(n^3)"
            else:
                worst_case = f"O(n^{combined_exp:.2f})"
        else:
            worst_case = f"O({time_term})"

        best_case = f"Ω({time_term})"
        average_case = f"Θ({time_term})"
        space_complexity = f"O({state_space_term})"

        complexities = {
            "worst_case": worst_case,
            "best_case": best_case,
            "average_case": average_case,
        }

        loop_explanation = ""
        if loop_depth_hint and loop_depth_hint > state_dimensions:
            loop_explanation = (
                f" Nested loops of depth {loop_depth_hint} iterate across intermediate states, "
                f"so runtime covers n^{loop_depth_hint} combinations even though the memo table captures "
                f"{state_dimensions} dimension(s)."
            )

        steps = [
            {
                "step": "State Space Analysis",
                "technique": "dp_state_space",
                "dimensions": state_dimensions,
                "state_dimensions": state_dimensions,
                "subproblems": state_space_term,
                "has_memoization": has_memoization,
                "memo_structures": memo_variables,
                "state_labels": state_labels,
                "explanation": (
                    f"DP table has {state_dimensions}D state space, "
                    f"resulting in {state_space_term} unique subproblems to solve and memoize."
                    f"{loop_explanation}"
                ),
            },
            {
                "step": "Time Complexity",
                "technique": "dp_time_analysis",
                "time_per_subproblem": subproblem_time,
                "total_time": worst_case,
                "explanation": (
                    f"Each of the {time_term} aggregated transitions requires {subproblem_time} "
                    f"work; combined complexity is {worst_case}."
                ),
            },
            {
                "step": "Space Complexity",
                "technique": "dp_space_analysis",
                "space_complexity": space_complexity,
                "space_optimization_note": (
                    "Space can sometimes be optimized (e.g., 2D DP to 1D) "
                    "if only previous row/column is needed."
                ),
            },
            {
                "step": "Bottom-Up vs Top-Down",
                "technique": "dp_approach",
                "note": (
                    "Bottom-up (tabulation) and top-down (memoization) have same "
                    "time complexity but differ in space constants and call overhead."
                ),
            },
        ]

        return complexities, steps, {}

    def _format_state_volume(self, labels: List[str]) -> str:
        normalized = [label or "n" for label in labels]
        if not normalized:
            return "n"

        if len(set(normalized)) == 1:
            base = normalized[0]
            if len(normalized) == 1:
                return base
            return f"{base}^{len(normalized)}"

        return " * ".join(normalized)

    def _dimension_to_term(self, dimension: int) -> str:
        if dimension <= 1:
            return "n"
        return f"n^{dimension}"

    def _int_or_default(self, value: Any, fallback: int) -> int:
        try:
            parsed = int(value)
            return max(parsed, 0)
        except (TypeError, ValueError):
            return fallback
