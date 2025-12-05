"""
Dynamic Programming complexity analysis strategy.
Analyzes subproblem count, memoization patterns, and state space dimensions.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.application.agents.strategies.base_strategy import ComplexityAnalysisStrategy
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
        """
        Initialize dynamic programming analysis strategy.
        """
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

        state_dimensions = patterns.get("state_dimensions", patterns.get("memo_dimensions", 1))
        has_memoization = patterns.get("has_memoization", True)
        memo_table_size = patterns.get("memo_table_size")
        subproblem_time = patterns.get("subproblem_time", "O(1)")

        if memo_table_size:
            base_term = memo_table_size
        elif state_dimensions == 1:
            base_term = "n"
        elif state_dimensions == 2:
            base_term = "n^2"
        elif state_dimensions == 3:
            base_term = "n^3"
        else:
            base_term = f"n^{state_dimensions}"

        def _extract_exponent(term: str) -> Optional[float]:
            cleaned = term.replace("O(", "").replace(")", "")
            if cleaned in {"n", "n¹", "n^1"}:
                return 1.0
            if cleaned in {"n^2", "n²"}:
                return 2.0
            if cleaned.startswith("n^"):
                try:
                    return float(cleaned.split("^")[1])
                except ValueError:
                    return None
            return None

        base_exp = _extract_exponent(base_term)
        time_exp = _extract_exponent(subproblem_time)

        if base_exp is not None and time_exp is not None:
            combined_exp = base_exp + time_exp
            if combined_exp == 2:
                worst_case = "O(n^2)"
            elif combined_exp == 3:
                worst_case = "O(n^3)"
            else:
                worst_case = f"O(n^{combined_exp:.2f})"
        else:
            worst_case = f"O({base_term})"

        best_case = f"Ω({base_term})"
        average_case = f"Θ({base_term})"
        space_complexity = f"O({base_term})"

        complexities = {
            "worst_case": worst_case,
            "best_case": best_case,
            "average_case": average_case,
        }

        steps = [
            {
                "step": "State Space Analysis",
                "technique": "dp_state_space",
                "dimensions": state_dimensions,
                "state_dimensions": state_dimensions,
                "subproblems": base_term,
                "has_memoization": has_memoization,
                "explanation": (
                    f"DP table has {state_dimensions}D state space, "
                    f"resulting in {base_term} unique subproblems to solve and memoize."
                ),
            },
            {
                "step": "Time Complexity",
                "technique": "dp_time_analysis",
                "time_per_subproblem": subproblem_time,
                "total_time": worst_case,
                "explanation": (
                    f"Each of the {base_term} subproblems requires {subproblem_time} "
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
