"""
Backtracking Analysis Strategy Module.
Estimates exponential search space complexity with pruning considerations.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.application.agents.strategies.base_strategy import \
    ComplexityAnalysisStrategy
from app.domain.services.complexity_service import ComplexityAnalysisService

logger = logging.getLogger(__name__)


class BacktrackingAnalysisStrategy(ComplexityAnalysisStrategy):
    """
    Strategy for analyzing backtracking algorithm complexity.
    """

    def __init__(
        self,
        complexity_service: Optional[ComplexityAnalysisService] = None,
        llm_service=None,
        enable_llm_peer_review: bool = False,
    ):
        """
        Initialize backtracking analysis strategy.
        """

        self._complexity_service = complexity_service or ComplexityAnalysisService()
        self._llm_service = llm_service
        self._enable_llm_peer_review = enable_llm_peer_review

    def analyze(
        self, ast_dict: Dict[str, Any], patterns: Dict[str, Any]
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, str]]:
        """
        Analyze backtracking algorithm complexity.
       """

        branching_factor = patterns.get("branching_factor", 2)
        max_depth = patterns.get("search_depth", 10)
        has_pruning = patterns.get("has_pruning", False)

        worst_case, analysis = (
            self._complexity_service.estimate_backtracking_complexity(
                branching_factor=branching_factor,
                max_depth=max_depth,
                pruning_enabled=has_pruning,
            )
        )

        complexities = {
            "worst_case": analysis["worst_case"],
            "best_case": analysis["best_case"],
            "average_case": analysis["average_case"],
        }

        steps = [
            {
                "step": "Search Space Analysis",
                "technique": "backtracking_analysis",
                "branching_factor": branching_factor,
                "max_depth": max_depth,
                "search_space_size": worst_case,
                "explanation": analysis["reasoning"],
            },
            {
                "step": "Space Complexity Analysis",
                "technique": "space_analysis",
                "space_complexity": analysis["space_complexity"],
                "explanation": (
                    f"Recursion stack depth is O({max_depth}) for backtracking. "
                    f"Additional space for state depends on problem representation."
                ),
            },
        ]

        if has_pruning:
            steps.append(
                {
                    "step": "Pruning Optimization (Branch & Bound)",
                    "technique": "optimization_analysis",
                    "pruning_effect": (
                        "Pruning reduces average-case complexity by eliminating "
                        "unpromising branches early. Worst-case remains exponential."
                    ),
                    "average_improvement": (
                        f"With effective pruning, average case may be {analysis['average_case']} "
                        f"vs worst {analysis['worst_case']}"
                    ),
                }
            )

        steps.append(
            {
                "step": "Complexity Characterization",
                "technique": "complexity_characterization",
                "expression": f"O({branching_factor}^{max_depth})",
                "note": "Exponential search space grows with branching factor raised to depth.",
            }
        )

        return complexities, steps, {}
