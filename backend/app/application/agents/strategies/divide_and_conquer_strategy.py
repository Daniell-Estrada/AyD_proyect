"""
Divide and Conquer complexity analysis strategy.
Uses Master Theorem and recursion tree generation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.application.agents.strategies.base_strategy import \
    ComplexityAnalysisStrategy
from app.domain.services.complexity_service import ComplexityAnalysisService

logger = logging.getLogger(__name__)


class DivideAndConquerAnalysisStrategy(ComplexityAnalysisStrategy):
    """
    Analyzes divide-and-conquer algorithms using Master Theorem.
    Generates recursion trees and applies recurrence relation solving.
    Supports optional LLM peer review for enhanced validation.
    """

    def __init__(
        self,
        complexity_service: Optional[ComplexityAnalysisService] = None,
        llm_service=None,
        enable_llm_peer_review: bool = False,
    ):
        """
        Initialize divide-and-conquer analysis strategy.
        """
        self._complexity_service = complexity_service or ComplexityAnalysisService()
        self._llm_service = llm_service
        self._enable_llm_peer_review = enable_llm_peer_review

    def analyze(
        self, ast_dict: Dict[str, Any], patterns: Dict[str, Any]
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, str]]:
        """
        Analyze divide-and-conquer algorithms using Master Theorem.

        """
        logger.info("Applying Master Theorem for divide-and-conquer analysis")

        a = patterns.get("recursive_calls_count", 2)
        b = patterns.get("division_factor", 2)
        k = patterns.get("combine_work_exponent", 1)

        complexity, analysis_detail = self._complexity_service.solve_recurrence(a, b, k)

        func_name = self._extract_function_name(ast_dict)
        _, mermaid_tree = self._complexity_service.build_recursion_tree(
            func_name=func_name or "T",
            initial_input="n",
            depth=4,
        )

        worst_case = analysis_detail.get("worst_case", complexity)
        best_case = analysis_detail.get("best_case", worst_case)
        complexities = {
            "worst_case": worst_case,
            "best_case": best_case,
            "average_case": worst_case,
        }

        steps = [
            {
                "step": "Master Theorem Application",
                "technique": "master_theorem",
                "recurrence": f"T(n) = {a}T(n/{b}) + O(n^{k})",
                "result": complexity,
                "details": analysis_detail,
            },
            {
                "step": "Recursion Tree Analysis",
                "technique": "recursion_tree",
                "description": f"Generated recursion tree with depth log_{b}(n)",
                "levels": f"log_{b}(n)",
                "work_per_level": f"n^{k}",
            },
        ]

        diagrams = {
            "recursion_tree": mermaid_tree,
        }

        if self._enable_llm_peer_review and self._llm_service:
            peer_review_step = self._get_llm_peer_review(
                complexities=complexities,
                steps=steps,
                paradigm="divide_and_conquer",
            )
            if peer_review_step:
                steps.append(peer_review_step)

        return complexities, steps, diagrams

    def _get_llm_peer_review(
        self,
        complexities: Dict[str, str],
        steps: List[Dict[str, Any]],
        paradigm: str,
    ) -> Dict[str, Any]:
        """
        Get LLM peer review of deterministic analysis results.
        """
        import json

        system_prompt = """You are a complexity analysis peer reviewer.
        You will receive DETERMINISTIC ANALYSIS RESULTS that are mathematically correct.
        Your role: assess reasoning clarity and completeness, not recalculate.
        
        Provide brief peer review in JSON:
        {"reasoning_clear": true/false, "suggestions": ["..."]}"""

        user_prompt = f"""Review this deterministic {paradigm} analysis:

        **Complexities:** {json.dumps(complexities)}
        **Steps:** {json.dumps(steps)}

        Is the reasoning clear? Suggest improvements."""

        try:
            response, _ = self._llm_service.invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            review = json.loads(json_str)

            return {
                "step": "LLM Peer Review",
                "technique": "llm_peer_review",
                "reasoning_clear": review.get("reasoning_clear", True),
                "suggestions": review.get("suggestions", []),
            }
        except Exception as e:
            logger.warning(f"LLM peer review failed: {e}")
            return None

    def _extract_function_name(self, ast_dict: Dict[str, Any]) -> Optional[str]:
        """
        Extract the main function name from AST.
        """
        if ast_dict.get("type") == "Program":
            for stmt in ast_dict.get("statements", []):
                if stmt.get("type") == "SubroutineDef":
                    return stmt.get("name", "T")
        return None
