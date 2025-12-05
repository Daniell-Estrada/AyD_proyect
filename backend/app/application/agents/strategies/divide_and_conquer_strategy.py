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
        per_level_work = self._format_theta_term(k)
        levels_description = f"log_{b}(n)"
        analysis_detail["work_per_level"] = per_level_work
        analysis_detail["levels"] = levels_description
        analysis_detail["per_level_reasoning"] = (
            f"{levels_description} levels × {per_level_work} per level ⇒ {complexity}"
        )

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
        complexities["tight_bounds"] = analysis_detail.get(
            "tight_bounds", complexity
        )

        master_step = {
            "step": "Master Theorem Application",
            "technique": "master_theorem",
            "recurrence": f"T(n) = {a}T(n/{b}) + O(n^{k})",
            "result": complexity,
            "details": dict(analysis_detail),
            "applies_to_cases": ["best_case", "average_case", "worst_case"],
        }

        recursion_tree_step = {
            "step": "Recursion Tree Analysis",
            "technique": "recursion_tree",
            "description": (
                f"Balanced recursion tree spans {levels_description} levels; each costs {per_level_work}, yielding {complexity}."
            ),
            "levels": levels_description,
            "work_per_level": per_level_work,
            "total_work": f"{levels_description} × {per_level_work}",
            "applies_to_cases": ["best_case", "average_case", "worst_case"],
        }

        steps = [master_step, recursion_tree_step]

        if self._detect_quicksort_pattern(ast_dict):
            complexities["best_case"] = "Ω(n log n)"
            complexities["average_case"] = "Θ(n log n)"
            complexities["worst_case"] = "O(n^2)"
            complexities["tight_bounds"] = "Θ(n log n)"

            balanced_detail = dict(analysis_detail)
            balanced_detail["applies_to"] = "balanced_partitions (best/avg)"
            balanced_detail["case_uniformity_reason"] = (
                "Balanced recurrence only covers inputs where pivots stay within a constant-factor of n/2."
            )
            balanced_detail["best_case_assumption"] = (
                "Best case assumes a deterministic pivot policy (e.g., median-of-medians or presorted input) that guarantees near-halves."
            )
            balanced_detail["average_case_assumption"] = (
                "Average case assumes random pivot selection so expected partitions remain balanced."
            )
            balanced_detail["tight_bound_statement"] = (
                "Balanced splits have log_2(n) levels and Θ(n) work per level, so Θ(n log n) is both an upper and lower bound for those cases."
            )

            master_step["step"] = "Balanced Partition Master Theorem (Best/Average)"
            master_step["details"] = balanced_detail
            master_step["applies_to_cases"] = ["best_case", "average_case"]
            master_step["note"] = (
                "Balanced Master Theorem case: applies when partitions stay within a constant factor of n/2."
            )
            recursion_tree_step["step"] = "Balanced Recursion Tree (Best/Average)"
            recursion_tree_step["description"] = (
                "Balanced recursion tree (best/avg case) with depth log_2(n) and Θ(n) work per level"
            )
            recursion_tree_step["applies_to_cases"] = ["best_case", "average_case"]

            steps.append(
                {
                    "step": "Balanced Case Tight Bound",
                    "technique": "bound_justification",
                    "description": (
                        "Balanced partitions incur Θ(n) work at each of log_2(n) levels, so Ω(n log n) ≤ T(n) ≤ O(n log n); thus Θ(n log n) is tight for best/avg."
                    ),
                    "applies_to_cases": ["best_case", "average_case"],
                    "tight_bound": "Θ(n log n)",
                }
            )

            steps.append(
                {
                    "step": "Pivot Imbalance Consideration",
                    "technique": "worst_case_adjustment",
                    "explanation": (
                        "Detected quicksort-style partitioning. Deterministic pivots "
                        "can degrade to O(n^2) when partitions are unbalanced, leading "
                        "to the recurrence T(n) = T(n-1) + Θ(n) handled in the next step."
                    ),
                }
            )
            steps.append(
                {
                    "step": "Worst-case Recurrence",
                    "technique": "recurrence_expansion",
                    "recurrence": "T(n) = T(n-1) + Θ(n)",
                    "result": "O(n^2)",
                    "details": {
                        "applies_to": "worst_case_unbalanced_partitions",
                        "depth": "n",
                        "levels": "n",
                        "work_per_level": "Θ(n)",
                        "total_work": "Θ(n^2)",
                    },
                }
            )
            steps.append(
                {
                    "step": "Worst-case Recursion Tree",
                    "technique": "recursion_tree",
                    "description": (
                        "Unbalanced partitions yield a linear-depth recursion tree: n levels × Θ(n) work ⇒ Θ(n^2)."
                    ),
                    "levels": "n",
                    "work_per_level": "Θ(n)",
                }
            )

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

    def _format_theta_term(self, exponent: Optional[float]) -> str:
        """Render Θ(n^k) style strings for per-level work descriptions."""

        if exponent is None:
            return "Θ(n)"

        try:
            value = float(exponent)
        except (TypeError, ValueError):
            return f"Θ(n^{exponent})"

        if abs(value) < 1e-9:
            return "Θ(1)"
        if abs(value - 1) < 1e-9:
            return "Θ(n)"

        if value.is_integer():
            exp_str = str(int(round(value)))
        else:
            exp_str = f"{value:.2f}".rstrip("0").rstrip(".")

        return f"Θ(n^{exp_str})"

    def _detect_quicksort_pattern(self, ast_dict: Dict[str, Any]) -> bool:
        program = ast_dict if isinstance(ast_dict, dict) else {}
        if program.get("type") != "Program":
            return False

        for stmt in program.get("statements", []):
            if stmt.get("type") != "SubroutineDef":
                continue
            name = (stmt.get("name") or "").lower()
            if "quick" not in name:
                continue

            body = stmt.get("body", [])
            if not body:
                continue

            has_partition = self._contains_call(body, "partition")
            recursive_calls = self._count_recursive_calls(body, stmt.get("name"))

            if has_partition and recursive_calls >= 2:
                return True

        return False

    def _contains_call(self, node: Any, target: str) -> bool:
        target_lower = target.lower()

        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type in {"CallStmt", "FuncCallExpr"}:
                name = (node.get("name") or "").lower()
                if target_lower in name:
                    return True
            for value in node.values():
                if isinstance(value, (dict, list)) and self._contains_call(
                    value, target
                ):
                    return True
        elif isinstance(node, list):
            for item in node:
                if self._contains_call(item, target):
                    return True

        return False

    def _count_recursive_calls(self, node: Any, func_name: Optional[str]) -> int:
        if not func_name:
            return 0

        count = 0
        target_lower = func_name.lower()

        def visit(current: Any) -> None:
            nonlocal count
            if isinstance(current, dict):
                node_type = current.get("type")
                if node_type in {"CallStmt", "FuncCallExpr"}:
                    name = (current.get("name") or "").lower()
                    if name == target_lower:
                        count += 1
                for value in current.values():
                    if isinstance(value, (dict, list)):
                        visit(value)
            elif isinstance(current, list):
                for item in current:
                    visit(item)

        visit(node)
        return count

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
