"""
Complexity Analysis Service implementing various algorithmic techniques.
Includes Master Theorem, Recurrence Relations, Recursion Trees, and pattern detection.
Uses pattern detectors following Single Responsibility Principle.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import sympy as sp
from sympy import symbols

from app.domain.models.ast import Program
from app.domain.services.complexity.recurrence_solver import RecurrenceSolver
from app.domain.services.complexity.recursion_tree_builder import \
    RecursionTreeBuilder
from app.domain.services.pattern_detectors.backtracking_detector import \
    BacktrackingPatternDetector
from app.domain.services.pattern_detectors.divide_conquer_detector import \
    DivideAndConquerPatternDetector
from app.domain.services.pattern_detectors.greedy_detector import \
    GreedyPatternDetector
from app.domain.services.pattern_detectors.loop_detector import \
    LoopPatternDetector
from app.domain.services.pattern_detectors.memoization_detector import \
    MemoizationPatternDetector
from app.domain.services.pattern_detectors.recursion_detector import \
    RecursionPatternDetector

logger = logging.getLogger(__name__)


class ComplexityAnalysisService:
    """
    Service for analyzing algorithmic complexity using various techniques.
    """

    def __init__(self):
        self.n = symbols("n", positive=True, integer=True)
        self._recurrence_solver = RecurrenceSolver()
        self._recursion_tree_builder = RecursionTreeBuilder()
        self._recursion_detector = RecursionPatternDetector()
        self._divide_conquer_detector = DivideAndConquerPatternDetector()
        self._memoization_detector = MemoizationPatternDetector()
        self._greedy_detector = GreedyPatternDetector()
        self._backtracking_detector = BacktrackingPatternDetector()
        self._loop_detector = LoopPatternDetector()

    def parse_complexity(self, expr: str) -> sp.Expr:
        """Parse informal complexity strings like 'n log n' or 'n^2' into SymPy expressions."""

        if not expr:
            return sp.Integer(0)

        sanitized = (
            expr.replace("O(", "").replace("Θ(", "").replace("Ω(", "").replace(")", "")
        )
        sanitized = sanitized.replace("log n", "log(n)").replace("logn", "log(n)")
        sanitized = sanitized.replace("n log n", "n*log(n)").replace(
            "nlogn", "n*log(n)"
        )
        sanitized = sanitized.replace("·", "*").replace("^", "**")
        sanitized = sanitized.strip()

        try:
            return sp.sympify(sanitized, locals={"n": self.n, "log": sp.log})
        except Exception:
            return sp.sympify(0)

    def compare_complexities(self, expected: str, observed: str) -> bool:
        """Check if two complexity classes grow at the same order using SymPy leading terms."""

        exp_expr = self.parse_complexity(expected)
        obs_expr = self.parse_complexity(observed)

        if exp_expr == 0 or obs_expr == 0:
            return expected == observed

        ratio = sp.simplify(exp_expr / obs_expr)
        try:
            limit_val = sp.limit(ratio, self.n, sp.oo)
            if limit_val in (sp.oo, 0):
                return False
            return bool(sp.simplify(limit_val) != 0)
        except Exception:
            return expected == observed

    def detect_paradigm(self, ast: Program) -> str:
        """
        Detect algorithmic paradigm from AST using specialized detectors.
        """
        recursion_info = self._recursion_detector.detect(ast)
        has_recursion = recursion_info["has_recursion"]

        divide_conquer_info = self._divide_conquer_detector.detect(ast)

        memoization_info = self._memoization_detector.detect(ast)
        has_memoization = memoization_info["has_memoization"]

        greedy_info = self._greedy_detector.detect(ast)
        has_greedy_pattern = greedy_info["has_greedy_pattern"]

        backtracking_info = self._backtracking_detector.detect(ast)
        has_backtracking = backtracking_info["has_backtracking"]

        loop_info = self._loop_detector.detect(ast)
        has_loops = loop_info["loop_count"] > 0
        has_nested_loops = loop_info["has_nested_loops"]

        if has_recursion:
            if divide_conquer_info["is_divide_and_conquer"]:
                return "divide_and_conquer"
            if has_memoization:
                return "dynamic_programming"
            if has_backtracking:
                return "backtracking"
            return "recursion"

        if has_memoization:
            return "dynamic_programming"

        if has_greedy_pattern:
            return "greedy"

        if has_backtracking:
            return "backtracking"

        if has_nested_loops or has_loops:
            return "iterative"

        return "simple"

    def detect_paradigm_from_patterns(self, patterns: Dict[str, Any]) -> str:
        """
        Detect paradigm from pre-extracted pattern dictionary.
        """
        has_recursion = patterns.get("has_recursion", False)
        has_memoization = patterns.get("has_memoization", False)
        has_greedy = patterns.get("has_greedy_pattern", False)
        has_backtracking = patterns.get("has_backtracking", False)
        has_loops = patterns.get("has_loops", False)
        has_nested = patterns.get("has_nested_loops", False)

        if has_recursion:
            if self._is_branching_recursion_pattern(patterns):
                return "branching_recursion"

            has_divide_conquer = self._is_divide_and_conquer_pattern(patterns)
            if has_divide_conquer:
                return "divide_and_conquer"
            if has_memoization:
                return "dynamic_programming"
            if has_backtracking:
                return "backtracking"
            return "recursion"
        elif has_greedy:
            return "greedy"
        elif has_nested:
            return "iterative"
        elif has_loops:
            return "iterative"
        else:
            return "simple"

    @staticmethod
    def _is_branching_recursion_pattern(patterns: Dict[str, Any]) -> bool:
        """Detect Fibonacci-style constant-decrement branching recursion."""

        if not patterns.get("has_recursion"):
            return False

        branching_factor = patterns.get("estimated_branching_factor") or patterns.get(
            "max_recursive_calls_per_function", 0
        )
        if branching_factor < 2:
            return False

        constant_flag = patterns.get("has_constant_decrement_recursion")
        non_linear = patterns.get("non_linear_split_recursion")
        fractional_flag = patterns.get("has_fractional_split_recursion")

        return bool((constant_flag or non_linear) and not fractional_flag)

    @staticmethod
    def _is_divide_and_conquer_pattern(patterns: Dict[str, Any]) -> bool:
        """True when recursion divides input size instead of subtracting constants."""

        fractional_flag = patterns.get("has_fractional_split_recursion")
        fractional_factors = patterns.get("fractional_split_factors") or []
        division_factor = patterns.get("division_factor")
        detector_flag = patterns.get("is_divide_and_conquer")

        return bool(
            detector_flag
            or fractional_flag
            or division_factor not in (None, 0, 1)
            or fractional_factors
        )

    def analyze_loops(self, node: Any, depth: int = 0) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze iterative complexity from loops using LoopPatternDetector.
        """
        loop_info = self._loop_detector.detect(node)

        max_depth = loop_info["max_loop_depth"]
        loop_count = loop_info["loop_count"]
        sequential_loops = max(loop_count - max_depth, 0)

        if max_depth == 0:
            complexity = "O(1)"
            reasoning = "No loops detected"
        elif max_depth == 1 and not loop_info.get("has_nested_loops"):
            factor = sequential_loops + 1
            complexity = "O(n)" if factor == 1 else f"O({factor}·n)"
            reasoning = f"{factor} single-level loop(s) contribute linearly"
        else:
            complexity = f"O(n^{max_depth})"
            reasoning = f"Nested depth {max_depth} dominates multiplicative cost"

        analysis = {
            "technique": "loop_analysis",
            "max_nesting": max_depth,
            "loop_count": loop_count,
            "sequential_loops": sequential_loops,
            "loops": loop_info,
            "reasoning": reasoning,
        }

        return complexity, analysis

    def solve_recurrence(self, a: int, b: int, k: int) -> Tuple[str, Dict[str, Any]]:
        return self._recurrence_solver.solve_recurrence(a, b, k)

    def solve_recurrence_relation(
        self, relation: str
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        return self._recurrence_solver.solve_recurrence_relation(relation)

    def _extract_complexity_class(self, solution, n) -> str:
        return self._recurrence_solver._extract_complexity_class(solution, n)

    def build_recursion_tree(
        self,
        func_name: str,
        initial_input: str = "n",
        branching_factor: int = 2,
        depth: int = 3,
        size_reduction: str = "n/2",
    ) -> Tuple[nx.DiGraph, str]:
        return self._recursion_tree_builder.build_recursion_tree(
            func_name=func_name,
            initial_input=initial_input,
            branching_factor=branching_factor,
            depth=depth,
            size_reduction=size_reduction,
        )

    def analyze_dynamic_programming(self, ast: Program) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze dynamic programming complexity based on subproblem structure.
        """
        memo_info = self._memoization_detector.detect(ast)

        state_dimensions = memo_info.get("state_dimensions", 1)

        if state_dimensions == 1:
            subproblem_count = "n"
            complexity = "O(n)"
            space_complexity = "O(n)"
        elif state_dimensions == 2:
            subproblem_count = "n^2"
            complexity = "O(n^2)"
            space_complexity = "O(n^2)"
        elif state_dimensions == 3:
            subproblem_count = "n^3"
            complexity = "O(n^3)"
            space_complexity = "O(n^3)"
        else:
            subproblem_count = f"n^{state_dimensions}"
            complexity = f"O(n^{state_dimensions})"
            space_complexity = f"O(n^{state_dimensions})"

        analysis = {
            "technique": "dynamic_programming",
            "state_dimensions": state_dimensions,
            "subproblems": subproblem_count,
            "time_complexity": complexity,
            "space_complexity": space_complexity,
            "has_memoization": memo_info.get("has_memoization", False),
            "reasoning": (
                f"DP with {state_dimensions}D state space requires computing "
                f"{subproblem_count} subproblems, each in constant/linear time."
            ),
        }

        return complexity, analysis

    def estimate_backtracking_complexity(
        self, branching_factor: int, max_depth: int, pruning_enabled: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Estimate backtracking complexity based on search tree structure.
        """

        worst_case = f"O({branching_factor}^{max_depth})"

        if pruning_enabled:
            average_case = worst_case
            best_case = "Ω(1)"
            reasoning = (
                f"Backtracking with pruning: worst-case explores up to {branching_factor}^{max_depth} "
                f"nodes. Pruning can cut branches early but worst-case remains exponential."
            )
        else:
            average_case = worst_case
            best_case = "Ω(1)"
            reasoning = (
                f"Exhaustive backtracking explores {branching_factor}^{max_depth} nodes "
                f"in the worst case without pruning."
            )

        analysis = {
            "technique": "backtracking",
            "branching_factor": branching_factor,
            "max_depth": max_depth,
            "pruning_enabled": pruning_enabled,
            "worst_case": worst_case,
            "average_case": average_case,
            "best_case": best_case,
            "reasoning": reasoning,
            "space_complexity": f"O({max_depth})",
        }

        return worst_case, analysis

    def estimate_greedy_complexity(
        self, n_elements: str, sorting_required: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Estimate greedy algorithm complexity based on sorting requirement.
        """
        if sorting_required:
            complexity = "O(n log n)"
            reasoning = (
                "Greedy algorithm with sorting: sorting dominates at O(n log n), "
                "followed by linear greedy selection."
            )
        else:
            complexity = "O(n)"
            reasoning = (
                "Greedy algorithm without sorting: single pass through elements "
                "making locally optimal choices in O(n)."
            )

        analysis = {
            "technique": "greedy",
            "n_elements": n_elements,
            "sorting_required": sorting_required,
            "time_complexity": complexity,
            "space_complexity": "O(1)" if not sorting_required else "O(log n)",
            "reasoning": reasoning,
            "approximation_note": (
                "Greedy algorithms may not always produce optimal solutions. "
                "Approximation ratio depends on problem structure."
            ),
        }

        return complexity, analysis

    def validate_with_simulation(
        self, complexity: str, test_sizes: List[int] = None
    ) -> Dict[str, Any]:
        """
        Validate complexity through simulation/counting (placeholder for actual execution).
        """

        results = {
            "technique": "simulation_validation",
            "predicted_complexity": complexity,
            "test_sizes": test_sizes or [10, 100, 1000],
            "valid": True,
            "measurements": [
                {"n": size, "operations": size, "matches_prediction": True}
                for size in test_sizes
            ],
            "reasoning": "Simulation validation placeholder - implement actual execution",
        }

        return results

    def validate_summation(
        self, expression: str, expected_result: str
    ) -> Dict[str, Any]:
        """Validate loop summation growth against an expected class using SymPy."""

        try:
            n = symbols("n", positive=True, integer=True)

            parsed_expr = self.parse_complexity(expression)
            if parsed_expr == 0:
                parsed_expr = n

            symbolic = sp.simplify(parsed_expr)
            complexity_class = self._extract_complexity_class(symbolic, n)

            matches = self.compare_complexities(
                expected_result, expression
            ) or self.compare_complexities(expected_result, complexity_class)

            return {
                "technique": "summation_validation",
                "expression": expression,
                "symbolic_result": str(symbolic),
                "complexity_class": complexity_class,
                "expected": expected_result,
                "valid": matches,
                "reasoning": f"Summation approximated as {symbolic}, classified as {complexity_class}",
            }
        except Exception as e:
            logger.error(f"Error validating summation: {e}")
            return {
                "technique": "summation_validation",
                "expression": expression,
                "error": str(e),
                "valid": False,
            }

    def verify_tight_bounds(
        self, worst_case: str, best_case: str, average_case: str
    ) -> Dict[str, Any]:
        """
        Verify if tight bounds (Θ notation) apply based on worst/best/average cases.
        """
        worst_clean = worst_case.replace("O(", "").replace(")", "").strip()
        best_clean = (
            best_case.replace("Ω(", "")
            .replace("Ω", "")
            .replace("O(", "")
            .replace(")", "")
            .strip()
        )

        has_tight_bounds = worst_clean == best_clean

        if has_tight_bounds:
            tight_bound = f"Θ({worst_clean})"
            reasoning = (
                f"Tight bounds apply: worst-case {worst_case} matches best-case {best_case}, "
                f"therefore the algorithm is {tight_bound}."
            )
        else:
            tight_bound = None
            reasoning = (
                f"No tight bounds: worst-case {worst_case} differs from best-case {best_case}. "
                f"Use separate O and Ω notation."
            )

        return {
            "technique": "tight_bounds_verification",
            "worst_case": worst_case,
            "best_case": best_case,
            "average_case": average_case,
            "has_tight_bounds": has_tight_bounds,
            "tight_bound": tight_bound,
            "reasoning": reasoning,
        }
