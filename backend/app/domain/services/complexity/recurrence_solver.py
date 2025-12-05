"""
Helpers for solving recurrence relations and applying the Master Theorem.
Separated from the main complexity service to keep responsibilities focused.
"""

import logging
import re
from typing import Any, Dict, Optional, Tuple

import sympy as sp
from sympy import Function, symbols

logger = logging.getLogger(__name__)


class RecurrenceSolver:
    """Encapsulates recurrence solving utilities (Master Theorem + rsolve)."""

    def __init__(self) -> None:
        self.n = symbols("n", positive=True, integer=True)
        self._epsilon = 1e-9

    def solve_recurrence(self, a: int, b: int, k: int) -> Tuple[str, Dict[str, Any]]:
        """
        Solve recurrence T(n) = aT(n/b) + n^k using Master Theorem.
        """
        log_b_a = sp.log(a, b)
        log_b_a_float = float(sp.N(log_b_a, 5))
        epsilon = 1e-6
        base_term = self._format_power_term(k)
        recurrence_term = self._format_theta_term(k)

        if abs(k - log_b_a_float) < epsilon:
            complexity = self._format_big_o_with_log(base_term)
            worst_case = complexity
            best_case = complexity
            case = 2
            reasoning = (
                f"Case 2: k={k} ≈ log_{b}({a}) = {log_b_a_float:.3f}. "
                f"Work is balanced across all {sp.ceiling(sp.log(self.n, b))} levels, "
                f"each contributing Θ({recurrence_term}), leading to logarithmic multiplication."
            )
        elif k < log_b_a_float:
            dom_term = self._format_power_term(log_b_a_float)
            complexity = f"O({dom_term})"
            worst_case = complexity
            best_case = complexity
            case = 1
            reasoning = (
                f"Case 1: k={k} < log_{b}({a}) = {log_b_a_float:.3f}. "
                f"Recursive subproblems dominate. With {a} subproblems of size n/{b}, "
                f"leaf-level work dominates the total cost."
            )
        else:
            complexity = f"O({base_term})"
            worst_case = complexity
            best_case = complexity
            case = 3
            reasoning = (
                f"Case 3: k={k} > log_{b}({a}) = {log_b_a_float:.3f}. "
                f"Work at the root level dominates. The combine step of Θ({recurrence_term}) "
                f"exceeds the cost of solving subproblems."
            )

        analysis = {
            "technique": "master_theorem",
            "recurrence": f"T(n) = {a}T(n/{b}) + Θ({recurrence_term})",
            "a": a,
            "b": b,
            "k": k,
            "log_b_a": log_b_a_float,
            "case": case,
            "complexity": complexity,
            "worst_case": worst_case,
            "best_case": best_case,
            "reasoning": reasoning,
            "levels": f"log_{b}(n)",
            "work_per_level": recurrence_term,
        }

        return complexity, analysis

    def solve_recurrence_relation(
        self, relation: str
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Solve general recurrence relation using SymPy's rsolve.
        """
        try:
            n = symbols("n", integer=True, positive=True)
            T = Function("T")

            parsed = self._parse_recurrence_relation(relation)
            if not parsed:
                return None, {
                    "technique": "recurrence_solving",
                    "relation": relation,
                    "error": "Unsupported recurrence format. Supported: T(n)=aT(n/b)+f(n) or T(n)=T(n-c)+f(n)",
                }

            equation = parsed["equation"]
            solution = sp.rsolve(equation, T(n))

            if solution:
                complexity_class = self._extract_complexity_class(solution, n)
                complexity = f"Θ({complexity_class})"
                worst_case = f"O({complexity_class})"
                best_case = f"Ω({complexity_class})"
                simplified_solution = sp.simplify(solution)
                reasoning = (
                    f"Solved recurrence to closed form: {simplified_solution}. "
                    f"Dominant term determines asymptotic behavior."
                )
            else:
                complexity = "Unable to solve"
                worst_case = "Unknown"
                best_case = "Unknown"
                reasoning = (
                    "SymPy could not find a closed-form solution for this recurrence"
                )

            analysis = {
                "technique": "recurrence_solving",
                "relation": relation,
                "normalized": parsed.get("normalized"),
                "equation": str(equation),
                "solution": str(solution) if solution else "No closed form",
                "complexity": complexity,
                "worst_case": worst_case,
                "best_case": best_case,
                "reasoning": reasoning,
            }

            return complexity, analysis

        except Exception as e:
            logger.error(f"Error solving recurrence '{relation}': {e}")
            return None, {
                "technique": "recurrence_solving",
                "relation": relation,
                "error": str(e),
            }

    def _extract_complexity_class(self, solution, n) -> str:
        """Extract asymptotic class from SymPy closed-form solution."""
        expanded = sp.expand(solution)
        solution_str = str(expanded).lower()

        if "2**n" in solution_str or "exp(n)" in solution_str:
            return "2^n"
        if "factorial" in solution_str:
            return "n!"
        if "log" in solution_str:
            if "n**3" in solution_str or "n^3" in solution_str:
                return "n^3 log n"
            if "n**2" in solution_str or "n^2" in solution_str:
                return "n^2 log n"
            if "n" in solution_str:
                return "n log n"
            return "log n"
        if "n**4" in solution_str or "n^4" in solution_str:
            return "n^4"
        if "n**3" in solution_str or "n^3" in solution_str:
            return "n^3"
        if "n**2" in solution_str or "n^2" in solution_str:
            return "n^2"
        if "n" in solution_str:
            return "n"
        return "1"

    def _parse_recurrence_relation(self, relation: str) -> Optional[Dict[str, Any]]:
        """Parse recurrence strings into SymPy equations."""
        n = symbols("n", integer=True, positive=True)
        T = Function("T")

        relation = relation.replace(" ", "")
        match = re.match(r"T\(n\)=T\(n-([0-9]+)\)\+(.+)", relation)
        if match:
            c = int(match.group(1))
            f_n_str = match.group(2)
            f_n = self._safe_parse_function(f_n_str, n)
            equation = sp.Eq(T(n), T(n - c) + f_n)
            return {"equation": equation, "normalized": f"T(n) = T(n-{c}) + {f_n_str}"}

        match = re.match(r"T\(n\)=([0-9]+)\*?T\(n/([0-9]+)\)\+(.+)", relation)
        if match:
            a = int(match.group(1))
            b = int(match.group(2))
            f_n_str = match.group(3)
            f_n = self._safe_parse_function(f_n_str, n)
            equation = sp.Eq(T(n), a * T(n / b) + f_n)
            return {
                "equation": equation,
                "normalized": f"T(n) = {a}T(n/{b}) + {f_n_str}",
            }

        return None

    def _safe_parse_function(self, f_n_str: str, n) -> Any:
        """Parse a polynomial/log expression into a SymPy term."""
        try:
            return sp.sympify(f_n_str, locals={"n": n, "log": sp.log})
        except Exception:
            return sp.Symbol("f(n)")

    def _format_power_term(self, exponent: float) -> str:
        """Render n^k without showing exponent 1 or 0."""

        if exponent is None:
            return "n"

        if abs(exponent) < self._epsilon:
            return "1"

        if abs(exponent - 1) < self._epsilon:
            return "n"

        rounded = round(exponent)
        if abs(exponent - rounded) < self._epsilon:
            return f"n^{int(rounded)}"

        value = f"n^{exponent:.2f}"
        if "." in value:
            base, frac = value.split(".")
            frac = frac.rstrip("0")
            value = base if not frac else f"{base}.{frac}"
        return value

    def _format_theta_term(self, exponent: float) -> str:
        term = self._format_power_term(exponent)
        return "n" if term == "1" else term

    def _format_big_o_with_log(self, base_term: str) -> str:
        if base_term == "1":
            return "O(log n)"
        return f"O({base_term} log n)"
