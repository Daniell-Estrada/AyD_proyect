"""
Backtracking pattern detector.
Identifies backtracking and state restoration patterns.
"""

from typing import Any, Dict, List

from app.domain.models.ast import (Assignment, FuncCallExpr, IfElse, Program,
                                   SubroutineDef)
from app.domain.services.pattern_detectors.base_detector import PatternDetector
from app.domain.services.pattern_detectors.recursion_detector import \
    RecursionPatternDetector


class BacktrackingPatternDetector(PatternDetector):
    """
    Detects backtracking algorithm patterns.
    Looks for multiple recursive calls and state restoration.
    """

    def detect(self, node: Any) -> Dict[str, Any]:
        """Detect backtracking patterns on the provided AST node."""
        result = {
            "has_backtracking": False,
            "has_state_restoration": False,
            "branching_factor": 1,
        }

        if isinstance(node, Program):
            for stmt in node.statements:
                if isinstance(stmt, SubroutineDef):
                    self._analyze_function(stmt, result)

        return result

    def _analyze_function(self, func: SubroutineDef, result: Dict[str, Any]) -> None:
        recursion_info = RecursionPatternDetector().detect(func)
        if not recursion_info["has_recursion"]:
            return

        recursive_calls = self._find_recursive_calls(func.body, func.name)
        if recursive_calls:
            result["has_backtracking"] = len(recursive_calls) >= 2
            result["branching_factor"] = max(len(recursive_calls), 1)

        if self._has_state_restoration(func.body):
            result["has_state_restoration"] = True
            result["has_backtracking"] = True

    def _find_recursive_calls(
        self, body: List[Any], func_name: str
    ) -> List[FuncCallExpr]:
        calls: List[FuncCallExpr] = []
        for stmt in body:
            if isinstance(stmt, FuncCallExpr) and stmt.name == func_name:
                calls.append(stmt)
            if (
                isinstance(stmt, Assignment)
                and isinstance(stmt.value, FuncCallExpr)
                and stmt.value.name == func_name
            ):
                calls.append(stmt.value)
            if hasattr(stmt, "cond"):
                calls.extend(
                    self._extract_calls_from_expr(getattr(stmt, "cond"), func_name)
                )
            if isinstance(stmt, IfElse):
                calls.extend(self._find_recursive_calls(stmt.then_branch, func_name))
                calls.extend(
                    self._find_recursive_calls(
                        getattr(stmt, "else_branch", []), func_name
                    )
                )
            if hasattr(stmt, "body"):
                calls.extend(
                    self._find_recursive_calls(getattr(stmt, "body"), func_name)
                )
        return calls

    def _extract_calls_from_expr(self, expr: Any, func_name: str) -> List[FuncCallExpr]:
        calls: List[FuncCallExpr] = []
        if expr is None:
            return calls

        if isinstance(expr, FuncCallExpr) and expr.name == func_name:
            calls.append(expr)

        if hasattr(expr, "args") and isinstance(expr, FuncCallExpr):
            for arg in expr.args:
                calls.extend(self._extract_calls_from_expr(arg, func_name))

        if hasattr(expr, "left") and hasattr(expr, "right"):
            calls.extend(self._extract_calls_from_expr(expr.left, func_name))
            calls.extend(self._extract_calls_from_expr(expr.right, func_name))

        return calls

    def _has_state_restoration(self, body: List[Any]) -> bool:
        seen_targets = set()
        for stmt in body:
            if isinstance(stmt, Assignment):
                key = str(getattr(stmt, "target", stmt))
                if key in seen_targets:
                    return True
                seen_targets.add(key)
            if hasattr(stmt, "body") and self._has_state_restoration(
                getattr(stmt, "body")
            ):
                return True
            if isinstance(stmt, IfElse):
                if self._has_state_restoration(
                    stmt.then_branch
                ) or self._has_state_restoration(getattr(stmt, "else_branch", [])):
                    return True
        return False
