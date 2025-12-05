"""
Recursion pattern detector.
Identifies recursive function calls and patterns.
"""

from typing import Any, Dict, List

from app.domain.models.ast import (Assignment, CallStmt, ForEachLoop, ForLoop,
                                   FuncCallExpr, IfElse, Program, ReturnStmt,
                                   SubroutineDef, WhileLoop)
from app.domain.services.pattern_detectors.base_detector import PatternDetector


class RecursionPatternDetector(PatternDetector):
    """
    Detects recursive patterns in algorithm implementations.
    Identifies self-calling functions and recursion depth.
    """

    def detect(self, node: Any) -> Dict[str, Any]:
        """
        Detect recursion patterns on the provided AST node.
        """
        result = {
            "has_recursion": False,
            "recursive_functions": [],
            "recursion_depth": 0,
        }

        if isinstance(node, SubroutineDef):
            func_name = node.name
            if self._contains_self_call(node.body, func_name):
                result["has_recursion"] = True
                result["recursive_functions"].append(func_name)

        elif isinstance(node, Program):
            for stmt in node.statements:
                if isinstance(stmt, SubroutineDef):
                    func_name = stmt.name
                    if self._contains_self_call(stmt.body, func_name):
                        result["has_recursion"] = True
                        result["recursive_functions"].append(func_name)

        return result

    def _contains_self_call(self, body: List[Any], func_name: str) -> bool:
        """
        Check if the function body contains self calls.
        """
        for stmt in body:
            if isinstance(stmt, (CallStmt, FuncCallExpr)) and stmt.name == func_name:
                return True

            if isinstance(stmt, Assignment):
                if self._contains_self_call_expr(stmt.value, func_name):
                    return True

            if isinstance(stmt, ReturnStmt):
                if self._contains_self_call_expr(stmt.value, func_name):
                    return True

            if isinstance(stmt, (ForLoop, ForEachLoop, WhileLoop)):
                if hasattr(stmt, "body") and self._contains_self_call(
                    stmt.body, func_name
                ):
                    return True
                if hasattr(stmt, "cond") and self._contains_self_call_expr(
                    stmt.cond, func_name
                ):
                    return True

            if isinstance(stmt, IfElse):
                if self._contains_self_call_expr(stmt.cond, func_name):
                    return True
                if self._contains_self_call(stmt.then_branch, func_name):
                    return True
                else_branch = getattr(
                    stmt, "else_branch", getattr(stmt, "else_body", [])
                )
                if self._contains_self_call(else_branch, func_name):
                    return True

        return False

    def _contains_self_call_expr(self, expr: Any, func_name: str) -> bool:
        """Recursively inspect expressions for self calls."""
        if isinstance(expr, (CallStmt, FuncCallExpr)):
            return expr.name == func_name

        if hasattr(expr, "args") and isinstance(expr, FuncCallExpr):
            return expr.name == func_name or any(
                self._contains_self_call_expr(arg, func_name) for arg in expr.args
            )

        if hasattr(expr, "left") and hasattr(expr, "right"):
            if self._contains_self_call_expr(expr.left, func_name):
                return True
            if self._contains_self_call_expr(expr.right, func_name):
                return True

        if hasattr(expr, "value"):
            return self._contains_self_call_expr(expr.value, func_name)

        return False
