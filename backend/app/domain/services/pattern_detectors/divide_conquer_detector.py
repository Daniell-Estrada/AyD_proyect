"""
Divide and conquer pattern detector.
Identifies algorithms that divide problems into smaller subproblems.
"""

from typing import Any, Dict, List

from app.domain.models.ast import (Assignment, BinOp, CallStmt, FuncCallExpr,
                                   IfElse, Program, ReturnStmt, SubroutineDef)
from app.domain.services.pattern_detectors.base_detector import PatternDetector


class DivideAndConquerPatternDetector(PatternDetector):
    """
    Detects divide-and-conquer algorithmic patterns.
    Looks for recursive calls with reduced input size (division or subtraction).
    """

    def detect(self, node: Any) -> Dict[str, Any]:
        """
        Detect divide-and-conquer patterns on the provided AST node.
        """
        result = {
            "is_divide_and_conquer": False,
            "division_factor": 1,
            "recursive_calls_count": 0,
        }

        if isinstance(node, SubroutineDef):
            return self._analyze_function(node)

        if isinstance(node, Program):
            aggregate = result.copy()
            for stmt in node.statements:
                if isinstance(stmt, SubroutineDef):
                    func_result = self._analyze_function(stmt)
                    if func_result["is_divide_and_conquer"]:
                        aggregate = func_result
                        break
            return aggregate

        return result

    def _analyze_function(self, node: SubroutineDef) -> Dict[str, Any]:
        result = {
            "is_divide_and_conquer": False,
            "division_factor": 1,
            "recursive_calls_count": 0,
        }

        func_name = node.name
        recursive_calls = self._find_recursive_calls(node.body, func_name)

        if recursive_calls:
            result["recursive_calls_count"] = len(recursive_calls)

            for call in recursive_calls:
                if self._is_dividing_call(call):
                    result["is_divide_and_conquer"] = True
                    result["division_factor"] = self._estimate_division_factor(call)
                    break

        return result

    def _find_recursive_calls(
        self, body: List[Any], func_name: str
    ) -> List[FuncCallExpr]:
        """
        Find all recursive calls in function body.
        """
        calls = []
        for stmt in body:
            calls.extend(self._extract_calls_from_statement(stmt, func_name))

        return calls

    def _extract_calls_from_statement(
        self, stmt: Any, func_name: str
    ) -> List[FuncCallExpr]:
        calls: List[FuncCallExpr] = []

        if isinstance(stmt, (CallStmt, FuncCallExpr)) and stmt.name == func_name:
            calls.append(
                stmt
                if isinstance(stmt, FuncCallExpr)
                else FuncCallExpr(name=stmt.name, args=stmt.args)
            )
            return calls

        if isinstance(stmt, Assignment):
            calls.extend(
                self._extract_calls_from_expr(getattr(stmt, "value", None), func_name)
            )
            return calls

        if isinstance(stmt, ReturnStmt):
            calls.extend(
                self._extract_calls_from_expr(getattr(stmt, "value", None), func_name)
            )
            return calls

        if hasattr(stmt, "cond"):
            calls.extend(
                self._extract_calls_from_expr(getattr(stmt, "cond"), func_name)
            )

        if hasattr(stmt, "body"):
            calls.extend(self._find_recursive_calls(getattr(stmt, "body"), func_name))

        if isinstance(stmt, IfElse):
            calls.extend(self._find_recursive_calls(stmt.then_branch, func_name))
            else_branch = getattr(stmt, "else_branch", getattr(stmt, "else_body", []))
            calls.extend(self._find_recursive_calls(else_branch, func_name))

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

        if hasattr(expr, "value"):
            calls.extend(self._extract_calls_from_expr(expr.value, func_name))

        return calls

    def _is_dividing_call(self, call: FuncCallExpr) -> bool:
        """
        Check if the recursive call reduces problem size via division or subtraction.
        """
        for arg in call.args:
            if isinstance(arg, BinOp):
                if arg.op in ["/", "//", "div", "-"]:
                    return True
            if hasattr(arg, "ranges"):
                return True

        return False

    def _estimate_division_factor(self, call: FuncCallExpr) -> int:
        """
        Estimate division factor from recursive call arguments.
        """

        for arg in call.args:
            if isinstance(arg, BinOp) and arg.op in ["/", "//", "div"]:
                return 2

        return 2
