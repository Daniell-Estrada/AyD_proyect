"""
Memoization pattern detector.
Identifies dynamic programming through memoization patterns.
"""

from typing import Any, Dict

from app.domain.models.ast import (
    ArrayAccess,
    ArrayTarget,
    ArrayVarDecl,
    Assignment,
    BinOp,
    Bool,
    CallMethod,
    FuncCallExpr,
    IfElse,
    LengthFunction,
    Null,
    Number,
    Program,
    ShortCircuitBinOp,
    String,
    UnOp,
    Var,
    VarDecl,
    VarTarget,
)
from app.domain.services.pattern_detectors.base_detector import PatternDetector


class MemoizationPatternDetector(PatternDetector):
    """
    Detects memoization patterns indicating dynamic programming.
    Looks for array/dict assignments indexed by parameters.
    """

    def detect(self, node: Any) -> Dict[str, Any]:
        """
        Detect memoization patterns on the provided AST node.
        """
        result = {
            "has_memoization": False,
            "memo_dimensions": 0,
            "memo_variables": [],
            "state_dimensions": 0,
            "memo_table_size": None,
            "state_labels": [],
        }

        self._array_dimensions: Dict[str, list[str]] = {}

        if isinstance(node, Program):
            self._traverse(node.statements, result)

        if result["memo_dimensions"]:
            result["state_dimensions"] = result["memo_dimensions"]
        elif result["state_labels"]:
            result["state_dimensions"] = len(result["state_labels"])
        elif result["has_memoization"]:
            result["state_dimensions"] = 1

        if not result["memo_table_size"] and result["state_labels"]:
            result["memo_table_size"] = self._format_table_volume(
                result["state_labels"]
            )

        return result

    def _traverse(self, statements, result: Dict[str, Any]) -> None:
        """Recursively traverse statements to find memoization assignments."""
        for stmt in statements:
            if isinstance(stmt, VarDecl):
                self._record_var_decl(stmt)

            if self._is_memoization_assignment(stmt):
                if isinstance(stmt, Assignment) and isinstance(
                    stmt.target, (ArrayAccess, ArrayTarget)
                ):
                    var_name = self._resolve_base_name(stmt.target)

                    if not self._uses_variable_indices(stmt.target):
                        continue

                    result["has_memoization"] = True

                    if var_name not in result["memo_variables"]:
                        result["memo_variables"].append(var_name)

                    labels = self._array_dimensions.get(var_name, [])
                    filtered_labels = [label or "n" for label in labels]
                    if filtered_labels and len(filtered_labels) >= len(
                        result["state_labels"] or []
                    ):
                        result["state_labels"] = filtered_labels
                        result["memo_table_size"] = self._format_table_volume(
                            filtered_labels
                        )

                    result["memo_dimensions"] = max(
                        result["memo_dimensions"],
                        self._count_array_dimensions(stmt.target),
                    )

            if hasattr(stmt, "body"):
                self._traverse(getattr(stmt, "body"), result)

            if isinstance(stmt, IfElse):
                self._traverse(stmt.then_branch, result)
                self._traverse(getattr(stmt, "else_branch", []), result)

    def _is_memoization_assignment(self, stmt: Any) -> bool:
        """
        Check if statement is a memoization assignment pattern.
        """
        if isinstance(stmt, Assignment):
            if isinstance(stmt.target, (ArrayAccess, ArrayTarget)):
                return True

        return False

    def _count_array_dimensions(self, array_access: ArrayAccess | ArrayTarget) -> int:
        """
        Count dimensions of array access.

        """
        if isinstance(array_access, ArrayTarget):
            return max(1, len(array_access.index or []))

        dims = max(1, len(array_access.index or []))
        current = array_access.array

        while isinstance(current, ArrayAccess):
            dims += max(1, len(current.index or []))
            current = current.array

        return dims

    def _uses_variable_indices(self, target: ArrayAccess | ArrayTarget) -> bool:
        indices = getattr(target, "index", []) or []
        if not indices:
            return False
        return any(self._expression_depends_on_identifier(index) for index in indices)

    def _resolve_base_name(self, target: ArrayAccess | ArrayTarget) -> str:
        candidate = getattr(target, "name", None)
        if candidate:
            return candidate

        base = getattr(target, "array", None)
        while base is not None:
            name = getattr(base, "name", None)
            if name:
                return name
            base = getattr(base, "array", None)

        return "memo"

    def _record_var_decl(self, decl: VarDecl) -> None:
        for item in getattr(decl, "items", []) or []:
            if isinstance(item, ArrayVarDecl):
                labels = [
                    self._dimension_label(dimension)
                    for dimension in item.dimensions or []
                ]
                self._array_dimensions[item.name] = [
                    label or "n" for label in labels
                ]

    def _dimension_label(self, spec: Any) -> str:
        if isinstance(spec, dict):
            if spec.get("type") == "range":
                return self._expr_to_str(spec.get("end")) or self._expr_to_str(spec.get("start"))
            return self._expr_to_str(spec.get("value")) or spec.get("name", "")
        return self._expr_to_str(spec)

    def _expr_to_str(self, expr: Any) -> str:
        if expr is None:
            return ""

        if isinstance(expr, Var):
            return expr.name

        if isinstance(expr, Number):
            return str(expr.value)

        if isinstance(expr, (int, float)):
            return str(expr)

        if isinstance(expr, str):
            return expr

        if isinstance(expr, BinOp):
            left = self._expr_to_str(expr.left)
            right = self._expr_to_str(expr.right)
            op = expr.op or ""
            if left and right:
                return f"({left} {op} {right})"
            return left or right

        if isinstance(expr, dict):
            expr_type = expr.get("type")
            if expr_type == "Var":
                return expr.get("name", "")
            if expr_type == "Number":
                value = expr.get("value")
                return str(value) if value is not None else ""
            if expr_type == "range":
                return self._expr_to_str(expr.get("end")) or self._expr_to_str(expr.get("start"))
            if "name" in expr:
                return expr["name"]

        if hasattr(expr, "name"):
            return getattr(expr, "name")

        return ""

    def _format_table_volume(self, labels: list[str]) -> str:
        filtered = [label or "n" for label in labels]
        if not filtered:
            return ""

        if len(set(filtered)) == 1:
            base = filtered[0]
            if len(filtered) == 1:
                return base
            return f"{base}^{len(filtered)}"

        return " * ".join(filtered)

    def _expression_depends_on_identifier(self, expr: Any) -> bool:
        if expr is None:
            return False

        if isinstance(expr, (Var, VarTarget)):
            return True

        if isinstance(expr, (ArrayAccess, ArrayTarget)):
            return True

        if isinstance(expr, (BinOp, ShortCircuitBinOp)):
            return self._expression_depends_on_identifier(expr.left) or self._expression_depends_on_identifier(
                expr.right
            )

        if isinstance(expr, UnOp):
            return self._expression_depends_on_identifier(expr.value)

        if isinstance(expr, FuncCallExpr):
            return any(self._expression_depends_on_identifier(arg) for arg in expr.args)

        if isinstance(expr, CallMethod):
            if self._expression_depends_on_identifier(expr.obj):
                return True
            return any(self._expression_depends_on_identifier(arg) for arg in expr.args)

        if isinstance(expr, LengthFunction):
            return self._expression_depends_on_identifier(expr.array)

        if isinstance(expr, (Number, String, Bool, Null)):
            return False

        if isinstance(expr, (int, float, bool, str)):
            return False

        if isinstance(expr, list):
            return any(self._expression_depends_on_identifier(item) for item in expr)

        if hasattr(expr, "__dict__"):
            for key, value in vars(expr).items():
                if key == "metadata":
                    continue
                if isinstance(value, list):
                    if any(self._expression_depends_on_identifier(item) for item in value):
                        return True
                else:
                    if self._expression_depends_on_identifier(value):
                        return True

        return False
