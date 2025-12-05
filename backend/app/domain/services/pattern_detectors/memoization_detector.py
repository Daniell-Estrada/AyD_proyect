"""
Memoization pattern detector.
Identifies dynamic programming through memoization patterns.
"""

from typing import Any, Dict

from app.domain.models.ast import ArrayAccess, ArrayTarget, Assignment, IfElse, Program
from app.domain.services.pattern_detectors.base_detector import PatternDetector


class MemoizationPatternDetector(PatternDetector):
    """
    Detects memoization patterns indicating dynamic programming.
    Looks for array/dict assignments indexed by parameters.
    """

    def detect(self, node: Any) -> Dict[str, Any]:
        """
        Detect memoization patterns in AST.

        Args:
            node: AST node to analyze

        Returns:
            Dictionary with keys:
                - has_memoization: boolean
                - memo_dimensions: estimated table dimensions (1D, 2D, etc.)
                - memo_variables: list of identified memoization array names
        """
        result = {
            "has_memoization": False,
            "memo_dimensions": 0,
            "memo_variables": [],
        }

        if isinstance(node, Program):
            self._traverse(node.statements, result)

        return result

    def _traverse(self, statements, result: Dict[str, Any]) -> None:
        """Recursively traverse statements to find memoization assignments."""
        for stmt in statements:
            if self._is_memoization_assignment(stmt):
                if isinstance(stmt, Assignment) and isinstance(stmt.target, (ArrayAccess, ArrayTarget)):
                    var_name = getattr(getattr(stmt.target, "array", None), "name", None)
                    if var_name is None:
                        var_name = getattr(stmt.target, "name", "memo")

                    if not self._looks_like_memo_table(var_name):
                        continue

                    result["has_memoization"] = True

                    if var_name not in result["memo_variables"]:
                        result["memo_variables"].append(var_name)

                    result["memo_dimensions"] = max(
                        result["memo_dimensions"],
                        self._count_array_dimensions(stmt.target),
                    )

            # Recurse into nested bodies
            if hasattr(stmt, "body"):
                self._traverse(getattr(stmt, "body"), result)

            if isinstance(stmt, IfElse):
                self._traverse(stmt.then_branch, result)
                self._traverse(getattr(stmt, "else_branch", []), result)

    def _is_memoization_assignment(self, stmt: Any) -> bool:
        """
        Check if statement is a memoization assignment pattern.

        Memoization typically involves:
            memo[param] = value  (1D)
            memo[i][j] = value   (2D)

        Args:
            stmt: Statement to check

        Returns:
            True if statement matches memoization pattern
        """
        if isinstance(stmt, Assignment):
            # Assignment to array indicates potential memoization
            if isinstance(stmt.target, (ArrayAccess, ArrayTarget)):
                return True
                
        return False

    def _count_array_dimensions(self, array_access: ArrayAccess | ArrayTarget) -> int:
        """
        Count dimensions of array access.

        Args:
            array_access: ArrayAccess node

        Returns:
            Number of dimensions (1 for arr[i], 2 for arr[i][j], etc.)
        """
        # Simplified: count based on index dimensions
        if isinstance(array_access, ArrayTarget):
            return max(1, len(array_access.index))

        dimensions = 1
        current = array_access.array
        while isinstance(current, ArrayAccess):
            dimensions += 1
            current = current.array
        return dimensions

    def _looks_like_memo_table(self, name: str) -> bool:
        if not name:
            return False
        lowered = name.lower()
        return any(keyword in lowered for keyword in ["dp", "memo", "cache", "table", "lookup"])
