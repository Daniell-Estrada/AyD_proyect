"""
Loop pattern detector.
Analyzes loop structures and nesting depth.
"""

from typing import Any, Dict

from app.domain.models.ast import (ForEachLoop, ForLoop, IfElse, Program,
                                   RepeatUntil, WhileLoop)
from app.domain.services.pattern_detectors.base_detector import PatternDetector


class LoopPatternDetector(PatternDetector):
    """
    Detects loop patterns and nesting structures.
    Calculates maximum nesting depth and total loop count.
    """

    def detect(self, node: Any) -> Dict[str, Any]:
        """
        Detect loop patterns on the provided AST node.
        """
        result = {
            "loop_count": 0,
            "max_loop_depth": 0,
            "has_nested_loops": False,
            "has_sequential_loops": False,
            "loop_types": [],
        }

        self._traverse_and_count_loops(node, result, depth=0)

        if result["max_loop_depth"] > 1:
            result["has_nested_loops"] = True

        if result["loop_count"] > result["max_loop_depth"]:
            result["has_sequential_loops"] = True

        return result

    def _traverse_and_count_loops(
        self, node: Any, result: Dict[str, Any], depth: int
    ) -> None:
        """
        Recursively traverse AST counting loops and tracking depth.
        """
        if isinstance(node, (ForLoop, ForEachLoop)):
            result["loop_count"] += 1
            result["max_loop_depth"] = max(result["max_loop_depth"], depth + 1)

            if "for" not in result["loop_types"]:
                result["loop_types"].append("for")

            for stmt in node.body:
                self._traverse_and_count_loops(stmt, result, depth + 1)

        elif isinstance(node, RepeatUntil):
            result["loop_count"] += 1
            result["max_loop_depth"] = max(result["max_loop_depth"], depth + 1)

            if "repeat_until" not in result["loop_types"]:
                result["loop_types"].append("repeat_until")

            for stmt in node.body:
                self._traverse_and_count_loops(stmt, result, depth + 1)

        elif isinstance(node, WhileLoop):
            result["loop_count"] += 1
            result["max_loop_depth"] = max(result["max_loop_depth"], depth + 1)

            if "while" not in result["loop_types"]:
                result["loop_types"].append("while")

            for stmt in node.body:
                self._traverse_and_count_loops(stmt, result, depth + 1)

        elif isinstance(node, Program):
            for stmt in node.statements:
                self._traverse_and_count_loops(stmt, result, depth)

        elif isinstance(node, IfElse):
            for stmt in node.then_branch:
                self._traverse_and_count_loops(stmt, result, depth)
            else_branch = getattr(node, "else_branch", getattr(node, "else_body", []))
            for stmt in else_branch:
                self._traverse_and_count_loops(stmt, result, depth)

        elif hasattr(node, "body"):
            for stmt in node.body:
                self._traverse_and_count_loops(stmt, result, depth)
