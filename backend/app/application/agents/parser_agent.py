"""
Parser Agent - Parses pseudocode into Abstract Syntax Tree (AST).
Implements Single Responsibility Principle.
"""

import json
import logging
from typing import Any, Dict

from app.application.agents.state import AgentState
from app.domain.models.ast import ASTNode, Program
from app.infrastructure.parser.language_parser import LanguageParser, ParserResult

logger = logging.getLogger(__name__)


class ParserAgent:
    """
    Agent responsible for parsing pseudocode into AST using Lark parser.
    Identifies patterns like loops, recursion, conditionals, etc.
    """

    def __init__(self):
        """Initialize Parser Agent."""
        self.parser = LanguageParser()
        self.name = "ParserAgent"

    def execute(self, state: AgentState) -> AgentState:
        """
        Execute parsing of pseudocode to AST.

        Args:
            state: Current agent state

        Returns:
            Updated state with parsed AST
        """
        logger.info(f"{self.name}: Starting parsing")

        pseudocode = state.get("translated_pseudocode")
        if not pseudocode:
            logger.error(f"{self.name}: No pseudocode to parse")
            errors = state.get("errors", [])
            errors.append("No pseudocode available for parsing")
            state["errors"] = errors
            state["parsing_success"] = False
            return state

        try:
            # Parse pseudocode to AST
            parse_result: ParserResult = self.parser.parse(pseudocode)
            ast = parse_result.ast

            # Convert AST to serializable dict
            ast_dict = self._ast_to_dict(ast)

            # Analyze AST patterns
            patterns = self._identify_patterns(ast)

            diagnostics = parse_result.diagnostics
            warning_messages = [warning.message for warning in diagnostics.warnings]

            # Update state
            state["parsed_ast"] = ast_dict
            state["parsing_success"] = True
            state["parsing_errors"] = None
            state["current_stage"] = "parsing_complete"
            state["parser_normalized_source"] = diagnostics.normalized_source
            state["parser_warnings"] = warning_messages

            # Store pattern information for next agents
            if "metadata" not in state:
                state["metadata"] = {}
            state["metadata"]["patterns"] = patterns

            logger.info(
                f"{self.name}: Parsing completed successfully. "
                f"Patterns detected: {', '.join(patterns.keys())}. "
                f"Warnings: {len(warning_messages)}"
            )

        except Exception as e:
            logger.error(f"{self.name}: Parsing failed: {e}")
            errors = state.get("errors", [])
            errors.append(f"Parsing failed: {str(e)}")
            state["errors"] = errors
            state["parsing_success"] = False
            state["parsing_errors"] = [str(e)]

        return state

    def _ast_to_dict(self, node: Any, depth: int = 0, max_depth: int = 50) -> Dict[str, Any]:
        """
        Convert AST node to serializable dictionary.

        Args:
            node: AST node
            depth: Current recursion depth
            max_depth: Maximum recursion depth to prevent infinite loops

        Returns:
            Dictionary representation of AST
        """
        if depth > max_depth:
            return {"type": "max_depth_exceeded"}

        if node is None:
            return None

        # Handle primitive types
        if isinstance(node, (int, float, str, bool)):
            return node

        # Handle lists
        if isinstance(node, list):
            return [self._ast_to_dict(item, depth + 1, max_depth) for item in node]

        # Handle dict (for indexer ranges, etc.)
        if isinstance(node, dict):
            return {k: self._ast_to_dict(v, depth + 1, max_depth) for k, v in node.items()}

        # Handle AST nodes
        if isinstance(node, ASTNode):
            result = {
                "type": type(node).__name__,
            }

            # Add all attributes except methods and private attributes
            for attr_name in dir(node):
                if attr_name.startswith("_") or callable(getattr(node, attr_name)):
                    continue

                if attr_name in ["metadata", "accept"]:
                    continue

                attr_value = getattr(node, attr_name)
                result[attr_name] = self._ast_to_dict(attr_value, depth + 1, max_depth)

            return result

        # Fallback for unknown types
        return str(node)

    def _identify_patterns(self, ast: Program) -> Dict[str, Any]:
        """
        Identify algorithmic patterns in AST.

        Args:
            ast: Program AST

        Returns:
            Dictionary of detected patterns
        """
        from app.domain.models.ast import (
            CallStmt,
            ForEachLoop,
            ForLoop,
            FuncCallExpr,
            IfElse,
            RepeatUntil,
            ReturnStmt,
            SubroutineDef,
            WhileLoop,
        )

        patterns = {
            "has_loops": False,
            "has_nested_loops": False,
            "max_loop_depth": 0,
            "has_recursion": False,
            "has_conditionals": False,
            "has_functions": False,
            "loop_count": 0,
            "recursive_functions": [],
            "recursive_call_counts": {},
            "max_recursive_calls_per_function": 0,
            "tail_recursive_functions": [],
            "has_sorting": False,
            "sorting_calls": [],
            "has_priority_queue": False,
            "priority_queue_calls": [],
            "call_targets": [],
            "has_sequential_loops": False,
            "has_repeat_until": False,
            "repeat_until_count": 0,
        }

        call_targets: set[str] = set()
        sorting_calls: set[str] = set()
        priority_queue_calls: set[str] = set()
        loops_per_depth: dict[int, int] = {}

        def analyze_node(node: Any, loop_depth: int = 0):
            if isinstance(node, (ForLoop, ForEachLoop, WhileLoop, RepeatUntil)):
                patterns["has_loops"] = True
                patterns["loop_count"] += 1
                patterns["max_loop_depth"] = max(patterns["max_loop_depth"], loop_depth + 1)

                loops_per_depth[loop_depth] = loops_per_depth.get(loop_depth, 0) + 1
                if loops_per_depth[loop_depth] > 1:
                    patterns["has_sequential_loops"] = True

                if loop_depth > 0:
                    patterns["has_nested_loops"] = True

                if isinstance(node, RepeatUntil):
                    patterns["has_repeat_until"] = True
                    patterns["repeat_until_count"] += 1

                # Analyze loop body
                for stmt in node.body:
                    analyze_node(stmt, loop_depth + 1)

            elif isinstance(node, IfElse):
                patterns["has_conditionals"] = True

                for stmt in node.then_branch:
                    analyze_node(stmt, loop_depth)
                for stmt in node.else_branch:
                    analyze_node(stmt, loop_depth)

            elif isinstance(node, SubroutineDef):
                patterns["has_functions"] = True
                func_name = node.name

                # Check for recursion
                recursion_info = self._analyze_self_calls(node.body, func_name)
                if recursion_info["call_count"] > 0:
                    patterns["has_recursion"] = True
                    patterns["recursive_functions"].append(func_name)
                    patterns["recursive_call_counts"][func_name] = recursion_info["call_count"]
                    patterns["max_recursive_calls_per_function"] = max(
                        patterns["max_recursive_calls_per_function"],
                        recursion_info["call_count"],
                    )

                    if recursion_info["call_count"] == recursion_info["tail_call_count"]:
                        patterns["tail_recursive_functions"].append(func_name)

                for stmt in node.body:
                    analyze_node(stmt, loop_depth)

            elif isinstance(node, Program):
                for stmt in node.statements:
                    analyze_node(stmt, loop_depth)

            elif isinstance(node, (CallStmt, FuncCallExpr)):
                call_name = getattr(node, "name", None)
                if call_name:
                    call_targets.add(call_name)
                    lowered = call_name.lower()
                    if "sort" in lowered:
                        patterns["has_sorting"] = True
                        sorting_calls.add(call_name)
                    if any(keyword in lowered for keyword in ("priority", "heap")):
                        patterns["has_priority_queue"] = True
                        priority_queue_calls.add(call_name)

        analyze_node(ast)

        if call_targets:
            patterns["call_targets"] = sorted(call_targets)
        if sorting_calls:
            patterns["sorting_calls"] = sorted(sorting_calls)
        if priority_queue_calls:
            patterns["priority_queue_calls"] = sorted(priority_queue_calls)

        return patterns

    def _analyze_self_calls(self, body: list, func_name: str) -> Dict[str, int]:
        """Collect recursion metrics (total and tail calls) for a function body."""
        from app.domain.models.ast import ASTNode, CallStmt, FuncCallExpr, ReturnStmt

        metrics = {
            "call_count": 0,
            "tail_call_count": 0,
        }

        def traverse(value: Any, tail_context: bool = False) -> int:
            call_total = 0

            if isinstance(value, ReturnStmt):
                return traverse(value.value, tail_context=True)

            if isinstance(value, (CallStmt, FuncCallExpr)):
                if value.name == func_name:
                    metrics["call_count"] += 1
                    if tail_context:
                        metrics["tail_call_count"] += 1
                    return 1
                return 0

            if isinstance(value, ASTNode):
                for attr_name, attr_value in vars(value).items():
                    if attr_name.startswith("_") or attr_value is None:
                        continue
                    call_total += traverse(attr_value, tail_context=False)
                return call_total

            if isinstance(value, dict):
                for dict_value in value.values():
                    call_total += traverse(dict_value, tail_context=False)
                return call_total

            if isinstance(value, (list, tuple)):
                for item in value:
                    call_total += traverse(item, tail_context=False)
                return call_total

            return 0

        for stmt in body:
            traverse(stmt, tail_context=False)

        return metrics
