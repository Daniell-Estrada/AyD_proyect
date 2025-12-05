"""
Parser Agent - Parses pseudocode into Abstract Syntax Tree (AST).
Implements Single Responsibility Principle.
"""

import logging
from typing import Any, Dict, List, Optional

from app.application.agents.state import AgentState
from app.domain.models.ast import (ASTNode, BinOp, CallStmt, ForEachLoop,
                                   ForLoop, FuncCallExpr, IfElse, Number, Program,
                                   RepeatUntil, ReturnStmt, SubroutineDef, Var, WhileLoop)
from app.domain.services.pattern_detectors.memoization_detector import \
    MemoizationPatternDetector
from app.infrastructure.parser.language_parser import (LanguageParser,
                                                       ParserResult)

logger = logging.getLogger(__name__)


class ParserAgent:
    """
    Agent responsible for parsing pseudocode into AST using Lark parser.
    Identifies patterns like loops, recursion, conditionals, etc.
    """

    def __init__(self):
        self.parser = LanguageParser()
        self.name = "ParserAgent"

    def execute(self, state: AgentState) -> AgentState:
        """
        Execute parsing of pseudocode into AST.
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
            parse_result: ParserResult = self.parser.parse(pseudocode)
            ast = parse_result.ast

            ast_dict = self._ast_to_dict(ast)
            patterns = self._identify_patterns(ast)

            diagnostics = parse_result.diagnostics
            warning_messages = [warning.message for warning in diagnostics.warnings]

            state["parsed_ast"] = ast_dict
            state["parsing_success"] = True
            state["parsing_errors"] = None
            state["current_stage"] = "parsing_complete"
            state["parser_normalized_source"] = diagnostics.normalized_source
            state["parser_warnings"] = warning_messages

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

    def _ast_to_dict(
        self, node: Any, depth: int = 0, max_depth: int = 50
    ) -> Dict[str, Any]:
        """
        Convert AST node to serializable dictionary.
        """
        if depth > max_depth:
            return {"type": "max_depth_exceeded"}

        if node is None:
            return None

        if isinstance(node, (int, float, str, bool)):
            return node

        if isinstance(node, list):
            return [self._ast_to_dict(item, depth + 1, max_depth) for item in node]

        if isinstance(node, dict):
            return {
                k: self._ast_to_dict(v, depth + 1, max_depth) for k, v in node.items()
            }

        if isinstance(node, ASTNode):
            result = {
                "type": type(node).__name__,
            }

            for attr_name in dir(node):
                if attr_name.startswith("_") or callable(getattr(node, attr_name)):
                    continue

                if attr_name in ["metadata", "accept"]:
                    continue

                attr_value = getattr(node, attr_name)
                result[attr_name] = self._ast_to_dict(attr_value, depth + 1, max_depth)

            return result

        return str(node)

    def _identify_patterns(self, ast: Program) -> Dict[str, Any]:
        """
        Identify algorithmic patterns in AST.
        """

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
            "has_constant_decrement_recursion": False,
            "has_fractional_split_recursion": False,
            "non_linear_split_recursion": False,
            "recursive_constant_decrements": [],
            "fractional_split_factors": [],
            "estimated_branching_factor": 0,
            "recursive_argument_shapes": [],
        }

        call_targets: set[str] = set()
        sorting_calls: set[str] = set()
        priority_queue_calls: set[str] = set()
        loops_per_depth: dict[int, int] = {}

        def analyze_node(node: Any, loop_depth: int = 0):
            if isinstance(node, (ForLoop, ForEachLoop, WhileLoop, RepeatUntil)):
                patterns["has_loops"] = True
                patterns["loop_count"] += 1
                patterns["max_loop_depth"] = max(
                    patterns["max_loop_depth"], loop_depth + 1
                )

                loops_per_depth[loop_depth] = loops_per_depth.get(loop_depth, 0) + 1
                if loops_per_depth[loop_depth] > 1:
                    patterns["has_sequential_loops"] = True

                if loop_depth > 0:
                    patterns["has_nested_loops"] = True

                if isinstance(node, RepeatUntil):
                    patterns["has_repeat_until"] = True
                    patterns["repeat_until_count"] += 1

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

                recursion_info = self._analyze_self_calls(node, func_name)
                if recursion_info["call_count"] > 0:
                    patterns["has_recursion"] = True
                    patterns["recursive_functions"].append(func_name)
                    patterns["recursive_call_counts"][func_name] = recursion_info[
                        "call_count"
                    ]
                    patterns["max_recursive_calls_per_function"] = max(
                        patterns["max_recursive_calls_per_function"],
                        recursion_info["call_count"],
                    )

                    if (
                        recursion_info["call_count"]
                        == recursion_info["tail_call_count"]
                    ):
                        patterns["tail_recursive_functions"].append(func_name)

                    patterns["estimated_branching_factor"] = max(
                        patterns["estimated_branching_factor"],
                        recursion_info["call_count"],
                    )

                    if recursion_info["constant_decrements"]:
                        patterns["has_constant_decrement_recursion"] = True
                        patterns["recursive_constant_decrements"].extend(
                            recursion_info["constant_decrements"]
                        )

                    if recursion_info["fractional_split_factors"]:
                        patterns["has_fractional_split_recursion"] = True
                        patterns["fractional_split_factors"].extend(
                            recursion_info["fractional_split_factors"]
                        )

                    if recursion_info["argument_shapes"]:
                        patterns["recursive_argument_shapes"].extend(
                            recursion_info["argument_shapes"]
                        )

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

            elif isinstance(node, ASTNode):
                for attr_name, attr_value in vars(node).items():
                    if attr_name.startswith("_") or attr_value is None:
                        continue
                    analyze_node(attr_value, loop_depth)

            elif isinstance(node, dict):
                for value in node.values():
                    analyze_node(value, loop_depth)

            elif isinstance(node, (list, tuple)):
                for item in node:
                    analyze_node(item, loop_depth)

        analyze_node(ast)

        memo_detector = MemoizationPatternDetector()
        memo_patterns = memo_detector.detect(ast)
        if memo_patterns:
            patterns.update(memo_patterns)
            if memo_patterns.get("memo_dimensions") and not patterns.get(
                "state_dimensions"
            ):
                patterns["state_dimensions"] = memo_patterns["memo_dimensions"]

        if call_targets:
            patterns["call_targets"] = sorted(call_targets)
        if sorting_calls:
            patterns["sorting_calls"] = sorted(sorting_calls)
        if priority_queue_calls:
            patterns["priority_queue_calls"] = sorted(priority_queue_calls)

        if patterns["recursive_constant_decrements"]:
            patterns["recursive_constant_decrements"] = sorted(
                {
                    int(value)
                    for value in patterns["recursive_constant_decrements"]
                    if value is not None
                }
            )

        if patterns["fractional_split_factors"]:
            patterns["fractional_split_factors"] = sorted(
                {
                    int(value)
                    for value in patterns["fractional_split_factors"]
                    if value not in (None, 0)
                }
            )

        patterns["estimated_branching_factor"] = max(
            patterns["estimated_branching_factor"],
            patterns["max_recursive_calls_per_function"],
        )

        if (
            patterns["has_constant_decrement_recursion"]
            and not patterns["has_fractional_split_recursion"]
            and patterns["max_recursive_calls_per_function"] >= 2
        ):
            patterns["non_linear_split_recursion"] = True

        return patterns

    def _analyze_self_calls(
        self, func: SubroutineDef, func_name: str
    ) -> Dict[str, Any]:
        """Collect recursion metrics (call distribution + argument shapes)."""

        metrics: Dict[str, Any] = {
            "call_count": 0,
            "tail_call_count": 0,
            "constant_decrements": [],
            "fractional_split_factors": [],
            "constant_decrement_calls": 0,
            "fractional_split_calls": 0,
            "argument_shapes": [],
        }

        parameters = [
            param.name
            for param in getattr(func, "parameters", [])
            if getattr(param, "name", None)
        ]
        body = func.body

        def traverse(value: Any, tail_context: bool = False) -> int:
            call_total = 0

            if isinstance(value, ReturnStmt):
                return traverse(value.value, tail_context=True)

            if isinstance(value, (CallStmt, FuncCallExpr)):
                if value.name == func_name:
                    metrics["call_count"] += 1
                    if tail_context:
                        metrics["tail_call_count"] += 1
                    arg_shape = self._classify_recursive_arguments(
                        getattr(value, "args", []), parameters, metrics
                    )
                    if arg_shape:
                        metrics["argument_shapes"].append(arg_shape)
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

    def _classify_recursive_arguments(
        self,
        args: Any,
        parameter_names: List[str],
        metrics: Dict[str, Any],
    ) -> list[str]:
        """Classify how each recursive argument transforms the input size."""

        shapes: list[str] = []
        if not args:
            return shapes

        for arg in args:
            shapes.append(
                self._categorize_recursive_argument(arg, parameter_names, metrics)
            )

        return shapes

    def _categorize_recursive_argument(
        self,
        arg: Any,
        parameter_names: list[str],
        metrics: Dict[str, Any],
    ) -> str:
        """Identify whether the recursive call divides or decrements the problem size."""

        if isinstance(arg, Var) and arg.name in parameter_names:
            return f"{arg.name}"

        if isinstance(arg, Number):
            return "constant"

        if isinstance(arg, BinOp):
            op = str(arg.op).lower()
            left_var = isinstance(arg.left, Var) and arg.left.name in parameter_names
            right_number = self._extract_number_value(arg.right)

            if left_var and op in {"-", "--"} and right_number is not None:
                decrement = abs(int(right_number))
                metrics["constant_decrements"].append(decrement)
                metrics["constant_decrement_calls"] += 1
                return f"{arg.left.name}-const"

            if left_var and op in {"/", "//", "div"} and right_number not in (0, None):
                metrics["fractional_split_factors"].append(int(right_number))
                metrics["fractional_split_calls"] += 1
                return f"{arg.left.name}/{int(right_number)}"

        return "generic"

    @staticmethod
    def _extract_number_value(value: Any) -> Optional[float]:
        """Extract numeric literal value regardless of wrapper type."""

        if isinstance(value, Number):
            try:
                return float(value.value)
            except (TypeError, ValueError):
                return None

        if isinstance(value, (int, float)):
            return float(value)

        return None
