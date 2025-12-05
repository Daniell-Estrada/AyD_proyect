"""
Diagram Generation Service for creating Mermaid diagrams.
Supports flowcharts, recursion trees, and architecture diagrams.
"""

import logging
from typing import Any, Dict, List, Optional

import networkx as nx

from app.domain.models.ast import (
    Assignment,
    ForEachLoop,
    ForLoop,
    IfElse,
    Program,
    ReturnStmt,
    SubroutineDef,
    WhileLoop,
)

logger = logging.getLogger(__name__)


class DiagramService:
    """
    Service for generating Mermaid diagrams from various sources.
    """

    def __init__(self):
        """Initialize diagram service."""
        self.node_counter = 0

    def reset_counter(self):
        """Reset node counter for new diagram."""
        self.node_counter = 0

    def _get_node_id(self) -> str:
        """Generate unique node ID."""
        self.node_counter += 1
        return f"node{self.node_counter}"

    def _register_node(
        self,
        graph: nx.DiGraph,
        label: str,
        shape: str = "process",
        node_id: Optional[str] = None,
    ) -> str:
        """Add a node with sanitized label/shape metadata and return its id."""

        node_id = node_id or self._get_node_id()
        graph.add_node(
            node_id,
            label=self._sanitize_label(label),
            shape=shape,
        )
        return node_id

    def _add_edge(
        self,
        graph: nx.DiGraph,
        source: str,
        target: str,
        label: Optional[str] = None,
    ) -> None:
        """Connect two nodes with optional edge label."""

        if label:
            graph.add_edge(source, target, label=self._sanitize_label(label))
        else:
            graph.add_edge(source, target)

    def _sanitize_label(self, label: Any) -> str:
        """Normalize diagram labels to ASCII-friendly text."""

        if label is None:
            return ""

        text = str(label).replace("\n", " ").strip()
        return text.replace('"', "'")

    def _format_mermaid_node(self, node_id: Any, label: str, shape: str) -> str:
        """Render a node line according to shape metadata."""

        if shape == "terminal":
            return f'    {node_id}(["{label}"])'
        if shape == "decision":
            return f"    {node_id}{{{{{label}}}}}"
        return f'    {node_id}["{label}"]'

    def _expr_to_label(self, expr: Any) -> str:
        """Best-effort conversion of AST fragments into human labels."""

        if hasattr(expr, "name"):
            return getattr(expr, "name")
        if hasattr(expr, "value"):
            return str(getattr(expr, "value"))
        return str(expr)

    # ===== Flowchart Generation =====

    def generate_flowchart(self, ast: Program) -> str:
        """
        Generate flowchart from AST.

        Args:
            ast: Program AST

        Returns:
            Mermaid flowchart string
        """
        self.reset_counter()
        graph = nx.DiGraph()

        start_id = self._register_node(graph, "Start", shape="terminal")
        prev_id = start_id

        for stmt in ast.statements:
            prev_id = self._process_statement_for_flowchart(stmt, prev_id, graph)

        end_id = self._register_node(graph, "End", shape="terminal")
        self._add_edge(graph, prev_id, end_id)

        return self.networkx_to_mermaid(graph, graph_type="flowchart TD")

    def _process_statement_for_flowchart(
        self,
        stmt: Any,
        prev_id: str,
        graph: nx.DiGraph,
        edge_label: Optional[str] = None,
    ) -> str:
        """Process single statement for flowchart."""

        if isinstance(stmt, Assignment):
            target = getattr(stmt.target, "name", "var")
            node_id = self._register_node(graph, f"{target} ← value")
            self._add_edge(graph, prev_id, node_id, edge_label)
            return node_id

        elif isinstance(stmt, ForEachLoop):
            collection_label = getattr(stmt.collection, "name", str(stmt.collection))
            condition_id = self._register_node(
                graph,
                f"for {stmt.var} in {collection_label}",
                shape="decision",
            )
            self._add_edge(graph, prev_id, condition_id, edge_label)

            if stmt.body:
                current_id = condition_id
                first = True
                for body_stmt in stmt.body:
                    current_id = self._process_statement_for_flowchart(
                        body_stmt,
                        current_id,
                        graph,
                        edge_label="iterate" if first else None,
                    )
                    first = False
            else:
                current_id = self._register_node(
                    graph, f"{stmt.var} body", shape="process"
                )
                self._add_edge(graph, condition_id, current_id, "iterate")

            self._add_edge(graph, current_id, condition_id)
            after_loop_id = self._register_node(
                graph, f"after {stmt.var} loop", shape="process"
            )
            self._add_edge(graph, condition_id, after_loop_id, "done")

            return after_loop_id

        elif isinstance(stmt, ForLoop):
            range_label = (
                f"{stmt.var} = {self._expr_to_label(stmt.start)} "
                f"to {self._expr_to_label(stmt.end)}"
            )
            condition_id = self._register_node(
                graph, range_label, shape="decision"
            )
            self._add_edge(graph, prev_id, condition_id, edge_label)

            if stmt.body:
                current_id = condition_id
                first = True
                for body_stmt in stmt.body:
                    current_id = self._process_statement_for_flowchart(
                        body_stmt,
                        current_id,
                        graph,
                        edge_label="iterate" if first else None,
                    )
                    first = False
            else:
                current_id = self._register_node(
                    graph, f"{stmt.var} loop body"
                )
                self._add_edge(graph, condition_id, current_id, "iterate")

            self._add_edge(graph, current_id, condition_id)
            after_loop_id = self._register_node(
                graph, f"after {stmt.var} loop", shape="process"
            )
            self._add_edge(graph, condition_id, after_loop_id, "done")

            return after_loop_id

        elif isinstance(stmt, WhileLoop):
            condition_id = self._register_node(
                graph, "condition?", shape="decision"
            )
            self._add_edge(graph, prev_id, condition_id, edge_label)

            if stmt.body:
                current_id = condition_id
                first = True
                for body_stmt in stmt.body:
                    current_id = self._process_statement_for_flowchart(
                        body_stmt,
                        current_id,
                        graph,
                        edge_label="true" if first else None,
                    )
                    first = False
            else:
                current_id = self._register_node(graph, "loop body")
                self._add_edge(graph, condition_id, current_id, "true")

            self._add_edge(graph, current_id, condition_id)
            after_loop_id = self._register_node(graph, "after while", shape="process")
            self._add_edge(graph, condition_id, after_loop_id, "false")

            return after_loop_id

        elif isinstance(stmt, IfElse):
            condition_id = self._register_node(
                graph, "condition?", shape="decision"
            )
            self._add_edge(graph, prev_id, condition_id, edge_label)
            merge_id = self._register_node(graph, "merge")

            if stmt.then_branch:
                then_current = condition_id
                first = True
                for then_stmt in stmt.then_branch:
                    then_current = self._process_statement_for_flowchart(
                        then_stmt,
                        then_current,
                        graph,
                        edge_label="true" if first else None,
                    )
                    first = False
                self._add_edge(graph, then_current, merge_id)
            else:
                self._add_edge(graph, condition_id, merge_id, "true")

            if stmt.else_branch:
                else_current = condition_id
                first = True
                for else_stmt in stmt.else_branch:
                    else_current = self._process_statement_for_flowchart(
                        else_stmt,
                        else_current,
                        graph,
                        edge_label="false" if first else None,
                    )
                    first = False
                self._add_edge(graph, else_current, merge_id)
            else:
                self._add_edge(graph, condition_id, merge_id, "false")

            return merge_id

        elif isinstance(stmt, ReturnStmt):
            node_id = self._register_node(graph, "return")
            self._add_edge(graph, prev_id, node_id, edge_label)
            return node_id

        else:
            # Generic statement
            stmt_type = type(stmt).__name__
            node_id = self._register_node(graph, stmt_type)
            self._add_edge(graph, prev_id, node_id, edge_label)
            return node_id

    # ===== Recursion Tree =====

    def generate_recursion_tree(
        self,
        func_name: str,
        branching_factor: int = 2,
        depth: int = 3,
        size_reduction: str = "n/2",
    ) -> str:
        """
        Generate recursion tree diagram.

        Args:
            func_name: Function name
            branching_factor: Number of recursive calls
            depth: Tree depth
            size_reduction: How input size reduces (e.g., "n/2", "n-1")

        Returns:
            Mermaid tree diagram string
        """
        self.reset_counter()
        graph = nx.DiGraph()

        def add_node(size: str, current_depth: int) -> str:
            node_label = f"{func_name}({size})"
            node_id = self._register_node(graph, node_label)

            if current_depth >= depth:
                return node_id

            for i in range(branching_factor):
                child_size = self._reduce_size(size, size_reduction)
                child_id = add_node(child_size, current_depth + 1)
                edge_label = f"branch {i + 1}" if branching_factor > 1 else None
                self._add_edge(graph, node_id, child_id, edge_label)

            return node_id

        add_node("n", 0)

        return self.networkx_to_mermaid(graph, graph_type="graph TD")

    def _reduce_size(self, size: str, reduction: str) -> str:
        """Apply size reduction formula."""
        if reduction == "n/2":
            if size == "n":
                return "n/2"
            elif "/" in size:
                # n/2 -> n/4
                parts = size.split("/")
                return f"{parts[0]}/{int(parts[1])*2}"
            else:
                return f"{size}/2"
        elif reduction == "n-1":
            if size == "n":
                return "n-1"
            else:
                return f"({size})-1"
        else:
            return size

    # ===== Call Graph =====

    def generate_call_graph(self, ast: Program) -> str:
        """
        Generate function call graph.

        Args:
            ast: Program AST

        Returns:
            Mermaid graph string
        """
        lines = ["graph LR"]

        # Extract functions
        functions = [
            stmt for stmt in ast.statements if isinstance(stmt, SubroutineDef)
        ]

        for func in functions:
            func_name = func.name or "anonymous"

            # Find function calls in body
            called_funcs = self._extract_function_calls(func.body)

            for called in called_funcs:
                lines.append(f'    {func_name} --> {called}')

        return "\n".join(lines)

    def _extract_function_calls(self, body: List[Any]) -> List[str]:
        """Extract function names called in body."""
        from app.domain.models.ast import CallStmt, FuncCallExpr

        calls = []

        for stmt in body:
            if isinstance(stmt, (CallStmt, FuncCallExpr)):
                calls.append(stmt.name)
            elif hasattr(stmt, "body"):
                calls.extend(self._extract_function_calls(stmt.body))

        return calls

    # ===== Architecture Diagram =====

    def generate_architecture_diagram(
        self, components: List[Dict[str, Any]]
    ) -> str:
        """
        Generate system architecture diagram.

        Args:
            components: List of component definitions with dependencies

        Returns:
            Mermaid C4 or flowchart diagram string
        """
        lines = ["graph TB"]

        for comp in components:
            comp_id = comp["id"]
            comp_name = comp["name"]
            comp_type = comp.get("type", "component")

            # Different shapes for different types
            if comp_type == "agent":
                lines.append(f'    {comp_id}[("{comp_name}")]')
            elif comp_type == "service":
                lines.append(f'    {comp_id}["{comp_name}"]')
            elif comp_type == "database":
                lines.append(f'    {comp_id}[("{comp_name}")]')
            else:
                lines.append(f'    {comp_id}["{comp_name}"]')

            # Add dependencies
            for dep in comp.get("depends_on", []):
                lines.append(f'    {comp_id} --> {dep}')

        return "\n".join(lines)

    # ===== Complexity Visualization =====

    def generate_complexity_comparison(
        self, complexities: Dict[str, str]
    ) -> str:
        """
        Generate visual comparison of complexities.

        Args:
            complexities: Dict mapping case names to complexity notations

        Returns:
            Mermaid diagram string
        """
        lines = ["graph LR"]
        lines.append(f'    Algorithm["Algorithm"]')

        for case, complexity in complexities.items():
            case_id = case.replace(" ", "_")
            lines.append(f'    {case_id}["{case}: {complexity}"]')
            lines.append(f'    Algorithm --> {case_id}')

        return "\n".join(lines)

    # ===== NetworkX to Mermaid =====

    def networkx_to_mermaid(
        self, G: nx.DiGraph, graph_type: str = "graph TD"
    ) -> str:
        """
        Convert NetworkX graph to Mermaid diagram.

        Args:
            G: NetworkX directed graph
            graph_type: Mermaid graph type (e.g., "graph TD", "flowchart LR")

        Returns:
            Mermaid diagram string
        """
        lines = [graph_type]

        for node, data in G.nodes(data=True):
            label = data.get("label", str(node))
            shape = data.get("shape", "process")
            lines.append(self._format_mermaid_node(node, label, shape))

        for u, v, edge_data in G.edges(data=True):
            edge_label = edge_data.get("label")
            if edge_label:
                lines.append(f'    {u} -->|{edge_label}| {v}')
            else:
                lines.append(f'    {u} --> {v}')

        return "\n".join(lines)
