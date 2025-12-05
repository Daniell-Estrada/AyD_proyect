"""
Utilities to build recursion trees and export them as Mermaid diagrams.
Separated from the main complexity service for clarity and reuse.
"""

import logging
from typing import Tuple

import networkx as nx

logger = logging.getLogger(__name__)


class RecursionTreeBuilder:
    """Build recursion trees for divide-and-conquer visualizations."""

    def build_recursion_tree(
        self,
        func_name: str,
        initial_input: str = "n",
        branching_factor: int = 2,
        depth: int = 3,
        size_reduction: str = "n/2",
    ) -> Tuple[nx.DiGraph, str]:
        G = nx.DiGraph()

        def add_node_recursive(node_id: str, size: str, current_depth: int):
            """Recursively add labeled nodes to the recursion tree."""
            if current_depth > depth:
                return

            label = f"{func_name}({size})"
            G.add_node(node_id, label=label, depth=current_depth, size=size)

            if current_depth == depth:
                return

            for i in range(branching_factor):
                child_id = f"{node_id}_{i}"
                child_size = self._reduce_size(size, size_reduction)
                add_node_recursive(child_id, child_size, current_depth + 1)
                G.add_edge(node_id, child_id)

        add_node_recursive("root", initial_input, 0)
        mermaid_diagram = self._graph_to_mermaid(G, func_name)
        return G, mermaid_diagram

    def _reduce_size(self, size: str, reduction: str) -> str:
        """Apply size reduction expression to derive child problem size."""
        if reduction == "n/2":
            if size == "n":
                return "n/2"
            if "/" in size:
                try:
                    parts = size.split("/")
                    numerator = parts[0]
                    denom = int(parts[1]) * 2
                    return f"{numerator}/{denom}"
                except (ValueError, IndexError):
                    return f"({size})/2"
            return f"({size})/2"

        if reduction == "n-1":
            if size == "n":
                return "n-1"
            if "-" in size:
                return f"({size})-1"
            return f"{size}-1"

        if reduction.startswith("n/"):
            try:
                divisor = int(reduction.split("/")[1])
                if size == "n":
                    return f"n/{divisor}"
                return f"({size})/{divisor}"
            except (ValueError, IndexError):
                pass

        return f"({size}) reduced"

    def _graph_to_mermaid(self, G: nx.DiGraph, func_name: str = "T") -> str:
        """Convert a recursion tree graph to Mermaid diagram syntax."""
        lines = ["graph TD"]

        for node_id, data in G.nodes(data=True):
            label = data.get("label", f"{func_name}(?)")
            safe_label = label.replace('"', "&quot;")
            lines.append(f'    {node_id}["{safe_label}"]')

        for parent, child in G.edges():
            lines.append(f"    {parent} --> {child}")

        lines.append(
            "    classDef default fill:#e1f5ff,stroke:#0066cc,stroke-width:2px"
        )
        return "\n".join(lines)
