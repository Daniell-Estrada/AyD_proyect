"""
Module that implements a concrete parser using Lark.
"""

import os

from lark import Lark


class LarkParser:
    """
    Concrete parser implementation using Lark.
    """

    def __init__(self):
        self._grammar_path = os.path.join(os.path.dirname(__file__), "grammar.lark")
        self._parser = self._initialize_parser()

    def _initialize_parser(self) -> Lark:
        """
        Initialize the Lark parser with LALR algorithm.
        Enables position tracking for better error messages and AST metadata.
        """
        return Lark.open(
            self._grammar_path,
            rel_to=__file__,
            parser="lalr",
            propagate_positions=True,
            maybe_placeholders=False,
        )

    def parse(self, code: str):
        """
        Parse the code and return the parse tree.
        """
        return self._parser.parse(code)
