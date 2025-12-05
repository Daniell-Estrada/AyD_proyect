"""High-level parser facade that returns fully transformed ASTs with diagnostics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from lark import LarkError

from app.domain.models.ast import Program
from app.infrastructure.parser.lark_parser import LarkParser
from app.infrastructure.parser.preprocessor import (
    ParserWarning,
    PseudocodeNormalizer,
)
from app.infrastructure.parser.transformer import ASTTransformer
from app.shared.exceptions import ParsingError
from app.shared.file_reader import FileReader


@dataclass
class ParserDiagnostics:
    """Metadata emitted after parsing to aid debugging and tooling."""

    normalized_source: str
    warnings: List[ParserWarning]


@dataclass
class ParserResult:
    """Tuple-like structure returned by :class:`LanguageParser`."""

    ast: Program
    diagnostics: ParserDiagnostics


class ILanguageParser(ABC):
    """
    Interface for the language parser.
    """

    @abstractmethod
    def parse(self, code: str) -> ParserResult:
        """
        Parse code and return the AST.
        """

    @abstractmethod
    def parse_file(self, file_path: str) -> ParserResult:
        """
        Parse a file and return the AST.
        """


class LanguageParser(ILanguageParser):
    """
    Main parser for the refactored language.
    Converts source code to Abstract Syntax Tree using Lark parser and custom transformer.
    """

    def __init__(self):
        """
        Initializes the components needed for parsing.
        """
        self._file_reader = FileReader()
        self._lark_parser = LarkParser()
        self._transformer = ASTTransformer()
        self._normalizer = PseudocodeNormalizer()

    def parse(self, code: str) -> ParserResult:
        """
        Parse code and return the AST.

        Args:
            code: Source code string to parse

        Returns:
            Program AST node containing the parsed structure

        Raises:
            ParsingError: If parsing or transformation fails
        """
        preprocessed = self._normalizer.normalize(code)

        try:
            parse_tree = self._lark_parser.parse(preprocessed.code)
            ast = self._transformer.transform(parse_tree)
            diagnostics = ParserDiagnostics(
                normalized_source=preprocessed.code,
                warnings=preprocessed.warnings,
            )
            return ParserResult(ast=ast, diagnostics=diagnostics)
        except LarkError as e:
            message = str(e)
            line = getattr(e, "line", None)
            column = getattr(e, "column", None)
            detail = f"Error parsing code at line {line}, column {column}: {message}"
            raise ParsingError(detail) from e
        except Exception as e:
            raise ParsingError(f"Error parsing code: {str(e)}") from e

    def parse_file(self, file_path: str) -> ParserResult:
        """
        Parse a file and return the AST.

        Args:
            file_path: Path to the source code file

        Returns:
            Program AST node containing the parsed structure

        Raises:
            ParsingError: If file reading or parsing fails
        """
        try:
            code = self._file_reader.read_file(file_path)
            return self.parse(code)
        except ParsingError:
            raise
        except Exception as e:
            raise ParsingError(f"Error parsing file {file_path}: {str(e)}") from e
