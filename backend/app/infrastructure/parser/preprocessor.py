"""Utility helpers to normalize and validate pseudocode before parsing.

The Lark grammar is intentionally strict to guarantee deterministic ASTs,
but real-world pseudocode tends to mix Unicode arrows, Pascal style
assignments (`:=`), or uppercase keywords.  The normalizer smooths out
those inconsistencies without mutating the program semantics so the
parser can operate on a clean source string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Optional


@dataclass
class ParserWarning:
    """Represents a non-fatal issue detected while preprocessing."""

    message: str
    line: Optional[int] = None


@dataclass
class PreprocessingResult:
    """Container returned by the pseudocode normalizer."""

    code: str
    warnings: List[ParserWarning] = field(default_factory=list)


class PseudocodeNormalizer:
    """Clean and validate raw pseudocode before it reaches the parser."""

    _ASSIGNMENT_TOKENS = ("←", "<-", ":=")

    _KEYWORD_MAP = {
        # Structural keywords
        "function": "function",
        "begin": "begin",
        "end": "end",
        "return": "return",
        "class": "class",
        "var": "var",
        "new": "new",
        "if": "if",
        "then": "then",
        "else": "else",
        "for": "for",
        "to": "to",
        "do": "do",
        "while": "while",
        "repeat": "repeat",
        "until": "until",
        # Logical operators and built-ins
        "and": "and",
        "or": "or",
        "not": "not",
        "true": "T",
        "false": "F",
        "null": "NULL",
    }

    _KEYWORD_PATTERN = re.compile(
        r"\b(" + "|".join(_KEYWORD_MAP.keys()) + r")\b",
        flags=re.IGNORECASE,
    )

    _CALL_STMT_PATTERN = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
    _CALL_PREFIX_BLACKLIST = {
        "call",
        "function",
        "if",
        "while",
        "for",
        "return",
        "else",
        "begin",
        "repeat",
        "until",
        "print",
        "length",
        "ceil",
        "floor",
        "concat",
        "substring",
        "strlen",
        "addnode",
        "addedge",
        "neighbors",
        "var",
        "class",
        "new",
    }

    _FOR_EACH_DESCRIPTOR_PATTERN = re.compile(
        r"^(?P<indent>\s*)for\s+each\s+(?P<descriptor>[A-Za-z_][A-Za-z0-9_]*)\s+"
        r"(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s+of\s+(?P<collection>.+?)\s+do\b",
        flags=re.IGNORECASE,
    )

    _FUNCTION_HEADER_PATTERN = re.compile(
        r"^\s*function\s+[A-Za-z_][A-Za-z0-9_]*\s*\((?P<params>[^)]*)\)",
        flags=re.IGNORECASE,
    )

    def normalize(self, source: str) -> PreprocessingResult:
        """Normalize whitespace, Unicode arrows, and keyword casing."""

        if not source:
            return PreprocessingResult(code="")

        warnings: List[ParserWarning] = []
        normalized_lines: List[str] = []
        current_function_params: List[str] = []

        for idx, raw_line in enumerate(source.splitlines(), start=1):
            line = raw_line.rstrip()
            line = line.replace("…", "...")
            line = line.replace("“", '"').replace("”", '"').replace("’", "'")
            line, comment_warning = self._strip_line_comment(line)
            if comment_warning:
                warnings.append(
                    ParserWarning(
                        message=comment_warning,
                        line=idx,
                    )
                )
            line = line.replace("\t", "    ")
            line = self._replace_assignment_tokens(line)
            line = self._standardize_keywords(line)
            current_function_params = self._maybe_capture_function_params(line, current_function_params)
            line, descriptor_warning = self._rewrite_for_each_descriptor(line, current_function_params)
            if descriptor_warning:
                warnings.append(ParserWarning(message=descriptor_warning, line=idx))
            line = self._ensure_call_prefix(line, idx, warnings)
            normalized_lines.append(line)

            if "<-" not in line:
                continue
            # Warn about uncommon assignment glyphs that were replaced
            if any(token in raw_line for token in self._ASSIGNMENT_TOKENS if token != "<-"):
                warnings.append(
                    ParserWarning(
                        message="Normalized assignment operator to '<-'",
                        line=idx,
                    )
                )

        normalized_source = "\n".join(normalized_lines).strip("\n") + "\n"
        if not normalized_source.strip():
            warnings.append(
                ParserWarning(
                    message="Input is empty after normalization; nothing to parse",
                    line=None,
                )
            )
        warnings.extend(self._validate_block_balance(normalized_source))
        return PreprocessingResult(code=normalized_source, warnings=warnings)

    def _replace_assignment_tokens(self, line: str) -> str:
        """Replace alternative assignment markers with the canonical '<-' token."""

        return line.replace("←", "<-").replace(":=", "<-")

    def _standardize_keywords(self, line: str) -> str:
        """Lower-case structural keywords while keeping identifiers intact."""

        def _replacer(match: re.Match[str]) -> str:
            keyword = match.group(0)
            canonical = self._KEYWORD_MAP.get(keyword.lower())
            return canonical if canonical is not None else keyword

        return self._KEYWORD_PATTERN.sub(_replacer, line)

    def _ensure_call_prefix(
        self, line: str, line_number: int, warnings: List[ParserWarning]
    ) -> str:
        """Insert explicit 'call' keyword for bare function-call statements."""

        stripped = line.lstrip()
        if not stripped or stripped.lower().startswith("call "):
            return line

        prefix = stripped.split(maxsplit=1)[0].lower()
        if prefix in self._CALL_PREFIX_BLACKLIST:
            return line

        match = self._CALL_STMT_PATTERN.match(stripped)
        if not match:
            return line

        name = match.group("name")
        if name.lower() in self._CALL_PREFIX_BLACKLIST:
            return line

        indent_length = len(line) - len(stripped)
        indent = line[:indent_length]
        updated = f"{indent}call {stripped}"

        warnings.append(
            ParserWarning(
                message=f"Se agregó la palabra clave 'call' antes de '{name}'",
                line=line_number,
            )
        )
        return updated

    def _validate_block_balance(self, source: str) -> List[ParserWarning]:
        """Emit warnings when begin/end or repeat/until blocks are unbalanced."""

        warnings: List[ParserWarning] = []

        begin_count = len(re.findall(r"\bbegin\b", source))
        end_count = len(re.findall(r"\bend\b", source))
        if begin_count != end_count:
            warnings.append(
                ParserWarning(
                    message=(
                        "Unbalanced 'begin'/'end' blocks detected "
                        f"(begin={begin_count}, end={end_count})"
                    )
                )
            )

        repeat_count = len(re.findall(r"\brepeat\b", source))
        until_count = len(re.findall(r"\buntil\b", source))
        if repeat_count != until_count:
            warnings.append(
                ParserWarning(
                    message=(
                        "Unbalanced 'repeat'/'until' blocks detected "
                        f"(repeat={repeat_count}, until={until_count})"
                    )
                )
            )

        return warnings

    def _maybe_capture_function_params(
        self, line: str, previous_params: List[str]
    ) -> List[str]:
        """Update cached function parameters when encountering a function header."""

        match = self._FUNCTION_HEADER_PATTERN.match(line)
        if not match:
            return previous_params

        params_field = match.group("params") or ""
        if not params_field.strip():
            return []

        params: List[str] = []
        for part in params_field.split(","):
            token = part.strip()
            if not token:
                continue

            token = token.split("[", 1)[0].strip()
            pieces = token.split()
            name = pieces[-1] if pieces else token
            if name:
                params.append(name)

        return params

    def _rewrite_for_each_descriptor(
        self, line: str, function_params: List[str]
    ) -> tuple[str, Optional[str]]:
        """Rewrite 'for each descriptor var of X do' into grammar-compliant syntax."""

        match = self._FOR_EACH_DESCRIPTOR_PATTERN.match(line)
        if not match:
            return line, None

        indent = match.group("indent") or ""
        descriptor = match.group("descriptor") or ""
        var_name = match.group("var") or "item"
        collection = (match.group("collection") or "").strip()

        iterable = collection
        descriptor_lower = descriptor.lower()

        if descriptor_lower in {"neighbor", "neighbour"}:
            graph_param = self._select_graph_parameter(function_params)
            if graph_param:
                iterable = f"neighbors({graph_param}, {collection})"
            else:
                iterable = f"neighbors({collection})"

        rewritten = f"{indent}for each {var_name} in {iterable} do"
        message = (
            f"Reescritura de bucle 'for each {descriptor} {var_name} of ...' a forma estándar"
        )
        return rewritten, message

    def _select_graph_parameter(self, params: List[str]) -> Optional[str]:
        """Return a graph-like parameter name when available."""

        for name in params:
            lowered = name.lower()
            if lowered in {"g", "graph"} or lowered.startswith("graph"):
                return name
        return None

    def _strip_line_comment(self, line: str) -> tuple[str, Optional[str]]:
        """Remove inline comments that start with '//' to satisfy the grammar."""

        comment_start = line.find("//")
        if comment_start == -1:
            return line, None

        stripped = line[:comment_start].rstrip()
        if stripped:
            return stripped, "Se eliminó un comentario en línea que iniciaba con '//'"
        return "", "Se eliminó un comentario en línea que iniciaba con '//'"
