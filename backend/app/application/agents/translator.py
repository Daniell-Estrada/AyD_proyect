"""
Translator Agent - Converts natural language descriptions to structured pseudocode.
Implements Single Responsibility Principle.
"""

import logging
import re
from typing import Iterable

from app.application.agents.base import BaseAgent
from app.application.agents.state import AgentState
from app.infrastructure.llm.llm_service import LLMService

logger = logging.getLogger(__name__)


RECURRENCE_KEYWORDS: Iterable[str] = (
    "árbol de recurrencia",
    "arbol de recurrencia",
    "recurrence tree",
    "resolve the recurrence",
    "resolver la recurrencia",
    "divide y vencerás",
    "divide y venceras",
    "divide-and-conquer recurrence",
    "recurrence relation",
    "recursion",
    "recursive",
    "recursivo",
    "recursiva",
)

ASSIGNMENT_EXCLUDED_PREFIXES: tuple[str, ...] = (
    "if",
    "else",
    "elseif",
    "elif",
    "while",
    "until",
    "repeat",
    "return",
    "procedure",
    "function",
    "class",
    "algorithm",
    "begin",
    "end",
    "call",
    "print",
)

FOR_ASSIGNMENT_PATTERN = re.compile(
    r"^(?P<lead>\s*for\s+(?:each\s+)?[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<tail>.+)$",
    flags=re.IGNORECASE,
)

SIMPLE_ASSIGNMENT_PATTERN = re.compile(
    r"^(?P<indent>\s*)(?P<lhs>[A-Za-z_][A-Za-z0-9_\[\]\.]*(?:\[[^\]]+\])*)\s*=\s*(?P<rhs>.+)$",
)

DECLARATION_ASSIGNMENT_PATTERN = re.compile(
    r"^(?P<indent>\s*)(?P<type>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<var>[A-Za-z_][A-Za-z0-9_\[\]\.]*(?:\[[^\]]+\])*)\s*=\s*(?P<rhs>.+)$",
)


def needs_recursion(input_text: str) -> bool:
    """Return True when the natural language request mentions recurrence analysis."""

    lowered = (input_text or "").lower()
    return any(keyword in lowered for keyword in RECURRENCE_KEYWORDS)


def pseudocode_contains_recursion(pseudocode: str) -> bool:
    """Heuristic check that looks for self-recursive calls inside the pseudocode."""

    if not pseudocode:
        return False

    function_pattern = re.compile(r"function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", flags=re.IGNORECASE)

    for match in function_pattern.finditer(pseudocode):
        name = match.group(1)
        body = pseudocode[match.end():]
        call_pattern = re.compile(rf"\b(?:call\s+)?{re.escape(name)}\s*\(", flags=re.IGNORECASE)
        if call_pattern.search(body):
            return True

    return False


class TranslatorAgent(BaseAgent):
    """
    Agent responsible for translating natural language algorithm descriptions
    into structured pseudocode following the grammar specification.
    """

    def __init__(self, llm_service: LLMService):
        """
        Initialize Translator Agent.

        Args:
            llm_service: LLM service instance
        """
        super().__init__(name="TranslatorAgent", llm_service=llm_service)
        self.llm_service = llm_service

    def execute(self, state: AgentState) -> AgentState:
        """
        Execute translation from natural language to pseudocode.

        Args:
            state: Current agent state

        Returns:
            Updated state with translated pseudocode
        """
        logger.info(f"{self.name}: Starting translation")

        user_input = state["user_input"]
        feedback = (state.get("user_feedback") or "").strip()
        previous_pseudocode = state.get("translated_pseudocode") or user_input

        if feedback:
            logger.info("%s: Applying reviewer feedback", self.name)
            try:
                recursion_required = needs_recursion(feedback) or needs_recursion(user_input)
                system_prompt = self._build_revision_system_prompt(recursion_required)
                user_prompt = self._build_revision_user_prompt(
                    previous_pseudocode,
                    feedback,
                    recursion_required,
                )

                response_text, metrics = self._invoke_llm(
                    state=state,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )

                pseudocode, reasoning = self._parse_response(response_text)

                if recursion_required and not pseudocode_contains_recursion(pseudocode):
                    logger.info(
                        "%s: Recursion requested in feedback; retrying revision with stronger emphasis",
                        self.name,
                    )
                    system_prompt = self._build_revision_system_prompt(recursion_required=True)
                    user_prompt = self._build_revision_user_prompt(
                        previous_pseudocode,
                        feedback + "\nAsegúrate de incluir una llamada recursiva explícita.",
                        enforce_recursion=True,
                    )
                    response_text, metrics = self._invoke_llm(
                        state=state,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                    pseudocode_retry, reasoning_retry = self._parse_response(response_text)
                    if pseudocode_contains_recursion(pseudocode_retry):
                        pseudocode, reasoning = pseudocode_retry, reasoning_retry

                state["translated_pseudocode"] = pseudocode
                state["translation_reasoning"] = reasoning or f"Applied reviewer feedback: {feedback}"
                state["current_stage"] = "translation_complete"
                state["user_feedback"] = None

                logger.info(
                    "%s: Revision completed. Tokens: %s",
                    self.name,
                    metrics.total_tokens,
                )

                return state

            except Exception as exc:
                logger.error("%s: Failed to apply reviewer feedback: %s", self.name, exc)
                self._append_error(state, f"Failed to apply reviewer feedback: {exc}")
                state["status"] = "failed"
                state["user_feedback"] = None
                return state

        # Skip translation if the input already resembles pseudocode
        if self._looks_like_pseudocode(user_input):
            logger.info(f"{self.name}: Input auto-detected as pseudocode, skipping translation")
            state["translated_pseudocode"] = user_input
            state["translation_reasoning"] = "Input auto-detected as pseudocode"
            state["current_stage"] = "translation_complete"
            state["user_feedback"] = None
            return state

        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(user_input)

        try:
            # Invoke LLM
            response_text, metrics = self._invoke_llm(
                state=state,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            # Parse response
            pseudocode, reasoning = self._parse_response(response_text)

            recursion_needed = needs_recursion(user_input)

            if recursion_needed and not pseudocode_contains_recursion(pseudocode):
                logger.info(
                    "%s: Recursion requested but not detected in first draft; retrying with recursion emphasis",
                    self.name,
                )

                system_prompt = self._build_system_prompt(recursion_required=True)
                user_prompt = self._build_user_prompt(user_input, enforce_recursion=True)

                response_text, metrics = self._invoke_llm(
                    state=state,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    use_fallback=True,
                )
                pseudocode_retry, reasoning_retry = self._parse_response(response_text)

                if pseudocode_contains_recursion(pseudocode_retry):
                    pseudocode, reasoning = pseudocode_retry, reasoning_retry
                else:
                    logger.warning(
                        "%s: Recursion still missing after retry; returning latest draft",
                        self.name,
                    )

            # Update state
            state["translated_pseudocode"] = pseudocode
            state["translation_reasoning"] = reasoning
            state["current_stage"] = "translation_complete"
            state["user_feedback"] = None

            logger.info(
                f"{self.name}: Translation completed successfully. "
                f"Cost: ${metrics.estimated_cost_usd:.4f}, "
                f"Tokens: {metrics.total_tokens}"
            )

        except Exception as e:
            logger.error(f"{self.name}: Translation failed: {e}")
            self._append_error(state, f"Translation failed: {str(e)}")
            state["status"] = "failed"
            state["user_feedback"] = None
            if not state.get("translation_retry_performed"):
                state["auto_retry_agent"] = True
                state["retry_reason"] = "translator_llm_error"
                state["translation_retry_performed"] = True

        return state

    def _looks_like_pseudocode(self, text: str) -> bool:
        """Heuristically determine if the input is already pseudocode."""

        stripped = text.strip()
        if not stripped:
            return False

        lowered = stripped.lower()

        # Quick rejection if the text is a single short sentence
        if len(stripped.split()) < 6 and "function" not in lowered:
            return False

        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if not lines:
            return False

        code_indicators = (
            "function",
            "procedure",
            "algorithm",
            "return",
            "for ",
            "while",
            "repeat",
            "until",
            "if",
            "else",
            "begin",
            "end",
            "<-",
            ":=",
            "var ",
            "class ",
        )

        indicator_hits = sum(1 for token in code_indicators if token in lowered)

        # Structural cues: multiple lines, indentation, or numbered steps
        has_structural_cues = any(
            line.startswith(("function", "procedure", "class", "for", "while", "if"))
            or "<-" in line
            or re.search(r"^[0-9]+\.\s", line)
            for line in lines
        )

        # Presence of code-like punctuation
        has_code_symbols = any(symbol in stripped for symbol in ("<-", ":=", "{}", "()", "[]"))

        long_enough = len(lines) >= 2 or len(stripped) > 120

        return (indicator_hits >= 3 and long_enough) or (has_structural_cues and (indicator_hits >= 2 or has_code_symbols))

    def _build_system_prompt(self, recursion_required: bool = False) -> str:
        """Build system prompt defining the translator's role."""
        base_prompt = """You are an expert algorithm translator. Your task is to convert natural language algorithm descriptions into structured pseudocode following a strict grammar.

**Grammar Rules (must follow exactly):**
- Use `function name(parameters)` for function definitions, followed by a `begin ... end` code block.
- Every control structure must wrap its body in `begin ... end`, even if the body is one line:
  - `if (condition) then` **begin ... end**, and if an `else` exists it must be `else` **begin ... end**.
  - `for ... do` **begin ... end**.
  - `while ... do` **begin ... end**.
  - `repeat` **begin ... end** `until (condition)`.
- Use `<-` for assignment (e.g., `x <- 5`).
- Use `for var <- start to end do` for ranged loops. You may also use `for each ... in ... do` as supported by the grammar; wrap every loop body in `begin ... end`.
- Use `while (condition) do` for while loops.
- Use `if (condition) then ... else ...` for conditionals, with the branch code blocks as noted above.
- Use `return expression` for return statements.
- Use `var name` for variable declarations before clause begin blocks of functions (e.g., `var count` or `var count, list[1..n], etc.`).
- Use `A[1..n]` for array declarations.
- Use `class Name { field1 field2 }` for class definitions.
- Use `Graph G` for graph declarations and `addNode(G, node)`, `addEdge(G, from, to)` for graph operations.
- Avoid dotted array indexing that mixes dots and brackets (e.g., `G.adj[i][j]` or `obj.field[k]`), because the grammar does not allow brackets after a dotted field. Use either pure bracketed arrays (`adj[i][j]`) or dotted field access without brackets (`graph.adj`).
- Use only ASCII operators: `!=` instead of `≠`, and represent infinity with a large constant (e.g., `INF <- 10^9`).
- Do not use reserved keywords as identifiers (e.g., avoid `end` as a variable name); prefer neutral names like `target`, `dest`, etc.
- Do not nest function definitions; declare any helper functions at the top level, separate from the main function.
- Do **not** emit Markdown fences or commentary inside the pseudocode. No `//` comments.
- Never emit dotted indexing that precedes a bracket (e.g., no `G.adj[u][k]` or `obj.field[i]`). If you need nested arrays, declare them as arrays directly (`adj[1..n][1..m]`) and index them without dots.
- When iterating adjacency, use explicit lengths and indices (e.g., `deg <- len(adj[u])`; `for k <- 1 to deg do` followed by `neighbor <- adj[u][k]`).
- Multidimensional arrays are allowed (e.g., `matrix[1..n][1..m]`); compute scalar bounds first (e.g., `n <- len(nodes)`, `m <- maxDegree`) and use those scalars in declarations (`adj[1..n][1..m]`).
- Avoid dotted expressions inside array bounds; compute scalar bounds before declaring arrays.

**Output Format (exact):**
1. Pseudocode only.
2. `---REASONING---` on its own line.
3. Brief reasoning.

**Example (shows required blocks):**
```
function findMax(A[1..n])
begin
    var max
    max <- A[1]
    for i <- 2 to n do
    begin
        if (A[i] > max) then
        begin
            max <- A[i]
        end
    end
    return max
end
```
---REASONING---
Translated as an iterative algorithm with a single loop to traverse the array. Used array indexing A[i] and comparison to find maximum value.
"""

        if recursion_required:
            base_prompt += """

**Additional Requirements:**
- The pseudocode must explicitly model the recurrence using recursive function calls.
- Do not replace the recursive expansion with iterative loops or simulations of the tree.
- When a recurrence tree is requested, ensure there is at least one function that calls itself (directly or via helper) to represent the recursive structure.
"""

        return base_prompt

    def _build_user_prompt(self, user_input: str, enforce_recursion: bool = False) -> str:
        """Build user prompt with the algorithm description."""
        prompt = f"""Translate the following algorithm description into structured pseudocode:

{user_input}

Hard requirements:
- Use `begin`/`end` blocks for every control structure body (both THEN and ELSE blocks, every loop body, repeat/until body).
- No Markdown fences, no extra commentary inside the pseudocode.
- Do not mix dots and brackets (no `G.adj[u][k]` or `obj.field[i]`). If you need nested arrays, declare them directly (`adj[1..n][1..m]`) and index without dots; dotted field access without brackets is allowed (`graph.adj`).
- Multidimensional arrays are allowed (e.g., `matrix[1..n][1..m]`). Prefer scalar bounds computed first (e.g., `n <- len(nodes)`, `m <- maxDegree`) and then declare (`adj[1..n][1..m]`).
- Avoid dotted expressions inside array bounds; compute scalar bounds before declaring arrays. When data has multiple fields, you may use parallel arrays (`edgeTo[k]`, `edgeW[k]`) instead of dotted field access if needed for clarity.

Return the pseudocode, then `---REASONING---`, then a brief explanation."""

        if enforce_recursion:
            prompt += """

Additional constraint: The pseudocode must use recursion to build the recurrence tree. Avoid while/for loops that merely iterate over tree levels; instead, express each recursive level as separate function calls following the recurrence relation.
"""

        return prompt

    def _build_revision_system_prompt(self, recursion_required: bool = False) -> str:
        """Prompt used when applying reviewer feedback to existing pseudocode."""
        base_prompt = """You are an expert pseudocode editor. Update existing pseudocode to satisfy reviewer feedback while preserving correctness.\n\nRules:\n- Maintain the provided pseudocode grammar (function declarations, begin/end blocks, <- assignments, loop formats).\n- Apply the requested changes faithfully, even if they require restructuring the algorithm.\n- Do not ignore the feedback.\n- Keep the output as clean pseudocode without additional commentary.\n\nOutput the revised pseudocode first, then the reasoning separated by ---REASONING---."""

        if recursion_required:
            base_prompt += "\n\nAdditional requirement: ensure the implementation uses explicit recursion to model the behaviour requested by the reviewer."

        return base_prompt

    def _build_revision_user_prompt(
        self,
        original_pseudocode: str,
        feedback: str,
        enforce_recursion: bool = False,
    ) -> str:
        """Construct user prompt for revision requests."""
        prompt = (
            "You are provided with existing pseudocode and reviewer feedback.\n"
            "Update the pseudocode to satisfy the feedback.\n\n"
            "Current pseudocode:\n"
            f"{original_pseudocode}\n\n"
            "Reviewer feedback:\n"
            f"{feedback}\n\n"
            "Return only the revised pseudocode following the standard grammar,"
            " then include ---REASONING--- and a short explanation of the changes."
        )

        if enforce_recursion:
            prompt += "\n\nReminder: Incorporate recursion explicitly if it is not already present."

        return prompt

    def _parse_response(self, response_text: str) -> tuple[str, str]:
        """
        Parse LLM response into pseudocode and reasoning.

        Args:
            response_text: Raw LLM response

        Returns:
            Tuple of (pseudocode, reasoning)
        """
        if "---REASONING---" in response_text:
            parts = response_text.split("---REASONING---", 1)
            pseudocode = parts[0].strip()
            reasoning = parts[1].strip()
        else:
            pseudocode = response_text.strip()
            reasoning = "No explicit reasoning provided"

        # Clean pseudocode (remove markdown code fences if present)
        if pseudocode.startswith("```"):
            lines = pseudocode.split("\n")
            # Remove first and last line (code fences)
            pseudocode = "\n".join(lines[1:-1]).strip()

        pseudocode = self._strip_line_comments(pseudocode)
        pseudocode = self._normalize_assignments(pseudocode)

        return pseudocode, reasoning

    def _strip_line_comments(self, code: str) -> str:
        """Remove inline // comments that violate the grammar before parsing."""
        cleaned_lines: list[str] = []
        for line in code.splitlines():
            comment_idx = line.find("//")
            if comment_idx != -1:
                line = line[:comment_idx]
            cleaned_lines.append(line.rstrip())
        return "\n".join(cleaned_lines).strip()

    def _normalize_assignments(self, code: str) -> str:
        """Convert single equals used for assignment into <- to satisfy the grammar."""
        normalized_lines: list[str] = []
        for raw_line in code.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                normalized_lines.append("")
                continue

            for_match = FOR_ASSIGNMENT_PATTERN.match(line)
            if for_match:
                normalized_lines.append(
                    f"{for_match.group('lead')} <- {for_match.group('tail').lstrip()}"
                )
                continue

            normalized_lines.append(self._normalize_assignment_line(line))

        return "\n".join(normalized_lines).strip()

    def _normalize_assignment_line(self, line: str) -> str:
        stripped = line.lstrip()
        if not stripped:
            return line

        first_word = stripped.split(None, 1)[0].lower()
        if first_word in ASSIGNMENT_EXCLUDED_PREFIXES:
            return line

        match = SIMPLE_ASSIGNMENT_PATTERN.match(line)
        if match:
            lhs = match.group("lhs")
            if lhs.lower() in ASSIGNMENT_EXCLUDED_PREFIXES:
                return line
            rhs = match.group("rhs").strip()
            return f"{match.group('indent')}{lhs} <- {rhs}"

        match = DECLARATION_ASSIGNMENT_PATTERN.match(line)
        if match:
            type_word = match.group("type")
            if type_word.lower() in ASSIGNMENT_EXCLUDED_PREFIXES:
                return line
            rhs = match.group("rhs").strip()
            return f"{match.group('indent')}{type_word} {match.group('var')} <- {rhs}"

        return line
