"""
Classifier Agent for Algorithm Paradigm Classification with Dual Approach
This module defines the ClassifierAgent which classifies algorithmic paradigms
using both deterministic methods based on AST pattern analysis and LLM-based
heuristic validation. The agent integrates with ComplexityAnalysisService for
deterministic classification and utilizes LLMService for probabilistic validation.
"""

import json
import logging
import re
from typing import Any, Dict

from app.application.agents.base import BaseAgent
from app.application.agents.state import AgentState
from app.domain.services.complexity_service import ComplexityAnalysisService
from app.infrastructure.llm.llm_service import LLMService

logger = logging.getLogger(__name__)


class ClassifierAgent(BaseAgent):
    """
    Agent responsible for classifying algorithmic paradigm using dual approach.
    """

    def __init__(
        self,
        llm_service: LLMService,
        complexity_service: ComplexityAnalysisService = None,
    ):
        super().__init__(name="ClassifierAgent", llm_service=llm_service)
        self.llm_service = llm_service
        self.complexity_service = complexity_service or ComplexityAnalysisService()

    def execute(self, state: AgentState) -> AgentState:
        """
        Execute paradigm classification using dual approach.
        """
        logger.info(f"{self.name}: Starting dual classification (deterministic + LLM)")

        ast_dict = state.get("parsed_ast")
        patterns = state.get("metadata", {}).get("patterns", {})

        if not ast_dict:
            logger.error(f"{self.name}: No AST available for classification")
            errors = state.get("errors", [])
            errors.append("No AST available for classification")
            state["errors"] = errors
            return state

        deterministic_paradigm = self._classify_deterministically(ast_dict, patterns)

        heuristic_paradigm = self._heuristic_classification(patterns)

        is_retry = bool(state.get("classification_retry_performed"))
        use_llm_validation = bool(self.llm_service) and not state.get(
            "disable_llm_validation", False
        )

        if use_llm_validation:
            system_prompt = (
                self._build_strict_system_prompt()
                if is_retry
                else self._build_system_prompt_with_guidance()
            )
            user_prompt = self._build_user_prompt_with_deterministic_guidance(
                ast_dict=ast_dict,
                patterns=patterns,
                deterministic_paradigm=deterministic_paradigm,
                heuristic_paradigm=heuristic_paradigm,
            )

            try:
                response_text, metrics = self._invoke_llm(
                    state=state,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    use_fallback=is_retry,
                )

                llm_paradigm, confidence, reasoning, parsed_cleanly = (
                    self._parse_response(response_text)
                )

                final_paradigm, final_confidence, final_reasoning = (
                    self._resolve_classification(
                        deterministic=deterministic_paradigm,
                        heuristic=heuristic_paradigm,
                        llm=llm_paradigm,
                        llm_confidence=confidence,
                        llm_reasoning=reasoning,
                        patterns=patterns,
                    )
                )

                (
                    final_paradigm,
                    final_confidence,
                    final_reasoning,
                ) = self._enforce_consistency(
                    patterns=patterns,
                    deterministic=deterministic_paradigm,
                    heuristic=heuristic_paradigm,
                    paradigm=final_paradigm,
                    confidence=final_confidence,
                    reasoning=final_reasoning,
                )

                if (not parsed_cleanly) and (
                    not state.get("classification_retry_performed")
                ):
                    state["auto_retry_agent"] = True
                    state["retry_reason"] = "classifier_parse_error"
                    state["classification_retry_performed"] = True

            except Exception as e:
                logger.warning(
                    f"LLM validation failed, using deterministic result: {e}"
                )
                final_paradigm = deterministic_paradigm or heuristic_paradigm
                final_confidence = 0.5
                final_reasoning = (
                    f"LLM failed: {e}. Using deterministic classification ({deterministic_paradigm}) "
                    f"with heuristic backup ({heuristic_paradigm})."
                )
                self._append_error(state, f"Classification failed: {e}")
        else:
            final_paradigm = deterministic_paradigm
            final_confidence = 0.9
            final_reasoning = f"Deterministic classification based on AST patterns: {deterministic_paradigm}"

        (
            final_paradigm,
            final_confidence,
            final_reasoning,
        ) = self._enforce_consistency(
            patterns=patterns,
            deterministic=deterministic_paradigm,
            heuristic=heuristic_paradigm,
            paradigm=final_paradigm,
            confidence=final_confidence,
            reasoning=final_reasoning,
        )

        state["paradigm"] = final_paradigm
        state["paradigm_confidence"] = final_confidence
        state["paradigm_reasoning"] = final_reasoning
        state["paradigm_deterministic"] = deterministic_paradigm
        state["paradigm_heuristic"] = heuristic_paradigm
        state["current_stage"] = "classification_complete"

        logger.info(
            f"{self.name}: Classification completed. "
            f"Final: {final_paradigm} (deterministic: {deterministic_paradigm}), "
            f"Confidence: {final_confidence:.2f}"
        )

        return state

    def _classify_deterministically(
        self, ast_dict: Dict[str, Any], patterns: Dict[str, Any]
    ) -> str:
        """
        Perform deterministic classification using ComplexityAnalysisService.
        """
        try:

            paradigm = self.complexity_service.detect_paradigm_from_patterns(patterns)

            logger.info(f"{self.name}: Deterministic classification result: {paradigm}")
            return paradigm

        except Exception as e:
            logger.warning(f"Deterministic classification failed: {e}")
            return "unknown"

    def _needs_llm_validation(
        self, deterministic_paradigm: str, patterns: Dict[str, Any]
    ) -> bool:
        """
        Determine if LLM validation is needed based on deterministic result.
        """
        if deterministic_paradigm == "unknown":
            return True

        has_multiple_patterns = (
            sum(
                [
                    patterns.get("has_recursion", False),
                    patterns.get("has_memoization", False),
                    patterns.get("has_greedy_pattern", False),
                    patterns.get("has_backtracking", False),
                ]
            )
            >= 2
        )

        return has_multiple_patterns

    def _resolve_classification(
        self,
        deterministic: str,
        heuristic: str,
        llm: str,
        llm_confidence: float,
        llm_reasoning: str,
        patterns: Dict[str, Any],
    ) -> tuple[str, float, str]:
        """
        Resolve conflicts between deterministic, heuristic, and LLM classifications.

        """
        if deterministic == llm:
            return (
                deterministic,
                min(llm_confidence + 0.1, 1.0),
                (f"Deterministic and LLM agree: {deterministic}. {llm_reasoning}"),
            )

        if llm not in {None, "", "unknown"} and llm_confidence >= 0.7:
            return (
                llm,
                llm_confidence,
                (
                    f"LLM classification chosen due to high confidence ({llm_confidence:.2f}) "
                    f"even though deterministic suggested {deterministic}. {llm_reasoning}"
                ),
            )

        if deterministic != "unknown":
            if llm_confidence < 0.7:
                return (
                    deterministic,
                    0.85,
                    (
                        f"Deterministic classification: {deterministic}. "
                        f"LLM suggested {llm} with low confidence ({llm_confidence:.2f})"
                    ),
                )
            else:
                return (
                    deterministic,
                    0.75,
                    (
                        f"Deterministic: {deterministic} (chosen). "
                        f"LLM suggested: {llm} ({llm_confidence:.2f}). "
                        f"LLM reasoning: {llm_reasoning}"
                    ),
                )

        if llm_confidence >= 0.6:
            return llm, llm_confidence, llm_reasoning

        return (
            heuristic,
            0.5,
            (
                f"Heuristic fallback: {heuristic}. "
                f"Deterministic: {deterministic}, LLM: {llm} (conf: {llm_confidence:.2f})"
            ),
        )

    def _build_system_prompt_with_guidance(self) -> str:
        """
        Build system prompt for LLM that includes deterministic guidance context.
        """
        return """You are an expert algorithm paradigm classifier.

        **CRITICAL**: You will receive DETERMINISTIC CLASSIFICATION RESULTS as guidance.
        These results are based on AST pattern analysis and are NOT absolute truth.

        Your task:
        1. Review the deterministic classification provided
        2. Analyze the AST and patterns independently
        3. Validate or critique the deterministic result
        4. Provide your classification with confidence and reasoning

        Classify algorithms into paradigms:
        - divide_and_conquer
        - branching_recursion
        - dynamic_programming
        - greedy
        - backtracking
        - iterative
        - recursion
        - simple

        **Output Format (JSON):**
        ```json
        {
          "paradigm": "paradigm_name",
          "confidence": 0.0-1.0,
          "reasoning": "Detailed explanation",
          "deterministic_agreement": "agree|disagree|uncertain",
          "deterministic_critique": "Assessment of baseline classification"
        }
        ```

        Be rigorous and indicate your confidence level clearly."""

    def _build_user_prompt_with_deterministic_guidance(
        self,
        ast_dict: Dict[str, Any],
        patterns: Dict[str, Any],
        deterministic_paradigm: str,
        heuristic_paradigm: str,
    ) -> str:
        """
        Build user prompt with AST data and deterministic guidance.
        """
        return f"""Classify the algorithmic paradigm for this algorithm:

        **AST:**
        ```json
        {json.dumps(ast_dict, indent=2)}
        ```

        **Detected Patterns:**
        ```json
        {json.dumps(patterns, indent=2)}
        ```

        **DETERMINISTIC CLASSIFICATION (for guidance):**
        - Paradigm: {deterministic_paradigm}
        - Heuristic: {heuristic_paradigm}

        **Instructions:**
        1. Review the deterministic classification above
        2. Analyze the AST and patterns yourself
        3. Validate or critique the deterministic result
        4. Provide your final classification with confidence

        Respond in JSON format."""

    def _enforce_consistency(
        self,
        patterns: Dict[str, Any],
        deterministic: str,
        heuristic: str,
        paradigm: str,
        confidence: float,
        reasoning: str,
    ) -> tuple[str, float, str]:
        """Apply structural arbitration after probabilistic resolution."""

        candidate = paradigm or heuristic or deterministic or "unknown"
        updated_reasoning = reasoning
        branch_struct = self._is_branching_recursion(patterns)

        if branch_struct and candidate != "branching_recursion":
            logger.info(
                "%s: Promoting paradigm to branching_recursion based on structural evidence",
                self.name,
            )
            candidate = "branching_recursion"
            confidence = max(confidence, 0.8)
            updated_reasoning = (
                "Structural recursion metadata (constant decrements + branching "
                "factor >= 2) requires branching_recursion classification. "
                f"Previous reasoning: {reasoning}"
            )
            return candidate, confidence, updated_reasoning

        if candidate == "divide_and_conquer" and not self._supports_divide_and_conquer(
            patterns
        ):
            fallback = (
                deterministic
                if deterministic not in {None, "", "unknown"}
                else heuristic or "recursion"
            )
            logger.info(
                "%s: Downgrading divide_and_conquer to %s due to missing balanced split evidence",
                self.name,
                fallback,
            )
            candidate = fallback
            confidence = min(confidence, 0.8)
            updated_reasoning = (
                "Balanced split evidence not found (no fractional division). "
                f"Switching to {fallback}. Original reasoning: {reasoning}"
            )

        if (
            patterns.get("has_recursion")
            and candidate in {"iterative", "simple"}
            and heuristic
            and heuristic != candidate
        ):
            logger.info(
                "%s: Overriding non-recursive prediction due to recursion evidence",
                self.name,
            )
            adjusted_reasoning = f"Override to {heuristic} because AST shows recursion. Original reasoning: {reasoning}"
            return heuristic, max(confidence, 0.6), adjusted_reasoning

        return candidate, confidence, updated_reasoning

    def _heuristic_classification(self, patterns: Dict[str, Any]) -> str:
        """
        Use pattern-based heuristics for classification.
        """
        if self._is_branching_recursion(patterns):
            return "branching_recursion"

        if patterns.get("has_recursion"):
            call_counts = patterns.get("recursive_call_counts", {})
            max_calls = max(call_counts.values(), default=1)

            if max_calls > 1:
                return "divide_and_conquer"

            if patterns.get("has_loops"):
                return "tail_recursive_iterative"

            return "iterative"

        elif patterns.get("has_nested_loops"):
            return "dynamic_programming"

        elif patterns.get("has_loops") and patterns.get("has_conditionals"):
            return "greedy"

        elif patterns.get("has_loops"):
            return "iterative"

        else:
            return "simple"

    @staticmethod
    def _is_branching_recursion(patterns: Dict[str, Any]) -> bool:
        if not patterns:
            return False

        branching_factor = patterns.get("estimated_branching_factor") or patterns.get(
            "max_recursive_calls_per_function", 0
        )
        if branching_factor < 2:
            return False

        constant_flag = patterns.get("has_constant_decrement_recursion")
        non_linear = patterns.get("non_linear_split_recursion")
        fractional_flag = patterns.get("has_fractional_split_recursion")

        return bool((constant_flag or non_linear) and not fractional_flag)

    @staticmethod
    def _supports_divide_and_conquer(patterns: Dict[str, Any]) -> bool:
        if not patterns or not patterns.get("has_recursion"):
            return False

        fractional_flag = bool(patterns.get("has_fractional_split_recursion"))
        detector_flag = bool(patterns.get("is_divide_and_conquer"))
        division_factor = patterns.get("division_factor")
        fractional_factors = patterns.get("fractional_split_factors") or []
        fractional_evidence = bool(
            detector_flag
            or fractional_flag
            or fractional_factors
            or (division_factor not in (None, 0, 1))
        )

        if fractional_evidence:
            return True

        call_targets = patterns.get("call_targets") or []
        sorting_calls = patterns.get("sorting_calls") or []
        merge_call = any("merge" in (name or "").lower() for name in call_targets)
        merge_stage = merge_call or any(
            "merge" in (name or "").lower() for name in sorting_calls
        )
        has_sorting = bool(patterns.get("has_sorting"))
        branching_calls = (patterns.get("max_recursive_calls_per_function") or 0) >= 2

        return bool(branching_calls and (merge_stage or has_sorting))

    def _build_system_prompt(self) -> str:
        """Build system prompt for paradigm classification."""
        return """You are an expert algorithm analyst specializing in paradigm classification.

            Your task is to classify algorithms into one of these paradigms:
            1. **divide_and_conquer**: Recursive algorithms that divide the problem into subproblems (e.g., MergeSort, QuickSort, Binary Search)
            2. **dynamic_programming**: Algorithms that solve overlapping subproblems with memoization (e.g., Fibonacci DP, Knapsack)
            3. **greedy**: Algorithms that make locally optimal choices (e.g., Dijkstra's, Prim's, Activity Selection)
            4. **backtracking**: Recursive algorithms with state exploration and backtracking (e.g., N-Queens, Sudoku Solver)
            5. **branch_and_bound**: Backtracking with pruning based on bounds
            6. **branching_recursion**: Multiple recursive calls with constant decrements (e.g., naive Fibonacci, Catalan numbers) that create exponential trees not covered by Master Theorem
            7. **tail_recursive_iterative**: Tail recursion used solely to simulate a single-pass iterative process (one self-call, shrinking input, no branching)
            8. **iterative**: Simple iterative algorithms without recursion
            9. **simple**: Constant time or simple sequential algorithms

            **Output Format (strict single-line JSON):**
            {"paradigm": "<one_of_the_above>", "confidence": 0.95, "reasoning": "brief justification"}

            Hard requirements:
            - Output **exactly one line** of JSON. No code fences, no Markdown, no prose before/after.
            - Do not insert newline characters inside the JSON. Do not escape as a string.
            - Keep the JSON compact (double quotes, commas, braces only).

            Analyze the AST structure and patterns to make your classification and then emit the single-line JSON."""

    def _build_strict_system_prompt(self) -> str:
        """Strict system prompt used on retry when the model previously emitted malformed output."""
        return """You are an expert algorithm analyst. Return exactly one line of compact JSON and nothing else.

            Valid paradigms: divide_and_conquer, dynamic_programming, greedy, backtracking, branch_and_bound, branching_recursion, tail_recursive_iterative, iterative, simple.

        Output format (one line, no markdown, no fences, no prose):
        {"paradigm": "<one_of_the_above>", "confidence": 0.95, "reasoning": "brief justification"}

        Hard constraints:
        - No code fences, markdown, bullet points, or explanations outside the JSON.
        - Do not wrap the JSON in quotes or additional characters.
        - Do not add newlines inside the JSON. Keep it on a single line.
        - If uncertain, choose the closest paradigm and lower the confidence, but still emit valid JSON."""

    def _build_user_prompt(
        self, ast_dict: Dict[str, Any], patterns: Dict[str, Any], heuristic: str
    ) -> str:
        """Build user prompt with AST and pattern information."""
        return f"""Classify the algorithmic paradigm for the following algorithm:
        ```json
        {json.dumps(ast_dict, indent=2)}
        ```

        **Detected Patterns:**
        ```json
        {json.dumps(patterns, indent=2)}
        ```

        **Heuristic Classification:** {heuristic}

        Provide your classification in JSON format with paradigm, confidence (0-1), and reasoning."""

    def _parse_response(self, response_text: str) -> tuple[str, float, str, bool]:
        """
        Parse LLM response.
        """

        def _extract_json_block(text: str) -> str | None:
            fenced = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if fenced:
                return fenced.group(1).strip()
            any_fence = re.search(r"```(.*?)```", text, re.DOTALL)
            if any_fence:
                return any_fence.group(1).strip()
            braces = re.search(r"\{.*\}", text, re.DOTALL)
            if braces:
                return braces.group(0).strip()
            return None

        json_candidate = _extract_json_block(response_text) or response_text.strip()

        try:
            cleaned = re.sub(r"[\x00-\x1f]+", " ", json_candidate)
            data = json.loads(cleaned)
            paradigm = data.get("paradigm", "unknown")
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "No reasoning provided")
            return paradigm, confidence, reasoning, True
        except Exception as e:
            logger.warning(f"Failed to parse classification response as JSON: {e}")

        paradigm = "unknown"
        confidence = 0.3
        reasoning = response_text[:500]

        paradigm_match = re.search(
            r'"?paradigm"?\s*[:=]\s*"?(?P<val>[A-Za-z_\-]+)"?',
            response_text,
            re.IGNORECASE,
        )
        if paradigm_match:
            paradigm = paradigm_match.group("val").strip()

        conf_match = re.search(
            r'"?confidence"?\s*[:=]\s*(?P<val>[0-9]+(?:\.[0-9]+)?)',
            response_text,
            re.IGNORECASE,
        )
        if conf_match:
            try:
                confidence = float(conf_match.group("val"))
            except ValueError:
                confidence = confidence

        reasoning_match = re.search(
            r'"?reasoning"?\s*[:=]\s*"(?P<val>.*?)"\s*\}?\s*$',
            response_text,
            re.IGNORECASE | re.DOTALL,
        )
        if reasoning_match:
            reasoning = reasoning_match.group("val").strip()

        return paradigm, confidence, reasoning, False
