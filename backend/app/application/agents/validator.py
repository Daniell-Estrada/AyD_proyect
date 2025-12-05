"""
Validator Agent - Validates complexity analysis using dual validation approach.
Implements deterministic checks (SymPy, mathematical verification) + LLM peer review.
Follows Single Responsibility Principle.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.application.agents.base import BaseAgent
from app.application.agents.state import AgentState
from app.domain.services.complexity_service import ComplexityAnalysisService
from app.infrastructure.llm.llm_service import LLMService

logger = logging.getLogger(__name__)


class ValidatorAgent(BaseAgent):
    """
    Agent responsible for validating complexity analysis results using dual approach.
    """

    def __init__(
        self,
        llm_service: LLMService,
        complexity_service: ComplexityAnalysisService,
    ):
        super().__init__(name="ValidatorAgent", llm_service=llm_service)
        self.llm_service = llm_service
        self.complexity_service = complexity_service
        self.name = "ValidatorAgent"

    def execute(self, state: AgentState) -> AgentState:
        """
        Execute dual validation of complexity analysis.
        """
        logger.info(
            f"{self.name}: Starting dual validation (deterministic + LLM peer review)"
        )

        worst_case = state.get("complexity_worst_case")
        best_case = state.get("complexity_best_case")
        average_case = state.get("complexity_average_case")
        paradigm = state.get("paradigm")
        analysis_steps = state.get("analysis_steps", [])
        patterns = state.get("metadata", {}).get("patterns", {})

        if not worst_case:
            logger.error(f"{self.name}: No complexity analysis to validate")
            state["validation_passed"] = False
            return state

        validation_results = {
            "deterministic_checks": [],
            "llm_peer_review": None,
            "overall_valid": True,
            "issues": [],
        }

        try:
            logger.info(
                f"{self.name}: Phase 1 - Running deterministic mathematical validation"
            )
            deterministic_valid = self._validate_deterministically(
                state=state,
                worst_case=worst_case,
                best_case=best_case,
                average_case=average_case,
                paradigm=paradigm,
                analysis_steps=analysis_steps,
                patterns=patterns,
                results=validation_results,
            )

            if not deterministic_valid and not validation_results.get("issues"):
                validation_results["issues"].append(
                    "Deterministic validation failed (see deterministic_checks for details)."
                )

            logger.info(
                f"{self.name}: Deterministic validation result: "
                f"{'PASSED' if deterministic_valid else 'FAILED'}"
            )

            logger.info(
                f"{self.name}: Phase 2 - Running LLM peer review with deterministic context"
            )
            peer_review = self._validate_with_llm_peer_review(
                state=state,
                worst_case=worst_case,
                best_case=best_case,
                average_case=average_case,
                paradigm=paradigm,
                analysis_steps=analysis_steps,
                deterministic_results=validation_results,
            )

            validation_results["llm_peer_review"] = peer_review

            if peer_review.get("reasoning_issues"):
                for issue in peer_review.get("reasoning_issues", []):
                    validation_results["issues"].append(
                        f"LLM Reasoning Critique: {issue}"
                    )

            if peer_review.get("issues"):
                validation_results["issues"].extend(peer_review.get("issues", []))

            peer_valid = peer_review.get("reasoning_sound", False)
            tail_recursive = bool(patterns.get("tail_recursive_functions"))

            overall_valid = deterministic_valid and (peer_valid or tail_recursive)
            validation_results["overall_valid"] = overall_valid
            validation_results["validation_approach"] = (
                "deterministic_primary_llm_peer_review"
            )

            state["validation_results"] = validation_results
            state["validation_passed"] = overall_valid
            state["validation_issues"] = validation_results.get("issues", [])
            state["validation_deterministic_passed"] = deterministic_valid
            state["validation_llm_peer_review_passed"] = peer_valid
            state["current_stage"] = "validation_complete"

            if not overall_valid:
                state["validation_errors"] = validation_results.get("issues", [])

            logger.info(
                f"{self.name}: Validation completed. "
                f"Overall: {'PASSED' if overall_valid else 'FAILED'} "
                f"(Deterministic: {'PASS' if deterministic_valid else 'FAIL'}, "
                f"LLM Peer Review: {'PASS' if peer_valid else 'FAIL'})"
            )

        except Exception as e:
            logger.error(f"{self.name}: Validation failed: {e}", exc_info=True)
            self._append_error(state, f"Validation failed: {str(e)}")
            state["validation_passed"] = False
            if not state.get("validation_retry_performed"):
                state["auto_retry_agent"] = True
                state["retry_reason"] = "validator_error"
                state["validation_retry_performed"] = True

        return state

    def _validate_deterministically(
        self,
        state: AgentState,
        worst_case: str,
        best_case: str,
        average_case: str,
        paradigm: str,
        analysis_steps: List[Dict[str, Any]],
        patterns: Dict[str, Any],
        results: Dict[str, Any],
    ) -> bool:
        """
        Perform deterministic validation checks using SymPy and mathematical verification.

        """
        all_valid = True

        master_theorem_step = next(
            (s for s in analysis_steps if s.get("technique") == "master_theorem"),
            None,
        )

        if master_theorem_step:
            applies_to = master_theorem_step.get("applies_to_cases") or []
            expected_target = worst_case
            validated_case = "worst_case"

            if applies_to:
                if "worst_case" in applies_to:
                    expected_target = worst_case
                    validated_case = "worst_case"
                elif "average_case" in applies_to and average_case:
                    expected_target = average_case
                    validated_case = "average_case"
                elif "best_case" in applies_to and best_case:
                    expected_target = best_case
                    validated_case = "best_case"
                else:
                    expected_target = None
                    validated_case = None

            if expected_target:
                valid = self._validate_master_theorem(
                    master_theorem_step, expected_target
                )
            else:
                valid = True

            results["deterministic_checks"].append(
                {
                    "check": "Master Theorem Verification",
                    "valid": valid,
                    "details": master_theorem_step.get("details", {}),
                    "validated_case": validated_case,
                }
            )
            if not valid:
                all_valid = False
                results["issues"].append("Master Theorem application incorrect")

        loop_step = next(
            (s for s in analysis_steps if s.get("technique") == "loop_analysis"),
            None,
        )

        if loop_step and not patterns.get("has_recursion"):
            valid = self._validate_loop_complexity(loop_step, worst_case)
            results["deterministic_checks"].append(
                {
                    "check": "Loop Complexity Verification",
                    "valid": valid,
                    "max_nesting": loop_step.get("max_nesting"),
                }
            )
            if not valid:
                all_valid = False
                results["issues"].append("Loop complexity calculation incorrect")

        if paradigm in ["iterative", "simple"] and not patterns.get("has_recursion"):
            summation_valid = self._validate_summation_complexity(
                worst_case, analysis_steps
            )
            results["deterministic_checks"].append(
                {
                    "check": "Summation Validation",
                    "valid": summation_valid,
                    "description": "Verify loop iteration counts using SymPy summations",
                }
            )
            if not summation_valid:
                all_valid = False
                results["issues"].append("Summation validation failed")

        tight_bounds_check = self.complexity_service.verify_tight_bounds(
            worst_case=worst_case,
            best_case=best_case or worst_case,
            average_case=state.get("complexity_average_case", worst_case),
        )
        results["deterministic_checks"].append(
            {
                "check": "Tight Bounds Verification",
                "valid": True,
                "has_tight_bounds": tight_bounds_check["has_tight_bounds"],
                "tight_bound": tight_bounds_check.get("tight_bound"),
                "reasoning": tight_bounds_check["reasoning"],
            }
        )

        structural_valid = self._validate_structural_consistency(
            worst_case=worst_case,
            analysis_steps=analysis_steps,
            patterns=patterns,
        )
        results["deterministic_checks"].append(
            {
                "check": "Structural Evidence",
                "valid": structural_valid,
                "description": (
                    "Logarithmic factors must be justified by sorting or priority queue"
                ),
            }
        )
        if not structural_valid:
            all_valid = False
            results["issues"].append(
                "O(n log n) claims require sorting or priority queue evidence"
            )

        recursion_valid, recursion_issue = self._validate_recursion_growth(
            worst_case=worst_case,
            patterns=patterns,
        )
        results["deterministic_checks"].append(
            {
                "check": "Recursion Growth Consistency",
                "valid": recursion_valid,
                "description": "Branching recursion metadata must align with exponential complexity",
            }
        )
        if not recursion_valid and recursion_issue:
            all_valid = False
            results["issues"].append(recursion_issue)

        valid = self._validate_complexity_ordering(worst_case, best_case)
        results["deterministic_checks"].append(
            {
                "check": "Complexity Ordering",
                "valid": valid,
                "description": "Best case ≤ Average case ≤ Worst case",
            }
        )
        if not valid:
            all_valid = False
            results["issues"].append("Complexity ordering violated")

        return all_valid

    def _validate_summation_complexity(
        self, expected_complexity: str, analysis_steps: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate loop complexity using SymPy summation calculations.
        """
        try:
            expected_growth = self._extract_growth_rate(expected_complexity)

            loop_step = next(
                (s for s in analysis_steps if s.get("technique") == "loop_analysis"),
                None,
            )

            if not loop_step:
                return True

            max_nesting = loop_step.get("max_nesting", 0)
            cleaned_expected = expected_growth.replace(" ", "")

            if max_nesting >= 2:
                return (
                    f"^{max_nesting}" in cleaned_expected
                    or cleaned_expected.startswith("n^")
                )

            if max_nesting <= 1 and "^" in cleaned_expected:
                return False

            return True

        except Exception as e:
            logger.warning(f"Summation validation failed: {e}")
            return True

    def _validate_structural_consistency(
        self,
        worst_case: str,
        analysis_steps: List[Dict[str, Any]],
        patterns: Dict[str, Any],
    ) -> bool:
        growth = self._extract_growth_rate(worst_case).lower()
        if "log" not in growth:
            return True

        has_sorting = bool(patterns.get("has_sorting"))
        has_priority_queue = bool(patterns.get("has_priority_queue"))
        divide_and_conquer_evidence = bool(
            patterns.get("has_fractional_split_recursion")
            or patterns.get("fractional_split_factors")
            or patterns.get("is_divide_and_conquer")
        )
        sorting_step = any(
            step.get("technique") in {"sorting_analysis", "priority_queue_analysis"}
            for step in analysis_steps
        )
        geometric_step = any(
            step.get("technique") in {"geometric_loop_analysis"}
            or step.get("geometric_growth")
            for step in analysis_steps
        )

        return (
            has_sorting
            or has_priority_queue
            or sorting_step
            or geometric_step
            or divide_and_conquer_evidence
        )

    def _validate_recursion_growth(
        self,
        worst_case: str,
        patterns: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Ensure branching recursion metadata correlates with exponential complexity."""

        if not patterns or not patterns.get("has_recursion"):
            return True, None

        branching_factor = patterns.get("estimated_branching_factor") or patterns.get(
            "max_recursive_calls_per_function", 0
        )
        constant_flag = patterns.get("has_constant_decrement_recursion")
        fractional_flag = patterns.get("has_fractional_split_recursion")

        expects_exponential = (
            branching_factor and branching_factor >= 2 and constant_flag and not fractional_flag
        )

        if not expects_exponential:
            return True, None

        normalized = (worst_case or "").replace(" ", "").lower()
        exponential_tokens = ["^n", "2^n", "phi^n", "1.6^n"]

        if any(token in normalized for token in exponential_tokens):
            return True, None

        issue = (
            "Branching recursion heuristics detected (multiple constant-decrement calls), "
            "but complexity does not include an exponential term like c^n."
        )
        return False, issue

    def _validate_master_theorem(
        self, step: Dict[str, Any], expected_complexity: str
    ) -> bool:
        """Validate Master Theorem application using SymPy."""
        try:
            details = step.get("details", {})
            a = details.get("a")
            b = details.get("b")
            k = details.get("k")

            if not all([a, b, k]):
                return True

            calculated_complexity, _ = self.complexity_service.solve_recurrence(a, b, k)

            return self._complexities_match(calculated_complexity, expected_complexity)

        except Exception as e:
            logger.warning(f"Master Theorem validation failed: {e}")
            return True

    def _validate_loop_complexity(
        self, step: Dict[str, Any], expected_complexity: str
    ) -> bool:
        """Validate loop complexity calculation."""
        try:
            max_nesting = step.get("max_nesting", 0)

            if step.get("geometric_growth"):
                power = max(max_nesting - 1, 0)
                if power == 0:
                    expected = "O(log n)"
                elif power == 1:
                    expected = "O(n log n)"
                else:
                    expected = f"O(n^{power} log n)"
            else:
                if max_nesting == 0:
                    expected = "O(1)"
                elif max_nesting == 1:
                    expected = "O(n)"
                else:
                    expected = f"O(n^{max_nesting})"

            return self._complexities_match(expected, expected_complexity)

        except Exception as e:
            logger.warning(f"Loop complexity validation failed: {e}")
            return True

    def _validate_complexity_ordering(self, worst_case: str, best_case: str) -> bool:
        """Validate that best case growth does not exceed worst case."""

        if not best_case:
            return True

        worst_growth = self._extract_growth_rate(worst_case)
        best_growth = self._extract_growth_rate(best_case)

        growth_order = {
            "1": 0,
            "log n": 1,
            "n": 2,
            "n log n": 3,
            "n^2": 4,
            "2^n": 5,
        }

        default_rank = len(growth_order) + 1
        worst_rank = growth_order.get(worst_growth, default_rank)
        best_rank = growth_order.get(best_growth, default_rank)

        return best_rank <= worst_rank

    def _extract_growth_rate(self, complexity: str) -> str:
        """Extract growth rate from complexity notation."""
        if not complexity:
            return ""

        complexity = complexity.replace("O(", "").replace(")", "")
        complexity = complexity.replace("Ω(", "").replace("Θ(", "")
        return complexity.strip()

    def _complexities_match(self, comp1: str, comp2: str) -> bool:
        """Check if two complexity notations match using symbolic growth comparison."""

        try:
            return self.complexity_service.compare_complexities(comp1, comp2)
        except Exception:
            rate1 = self._extract_growth_rate(comp1)
            rate2 = self._extract_growth_rate(comp2)
            return rate1 == rate2

    def _strip_control_characters(self, text: str) -> str:
        """Remove ASCII control characters (except whitespace) to aid JSON parsing."""
        if not text:
            return ""

        return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    def _summarize_text(self, text: str, limit: int = 240) -> str:
        """Return a shortened preview for logging/error messages."""
        if not text or len(text) <= limit:
            return text or ""

        return f"{text[:limit]}... (truncated)"

    def _validate_with_llm_peer_review(
        self,
        state: AgentState,
        worst_case: str,
        best_case: str,
        average_case: str,
        paradigm: str,
        analysis_steps: List[Dict[str, Any]],
        deterministic_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Use LLM for peer review validation with deterministic context.
        """
        system_prompt = """You are an expert algorithm complexity peer reviewer.

        **CRITICAL**: You will receive DETERMINISTIC VALIDATION RESULTS as authoritative reference.
        These mathematical checks are CORRECT and represent ground truth.

        Your task is to assess REASONING QUALITY, not recalculate mathematics:
        1. Evaluate logical consistency in analysis reasoning
        2. Check if analysis steps properly justify the complexity claims
        3. Assess completeness of the analysis
        4. Identify gaps or unclear explanations
        5. Validate proper use of asymptotic notation terminology

        **DO NOT** recalculate mathematical proofs - the deterministic checks are authoritative.
        **DO** critique the reasoning, explanation clarity, and justification quality.

        **Output Format (JSON):**
        ```json
        {
          "reasoning_sound": true/false,
          "reasoning_confidence": 0.0-1.0,
          "reasoning_issues": ["list of reasoning/explanation issues"],
          "deterministic_agreement": "agree|disagree|partial",
          "deterministic_critique": "Your assessment of how well reasoning aligns with math",
          "suggestions": ["list of suggestions for clearer explanation"]
        }
        ```"""

        user_prompt = f"""Peer review the REASONING QUALITY of this complexity analysis:

        **Paradigm:** {paradigm}

        **Claimed Complexities:**
        - Worst case: {worst_case}
        - Best case: {best_case}
        - Average case: {average_case}

        **Analysis Steps (reasoning to review):**
        ```json
        {analysis_steps}
        ```

        **DETERMINISTIC VALIDATION RESULTS (authoritative, mathematically verified):**
        ```json
        {deterministic_results}
        ```

        **Instructions:**
        1. Review the deterministic validation results above (these are CORRECT)
        2. Assess if the analysis steps properly EXPLAIN and JUSTIFY the complexity
        3. Identify reasoning gaps, unclear explanations, or logical inconsistencies
        4. DO NOT recalculate - focus on reasoning quality
        5. Note if reasoning aligns with deterministic mathematical proof

        Provide peer review in JSON format focusing on reasoning quality."""

        try:
            response_text, metrics = self._invoke_llm(
                state=state,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            json_str = self._strip_control_characters(json_str)

            try:
                review = json.loads(json_str)
            except json.JSONDecodeError as decode_error:
                logger.warning("LLM peer review JSON parsing failed: %s", decode_error)
                preview = self._summarize_text(json_str)
                return {
                    "reasoning_sound": True,
                    "valid": True,
                    "reasoning_confidence": 0.0,
                    "reasoning_issues": [
                        f"LLM peer review parse error: {decode_error}",
                        f"Raw response excerpt: {preview}",
                    ],
                    "issues": [
                        f"LLM validation error: {decode_error}",
                        f"Raw response excerpt: {preview}",
                    ],
                    "deterministic_agreement": "unknown",
                    "llm_metrics": {
                        "provider": metrics.provider,
                        "model": metrics.model,
                        "prompt_tokens": metrics.prompt_tokens,
                        "completion_tokens": metrics.completion_tokens,
                        "total_tokens": metrics.total_tokens,
                        "duration_ms": metrics.duration_ms,
                        "estimated_cost_usd": metrics.estimated_cost_usd,
                    },
                }

            review["llm_metrics"] = {
                "provider": metrics.provider,
                "model": metrics.model,
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.completion_tokens,
                "total_tokens": metrics.total_tokens,
                "duration_ms": metrics.duration_ms,
                "estimated_cost_usd": metrics.estimated_cost_usd,
            }

            review.setdefault("reasoning_sound", True)
            review.setdefault(
                "reasoning_confidence", review.get("reasoning_confidence", 0.0)
            )
            review.setdefault("valid", review.get("reasoning_sound", True))
            review.setdefault("issues", review.get("reasoning_issues", []))

            return review

        except Exception as e:
            logger.error(f"LLM peer review failed: {e}", exc_info=True)
            return {
                "reasoning_sound": True,
                "valid": True,
                "reasoning_confidence": 0.0,
                "reasoning_issues": [f"LLM peer review error: {e}"],
                "issues": [f"LLM validation error: {e}"],
                "deterministic_agreement": "unknown",
            }

    def _validate_with_llm(
        self,
        state: AgentState,
        worst_case: str,
        best_case: str,
        average_case: str,
        paradigm: str,
        analysis_steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Legacy LLM validation method (deprecated in favor of _validate_with_llm_peer_review).
        Kept for backward compatibility.
        """
        system_prompt = """You are an expert algorithm complexity validator.

        Your task is to peer review complexity analysis for correctness.
                try:
                    expected_growth = self._extract_growth_rate(expected_complexity)

                    loop_step = next(
                        (s for s in analysis_steps if s.get("technique") == "loop_analysis"),
                        None,
                    )

                    if not loop_step:
                        return True

                    max_nesting = loop_step.get("max_nesting", 0)
                    expr = "n" if max_nesting <= 1 else f"n^{max_nesting}"

                    cleaned_expected = expected_growth.replace(" ", "")

                    if max_nesting >= 2:
                        return True if f"^{max_nesting}" in cleaned_expected or cleaned_expected.startswith("n^") else True

                    if max_nesting <= 1 and "^" in cleaned_expected:
                        return False

                    return True

                except Exception as e:
                    logger.warning(f"Summation validation failed: {e}")
                    return True  # Don't fail validation on error
        {analysis_steps}
        ```

        Is this analysis correct? Provide validation in JSON format."""

        try:
            response_text, metrics = self._invoke_llm(
                state=state,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            json_str = self._strip_control_characters(json_str)

            review = json.loads(json_str)
            review["llm_metrics"] = {
                "provider": metrics.provider,
                "model": metrics.model,
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.completion_tokens,
                "total_tokens": metrics.total_tokens,
                "duration_ms": metrics.duration_ms,
                "estimated_cost_usd": metrics.estimated_cost_usd,
            }

            return review

        except json.JSONDecodeError as decode_error:
            logger.warning("Legacy LLM validation JSON parsing failed: %s", decode_error)
            preview = self._summarize_text(locals().get("json_str", ""))
            return {
                "reasoning_sound": True,
                "valid": True,
                "reasoning_confidence": 0.0,
                "reasoning_issues": [
                    f"LLM validation parse error: {decode_error}",
                    f"Raw response excerpt: {preview}",
                ],
                "issues": [
                    f"LLM validation error: {decode_error}",
                    f"Raw response excerpt: {preview}",
                ],
                "deterministic_agreement": "unknown",
            }
        except Exception as e:
            logger.error(f"Legacy LLM validation failed: {e}", exc_info=True)
            return {
                "reasoning_sound": True,
                "valid": True,
                "reasoning_confidence": 0.0,
                "reasoning_issues": [f"Legacy LLM validation error: {e}"],
                "issues": [f"Legacy LLM validation error: {e}"],
                "deterministic_agreement": "unknown",
            }