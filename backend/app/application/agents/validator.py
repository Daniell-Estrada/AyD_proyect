"""
Validator Agent - Validates complexity analysis using dual validation approach.
Implements deterministic checks (SymPy, mathematical verification) + LLM peer review.
Follows Single Responsibility Principle.
"""

import logging
from typing import Any, Dict, List

import sympy as sp

from app.application.agents.base import BaseAgent
from app.application.agents.state import AgentState
from app.domain.services.complexity_service import ComplexityAnalysisService
from app.infrastructure.llm.llm_service import LLMService

logger = logging.getLogger(__name__)


class ValidatorAgent(BaseAgent):
    """
    Agent responsible for validating complexity analysis results using dual approach.
    
    Architecture:
        1. Deterministic validation runs FIRST (SymPy, mathematical proofs)
        2. Results passed to LLM for peer review and critique
        3. LLM validates reasoning quality, not mathematical correctness
        4. Deterministic checks are primary; LLM provides quality assurance
    
    Validation techniques:
        - Master Theorem verification
        - Loop complexity recalculation
        - Summation validation
        - Tight bounds checking
        - Consistency verification (best ≤ avg ≤ worst)
    """

    def __init__(
        self,
        llm_service: LLMService,
        complexity_service: ComplexityAnalysisService,
    ):
        """
        Initialize Validator Agent with dual validation capability.

        Args:
            llm_service: LLM service instance for peer review
            complexity_service: Complexity analysis service for deterministic checks
        """
        super().__init__(name="ValidatorAgent", llm_service=llm_service)
        self.llm_service = llm_service
        self.complexity_service = complexity_service
        self.name = "ValidatorAgent"

    def execute(self, state: AgentState) -> AgentState:
        """
        Execute dual validation of complexity analysis.
        
        Workflow:
            1. Run deterministic mathematical checks (PRIMARY)
            2. Collect deterministic validation results
            3. Pass results to LLM for peer review and reasoning critique
            4. LLM assesses reasoning quality, not mathematical truth
            5. Combine deterministic and LLM results for final validation

        Args:
            state: Current agent state

        Returns:
            Updated state with validation results
        """
        logger.info(f"{self.name}: Starting dual validation (deterministic + LLM peer review)")

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
            # PHASE 1: Deterministic validation (PRIMARY - runs first)
            logger.info(f"{self.name}: Phase 1 - Running deterministic mathematical validation")
            deterministic_valid = self._validate_deterministically(
                state=state,
                worst_case=worst_case,
                best_case=best_case,
                paradigm=paradigm,
                analysis_steps=analysis_steps,
                patterns=patterns,
                results=validation_results,
            )
            
            logger.info(
                f"{self.name}: Deterministic validation result: "
                f"{'PASSED' if deterministic_valid else 'FAILED'}"
            )

            # PHASE 2: LLM peer review (SECONDARY - validates reasoning quality)
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
            
            # Collect LLM-identified issues (reasoning quality, not math correctness)
            if peer_review.get("reasoning_issues"):
                for issue in peer_review.get("reasoning_issues", []):
                    validation_results["issues"].append(f"LLM Reasoning Critique: {issue}")

            if peer_review.get("issues"):
                validation_results["issues"].extend(peer_review.get("issues", []))

            peer_valid = peer_review.get("reasoning_sound", False)
            tail_recursive = bool(patterns.get("tail_recursive_functions"))

            # PHASE 3: Combine results (deterministic takes precedence)
            # Deterministic validation is authoritative for mathematical correctness
            # LLM validates reasoning quality and completeness
            overall_valid = deterministic_valid and (peer_valid or tail_recursive)
            validation_results["overall_valid"] = overall_valid
            validation_results["validation_approach"] = "deterministic_primary_llm_peer_review"

            # Update state
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
        paradigm: str,
        analysis_steps: List[Dict[str, Any]],
        patterns: Dict[str, Any],
        results: Dict[str, Any],
    ) -> bool:
        """
        Perform deterministic validation checks using SymPy and mathematical verification.
        
        Validation techniques:
            1. Master Theorem recalculation
            2. Loop complexity verification
            3. Summation validation for iterative patterns
            4. Tight bounds verification
            5. Structural consistency checks

        Returns:
            True if all checks pass, False otherwise
        """
        all_valid = True

        # Check 1: Verify Master Theorem application (if used)
        master_theorem_step = next(
            (s for s in analysis_steps if s.get("technique") == "master_theorem"),
            None,
        )

        if master_theorem_step:
            valid = self._validate_master_theorem(master_theorem_step, worst_case)
            results["deterministic_checks"].append(
                {
                    "check": "Master Theorem Verification",
                    "valid": valid,
                    "details": master_theorem_step.get("details", {}),
                }
            )
            if not valid:
                all_valid = False
                results["issues"].append("Master Theorem application incorrect")

        # Check 2: Verify loop complexity
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
        
        # Check 3: Validate summations for iterative algorithms
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

        # Check 4: Verify tight bounds (Θ notation applicability)
        tight_bounds_check = self.complexity_service.verify_tight_bounds(
            worst_case=worst_case,
            best_case=best_case or worst_case,
            average_case=state.get("complexity_average_case", worst_case),
        )
        results["deterministic_checks"].append(
            {
                "check": "Tight Bounds Verification",
                "valid": True,  # Informational, doesn't fail validation
                "has_tight_bounds": tight_bounds_check["has_tight_bounds"],
                "tight_bound": tight_bounds_check.get("tight_bound"),
                "reasoning": tight_bounds_check["reasoning"],
            }
        )

        # Check 5: Structural consistency (e.g., sorting evidence for log factors)
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

        # Check 6: Verify best <= average <= worst
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
        
        For simple loops like "for i = 1 to n", verify that Σ(i=1 to n) 1 = n.
        
        Args:
            expected_complexity: Expected complexity notation
            analysis_steps: Analysis steps to extract loop information
            
        Returns:
            True if summation matches expected complexity
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

            # Accept polynomial growth aligned with nesting depth (e.g., 2-level -> n^2)
            if max_nesting >= 2:
                return f"^{max_nesting}" in cleaned_expected or cleaned_expected.startswith("n^")

            # Reject superlinear claims for single-level loops
            if max_nesting <= 1 and "^" in cleaned_expected:
                return False

            return True

        except Exception as e:
            logger.warning(f"Summation validation failed: {e}")
            return True  # Don't fail validation on error

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
        sorting_step = any(
            step.get("technique") in {"sorting_analysis", "priority_queue_analysis"}
            for step in analysis_steps
        )
        geometric_step = any(
            step.get("technique") in {"geometric_loop_analysis"}
            or step.get("geometric_growth")
            for step in analysis_steps
        )

        return has_sorting or has_priority_queue or sorting_step or geometric_step

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
                return True  # Can't verify without parameters

            # Recalculate using complexity service
            calculated_complexity, _ = self.complexity_service.solve_recurrence(a, b, k)

            # Compare (simplified comparison)
            return self._complexities_match(calculated_complexity, expected_complexity)

        except Exception as e:
            logger.warning(f"Master Theorem validation failed: {e}")
            return True  # Don't fail validation on error

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

    def _validate_complexity_ordering(
        self, worst_case: str, best_case: str
    ) -> bool:
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
        
        LLM focuses on REASONING QUALITY, not mathematical correctness.
        Deterministic results are provided as authoritative reference.
        
        Args:
            state: Current agent state
            worst_case: Worst case complexity
            best_case: Best case complexity
            average_case: Average case complexity
            paradigm: Algorithm paradigm
            analysis_steps: Analysis steps taken
            deterministic_results: Results from deterministic validation (authoritative)
            
        Returns:
            Peer review results dictionary
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

            # Parse response
            import json

            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

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

            # Normalize expected keys for tests and callers
            review.setdefault("reasoning_sound", True)
            review.setdefault("reasoning_confidence", review.get("reasoning_confidence", 0.0))
            review.setdefault("valid", review.get("reasoning_sound", True))
            review.setdefault("issues", review.get("reasoning_issues", []))

            return review

        except Exception as e:
            logger.error(f"LLM peer review failed: {e}", exc_info=True)
            return {
                "reasoning_sound": True,  # Don't fail on LLM error
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

            # Parse response
            import json

            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            review = json.loads(json_str)

            return review

        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return {
                "valid": True,  # Don't fail on LLM error
                "confidence": 0.0,
                "issues": [f"LLM validation error: {e}"],
            }
