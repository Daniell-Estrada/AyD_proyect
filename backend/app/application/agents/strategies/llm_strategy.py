"""
LLM-based complexity analysis strategy.
Uses Large Language Models when paradigm is unknown or complex.
Follows dual-validation architecture: deterministic analysis provides guidance,
LLM validates and enhances with advanced reasoning.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.application.agents.state import AgentState
from app.application.agents.strategies.base_strategy import \
    ComplexityAnalysisStrategy
from app.domain.services.complexity_service import ComplexityAnalysisService
from app.infrastructure.llm.llm_service import LLMService

logger = logging.getLogger(__name__)


class LLMAnalysisStrategy(ComplexityAnalysisStrategy):
    """
    Analyzes algorithm complexity using LLM reasoning when rule-based strategies are insufficient.
    """

    def __init__(
        self,
        llm_service: LLMService,
        state: AgentState,
        complexity_service: Optional[ComplexityAnalysisService] = None,
    ):
        """
        Initialize LLM-based analysis strategy.

        Args:
            llm_service: Service for invoking language models
            state: Current agent state for metric tracking
            complexity_service: Optional service for deterministic analysis guidance
        """
        self._llm_service = llm_service
        self._state = state
        self._complexity_service = complexity_service or ComplexityAnalysisService()

    def analyze(
        self, ast_dict: Dict[str, Any], patterns: Dict[str, Any]
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, str]]:
        """
        Analyze algorithm complexity using dual approach: deterministic + LLM.
        """
        logger.info("Using LLM strategy with deterministic guidance")

        deterministic_results = self._run_deterministic_analysis(ast_dict, patterns)

        paradigm = self._state.get("paradigm", "unknown")

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt_with_guidance(
            ast_dict=ast_dict,
            patterns=patterns,
            paradigm=paradigm,
            deterministic_guidance=deterministic_results,
        )

        response_text, metrics = self._llm_service.invoke(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        self._record_metrics(metrics)

        complexities, steps = self._parse_llm_response(response_text)

        if deterministic_results["steps"] and not steps:
            steps.insert(
                0,
                {
                    "step": "Deterministic Baseline Analysis",
                    "technique": "deterministic_guidance",
                    "complexities": deterministic_results["complexities"],
                    "reasoning": "Initial deterministic analysis provided as guidance for LLM validation",
                },
            )

        return complexities, steps, deterministic_results.get("diagrams", {})

    def _run_deterministic_analysis(
        self, ast_dict: Dict[str, Any], patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run deterministic complexity analysis to guide LLM.

        This baseline provides:
            - Loop-based complexity estimates
            - Recurrence detection and Master Theorem application
            - Pattern-based heuristics

        Args:
            ast_dict: Abstract Syntax Tree dictionary
            patterns: Detected algorithmic patterns

        Returns:
            Dictionary with deterministic analysis results
        """
        from app.domain.models.ast import Program

        results = {
            "complexities": {},
            "steps": [],
            "diagrams": {},
        }

        try:
            if patterns.get("has_loops"):
                loop_count = patterns.get("loop_count", 1)
                nested = patterns.get("has_nested_loops", False)

                if nested and loop_count >= 2:
                    complexity = f"O(n^{loop_count})"
                elif loop_count == 1:
                    complexity = "O(n)"
                else:
                    complexity = "O(n)"

                results["complexities"]["loop_based"] = complexity
                results["steps"].append(
                    {
                        "technique": "loop_analysis",
                        "result": complexity,
                        "loops_detected": loop_count,
                    }
                )

            if patterns.get("has_recursion"):
                recursive_calls = patterns.get("recursive_calls_count", 2)
                division_factor = patterns.get("division_factor", 2)
                combine_work = patterns.get("combine_work_exponent", 1)

                complexity, details = self._complexity_service.solve_recurrence(
                    a=recursive_calls,
                    b=division_factor,
                    k=combine_work,
                )

                results["complexities"]["recursion_based"] = complexity
                results["steps"].append(
                    {
                        "technique": "master_theorem_heuristic",
                        "result": complexity,
                        "details": details,
                    }
                )

            if not results["complexities"]:
                results["complexities"]["default"] = "O(n)"
                results["steps"].append(
                    {
                        "technique": "heuristic",
                        "result": "O(n)",
                        "note": "Default linear assumption",
                    }
                )

        except Exception as e:
            logger.warning(f"Deterministic analysis failed: {e}")
            results["complexities"]["error"] = "Unable to determine"

        return results

    def _build_system_prompt(self) -> str:
        """
        Build system prompt instructing LLM how to analyze complexity with deterministic guidance.

        Returns:
            System prompt string with role and format instructions
        """
        return """You are an expert algorithm complexity analyst with deep knowledge of:
- Asymptotic analysis (Big-O, Omega, Theta notation)
- Master Theorem and recurrence relations
- Loop analysis and amortized analysis
- Advanced algorithmic paradigms

**CRITICAL**: You will receive DETERMINISTIC ANALYSIS RESULTS as guidance context.
These results are NOT absolute truth but serve as baseline reference points.

Your task:
1. Review the deterministic analysis provided
2. Conduct your own independent complexity analysis
3. Validate, critique, or enhance the deterministic results
4. Provide rigorous mathematical justification
5. Note where you agree/disagree with deterministic baseline

Analyze the given algorithm and provide complexity analysis for:
- Worst case (Big-O)
- Best case (Omega)
- Average case (Theta)
- Tight bounds if applicable

**Output Format (JSON):**
```json
{
  "worst_case": "O(...)",
  "best_case": "Ω(...)",
  "average_case": "Θ(...)",
  "tight_bounds": "...",
  "deterministic_agreement": "agree|disagree|partial",
  "deterministic_critique": "Your assessment of the baseline analysis",
  "analysis_steps": [
    {
      "step": "Step description",
      "technique": "technique_used",
      "reasoning": "Detailed reasoning",
      "deterministic_comparison": "How this compares to baseline"
    }
  ]
}
```

Provide rigorous mathematical justification and clearly indicate your confidence level."""

    def _build_user_prompt_with_guidance(
        self,
        ast_dict: Dict[str, Any],
        patterns: Dict[str, Any],
        paradigm: str,
        deterministic_guidance: Dict[str, Any],
    ) -> str:
        """
        Build user prompt with algorithm details AND deterministic guidance.
        """
        return f"""Analyze the complexity of this algorithm:

        **Paradigm:** {paradigm}

        **AST:**
        ```json
        {json.dumps(ast_dict, indent=2)}
        ```

        **Detected Patterns:**
        ```json
        {json.dumps(patterns, indent=2)}
        ```

        **DETERMINISTIC BASELINE ANALYSIS (for guidance, not absolute truth):**
        ```json
        {json.dumps(deterministic_guidance, indent=2)}
        ```

        **Instructions:**
        1. Review the deterministic baseline above
        2. Conduct your own independent analysis
        3. Validate or critique the baseline results
        4. Provide your final complexity assessment with justification
        5. Indicate agreement level with deterministic baseline

        Provide detailed complexity analysis in JSON format with rigorous justification."""

    def _parse_llm_response(
        self, response_text: str
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """
        Parse LLM JSON response into structured complexity data.
        """
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            data = json.loads(json_str)

            complexities = {
                "worst_case": data.get("worst_case", "O(?)"),
                "best_case": data.get("best_case", "Ω(?)"),
                "average_case": data.get("average_case", "Θ(?)"),
                "tight_bounds": data.get("tight_bounds"),
            }

            steps = data.get("analysis_steps", [])

            return complexities, steps

        except Exception as e:
            logger.error(f"Failed to parse LLM analysis response: {e}")
            return {
                "worst_case": "O(?)",
                "best_case": "Ω(?)",
                "average_case": "Θ(?)",
            }, [
                {
                    "step": "LLM Analysis Failed",
                    "technique": "llm_fallback",
                    "error": str(e),
                }
            ]

    def _record_metrics(self, metrics: Any) -> None:
        """
        Record LLM usage metrics in state.
        """
        self._state["total_cost_usd"] = (
            self._state.get("total_cost_usd", 0.0) + metrics.estimated_cost_usd
        )
        self._state["total_tokens"] = (
            self._state.get("total_tokens", 0) + metrics.total_tokens
        )
        self._state["total_duration_ms"] = (
            self._state.get("total_duration_ms", 0.0) + metrics.duration_ms
        )

        agent_metrics = self._state.get("agent_metrics", [])
        agent_metrics.append(metrics)
        self._state["agent_metrics"] = agent_metrics
