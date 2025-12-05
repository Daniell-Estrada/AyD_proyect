"""
Analyzer Agent - Analyzes algorithm complexity using various techniques.
Implements Single Responsibility Principle and Strategy Pattern.
"""

import logging
import re

from app.application.agents.base import BaseAgent
from app.application.agents.state import AgentState
from app.application.agents.strategies.llm_strategy import LLMAnalysisStrategy
from app.application.agents.strategies.strategy_factory import \
    AnalysisStrategyFactory
from app.domain.services.complexity_service import ComplexityAnalysisService
from app.domain.services.diagram_service import DiagramService
from app.infrastructure.llm.llm_service import LLMService

_COMPLEXITY_WRAP_RE = re.compile(r"^(?P<symbol>[OΘΩ])\((?P<body>.+)\)$")
_UNIT_EXP_RE = re.compile(r"n\^\{?1\}?(?=(?:[^0-9]|$))")

logger = logging.getLogger(__name__)


class AnalyzerAgent(BaseAgent):
    """
    Agent responsible for analyzing algorithm complexity.
    Delegates to paradigm-specific strategies via Strategy Pattern.
    """

    def __init__(
        self,
        llm_service: LLMService,
        complexity_service: ComplexityAnalysisService,
        diagram_service: DiagramService,
    ):
        super().__init__(name="AnalyzerAgent", llm_service=llm_service)
        self.llm_service = llm_service
        self.complexity_service = complexity_service
        self.diagram_service = diagram_service
        self.name = "AnalyzerAgent"

        self._strategy_factory = AnalysisStrategyFactory(
            complexity_service=complexity_service,
            llm_service=llm_service,
        )

    def execute(self, state: AgentState) -> AgentState:
        """
        Execute complexity analysis using appropriate strategy.
        """
        logger.info(f"{self.name}: Starting analysis")

        paradigm = state.get("paradigm", "unknown")
        ast_dict = state.get("parsed_ast")
        patterns = state.get("metadata", {}).get("patterns", {})
        analysis_paradigm = self._select_analysis_paradigm(paradigm, patterns)

        state["analysis_paradigm"] = analysis_paradigm
        if analysis_paradigm != paradigm:
            state["analysis_paradigm_override"] = analysis_paradigm
        else:
            state.pop("analysis_paradigm_override", None)

        if not ast_dict:
            logger.error(f"{self.name}: No AST available for analysis")
            self._append_error(state, "No AST available for analysis")
            return state

        try:
            use_llm_for_recursion = patterns.get("has_recursion") and paradigm in {
                "iterative",
                "simple",
            }
            force_llm = bool(state.get("analysis_force_llm_fallback"))

            if use_llm_for_recursion or force_llm:
                logger.info(
                    "%s: Using LLM strategy (forced=%s, recursion=%s) for paradigm %s",
                    self.name,
                    force_llm,
                    use_llm_for_recursion,
                    paradigm,
                )
                strategy = LLMAnalysisStrategy(self.llm_service, state)
            else:
                strategy = self._strategy_factory.create_strategy(
                    analysis_paradigm, state
                )
            complexities, steps, diagrams = strategy.analyze(ast_dict, patterns)
            complexities = self._normalize_complexities(complexities)

            state["complexity_worst_case"] = complexities.get("worst_case", "O(?)")
            state["complexity_best_case"] = complexities.get("best_case", "Ω(?)")
            state["complexity_average_case"] = complexities.get("average_case", "Θ(?)")
            state["tight_bounds"] = complexities.get("tight_bounds")

            existing_steps = state.get("analysis_steps", [])
            state["analysis_steps"] = existing_steps + steps

            state["diagrams"] = diagrams
            state["current_stage"] = "analysis_complete"

            worst_case = complexities.get("worst_case")
            if (
                patterns.get("has_loops")
                and isinstance(worst_case, str)
                and worst_case.strip().lower()
                in {"o(1)", "θ(1)", "omega(1)", "Ω(1)", "O(1)"}
                and not state.get("analysis_retry_performed")
            ):
                logger.info(
                    "%s: Inconsistency detected (loops present, got %s). Scheduling auto-retry with LLM fallback.",
                    self.name,
                    worst_case,
                )
                state["auto_retry_agent"] = True
                state["analysis_force_llm_fallback"] = True
                state["analysis_retry_reason"] = (
                    "loops_detected_but_constant_complexity"
                )
            else:
                state.pop("auto_retry_agent", None)

            logger.info(
                f"{self.name}: Analysis completed using {analysis_paradigm} strategy. "
                f"Worst case: {complexities.get('worst_case')}"
            )

        except Exception as e:
            logger.error(f"{self.name}: Analysis failed: {e}", exc_info=True)
            self._append_error(state, f"Analysis failed: {str(e)}")

        if state.get("auto_retry_performed"):
            state["analysis_retry_performed"] = True

        return state

    def _select_analysis_paradigm(
        self, classifier_paradigm: str, patterns: dict
    ) -> str:
        """Override classifier choice when structure indicates branching recursion."""

        if self._detect_branching_recursion(patterns):
            if classifier_paradigm != "branching_recursion":
                logger.info(
                    "%s: Detected non-linear split recursion. Using branching analysis.",
                    self.name,
                )
            return "branching_recursion"

        return classifier_paradigm

    def _normalize_complexities(self, complexities: dict | None) -> dict:
        """Polish notation: drop n^1, prefer Ω/Θ, infer tight bounds."""

        normalized = dict(complexities or {})

        for key in ("worst_case", "best_case", "average_case", "tight_bounds"):
            normalized[key] = self._normalize_complexity_string(normalized.get(key))

        normalized["best_case"] = self._apply_preferred_symbol(
            normalized.get("best_case"), "Ω"
        )
        normalized["average_case"] = self._apply_preferred_symbol(
            normalized.get("average_case"), "Θ"
        )
        normalized["worst_case"] = self._apply_preferred_symbol(
            normalized.get("worst_case"), "O"
        )

        if not normalized.get("tight_bounds"):
            inner_values = [
                self._extract_inner_expression(normalized.get(key))
                for key in ("worst_case", "best_case", "average_case")
            ]
            if (
                all(inner_values)
                and len(set(inner_values)) == 1
                and "?" not in inner_values[0]
            ):
                normalized["tight_bounds"] = f"Θ({inner_values[0]})"

        return normalized

    def _normalize_complexity_string(self, value: str | None) -> str | None:
        if not isinstance(value, str):
            return value

        cleaned = value.strip().replace("n¹", "n")
        cleaned = _UNIT_EXP_RE.sub("n", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def _apply_preferred_symbol(self, value: str | None, preferred: str) -> str | None:
        if not isinstance(value, str) or not preferred:
            return value

        match = _COMPLEXITY_WRAP_RE.match(value.strip())
        if not match:
            return value

        symbol = match.group("symbol")
        body = match.group("body").strip()

        if preferred in {"Ω", "Θ"} and symbol == "O":
            symbol = preferred

        return f"{symbol}({body})"

    def _extract_inner_expression(self, value: str | None) -> str | None:
        if not isinstance(value, str):
            return None

        match = _COMPLEXITY_WRAP_RE.match(value.strip())
        if not match:
            return None

        return match.group("body").strip()

    @staticmethod
    def _detect_branching_recursion(patterns: dict) -> bool:
        """Identify Fibonacci/Catalan-style recurrences with constant decrements."""

        if not patterns or not patterns.get("has_recursion"):
            return False

        branching = patterns.get("estimated_branching_factor") or patterns.get(
            "max_recursive_calls_per_function", 0
        )
        if not branching or branching < 2:
            return False

        constant_flag = patterns.get("has_constant_decrement_recursion")
        decrement_values = patterns.get("recursive_constant_decrements") or []
        fractional_flag = patterns.get("has_fractional_split_recursion")
        non_linear = patterns.get("non_linear_split_recursion")

        return bool(
            not fractional_flag
            and decrement_values
            and (non_linear or constant_flag)
        )
