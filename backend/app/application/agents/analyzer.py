"""
Analyzer Agent - Analyzes algorithm complexity using various techniques.
Implements Single Responsibility Principle and Strategy Pattern.
"""

import logging

from app.application.agents.base import BaseAgent
from app.application.agents.state import AgentState
from app.application.agents.strategies.llm_strategy import LLMAnalysisStrategy
from app.application.agents.strategies.strategy_factory import \
    AnalysisStrategyFactory
from app.domain.services.complexity_service import ComplexityAnalysisService
from app.domain.services.diagram_service import DiagramService
from app.infrastructure.llm.llm_service import LLMService

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
        """
        Initialize Analyzer Agent with strategy factory.
        """
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
                strategy = self._strategy_factory.create_strategy(paradigm, state)
            complexities, steps, diagrams = strategy.analyze(ast_dict, patterns)

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
                in {"o(1)", "θ(1)", "omega(1)", "Ω(1)", "o(1)"}
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
                f"{self.name}: Analysis completed using {paradigm} strategy. "
                f"Worst case: {complexities.get('worst_case')}"
            )

        except Exception as e:
            logger.error(f"{self.name}: Analysis failed: {e}", exc_info=True)
            self._append_error(state, f"Analysis failed: {str(e)}")

        if state.get("auto_retry_performed"):
            state["analysis_retry_performed"] = True

        return state
