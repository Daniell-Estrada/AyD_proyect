"""
Strategy factory for creating appropriate complexity analysis strategies.
Implements Factory Pattern for strategy instantiation.
"""

from typing import Dict

from app.application.agents.state import AgentState
from app.application.agents.strategies.backtracking_strategy import \
    BacktrackingAnalysisStrategy
from app.application.agents.strategies.base_strategy import \
    ComplexityAnalysisStrategy
from app.application.agents.strategies.branching_recursion_strategy import \
    BranchingRecursionAnalysisStrategy
from app.application.agents.strategies.divide_and_conquer_strategy import \
    DivideAndConquerAnalysisStrategy
from app.application.agents.strategies.dynamic_programming_strategy import \
    DynamicProgrammingAnalysisStrategy
from app.application.agents.strategies.greedy_strategy import \
    GreedyAnalysisStrategy
from app.application.agents.strategies.iterative_strategy import \
    IterativeAnalysisStrategy
from app.application.agents.strategies.llm_strategy import LLMAnalysisStrategy
from app.domain.services.complexity_service import ComplexityAnalysisService
from app.infrastructure.llm.llm_service import LLMService


class AnalysisStrategyFactory:
    """
    Factory for creating complexity analysis strategies based on paradigm.
    Implements Factory Method pattern with dependency injection.
    """

    _STRATEGY_MAP: Dict[str, type] = {
        "divide_and_conquer": DivideAndConquerAnalysisStrategy,
        "dynamic_programming": DynamicProgrammingAnalysisStrategy,
        "iterative": IterativeAnalysisStrategy,
        "simple": IterativeAnalysisStrategy,
        "greedy": GreedyAnalysisStrategy,
        "backtracking": BacktrackingAnalysisStrategy,
        "branching_recursion": BranchingRecursionAnalysisStrategy,
    }

    def __init__(
        self,
        complexity_service: ComplexityAnalysisService,
        llm_service: LLMService,
        enable_llm_peer_review: bool = False,
    ):
        self._complexity_service = complexity_service
        self._llm_service = llm_service
        self._enable_llm_peer_review = enable_llm_peer_review

    def create_strategy(
        self, paradigm: str, state: AgentState
    ) -> ComplexityAnalysisStrategy:
        """
        Create appropriate analysis strategy for given paradigm.
        """
        strategy_class = self._STRATEGY_MAP.get(paradigm)

        if strategy_class is None:
            return LLMAnalysisStrategy(
                llm_service=self._llm_service,
                state=state,
                complexity_service=self._complexity_service,
            )

        if strategy_class == LLMAnalysisStrategy:
            return strategy_class(
                llm_service=self._llm_service,
                state=state,
                complexity_service=self._complexity_service,
            )
        else:
            return strategy_class(
                complexity_service=self._complexity_service,
                llm_service=self._llm_service,
                enable_llm_peer_review=self._enable_llm_peer_review,
            )
