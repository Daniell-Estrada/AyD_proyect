"""
Dependency Injection Container implementing Factory and Singleton patterns.
Provides centralized dependency management with lazy initialization.
"""

from functools import lru_cache
from typing import Optional

from app.application.use_cases.analysis_use_cases import AnalysisResultUseCases
from app.application.use_cases.hitl_use_cases import HitlUseCases
from app.application.use_cases.session_use_cases import SessionUseCases
from app.domain.repositories.interfaces import (AgentEventRepository,
                                                AnalysisResultRepository,
                                                SessionRepository)
from app.infrastructure.persistence.repositories import (
    MongoAgentEventRepository, MongoAnalysisResultRepository,
    MongoSessionRepository)


class DependencyContainer:
    """
    Centralized DI container for managing application dependencies.
    Implements Factory and Singleton patterns with lazy initialization.
    """

    _instance: Optional["DependencyContainer"] = None

    def __new__(cls) -> "DependencyContainer":
        """
        Ensure single instance of container (Singleton pattern).

        Returns:
            Singleton DependencyContainer instance
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @lru_cache(maxsize=1)
    def get_session_repository(self) -> SessionRepository:
        """
        Get singleton session repository instance.
        """
        return MongoSessionRepository()

    @lru_cache(maxsize=1)
    def get_analysis_repository(self) -> AnalysisResultRepository:
        """
        Get singleton analysis result repository instance.
        """
        return MongoAnalysisResultRepository()

    @lru_cache(maxsize=1)
    def get_agent_event_repository(self) -> AgentEventRepository:
        """
        Get singleton agent event repository instance.
        """
        return MongoAgentEventRepository()

    @lru_cache(maxsize=1)
    def get_session_use_cases(self) -> SessionUseCases:
        """
        Get singleton session use cases instance.
        """
        return SessionUseCases(self.get_session_repository())

    @lru_cache(maxsize=1)
    def get_analysis_use_cases(self) -> AnalysisResultUseCases:
        """
        Get singleton analysis result use cases instance.
        """
        return AnalysisResultUseCases(self.get_analysis_repository())

    @lru_cache(maxsize=1)
    def get_hitl_use_cases(self) -> HitlUseCases:
        """
        Get singleton HITL (Human-in-the-Loop) use cases instance.
        """
        return HitlUseCases(self.get_session_repository())

    def clear_cache(self) -> None:
        """
        Clear all cached dependencies.
        Useful for testing or resetting container state.
        """
        self.get_session_repository.cache_clear()
        self.get_analysis_repository.cache_clear()
        self.get_agent_event_repository.cache_clear()
        self.get_session_use_cases.cache_clear()
        self.get_analysis_use_cases.cache_clear()
        self.get_hitl_use_cases.cache_clear()


_container = DependencyContainer()


@lru_cache(maxsize=1)
def get_session_repository() -> SessionRepository:
    """
    Legacy function: Get session repository instance.
    """
    return _container.get_session_repository()


@lru_cache(maxsize=1)
def get_analysis_repository() -> AnalysisResultRepository:
    """
    Legacy function: Get analysis result repository instance.
    """
    return _container.get_analysis_repository()


@lru_cache(maxsize=1)
def get_agent_event_repository() -> AgentEventRepository:
    """
    Legacy function: Get agent event repository instance.
    """
    return _container.get_agent_event_repository()


@lru_cache(maxsize=1)
def get_session_use_cases() -> SessionUseCases:
    """
    Legacy function: Get session use cases instance.
    """
    return _container.get_session_use_cases()


@lru_cache(maxsize=1)
def get_analysis_use_cases() -> AnalysisResultUseCases:
    """
    Legacy function: Get analysis use cases instance.
    """
    return _container.get_analysis_use_cases()


@lru_cache(maxsize=1)
def get_hitl_use_cases() -> HitlUseCases:
    """
    Legacy function: Get HITL use cases instance.
    """
    return _container.get_hitl_use_cases()
