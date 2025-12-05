"""Repository contracts for persistence adapters in a hexagonal architecture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from app.domain.models.analysis import AnalysisResult, AgentEvent, HitlResponse, Session, SessionStatus


class SessionRepository(ABC):
    """Persistence port for analysis sessions."""

    @abstractmethod
    async def create(self, session: Session) -> Session:
        raise NotImplementedError

    @abstractmethod
    async def get(self, session_id: str) -> Optional[Session]:
        raise NotImplementedError

    @abstractmethod
    async def update_status(
        self, session_id: str, status: SessionStatus, error: Optional[str] = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def update_stage(self, session_id: str, stage: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def append_hitl_response(self, session_id: str, response: HitlResponse) -> None:
        raise NotImplementedError

    @abstractmethod
    async def list_recent(self, limit: int = 10) -> List[Session]:
        raise NotImplementedError


class AnalysisResultRepository(ABC):
    """Port for storing and retrieving analysis outputs."""

    @abstractmethod
    async def save(self, result: AnalysisResult) -> str:
        raise NotImplementedError

    @abstractmethod
    async def latest_for_session(self, session_id: str) -> Optional[AnalysisResult]:
        raise NotImplementedError


class AgentEventRepository(ABC):
    """Port for persisting agent lifecycle events."""

    @abstractmethod
    async def log(self, event: AgentEvent) -> str:
        raise NotImplementedError

    @abstractmethod
    async def list_for_session(self, session_id: str) -> List[AgentEvent]:
        raise NotImplementedError
