"""Application use cases for session lifecycle management."""

from typing import List, Optional

from app.domain.models.analysis import Session, SessionStatus
from app.domain.repositories.interfaces import SessionRepository


class SessionUseCases:
    """Coordinate session persistence through the repository port."""

    def __init__(self, repository: SessionRepository):
        self.repository = repository

    async def create_session(self, session: Session) -> Session:
        return await self.repository.create(session)

    async def get_session(self, session_id: str) -> Optional[Session]:
        return await self.repository.get(session_id)

    async def list_recent_sessions(self, limit: int = 10) -> List[Session]:
        return await self.repository.list_recent(limit)

    async def mark_status(
        self, session_id: str, status: SessionStatus, error: str | None = None
    ) -> None:
        await self.repository.update_status(session_id, status, error)

    async def update_stage(self, session_id: str, stage: str) -> None:
        await self.repository.update_stage(session_id, stage)
