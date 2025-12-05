"""Use cases for storing and retrieving analysis outputs."""

from __future__ import annotations

from typing import Optional

from app.domain.models.analysis import AnalysisResult
from app.domain.repositories.interfaces import AnalysisResultRepository


class AnalysisResultUseCases:
    def __init__(self, repository: AnalysisResultRepository):
        self.repository = repository

    async def save_result(self, result: AnalysisResult) -> str:
        return await self.repository.save(result)

    async def fetch_latest(self, session_id: str) -> Optional[AnalysisResult]:
        return await self.repository.latest_for_session(session_id)
