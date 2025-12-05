"""Use cases focused on managing human-in-the-loop callbacks."""


from app.domain.models.analysis import HitlResponse
from app.domain.repositories.interfaces import SessionRepository


class HitlUseCases:
    """Persist reviewer actions back into the session aggregate."""

    def __init__(self, repository: SessionRepository):
        self.repository = repository

    async def record_response(self, response: HitlResponse) -> None:
        await self.repository.append_hitl_response(response.session_id, response)
