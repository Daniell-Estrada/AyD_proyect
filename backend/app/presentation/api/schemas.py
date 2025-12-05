"""Pydantic request/response models exposed by the API."""

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.domain.models.analysis import HitlAction


class StartAnalysisRequest(BaseModel):
    """Payload for initiating an analysis run via HTTP."""

    user_input: str = Field(
        ..., min_length=5, description="Algorithm description or pseudocode"
    )
    socket_id: Optional[str] = Field(
        default=None,
        description="Existing Socket.IO session id for streaming updates",
    )


class StartAnalysisResponse(BaseModel):
    """Response returned after enqueuing a workflow."""

    session_id: str
    status: str = "processing"


class HitlResponseRequest(BaseModel):
    """Body for submitting HITL decisions over HTTP."""

    action: HitlAction
    feedback: Optional[str] = None
    edited_output: Optional[Any] = None
    stage: Optional[str] = None


__all__ = [
    "StartAnalysisRequest",
    "StartAnalysisResponse",
    "HitlResponseRequest",
]
