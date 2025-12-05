"""Shared HITL state management for pending approvals."""

import asyncio
from typing import Any, Dict, Optional


class HitlState:
    """Tracks pending HITL checkpoints and responses per session."""

    def __init__(self) -> None:
        self._pending_events: Dict[str, asyncio.Event] = {}
        self._responses: Dict[str, Dict[str, Any]] = {}
        self._pending_stage: Dict[str, str] = {}
        self._pending_payload: Dict[str, Dict[str, Any]] = {}

    def create_pending(
        self,
        session_id: str,
        stage: str,
        payload: Dict[str, Any],
    ) -> asyncio.Event:
        """Register a pending HITL checkpoint and return the awaiting event."""

        event = asyncio.Event()
        self._pending_events[session_id] = event
        self._pending_stage[session_id] = stage
        self._pending_payload[session_id] = payload
        return event

    def resolve(self, session_id: str, response: Dict[str, Any]) -> None:
        """Store a reviewer response and signal waiting workflows."""

        self._responses[session_id] = response
        pending_event = self._pending_events.get(session_id)
        if pending_event:
            pending_event.set()

    def clear(self, session_id: str) -> None:
        """Remove every pending artifact for a session."""

        self._pending_events.pop(session_id, None)
        self._responses.pop(session_id, None)
        self._pending_stage.pop(session_id, None)
        self._pending_payload.pop(session_id, None)

    def pending_payload(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._pending_payload.get(session_id)

    def pending_stage(self, session_id: str) -> Optional[str]:
        return self._pending_stage.get(session_id)

    def response(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._responses.get(session_id)

    def event(self, session_id: str) -> Optional[asyncio.Event]:
        return self._pending_events.get(session_id)

    def reset(self) -> None:
        """Clear all pending HITL tracking structures."""

        self._pending_events.clear()
        self._responses.clear()
        self._pending_stage.clear()
        self._pending_payload.clear()


__all__ = ["HitlState"]
