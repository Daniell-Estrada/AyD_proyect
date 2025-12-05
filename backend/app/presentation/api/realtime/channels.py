"""Utilities for tracking socket subscribers per session."""

from typing import Dict, Optional, Set


class SessionChannelManager:
    """Book-keeps socket subscriptions for workflow sessions."""

    def __init__(self) -> None:
        self._session_channels: Dict[str, Set[str]] = {}
        self._socket_index: Dict[str, Set[str]] = {}

    def register(self, session_id: str, sid: Optional[str]) -> None:
        """Associate a socket id with a session."""

        if not sid:
            return

        self._session_channels.setdefault(session_id, set()).add(sid)
        self._socket_index.setdefault(sid, set()).add(session_id)

    def unregister_sid(self, sid: str) -> None:
        """Remove a socket from all subscribed sessions."""

        sessions = self._socket_index.pop(sid, set())
        for session_id in sessions:
            watchers = self._session_channels.get(session_id)
            if not watchers:
                continue
            watchers.discard(sid)
            if not watchers:
                self._session_channels.pop(session_id, None)

    def clear_session(self, session_id: str) -> None:
        """Remove every subscription for a finished session."""

        watchers = self._session_channels.pop(session_id, set())
        for sid in watchers:
            sessions = self._socket_index.get(sid)
            if not sessions:
                continue
            sessions.discard(session_id)
            if not sessions:
                self._socket_index.pop(sid, None)

    def iter_targets(self, session_id: str, sid: Optional[str]) -> Set[str]:
        """Return sockets interested in a session, including the requester."""

        targets = set(self._session_channels.get(session_id, set()))
        if sid:
            targets.add(sid)
            self.register(session_id, sid)
        return targets

    def has_channel(self, session_id: str) -> bool:
        """Return True if session already has subscribers."""

        return bool(self._session_channels.get(session_id))

    def reset(self) -> None:
        """Clear every subscription (used mainly by tests)."""

        self._session_channels.clear()
        self._socket_index.clear()


__all__ = ["SessionChannelManager"]
