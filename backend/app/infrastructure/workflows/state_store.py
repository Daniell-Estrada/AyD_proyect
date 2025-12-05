"""
Module providing persistence for workflow state snapshots using
in-memory or Redis-backed stores.
"""

import json
import logging
import threading
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Protocol

import redis

from app.application.agents.state import AgentState
from app.shared.config import settings

logger = logging.getLogger(__name__)

WorkflowStateSnapshot = Dict[str, Any]


class WorkflowStateStore(Protocol):
    """Protocol describing the required persistence operations."""

    def save_state(self, session_id: str, stage: str, state: AgentState) -> None:
        """Persist the most recent workflow snapshot for the given session."""

    def load_latest(self, session_id: str) -> Optional[WorkflowStateSnapshot]:
        """Return the latest snapshot for a session if available."""

    def load_history(self, session_id: str) -> List[WorkflowStateSnapshot]:
        """Return all cached snapshots for inspection or replay."""

    def clear_state(self, session_id: str) -> None:
        """Remove cached snapshots once a session is finalized."""


class InMemoryWorkflowStateStore(WorkflowStateStore):
    """Lock-protected in-memory store used for tests and local development."""

    def __init__(self, max_history: int = 25):
        self._max_history = max_history
        self._store: Dict[str, Deque[WorkflowStateSnapshot]] = {}
        self._lock = threading.Lock()

    def save_state(self, session_id: str, stage: str, state: AgentState) -> None:
        snapshot = _build_snapshot(stage, state)
        with self._lock:
            bucket = self._store.setdefault(
                session_id,
                deque(maxlen=self._max_history),
            )
            bucket.appendleft(snapshot)

    def load_latest(self, session_id: str) -> Optional[WorkflowStateSnapshot]:
        with self._lock:
            bucket = self._store.get(session_id)
            return bucket[0] if bucket else None

    def load_history(self, session_id: str) -> List[WorkflowStateSnapshot]:
        with self._lock:
            bucket = self._store.get(session_id)
            return list(bucket) if bucket else []

    def clear_state(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)


class RedisWorkflowStateStore(WorkflowStateStore):
    """Redis-backed store that supports multi-process workflows and TTL."""

    def __init__(
        self,
        url: str,
        namespace: str,
        ttl_seconds: int,
        max_history: int = 25,
    ):
        if redis is None:
            raise ImportError(
                "redis package is required for RedisWorkflowStateStore; install redis-py or disable persistence"
            )
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._namespace = namespace
        self._ttl = ttl_seconds
        self._max_history = max_history

    def save_state(self, session_id: str, stage: str, state: AgentState) -> None:
        snapshot = _build_snapshot(stage, state)
        payload = json.dumps(snapshot)
        latest_key = self._key(session_id, "latest")
        history_key = self._key(session_id, "history")

        with self._client.pipeline() as pipe:
            pipe.setex(latest_key, self._ttl, payload)
            pipe.lpush(history_key, payload)
            pipe.ltrim(history_key, 0, self._max_history - 1)
            pipe.expire(history_key, self._ttl)
            pipe.execute()

    def load_latest(self, session_id: str) -> Optional[WorkflowStateSnapshot]:
        payload = self._client.get(self._key(session_id, "latest"))
        return json.loads(payload) if payload else None

    def load_history(self, session_id: str) -> List[WorkflowStateSnapshot]:
        entries = self._client.lrange(
            self._key(session_id, "history"),
            0,
            self._max_history - 1,
        )
        return [json.loads(item) for item in entries]

    def clear_state(self, session_id: str) -> None:
        self._client.delete(
            self._key(session_id, "latest"),
            self._key(session_id, "history"),
        )

    def ping(self) -> None:
        """Expose ping for health checks during initialization."""
        self._client.ping()

    def _key(self, session_id: str, suffix: str) -> str:
        return f"{self._namespace}:{session_id}:{suffix}"


def create_workflow_state_store() -> WorkflowStateStore:
    """Factory that builds the appropriate store based on configuration."""

    if not settings.enable_state_persistence:
        logger.info("Workflow state persistence disabled; using in-memory store")
        return InMemoryWorkflowStateStore()

    if redis is None:
        logger.warning(
            "Redis dependency is missing; falling back to in-memory workflow store. "
            "Install redis-py or set ENABLE_STATE_PERSISTENCE=false to silence this warning."
        )
        return InMemoryWorkflowStateStore()

    try:
        redis_store = RedisWorkflowStateStore(
            url=settings.redis_url,
            namespace=settings.redis_namespace,
            ttl_seconds=settings.redis_state_ttl,
        )
        redis_store.ping()
        logger.info(
            "Workflow state persistence enabled via Redis namespace '%s'",
            settings.redis_namespace,
        )
        return redis_store
    except Exception as exc:  
        logger.warning(
            "Redis workflow store unavailable (%s); falling back to in-memory store",
            exc,
        )
        return InMemoryWorkflowStateStore()


def _build_snapshot(stage: str, state: AgentState) -> WorkflowStateSnapshot:
    """Create a serializable snapshot payload for persistence."""

    session_id = state.get("session_id", "")
    safe_state = _make_json_safe(state)
    return {
        "session_id": session_id,
        "stage": stage,
        "state": safe_state,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _make_json_safe(value: Any) -> Any:
    """Recursively convert complex structures into JSON-friendly objects."""

    if isinstance(value, dict):
        return {str(key): _make_json_safe(val) for key, val in value.items()}

    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]

    if isinstance(value, set):
        return [_make_json_safe(item) for item in value]

    if is_dataclass(value):
        return _make_json_safe(asdict(value))

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


__all__ = [
    "WorkflowStateStore",
    "InMemoryWorkflowStateStore",
    "RedisWorkflowStateStore",
    "create_workflow_state_store",
]
