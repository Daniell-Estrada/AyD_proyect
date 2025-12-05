"""MongoDB-backed repository adapters implementing domain ports."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.domain.models.analysis import (
    AnalysisResult,
    AgentEvent,
    ComplexitySummary,
    HitlResponse,
    Session,
    SessionStatus,
)
from app.domain.repositories.interfaces import (
    AnalysisResultRepository,
    AgentEventRepository,
    SessionRepository,
)
from app.infrastructure.persistence.mongodb_service import MongoDBService, mongodb_service


def _to_session(doc: Dict[str, Any]) -> Session:
    return Session(
        session_id=doc["session_id"],
        user_input=doc["user_input"],
        status=SessionStatus(doc.get("status", SessionStatus.PENDING.value)),
        current_stage=doc.get("current_stage", "initialization"),
        metadata=doc.get("metadata", {}),
        workflow_state=doc.get("workflow_state", {}),
        total_cost_usd=doc.get("total_cost_usd", 0.0),
        total_tokens=doc.get("total_tokens", 0),
        total_duration_ms=doc.get("total_duration_ms", 0.0),
        hitl_approvals=doc.get("hitl_approvals", []),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
    )


def _to_analysis_result(doc: Dict[str, Any]) -> AnalysisResult:
    complexity = doc.get("complexities", {}) or {}
    summary = ComplexitySummary(
        worst_case=complexity.get("worst_case"),
        best_case=complexity.get("best_case"),
        average_case=complexity.get("average_case"),
        tight_bounds=complexity.get("tight_bounds"),
    )

    return AnalysisResult(
        session_id=doc["session_id"],
        algorithm_name=doc.get("algorithm_name", "Unknown"),
        pseudocode=doc.get("pseudocode", ""),
        ast=doc.get("ast", {}),
        paradigm=doc.get("paradigm", "unknown"),
        complexity=summary,
        analysis_steps=doc.get("analysis_steps", []),
        diagrams=doc.get("diagrams", {}),
        validation=doc.get("validation_results", {}),
        metadata=doc.get("metadata", {}),
        created_at=doc.get("created_at"),
    )


def _to_agent_event(doc: Dict[str, Any]) -> AgentEvent:
    payload = doc.get("payload") or doc.get("output_data") or {}
    stage = doc.get("stage") or doc.get("action") or "unknown"
    status = doc.get("status") or payload.get("status") or "started"

    return AgentEvent(
        session_id=doc["session_id"],
        stage=stage,
        agent_name=doc.get("agent_name", "unknown"),
        status=status,
        payload=payload,
        timestamp=doc.get("timestamp"),
    )


class MongoSessionRepository(SessionRepository):
    """Session repository backed by MongoDBService."""

    def __init__(self, service: MongoDBService | None = None):
        self._service = service or mongodb_service

    async def create(self, session: Session) -> Session:
        await self._service.create_session(
            session_id=session.session_id,
            user_input=session.user_input,
            metadata=session.metadata,
        )
        return session

    async def get(self, session_id: str) -> Optional[Session]:
        doc = await self._service.get_session(session_id)
        if not doc:
            return None
        doc.pop("_id", None)
        return _to_session(doc)

    async def update_status(
        self, session_id: str, status: SessionStatus, error: str | None = None
    ) -> None:
        await self._service.update_session_status(session_id, status.value, error=error)

    async def update_stage(self, session_id: str, stage: str) -> None:
        await self._service.update_session_stage(session_id, stage)

    async def append_hitl_response(self, session_id: str, response: HitlResponse) -> None:
        await self._service.record_hitl_approval(
            session_id=session_id,
            stage=response.stage,
            action=response.action.value,
            feedback=response.feedback,
        )

    async def list_recent(self, limit: int = 10) -> List[Session]:
        docs = await self._service.get_recent_sessions(limit=limit)
        return [_to_session({k: v for k, v in doc.items() if k != "_id"}) for doc in docs]


class MongoAnalysisResultRepository(AnalysisResultRepository):
    """Persist analysis outputs via MongoDBService."""
    def __init__(self, service: MongoDBService | None = None):
        self._service = service or mongodb_service

    async def save(self, result: AnalysisResult) -> str:
        return await self._service.save_analysis_result(
            session_id=result.session_id,
            algorithm_name=result.algorithm_name,
            pseudocode=result.pseudocode,
            ast=result.ast,
            paradigm=result.paradigm,
            complexities=result.complexity.__dict__ if hasattr(result.complexity, "__dict__") else result.complexity,
            analysis_steps=result.analysis_steps,
            diagrams=result.diagrams,
            validation_results=result.validation,
            metadata=result.metadata,
        )

    async def latest_for_session(self, session_id: str) -> Optional[AnalysisResult]:
        doc = await self._service.get_latest_analysis_result(session_id)
        if not doc:
            return None
        doc.pop("_id", None)
        return _to_analysis_result(doc)


class MongoAgentEventRepository(AgentEventRepository):
    """Log agent lifecycle events into MongoDB."""
    def __init__(self, service: MongoDBService | None = None):
        self._service = service or mongodb_service

    async def log(self, event: AgentEvent) -> str:
        return await self._service.log_agent_action(
            session_id=event.session_id,
            agent_name=event.agent_name,
            stage=event.stage,
            status=event.status,
            payload=event.payload,
        )

    async def list_for_session(self, session_id: str) -> List[AgentEvent]:
        entries = await self._service.get_agent_logs(session_id)
        return [_to_agent_event(entry) for entry in entries]
