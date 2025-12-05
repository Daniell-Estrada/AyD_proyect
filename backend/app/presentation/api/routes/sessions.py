"""Session-centric endpoints split from the monolithic app module."""

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

from app.presentation.api import dependencies as api_dependencies
from app.presentation.api.schemas import (HitlResponseRequest,
                                          StartAnalysisRequest,
                                          StartAnalysisResponse)
from app.presentation.api.services.orchestrator import WorkflowOrchestrator
from app.presentation.api.socket_server import emit_to_socket


def create_sessions_router(orchestrator: WorkflowOrchestrator) -> APIRouter:
    router = APIRouter(prefix="/sessions", tags=["sessions"])

    @router.post("", response_model=StartAnalysisResponse, status_code=201)
    async def create_session_endpoint(payload: StartAnalysisRequest):
        session_id = await orchestrator.start_analysis(
            user_input=payload.user_input,
            sid=payload.socket_id,
        )

        if payload.socket_id:
            await emit_to_socket(
                "session_created",
                {"session_id": session_id},
                payload.socket_id,
            )

        return StartAnalysisResponse(session_id=session_id)

    @router.get("")
    async def list_sessions(limit: int = 10):
        sessions = await api_dependencies.session_use_cases.list_recent_sessions(
            limit=limit
        )
        return {
            "sessions": jsonable_encoder(sessions),
            "count": len(sessions),
        }

    @router.get("/{session_id}")
    async def get_session(session_id: str):
        session = await api_dependencies.session_use_cases.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return jsonable_encoder(session)

    @router.get("/{session_id}/results")
    async def get_analysis_results(session_id: str):
        result = await api_dependencies.analysis_use_cases.fetch_latest(session_id)
        if not result:
            raise HTTPException(status_code=404, detail="No results found for session")
        return jsonable_encoder(result)

    @router.get("/{session_id}/events")
    async def get_session_events(session_id: str):
        events = await api_dependencies.agent_event_repository.list_for_session(
            session_id
        )
        return {
            "events": jsonable_encoder(events),
            "count": len(events),
        }

    @router.post("/{session_id}/hitl")
    async def respond_hitl_http(session_id: str, payload: HitlResponseRequest):
        action_value = (
            payload.action.value
            if hasattr(payload.action, "value")
            else str(payload.action)
        )

        await orchestrator.process_hitl_response(
            session_id=session_id,
            action=action_value,
            feedback=payload.feedback,
            edited_output=payload.edited_output,
            stage=payload.stage,
        )

        return {"session_id": session_id, "status": "received"}

    @router.get("/{session_id}/hitl/pending")
    async def get_pending_hitl_request(session_id: str):
        payload = orchestrator.pending_hitl_payload(session_id)
        return {
            "session_id": session_id,
            "pending": bool(payload),
            "request": payload,
        }

    @router.get("/{session_id}/hitl/history")
    async def get_hitl_history(session_id: str):
        session = await api_dependencies.session_use_cases.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        approvals = session.hitl_approvals or []
        return {
            "session_id": session_id,
            "approvals": jsonable_encoder(approvals),
            "count": len(approvals),
        }

    return router


__all__ = ["create_sessions_router"]
