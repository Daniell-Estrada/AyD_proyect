"""Socket.IO server setup isolated from the FastAPI application module."""

import logging
from typing import Any, Dict, Optional

import socketio
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

from app.presentation.api import dependencies as api_dependencies
from app.presentation.api.services.orchestrator import WorkflowOrchestrator
from app.shared.config import settings

logger = logging.getLogger(__name__)

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=settings.cors_origins_list,
    logger=settings.api_debug,
    engineio_logger=settings.api_debug,
)

socket_app: Optional[socketio.ASGIApp] = None
_orchestrator: Optional[WorkflowOrchestrator] = None


def init_socket_app(
    app: FastAPI, orchestrator: WorkflowOrchestrator
) -> socketio.ASGIApp:
    """Bind Socket.IO to the FastAPI app and configure callbacks."""

    global socket_app, _orchestrator
    _orchestrator = orchestrator
    orchestrator.set_broadcaster(_broadcast_event)
    socket_app = socketio.ASGIApp(socketio_server=sio, other_asgi_app=app)
    return socket_app


async def emit_to_socket(event_name: str, payload: Dict[str, Any], sid: str) -> None:
    """Convenience helper for REST endpoints emitting to a single socket."""

    await sio.emit(event_name, payload, room=sid)


def _require_orchestrator() -> WorkflowOrchestrator:
    if not _orchestrator:
        raise RuntimeError("Socket server used before orchestrator initialization")
    return _orchestrator


async def _broadcast_event(
    event_name: str,
    payload: Dict[str, Any],
    session_id: str,
    sid: Optional[str],
) -> None:
    """Emit an event to every socket subscribed to the session."""

    orchestrator = _require_orchestrator()
    targets = orchestrator.channel_manager.iter_targets(session_id, sid)
    if not targets:
        return

    for target in targets:
        await sio.emit(event_name, payload, room=target)


@sio.event
async def connect(sid, _environ):
    """Handle client connection."""

    logger.info("Client connected: %s", sid)
    await sio.emit("connection_established", {"sid": sid}, room=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection."""

    logger.info("Client disconnected: %s", sid)
    orchestrator = _require_orchestrator()
    orchestrator.channel_manager.unregister_sid(sid)


@sio.event
async def start_analysis(sid, data):
    """Start complexity analysis workflow via Socket.IO."""

    orchestrator = _require_orchestrator()

    try:
        user_input = (data or {}).get("user_input")
        if not user_input:
            await sio.emit(
                "error",
                {"message": "user_input is required"},
                room=sid,
            )
            return

        session_id = await orchestrator.start_analysis(user_input=user_input, sid=sid)

        await sio.emit(
            "session_created",
            {"session_id": session_id},
            room=sid,
        )

    except Exception as exc:
        logger.error("Error starting analysis: %s", exc)
        await sio.emit("error", {"message": str(exc)}, room=sid)


@sio.event
async def hitl_respond(sid, data):
    """Respond to HITL approval request via Socket.IO."""

    orchestrator = _require_orchestrator()

    try:
        session_id = (data or {}).get("session_id")
        action = (data or {}).get("action")
        stage = (data or {}).get("stage") or orchestrator.hitl_state.pending_stage(
            session_id
        )
        feedback = (data or {}).get("feedback")
        edited_output = (data or {}).get("edited_output")

        if not session_id or not action:
            await sio.emit(
                "error",
                {"message": "session_id and action are required"},
                room=sid,
            )
            return

        await orchestrator.process_hitl_response(
            session_id=session_id,
            action=action,
            feedback=feedback,
            edited_output=edited_output,
            stage=stage,
        )

    except Exception as exc:
        logger.error("Error processing HITL response: %s", exc)
        await sio.emit("error", {"message": str(exc)}, room=sid)


@sio.event
async def subscribe_session(sid, data):
    """Allow late joiners to receive session history and pending HITL info."""

    orchestrator = _require_orchestrator()
    session_id = (data or {}).get("session_id")
    if not session_id:
        await sio.emit(
            "error",
            {"message": "session_id is required", "code": "missing_session"},
            room=sid,
        )
        return

    try:
        session = await api_dependencies.session_use_cases.get_session(session_id)
        if not session:
            await sio.emit(
                "error",
                {"message": "Session not found", "session_id": session_id},
                room=sid,
            )
            return

        orchestrator.channel_manager.register(session_id, sid)

        events = await api_dependencies.agent_event_repository.list_for_session(
            session_id
        )
        await sio.emit(
            "workflow_history",
            {
                "session_id": session_id,
                "events": jsonable_encoder(events),
                "count": len(events),
            },
            room=sid,
        )

        pending_payload = orchestrator.pending_hitl_payload(session_id)
        if pending_payload:
            await sio.emit(
                "hitl_request",
                pending_payload,
                room=sid,
            )

        await sio.emit(
            "session_snapshot",
            {
                "session_id": session_id,
                "session": jsonable_encoder(session),
            },
            room=sid,
        )

    except Exception as exc:
        logger.error("Failed to subscribe session %s: %s", session_id, exc)
        await sio.emit(
            "error",
            {"message": "Unable to subscribe to session"},
            room=sid,
        )


__all__ = ["sio", "socket_app", "init_socket_app", "emit_to_socket"]
