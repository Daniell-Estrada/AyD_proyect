"""
FastAPI application with Socket.IO for real-time complexity analysis.
Implements HITL (Human-in-the-Loop) workflow with WebSocket communication.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Set

import socketio
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.application.workflows.analysis_workflow import AnalysisWorkflow
from app.domain.models.analysis import (
    AgentEvent,
    AnalysisResult,
    ComplexitySummary,
    HitlAction,
    HitlResponse,
    Session,
    SessionStatus,
)
from app.infrastructure.persistence.mongodb_service import mongodb_service
from app.shared.config import settings
from app.shared.di import (
    get_analysis_use_cases,
    get_agent_event_repository,
    get_hitl_use_cases,
    get_session_use_cases,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ===== Dependency Singletons =====

session_use_cases = get_session_use_cases()
analysis_use_cases = get_analysis_use_cases()
hitl_use_cases = get_hitl_use_cases()
agent_event_repository = get_agent_event_repository()


class StartAnalysisRequest(BaseModel):
    """Payload for initiating an analysis run via HTTP."""

    user_input: str = Field(..., min_length=5, description="Algorithm description or pseudocode")
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


# ===== Lifespan Context Manager =====


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting application...")
    await mongodb_service.connect()
    logger.info("Application started successfully")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    await mongodb_service.disconnect()
    logger.info("Application shutdown complete")


# ===== FastAPI App =====

app = FastAPI(
    title="Algorithm Complexity Analyzer API",
    description="AI-powered algorithm complexity analysis with multi-agent system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Socket.IO Setup =====

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=settings.cors_origins_list,
    logger=settings.api_debug,
    engineio_logger=settings.api_debug,
)

socket_app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=app,
)


# ===== Global State =====

# Store active workflows by session_id
active_workflows: Dict[str, Any] = {}

# Store HITL pending requests
hitl_pending: Dict[str, asyncio.Event] = {}
hitl_responses: Dict[str, Dict[str, Any]] = {}
hitl_pending_stage: Dict[str, str] = {}
hitl_pending_payload: Dict[str, Dict[str, Any]] = {}

# Track which sockets subscribe to a session for broadcasting updates
session_channels: Dict[str, Set[str]] = {}
socket_session_index: Dict[str, Set[str]] = {}


def _register_session_channel(session_id: str, sid: Optional[str]) -> None:
    """Associate a socket with a session for future broadcasts."""

    if not sid:
        return

    session_channels.setdefault(session_id, set()).add(sid)
    socket_session_index.setdefault(sid, set()).add(session_id)


def _unregister_session_channel(sid: str) -> None:
    """Remove a socket from any subscribed session rooms."""

    sessions = socket_session_index.pop(sid, set())
    for session_id in sessions:
        watchers = session_channels.get(session_id)
        if not watchers:
            continue
        watchers.discard(sid)
        if not watchers:
            session_channels.pop(session_id, None)


def _clear_session_channels(session_id: str) -> None:
    """Remove all socket subscriptions for a finished session."""

    watchers = session_channels.pop(session_id, set())
    for sid in watchers:
        sessions = socket_session_index.get(sid)
        if not sessions:
            continue
        sessions.discard(session_id)
        if not sessions:
            socket_session_index.pop(sid, None)


def _iter_session_targets(session_id: str, sid: Optional[str]) -> Set[str]:
    """Return sockets interested in a session, including the explicit sid."""

    targets = set(session_channels.get(session_id, set()))
    if sid:
        targets.add(sid)
        _register_session_channel(session_id, sid)
    return targets


async def _broadcast_event(
    event_name: str,
    payload: Dict[str, Any],
    session_id: str,
    sid: Optional[str],
) -> None:
    """Emit an event to all sockets watching a session."""

    targets = _iter_session_targets(session_id, sid)
    if not targets:
        return

    for target in targets:
        await sio.emit(event_name, payload, room=target)


async def start_analysis_workflow(
    user_input: str,
    sid: Optional[str] = None,
) -> str:
    """Create a session, bootstrap the workflow, and return the session id."""

    session_id = str(uuid.uuid4())
    logger.info(f"Starting analysis for session: {session_id}")
    metadata = {"channel": "socket", "socket_id": sid} if sid else {"channel": "rest"}

    session = Session(
        session_id=session_id,
        user_input=user_input,
        metadata=metadata,
    )
    await session_use_cases.create_session(session)

    loop = asyncio.get_running_loop()
    workflow = _build_workflow(session_id=session_id, sid=sid, loop=loop)
    active_workflows[session_id] = workflow

    asyncio.create_task(
        run_workflow_async(
            sid=sid,
            session_id=session_id,
            workflow=workflow,
            user_input=user_input,
        )
    )

    return session_id


def _build_workflow(session_id: str, sid: Optional[str], loop: asyncio.AbstractEventLoop) -> AnalysisWorkflow:
    """Construct a workflow with channel-aware callbacks."""

    def hitl_callback(stage, agent_name, output, reasoning):
        future = asyncio.run_coroutine_threadsafe(
            request_hitl_approval(
                sid=sid,
                session_id=session_id,
                stage=stage,
                agent_name=agent_name,
                output=output,
                reasoning=reasoning,
            ),
            loop,
        )
        return future.result()

    def workflow_event_callback(stage, agent_name, status, payload):
        coro = emit_agent_update(
            session_id=session_id,
            stage=stage,
            agent_name=agent_name,
            status=status,
            payload=payload,
            sid=sid,
        )
        asyncio.run_coroutine_threadsafe(coro, loop)

    return AnalysisWorkflow(
        hitl_callback=hitl_callback,
        event_callback=workflow_event_callback,
    )


async def emit_agent_update(
    session_id: str,
    stage: str,
    agent_name: str,
    status: str,
    payload: Optional[Dict[str, Any]],
    sid: Optional[str],
) -> None:
    """Emit agent lifecycle events when a socket channel is available."""

    safe_payload = jsonable_encoder(payload or {})

    message = {
        "session_id": session_id,
        "stage": stage,
        "agent_name": agent_name,
        "status": status,
        "payload": safe_payload,
    }

    try:
        event = AgentEvent(
            session_id=session_id,
            stage=stage,
            agent_name=agent_name,
            status=status,
            payload=safe_payload,
        )
        await agent_event_repository.log(event)
    except Exception as exc:  # pragma: no cover - logging only
        logger.warning(f"Failed to persist agent event for {agent_name}: {exc}")

    await _broadcast_event("agent_update", message, session_id, sid)


async def process_hitl_response_message(
    *,
    session_id: str,
    action: str,
    feedback: Optional[str],
    edited_output: Optional[Any],
    stage: Optional[str],
) -> None:
    """Record a HITL response and unblock the waiting workflow."""

    resolved_stage = stage or hitl_pending_stage.get(session_id)
    response_payload = {
        "action": action,
        "feedback": feedback,
        "edited_output": edited_output,
        "stage": resolved_stage,
    }
    hitl_responses[session_id] = response_payload

    pending_event = hitl_pending.get(session_id)
    if pending_event:
        pending_event.set()

    logger.info(f"HITL response received for session {session_id}: {action}")

    try:
        action_enum = HitlAction(action)
    except ValueError:
        action_enum = HitlAction.APPROVE

    hitl_response = HitlResponse(
        session_id=session_id,
        stage=resolved_stage or "unknown",
        action=action_enum,
        feedback=feedback,
        edited_output=edited_output,
    )

    await hitl_use_cases.record_response(hitl_response)

    resolution_payload = jsonable_encoder(
        {
            "session_id": session_id,
            "stage": hitl_response.stage,
            "action": hitl_response.action.value,
            "feedback": hitl_response.feedback,
            "edited_output": hitl_response.edited_output,
        }
    )

    await _broadcast_event("hitl_resolved", resolution_payload, session_id, None)

    hitl_pending.pop(session_id, None)
    hitl_responses.pop(session_id, None)
    hitl_pending_stage.pop(session_id, None)
    hitl_pending_payload.pop(session_id, None)


# ===== Socket.IO Events =====


@sio.event
async def connect(sid, _environ):
    """Handle client connection."""
    logger.info(f"Client connected: {sid}")
    await sio.emit("connection_established", {"sid": sid}, room=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {sid}")
    _unregister_session_channel(sid)


@sio.event
async def start_analysis(sid, data):
    """
    Start complexity analysis workflow.

    Args:
        sid: Socket ID
        data: {user_input: str}
    """
    try:
        user_input = data.get("user_input")
        if not user_input:
            await sio.emit(
                "error",
                {"message": "user_input is required"},
                room=sid,
            )
            return

        session_id = await start_analysis_workflow(
            user_input=user_input,
            sid=sid,
        )

        await sio.emit(
            "session_created",
            {"session_id": session_id},
            room=sid,
        )

    except Exception as e:
        logger.error(f"Error starting analysis: {e}")
        await sio.emit("error", {"message": str(e)}, room=sid)


@sio.event
async def hitl_respond(sid, data):
    """
    Respond to HITL approval request.

    Args:
        sid: Socket ID
        data: {session_id: str, action: str, feedback: str, edited_output: Any}
    """
    try:
        session_id = data.get("session_id")
        action = data.get("action")
        stage = data.get("stage") or hitl_pending_stage.get(session_id)
        feedback = data.get("feedback")
        edited_output = data.get("edited_output")

        if not session_id or not action:
            await sio.emit(
                "error",
                {"message": "session_id and action are required"},
                room=sid,
            )
            return

        await process_hitl_response_message(
            session_id=session_id,
            action=action,
            feedback=feedback,
            edited_output=edited_output,
            stage=stage,
        )

    except Exception as e:
        logger.error(f"Error processing HITL response: {e}")
        await sio.emit("error", {"message": str(e)}, room=sid)


@sio.event
async def subscribe_session(sid, data):
    """Allow late joiners to receive session history and pending HITL info."""

    session_id = (data or {}).get("session_id")
    if not session_id:
        await sio.emit(
            "error",
            {"message": "session_id is required", "code": "missing_session"},
            room=sid,
        )
        return

    try:
        session = await session_use_cases.get_session(session_id)
        if not session:
            await sio.emit(
                "error",
                {"message": "Session not found", "session_id": session_id},
                room=sid,
            )
            return

        _register_session_channel(session_id, sid)

        events = await agent_event_repository.list_for_session(session_id)
        await sio.emit(
            "workflow_history",
            {
                "session_id": session_id,
                "events": jsonable_encoder(events),
                "count": len(events),
            },
            room=sid,
        )

        if session_id in hitl_pending_payload:
            await sio.emit(
                "hitl_request",
                hitl_pending_payload[session_id],
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

    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.error("Failed to subscribe session %s: %s", session_id, exc)
        await sio.emit(
            "error",
            {"message": "Unable to subscribe to session"},
            room=sid,
        )


# ===== Helper Functions =====


async def run_workflow_async(
    sid: Optional[str],
    session_id: str,
    workflow: AnalysisWorkflow,
    user_input: str,
):
    """
    Run workflow asynchronously and emit updates.

    Args:
        sid: Socket ID
        session_id: Session ID
        workflow: Workflow instance
        user_input: User input
    """
    try:
        # Update session status
        await session_use_cases.mark_status(session_id, SessionStatus.PROCESSING)

        await _broadcast_event(
            "agent_update",
            {
                "session_id": session_id,
                "stage": "started",
                "message": "Analysis workflow started",
            },
            session_id,
            sid,
        )

        # Run workflow (blocking, so run in executor)
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(
            None, workflow.run, user_input, session_id
        )

        # Save results to database
        if final_state.get("status") == "completed":
            await save_analysis_results(session_id, final_state)
            await session_use_cases.mark_status(session_id, SessionStatus.COMPLETED)

            # Emit completion
            await _broadcast_event(
                "analysis_complete",
                {
                    "session_id": session_id,
                    "result": final_state.get("final_output"),
                },
                session_id,
                sid,
            )
        else:
            await session_use_cases.mark_status(
                session_id,
                SessionStatus.FAILED,
                error=str(final_state.get("errors")),
            )

            await _broadcast_event(
                "analysis_failed",
                {
                    "session_id": session_id,
                    "errors": final_state.get("errors", []),
                },
                session_id,
                sid,
            )

    except Exception as e:
        logger.error(f"Workflow execution error: {e}")
        await session_use_cases.mark_status(session_id, SessionStatus.FAILED, error=str(e))
        await _broadcast_event(
            "error",
            {"session_id": session_id, "message": str(e)},
            session_id,
            sid,
        )

    finally:
        # Cleanup
        if session_id in active_workflows:
            del active_workflows[session_id]
        _clear_session_channels(session_id)


async def request_hitl_approval(
    sid: Optional[str],
    session_id: str,
    stage: str,
    agent_name: str,
    output: Any,
    reasoning: str,
) -> Dict[str, Any]:
    """
    Request HITL approval and wait for response.

    Args:
        sid: Socket ID
        session_id: Session ID
        stage: Current workflow stage
        agent_name: Agent requesting approval
        output: Output to approve
        reasoning: Agent's reasoning

    Returns:
        HITL response dict
    """
    if not sid:
        logger.info(
            "No realtime channel for session %s at stage %s; auto-approving",
            session_id,
            stage,
        )
        return {"action": "approve", "stage": stage}

    # Create event for this request
    event = asyncio.Event()
    hitl_pending[session_id] = event
    hitl_pending_stage[session_id] = stage

    request_payload = jsonable_encoder(
        {
            "session_id": session_id,
            "stage": stage,
            "agent_name": agent_name,
            "output": output,
            "reasoning": reasoning,
            "requested_at": datetime.utcnow().isoformat(),
        }
    )
    hitl_pending_payload[session_id] = request_payload

    await _broadcast_event("hitl_request", request_payload, session_id, sid)

    logger.info(f"HITL request sent for session {session_id}, stage {stage}")

    # Wait for response (with timeout)
    try:
        await asyncio.wait_for(event.wait(), timeout=300)  # 5 minute timeout

        # Get response
        response = hitl_responses.get(session_id, {"action": "approve", "stage": stage})

        try:
            action_enum = HitlAction(response.get("action", "approve"))
        except ValueError:
            action_enum = HitlAction.APPROVE

        hitl_response = HitlResponse(
            session_id=session_id,
            stage=response.get("stage") or stage,
            action=action_enum,
            feedback=response.get("feedback"),
            edited_output=response.get("edited_output"),
        )

        await hitl_use_cases.record_response(hitl_response)

        # Cleanup
        hitl_pending.pop(session_id, None)
        hitl_responses.pop(session_id, None)
        hitl_pending_stage.pop(session_id, None)
        hitl_pending_payload.pop(session_id, None)

        return response

    except asyncio.TimeoutError:
        logger.warning(f"HITL request timeout for session {session_id}")
        hitl_pending.pop(session_id, None)
        hitl_pending_stage.pop(session_id, None)
        hitl_pending_payload.pop(session_id, None)
        return {"action": "approve", "stage": stage}


async def save_analysis_results(session_id: str, final_state: Dict[str, Any]):
    """Save analysis results to database."""
    final_output = final_state.get("final_output", {})

    complexity_dict = final_output.get("complexity", {}) or {}
    complexity = ComplexitySummary(
        worst_case=complexity_dict.get("worst_case"),
        best_case=complexity_dict.get("best_case"),
        average_case=complexity_dict.get("average_case"),
        tight_bounds=complexity_dict.get("tight_bounds"),
    )

    result = AnalysisResult(
        session_id=session_id,
        algorithm_name=final_output.get("algorithm_name", "Unknown"),
        pseudocode=final_state.get("translated_pseudocode", ""),
        ast=final_state.get("parsed_ast", {}),
        paradigm=final_state.get("paradigm", "unknown"),
        complexity=complexity,
        analysis_steps=final_state.get("analysis_steps", []),
        diagrams=final_state.get("diagrams", {}),
        validation=final_state.get("validation_results", {}),
        metadata={
            "total_cost_usd": final_state.get("total_cost_usd"),
            "total_tokens": final_state.get("total_tokens"),
            "total_duration_ms": final_state.get("total_duration_ms"),
        },
    )

    await analysis_use_cases.save_result(result)

    complexity_payload = {
        key: value for key, value in complexity_dict.items() if value is not None
    }

    await mongodb_service.update_session(
        session_id,
        {
            "metadata.algorithm_name": result.algorithm_name,
            "metadata.paradigm": final_state.get("paradigm", "unknown"),
            "metadata.latest_complexity": complexity_payload,
            "metadata.last_completed_at": datetime.utcnow(),
        },
    )

    # Update session metrics
    await mongodb_service.add_session_metrics(
        session_id=session_id,
        cost_usd=final_state.get("total_cost_usd", 0.0),
        tokens=final_state.get("total_tokens", 0),
        duration_ms=final_state.get("total_duration_ms", 0.0),
    )


# ===== REST API Endpoints =====


@app.post("/sessions", response_model=StartAnalysisResponse, status_code=201)
async def create_session_endpoint(payload: StartAnalysisRequest):
    """HTTP endpoint to trigger a new workflow run."""

    session_id = await start_analysis_workflow(
        user_input=payload.user_input,
        sid=payload.socket_id,
    )

    if payload.socket_id:
        await sio.emit(
            "session_created",
            {"session_id": session_id},
            room=payload.socket_id,
        )

    return StartAnalysisResponse(session_id=session_id)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Algorithm Complexity Analyzer API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database": "connected" if mongodb_service._connected else "disconnected",
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information."""
    session = await session_use_cases.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return jsonable_encoder(session)


@app.get("/sessions/{session_id}/results")
async def get_analysis_results(session_id: str):
    """Get analysis results for a session."""
    result = await analysis_use_cases.fetch_latest(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="No results found for session")

    return jsonable_encoder(result)


@app.get("/sessions/{session_id}/events")
async def get_session_events(session_id: str):
    """Return chronological agent events for a session."""

    events = await agent_event_repository.list_for_session(session_id)
    return {
        "events": jsonable_encoder(events),
        "count": len(events),
    }


@app.get("/sessions")
async def list_sessions(limit: int = 10):
    """List recent sessions."""
    sessions = await session_use_cases.list_recent_sessions(limit=limit)
    return {
        "sessions": jsonable_encoder(sessions),
        "count": len(sessions),
    }


@app.post("/sessions/{session_id}/hitl")
async def respond_hitl_http(session_id: str, payload: HitlResponseRequest):
    """HTTP endpoint allowing reviewers to resolve HITL checkpoints."""

    action_value = payload.action.value if isinstance(payload.action, HitlAction) else str(payload.action)

    await process_hitl_response_message(
        session_id=session_id,
        action=action_value,
        feedback=payload.feedback,
        edited_output=payload.edited_output,
        stage=payload.stage,
    )

    return {"session_id": session_id, "status": "received"}


@app.get("/sessions/{session_id}/hitl/pending")
async def get_pending_hitl_request(session_id: str):
    """Return pending HITL request payload, if any."""

    payload = hitl_pending_payload.get(session_id)
    return {
        "session_id": session_id,
        "pending": bool(payload),
        "request": payload,
    }


@app.get("/sessions/{session_id}/hitl/history")
async def get_hitl_history(session_id: str):
    """Retrieve recorded HITL approvals for a session."""

    session = await session_use_cases.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    approvals = session.hitl_approvals or []
    return {
        "session_id": session_id,
        "approvals": jsonable_encoder(approvals),
        "count": len(approvals),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.presentation.api.app:socket_app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level=settings.log_level.lower(),
    )
