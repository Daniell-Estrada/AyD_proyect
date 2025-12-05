"""Workflow orchestration helpers extracted from the monolithic app module."""


import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional

from fastapi.encoders import jsonable_encoder

from app.application.workflows.analysis_workflow import AnalysisWorkflow
from app.domain.models.analysis import (AgentEvent, AnalysisResult,
                                        ComplexitySummary, HitlAction,
                                        HitlResponse, Session, SessionStatus)
from app.infrastructure.persistence.mongodb_service import mongodb_service
from app.presentation.api import dependencies as api_dependencies
from app.presentation.api.hitl.state import HitlState
from app.presentation.api.realtime.channels import SessionChannelManager

logger = logging.getLogger(__name__)

BroadcastFn = Callable[[str, Dict[str, Any], str, Optional[str]], Awaitable[None]]


async def _noop_broadcast(*_args, **_kwargs):
    return None


class WorkflowOrchestrator:
    """Encapsulates workflow lifecycle, persistence, and HITL coordination."""

    def __init__(
        self,
        *,
        channel_manager: SessionChannelManager,
        hitl_state: HitlState,
        broadcast_fn: Optional[BroadcastFn] = None,
    ) -> None:
        self.channel_manager = channel_manager
        self.hitl_state = hitl_state
        self._broadcast = broadcast_fn or _noop_broadcast
        self._active_workflows: Dict[str, AnalysisWorkflow] = {}

    def set_broadcaster(self, broadcast_fn: BroadcastFn) -> None:
        """Inject the Socket.IO broadcaster after server initialization."""

        self._broadcast = broadcast_fn

    async def start_analysis(self, user_input: str, sid: Optional[str] = None) -> str:
        """Create a session, bootstrap the workflow, and return the session id."""

        session_id = str(uuid.uuid4())
        logger.info("Starting analysis for session: %s", session_id)
        metadata = (
            {"channel": "socket", "socket_id": sid} if sid else {"channel": "rest"}
        )

        session = Session(
            session_id=session_id,
            user_input=user_input,
            metadata=metadata,
        )
        await api_dependencies.session_use_cases.create_session(session)

        loop = asyncio.get_running_loop()
        workflow = self._build_workflow(session_id=session_id, sid=sid, loop=loop)
        self._active_workflows[session_id] = workflow

        asyncio.create_task(
            self._run_workflow_async(
                sid=sid,
                session_id=session_id,
                workflow=workflow,
                user_input=user_input,
            )
        )

        return session_id

    def _build_workflow(
        self,
        *,
        session_id: str,
        sid: Optional[str],
        loop: asyncio.AbstractEventLoop,
    ) -> AnalysisWorkflow:
        """Construct a workflow with channel-aware callbacks."""

        def hitl_callback(stage, agent_name, output, reasoning):
            future = asyncio.run_coroutine_threadsafe(
                self.request_hitl_approval(
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
            coro = self.emit_agent_update(
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
        self,
        *,
        session_id: str,
        stage: str,
        agent_name: str,
        status: str,
        payload: Optional[Dict[str, Any]],
        sid: Optional[str],
    ) -> None:
        """Emit agent lifecycle events and persist them."""

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
            await api_dependencies.agent_event_repository.log(event)
        except Exception as exc:  
            logger.warning("Failed to persist agent event for %s: %s", agent_name, exc)

        await self._broadcast("agent_update", message, session_id, sid)

    async def _run_workflow_async(
        self,
        *,
        sid: Optional[str],
        session_id: str,
        workflow: AnalysisWorkflow,
        user_input: str,
    ) -> None:
        """Run workflow asynchronously and emit updates."""

        try:
            await api_dependencies.session_use_cases.mark_status(
                session_id, SessionStatus.PROCESSING
            )

            await self._broadcast(
                "agent_update",
                {
                    "session_id": session_id,
                    "stage": "started",
                    "message": "Analysis workflow started",
                },
                session_id,
                sid,
            )

            loop = asyncio.get_event_loop()
            final_state = await loop.run_in_executor(
                None, workflow.run, user_input, session_id
            )

            if final_state.get("status") == "completed":
                await self.save_analysis_results(session_id, final_state)
                await api_dependencies.session_use_cases.mark_status(
                    session_id, SessionStatus.COMPLETED
                )

                await self._broadcast(
                    "analysis_complete",
                    {
                        "session_id": session_id,
                        "result": final_state.get("final_output"),
                    },
                    session_id,
                    sid,
                )
            else:
                await api_dependencies.session_use_cases.mark_status(
                    session_id,
                    SessionStatus.FAILED,
                    error=str(final_state.get("errors")),
                )

                await self._broadcast(
                    "analysis_failed",
                    {
                        "session_id": session_id,
                        "errors": final_state.get("errors", []),
                    },
                    session_id,
                    sid,
                )

        except Exception as exc:  
            logger.error("Workflow execution error: %s", exc)
            await api_dependencies.session_use_cases.mark_status(
                session_id, SessionStatus.FAILED, error=str(exc)
            )
            await self._broadcast(
                "error",
                {"session_id": session_id, "message": str(exc)},
                session_id,
                sid,
            )

        finally:
            self._active_workflows.pop(session_id, None)
            self.channel_manager.clear_session(session_id)

    async def request_hitl_approval(
        self,
        *,
        sid: Optional[str],
        session_id: str,
        stage: str,
        agent_name: str,
        output: Any,
        reasoning: str,
    ) -> Dict[str, Any]:
        """Request HITL approval and wait for response."""

        if not sid:
            logger.info(
                "No realtime channel for session %s at stage %s; auto-approving",
                session_id,
                stage,
            )
            return {"action": "approve", "stage": stage}

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
        event = self.hitl_state.create_pending(session_id, stage, request_payload)

        await self._broadcast("hitl_request", request_payload, session_id, sid)
        logger.info("HITL request sent for session %s, stage %s", session_id, stage)

        try:
            await asyncio.wait_for(event.wait(), timeout=300)
            response = self.hitl_state.response(session_id) or {
                "action": "approve",
                "stage": stage,
            }

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

            await api_dependencies.hitl_use_cases.record_response(hitl_response)
            self.hitl_state.clear(session_id)
            return response

        except asyncio.TimeoutError:
            logger.warning("HITL request timeout for session %s", session_id)
            self.hitl_state.clear(session_id)
            return {"action": "approve", "stage": stage}

    async def process_hitl_response(
        self,
        *,
        session_id: str,
        action: str,
        feedback: Optional[str],
        edited_output: Optional[Any],
        stage: Optional[str],
    ) -> None:
        """Record a HITL response and unblock the waiting workflow."""

        resolved_stage = stage or self.hitl_state.pending_stage(session_id)
        response_payload = {
            "action": action,
            "feedback": feedback,
            "edited_output": edited_output,
            "stage": resolved_stage,
        }
        self.hitl_state.resolve(session_id, response_payload)

        logger.info("HITL response received for session %s: %s", session_id, action)

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
        await api_dependencies.hitl_use_cases.record_response(hitl_response)

        resolution_payload = jsonable_encoder(
            {
                "session_id": session_id,
                "stage": hitl_response.stage,
                "action": hitl_response.action.value,
                "feedback": hitl_response.feedback,
                "edited_output": hitl_response.edited_output,
            }
        )

        await self._broadcast("hitl_resolved", resolution_payload, session_id, None)
        self.hitl_state.clear(session_id)

    async def save_analysis_results(
        self, session_id: str, final_state: Dict[str, Any]
    ) -> None:
        """Persist analysis result aggregates."""

        final_output = final_state.get("final_output", {})

        complexity_dict = final_output.get("complexity", {}) or {}
        complexity = ComplexitySummary(
            worst_case=complexity_dict.get("worst_case"),
            best_case=complexity_dict.get("best_case"),
            average_case=complexity_dict.get("average_case"),
            tight_bounds=complexity_dict.get("tight_bounds"),
        )

        paradigm_data = final_output.get("paradigm")
        if isinstance(paradigm_data, dict):
            paradigm_name = (
                paradigm_data.get("name")
                or final_state.get("paradigm")
                or "unknown"
            )
            paradigm_data = {
                "name": paradigm_name,
                "confidence": paradigm_data.get("confidence")
                or final_state.get("paradigm_confidence"),
                "reasoning": paradigm_data.get("reasoning")
                or final_state.get("paradigm_reasoning"),
            }
        else:
            paradigm_name = paradigm_data or final_state.get("paradigm") or "unknown"
            paradigm_data = {
                "name": paradigm_name,
                "confidence": final_state.get("paradigm_confidence"),
                "reasoning": final_state.get("paradigm_reasoning"),
            }

        result = AnalysisResult(
            session_id=session_id,
            algorithm_name=final_output.get("algorithm_name", "Unknown"),
            pseudocode=final_state.get("translated_pseudocode", ""),
            ast=final_state.get("parsed_ast", {}),
            paradigm=paradigm_data,
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

        await api_dependencies.analysis_use_cases.save_result(result)

        complexity_payload = {
            key: value for key, value in complexity_dict.items() if value is not None
        }

        await mongodb_service.update_session(
            session_id,
            {
                "metadata.algorithm_name": result.algorithm_name,
                "metadata.paradigm": paradigm_name,
                "metadata.paradigm_confidence": paradigm_data.get("confidence"),
                "metadata.paradigm_reasoning": paradigm_data.get("reasoning"),
                "metadata.paradigm_descriptor": paradigm_data,
                "metadata.latest_complexity": complexity_payload,
                "metadata.last_completed_at": datetime.utcnow(),
            },
        )

        await mongodb_service.add_session_metrics(
            session_id=session_id,
            cost_usd=final_state.get("total_cost_usd", 0.0),
            tokens=final_state.get("total_tokens", 0),
            duration_ms=final_state.get("total_duration_ms", 0.0),
        )

    def pending_hitl_payload(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Expose pending HITL payloads for HTTP/Sockets."""

        return self.hitl_state.pending_payload(session_id)


__all__ = ["WorkflowOrchestrator", "BroadcastFn"]
