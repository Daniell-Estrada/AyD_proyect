"""
Agent node handler implementing common agent execution logic.
Centralizes event emission and error handling for agent nodes.
"""

import logging
from typing import Any, Callable, Dict, Optional

from app.application.agents.state import AgentState

logger = logging.getLogger(__name__)


class AgentNodeHandler:
    """
    Handles agent node execution with standardized event emission.
    Provides reusable execution wrapper for all agent nodes,
    centralizing start/completion events and error handling.
    """

    def __init__(
        self,
        event_callback: Optional[
            Callable[[str, str, str, Dict[str, Any]], None]
        ] = None,
        state_recorder: Optional[Callable[[str, AgentState], None]] = None,
    ):
        self._event_callback = event_callback
        self._state_recorder = state_recorder

    def execute_agent(
        self,
        stage: str,
        agent_name: str,
        agent_fn: Callable[[AgentState], AgentState],
        state: AgentState,
        payload_builder: Optional[Callable[[AgentState], Dict[str, Any]]] = None,
        requires_review: bool = False,
    ) -> AgentState:
        """
        Execute agent with event emission and error handling.
        """
        baseline_cost = state.get("total_cost_usd", 0.0)
        baseline_tokens = state.get("total_tokens", 0)
        baseline_duration = state.get("total_duration_ms", 0.0)

        started_payload: Dict[str, Any] = {
            "session_id": state.get("session_id"),
            "stage": stage,
            "message": f"{agent_name} is processing...",
        }

        self._emit_event(
            stage=stage,
            agent_name=agent_name,
            status="started",
            payload=started_payload,
        )

        try:
            updated_state = agent_fn(state)

            if updated_state.get("auto_retry_agent") and not updated_state.get(
                "auto_retry_performed"
            ):
                retry_limit = updated_state.get("max_retries", 1)
                current_retries = updated_state.get("retry_count", 0)
                if current_retries < retry_limit:
                    updated_state["retry_count"] = current_retries + 1
                    updated_state["auto_retry_performed"] = True
                    updated_state["auto_retry_agent"] = False
                    updated_state = agent_fn(updated_state)

            delta_cost = max(
                0.0,
                updated_state.get("total_cost_usd", 0.0) - baseline_cost,
            )
            delta_tokens = max(
                0,
                updated_state.get("total_tokens", 0) - baseline_tokens,
            )
            delta_duration = max(
                0.0,
                updated_state.get("total_duration_ms", 0.0) - baseline_duration,
            )

            completed_payload: Dict[str, Any] = {
                "session_id": state.get("session_id"),
                "stage": stage,
                "message": f"{agent_name} completed successfully",
                "errors": updated_state.get("errors", []),
            }

            if delta_cost:
                completed_payload["cost_usd"] = delta_cost
            if delta_tokens:
                completed_payload["tokens"] = delta_tokens
            if delta_duration:
                completed_payload["duration_ms"] = delta_duration

            if delta_cost or delta_tokens or delta_duration:
                completed_payload["metrics"] = {
                    "cost_usd": delta_cost,
                    "tokens": delta_tokens,
                    "duration_ms": delta_duration,
                }

            if payload_builder:
                try:
                    additional_payload = payload_builder(updated_state)
                    if additional_payload:
                        completed_payload.update(additional_payload)
                except Exception as exc:
                    logger.warning(
                        "%s: payload builder failed: %s",
                        agent_name,
                        exc,
                    )

            retry_reason = updated_state.get(
                "analysis_retry_reason"
            ) or updated_state.get("retry_reason")
            if retry_reason:
                completed_payload["retry_reason"] = retry_reason

            if requires_review:
                completed_payload["message"] = f"{agent_name} output ready for review"
                completed_payload["requires_review"] = True

            self._emit_event(
                stage=stage,
                agent_name=agent_name,
                status="completed",
                payload=completed_payload,
            )

            self._record_state(stage, updated_state)

            return updated_state

        except Exception as e:
            logger.error(f"{agent_name} failed: {e}", exc_info=True)

            errors = state.get("errors", [])
            errors.append(f"{agent_name} execution failed: {str(e)}")
            state["errors"] = errors

            failure_payload: Dict[str, Any] = {
                "session_id": state.get("session_id"),
                "stage": stage,
                "message": f"{agent_name} failed",
                "error": str(e),
            }

            self._emit_event(
                stage=stage,
                agent_name=agent_name,
                status="failed",
                payload=failure_payload,
            )

            self._record_state(stage, state)

            return state

    def _emit_event(
        self, stage: str, agent_name: str, status: str, payload: Dict[str, Any]
    ) -> None:
        """
        Emit event through callback if configured.
        """
        if self._event_callback:
            try:
                self._event_callback(stage, agent_name, status, payload)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")

    def _record_state(self, stage: str, state: AgentState) -> None:
        """Forward state snapshots to the shared recorder when configured."""

        if self._state_recorder is None:
            return

        try:
            self._state_recorder(stage, state)
        except Exception as exc:
            logger.warning("State recorder failed: %s", exc)
