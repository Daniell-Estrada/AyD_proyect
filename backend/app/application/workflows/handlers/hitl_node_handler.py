"""
Module for handling Human-in-the-Loop (HITL) checkpoint nodes in a workflow.
"""

import logging
from typing import Any, Callable, Dict, Optional

from app.application.agents.state import AgentState

logger = logging.getLogger(__name__)


class HitlNodeHandler:
    """
    Handles Human-in-the-Loop (HITL) checkpoint nodes in a workflow.
    """

    def __init__(
        self,
        hitl_callback: Optional[Callable[[str, str, Any, str], Dict[str, Any]]] = None,
        event_callback: Optional[
            Callable[[str, str, str, Dict[str, Any]], None]
        ] = None,
    ):
        self._hitl_callback = hitl_callback
        self._event_callback = event_callback

    def request_review(
        self,
        stage: str,
        agent_name: str,
        state: AgentState,
        output_key: str,
        reasoning_key: str,
    ) -> AgentState:
        """
        Request Human-in-the-Loop (HITL) review for a workflow stage.
        """
        if not self._hitl_callback:
            state["approval_received"] = True
            return state

        output = state.get(output_key, "")
        reasoning = state.get(reasoning_key, "")

        state["approval_received"] = False
        state["user_feedback"] = None

        def _preview(value: Any) -> str:
            text = "" if value is None else str(value)
            return text[:400]

        self._emit_hitl_event(
            stage=stage,
            agent_name=agent_name,
            status="pending",
            payload={
                "session_id": state.get("session_id"),
                "reasoning": reasoning,
                "output": output,
                "output_preview": _preview(output),
            },
        )

        try:
            response = self._hitl_callback(
                stage=stage,
                agent_name=agent_name,
                output=output,
                reasoning=reasoning,
            )
        except Exception as e:
            logger.error(f"HITL callback failed: {e}", exc_info=True)
            response = {"action": "approve", "feedback": None}

        self._emit_hitl_event(
            stage=stage,
            agent_name=agent_name,
            status="resolved",
            payload={
                "session_id": state.get("session_id"),
                "action": response.get("action"),
                "feedback": response.get("feedback"),
                "edited_output": response.get("edited_output"),
            },
        )

        action = response.get("action")
        if action not in {"approve", "deny", "edit", "recommend"}:
            logger.warning(
                "Unexpected HITL action '%s' received for stage %s; forcing deny",
                action,
                stage,
            )
            action = "deny"

        state["approval_received"] = action in {"approve", "edit", "recommend"}
        state["user_feedback"] = response.get("feedback")

        if action == "edit" and response.get("edited_output"):
            state[output_key] = response["edited_output"]
            state["approval_received"] = True

        hitl_approvals = state.get("hitl_approvals", [])
        hitl_approvals.append(response)
        state["hitl_approvals"] = hitl_approvals

        return state

    def check_approval(self, state: AgentState) -> str:
        """
        Check HITL approval status for conditional routing.
        """
        if state.get("approval_received"):
            last_approval = state.get("hitl_approvals", [{}])[-1]
            if last_approval.get("action") in {"edit", "recommend"}:
                return "edited"
            return "approved"
        return "denied"

    def _emit_hitl_event(
        self, stage: str, agent_name: str, status: str, payload: Dict[str, Any]
    ) -> None:
        """
        Emit HITL event through callback if configured.
        """
        if self._event_callback:
            try:
                self._event_callback(stage, agent_name, status, payload)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
