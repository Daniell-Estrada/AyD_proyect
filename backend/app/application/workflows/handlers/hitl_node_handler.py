"""
HITL (Human-in-the-Loop) node handler.
Manages human review checkpoints in workflow.
"""

import logging
from typing import Any, Callable, Dict, Optional

from app.application.agents.state import AgentState

logger = logging.getLogger(__name__)


class HitlNodeHandler:
    """
    Handles Human-in-the-Loop checkpoint nodes.
    
    Requests human review, captures feedback, and updates state
    with approval decisions and edited outputs.
    """

    def __init__(
        self,
        hitl_callback: Optional[Callable[[str, str, Any, str], Dict[str, Any]]] = None,
        event_callback: Optional[Callable[[str, str, str, Dict[str, Any]], None]] = None,
    ):
        """
        Initialize HITL node handler.

        Args:
            hitl_callback: Optional callback for HITL requests.
                          Signature: (stage, agent_name, output, reasoning) -> {action, feedback, edited_output}
            event_callback: Optional callback for event emission.
                           Signature: (stage, agent_name, status, payload) -> None
        """
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
        Request human review at workflow checkpoint.

        Emits pending event, invokes callback for human decision,
        records approval/feedback, and updates state.

        Args:
            stage: Workflow stage name
            agent_name: Name of agent under review
            state: Current workflow state
            output_key: State key containing agent output to review
            reasoning_key: State key containing agent reasoning

        Returns:
            Updated state with approval decision and feedback
        """
        # Auto-approve if no callback configured
        if not self._hitl_callback:
            state["approval_received"] = True
            return state

        # Extract output and reasoning from state
        output = state.get(output_key, "")
        reasoning = state.get(reasoning_key, "")

        # Reset approval markers before requesting feedback.
        state["approval_received"] = False
        state["user_feedback"] = None

        def _preview(value: Any) -> str:
            text = "" if value is None else str(value)
            return text[:400]

        # Emit pending event
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

        # Invoke HITL callback
        try:
            response = self._hitl_callback(
                stage=stage,
                agent_name=agent_name,
                output=output,
                reasoning=reasoning,
            )
        except Exception as e:
            logger.error(f"HITL callback failed: {e}", exc_info=True)
            # Default to approval on error
            response = {"action": "approve", "feedback": None}

        # Emit resolved event
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

        # Process response
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

        # Apply edits if provided
        if action == "edit" and response.get("edited_output"):
            state[output_key] = response["edited_output"]
            state["approval_received"] = True

        # Record HITL approval
        hitl_approvals = state.get("hitl_approvals", [])
        hitl_approvals.append(response)
        state["hitl_approvals"] = hitl_approvals

        return state

    def check_approval(self, state: AgentState) -> str:
        """
        Check HITL approval status for conditional routing.

        Args:
            state: Current workflow state

        Returns:
            Routing decision: "approved", "denied", or "edited"
        """
        if state.get("approval_received"):
            # Check if output was edited
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

        Args:
            stage: Workflow stage name
            agent_name: Name of agent under review
            status: Event status (pending, resolved)
            payload: Additional event data
        """
        if self._event_callback:
            try:
                self._event_callback(stage, agent_name, status, payload)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
