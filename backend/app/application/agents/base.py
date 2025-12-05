"""Shared helpers for agent implementations."""

from abc import ABC
from typing import Optional, Tuple

from app.application.agents.state import AgentState
from app.infrastructure.llm.llm_service import LLMService, LLMUsageMetrics


class BaseAgent(ABC):
    """Provide metric tracking and convenience LLM helpers for agents."""

    def __init__(self, name: str, llm_service: Optional[LLMService] = None):
        self.name = name
        self._llm_service = llm_service

    def _record_metrics(self, state: AgentState, metrics: LLMUsageMetrics) -> None:
        """Accumulate LLM cost/latency metrics on the shared state."""

        state["total_cost_usd"] = (
            state.get("total_cost_usd", 0.0) + metrics.estimated_cost_usd
        )
        state["total_tokens"] = state.get("total_tokens", 0) + metrics.total_tokens
        state["total_duration_ms"] = (
            state.get("total_duration_ms", 0.0) + metrics.duration_ms
        )

        agent_metrics = state.get("agent_metrics", [])
        agent_metrics.append(metrics)
        state["agent_metrics"] = agent_metrics

    def _invoke_llm(
        self,
        state: AgentState,
        system_prompt: str,
        user_prompt: str,
        use_fallback: bool = False,
    ) -> Tuple[str, LLMUsageMetrics]:
        """Invoke the configured LLM and automatically record metrics."""

        if self._llm_service:
            print(
                "primary provider/model:",
                self._llm_service.primary_provider,
                self._llm_service.primary_model,
            )
            print(
                "fallback provider/model:",
                self._llm_service.fallback_provider,
                self._llm_service.fallback_model,
            )

        if not self._llm_service:
            raise RuntimeError("LLM service is not configured for this agent")

        response_text, metrics = self._llm_service.invoke(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            use_fallback=use_fallback,
        )
        self._record_metrics(state, metrics)
        print("LLM Response:", response_text)
        print("LLM Metrics:", metrics)
        return response_text, metrics

    def _append_error(self, state: AgentState, message: str) -> None:
        """Utility to accumulate human-readable errors."""

        errors = state.get("errors", [])
        errors.append(message)
        state["errors"] = errors
