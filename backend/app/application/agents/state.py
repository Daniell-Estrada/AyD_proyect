"""
Agent system state definitions for LangGraph workflow.
Defines the shared state passed between agents in the analysis pipeline.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from app.infrastructure.llm.llm_service import LLMUsageMetrics


@dataclass
class LLMUsageMetrics:
    provider: str = "unknown"
    model: str = "unknown"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    estimated_cost_usd: float = 0.0


AgentState = TypedDict(
    "AgentState",
    {
        "session_id": str,
        "user_input": str,
        "metadata": Dict[str, Any],
        "translated_pseudocode": Optional[str],
        "translation_reasoning": Optional[str],
        "parsed_ast": Optional[Dict[str, Any]],
        "parsing_success": bool,
        "parsing_errors": Optional[List[str]],
        "parser_warnings": Optional[List[str]],
        "paradigm": Optional[str],
        "paradigm_confidence": Optional[float],
        "paradigm_reasoning": Optional[str],
        "complexity_best_case": Optional[str],
        "complexity_worst_case": Optional[str],
        "complexity_average_case": Optional[str],
        "tight_bounds": Optional[str],
        "analysis_technique": Optional[str],
        "analysis_steps": List[Dict[str, Any]],
        "recurrence_relation": Optional[str],
        "validation_results": Dict[str, Any],
        "validation_passed": bool,
        "validation_errors": Optional[List[str]],
        "diagrams": Dict[str, str],
        "algorithm_name": Optional[str],
        "summary": Optional[str],
        "detailed_explanation": Optional[str],
        "hitl_approvals": List[Dict[str, Any]],
        "current_stage": str,
        "requires_approval": bool,
        "approval_received": Optional[bool],
        "user_feedback": Optional[str],
        "total_cost_usd": float,
        "total_tokens": int,
        "total_duration_ms": float,
        "agent_metrics": List[LLMUsageMetrics],
        "errors": List[str],
        "retry_count": int,
        "max_retries": int,
        "status": str,
        "final_output": Optional[Dict[str, Any]],
    },
    total=False,
)
