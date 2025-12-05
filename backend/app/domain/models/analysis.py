"""
Domain models describing sessions, results, and HITL events.
Defines data structures for tracking analysis sessions, results,
and human-in-the-loop interactions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class SessionStatus(str, Enum):
    """
    Canonical session lifecycle states.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Session:
    """
    Analysis session aggregate tracked across layers.

    Represents a single algorithm analysis request from initiation
    through completion. Aggregates workflow state, metrics, and HITL approvals.
    """

    session_id: str
    user_input: str
    status: SessionStatus = SessionStatus.PENDING
    current_stage: str = "initialization"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    workflow_state: Dict[str, Any] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    hitl_approvals: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ComplexitySummary:
    """
    Normalized representation of algorithm complexity bounds.

    Encapsulates Big-O, Omega, and Theta notation results.
    """

    worst_case: Optional[str] = None
    best_case: Optional[str] = None
    average_case: Optional[str] = None
    tight_bounds: Optional[str] = None  


@dataclass
class AnalysisResult:
    """
    Complete algorithm complexity analysis output.

    Aggregates all analysis artifacts including AST, complexity bounds,
    step-by-step analysis, diagrams, and validation results.
    """

    session_id: str
    algorithm_name: str
    pseudocode: str
    ast: Dict[str, Any]
    paradigm: Union[str, Dict[str, Any]]
    complexity: ComplexitySummary
    analysis_steps: List[Dict[str, Any]]
    diagrams: Dict[str, str]
    validation: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class AgentEventStatus(str, Enum):
    """
    Lifecycle states emitted by agents and workflow nodes.
    """

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"
    RESOLVED = "resolved"


@dataclass
class AgentEvent:
    """
    Event raised whenever an agent or HITL checkpoint changes state.
    Enables real-time progress tracking, debugging, and audit trails.
    """

    session_id: str
    stage: str
    agent_name: str
    status: AgentEventStatus
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class HitlAction(str, Enum):
    """
    Human-in-the-Loop decision actions.
    """

    APPROVE = "approve"
    DENY = "deny"
    EDIT = "edit"


@dataclass
class HitlResponse:
    """
    Decision captured from a human reviewer during HITL checkpoint.

    Records human intervention, feedback, and corrections made
    during workflow execution.
    """

    session_id: str
    stage: str
    action: HitlAction
    feedback: Optional[str] = None
    edited_output: Optional[Any] = None
    resolved_at: datetime = field(default_factory=datetime.utcnow)
