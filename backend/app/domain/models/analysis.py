"""
Domain models describing sessions, results, and HITL events.

This module defines the core domain entities for algorithm analysis:
    - Session: Tracks analysis workflow state and progress
    - AnalysisResult: Stores complete complexity analysis output
    - AgentEvent: Records agent lifecycle events
    - HitlResponse: Captures human-in-the-loop decisions

These models follow Domain-Driven Design principles and represent
the ubiquitous language of the algorithm analysis domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SessionStatus(str, Enum):
    """
    Canonical session lifecycle states.
    
    State transitions:
        PENDING -> PROCESSING -> COMPLETED
                            -> FAILED
    
    Invariants:
        - New sessions start in PENDING state
        - COMPLETED and FAILED are terminal states
        - PROCESSING indicates active workflow execution
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
    
    Persistence:
        Fields are persisted to MongoDB and exposed through REST API.
        All fields remain part of the contract even if not referenced locally.
    
    Invariants:
        - session_id must be unique across all sessions
        - status must follow valid state transitions
        - created_at <= updated_at
        - total_cost_usd >= 0
        - total_tokens >= 0
    
    Example:
        >>> session = Session(
        ...     session_id="abc123",
        ...     user_input="Calculate complexity of merge sort",
        ...     status=SessionStatus.PENDING
        ... )
    """

    session_id: str
    user_input: str
    status: SessionStatus = SessionStatus.PENDING
    current_stage: str = "initialization"  # noqa: F841 - persisted contract
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(  # noqa: F841 - persisted contract
        default_factory=datetime.utcnow
    )
    updated_at: datetime = field(  # noqa: F841 - persisted contract
        default_factory=datetime.utcnow
    )
    workflow_state: Dict[str, Any] = field(  # noqa: F841 - stored workflow state
        default_factory=dict
    )
    total_cost_usd: float = 0.0  # noqa: F841 - aggregated LLM usage costs
    total_tokens: int = 0  # aggregated LLM token consumption
    total_duration_ms: float = 0.0  # noqa: F841 - aggregated processing time
    hitl_approvals: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ComplexitySummary:
    """
    Normalized representation of algorithm complexity bounds.
    
    Encapsulates Big-O, Omega, and Theta notation results.
    
    Attributes:
        worst_case: Upper bound (Big-O notation), e.g., "O(n log n)"
        best_case: Lower bound (Omega notation), e.g., "Ω(n)"
        average_case: Average case (Theta notation), e.g., "Θ(n log n)"
        tight_bounds: Tight asymptotic bound when worst = average = best
    
    Example:
        >>> complexity = ComplexitySummary(
        ...     worst_case="O(n log n)",
        ...     best_case="Ω(n log n)",
        ...     average_case="Θ(n log n)",
        ...     tight_bounds="Θ(n log n)"
        ... )
    """

    worst_case: Optional[str] = None
    best_case: Optional[str] = None
    average_case: Optional[str] = None
    tight_bounds: Optional[str] = None  # noqa: F841 - optional API field


@dataclass
class AnalysisResult:
    """
    Complete algorithm complexity analysis output.
    
    Aggregates all analysis artifacts including AST, complexity bounds,
    step-by-step analysis, diagrams, and validation results.
    
    Persistence:
        Stored in MongoDB for historical queries and API retrieval.
    
    Invariants:
        - session_id must reference valid Session
        - ast must be valid JSON-serializable dictionary
        - paradigm must be recognized algorithmic paradigm
        - complexity must contain at least worst_case
    
    Example:
        >>> result = AnalysisResult(
        ...     session_id="abc123",
        ...     algorithm_name="MergeSort",
        ...     pseudocode="...",
        ...     ast={...},
        ...     paradigm="divide_and_conquer",
        ...     complexity=ComplexitySummary(worst_case="O(n log n)"),
        ...     analysis_steps=[...],
        ...     diagrams={"recursion_tree": "graph TD..."},
        ...     validation={"valid": True}
        ... )
    """

    session_id: str
    algorithm_name: str
    pseudocode: str
    ast: Dict[str, Any]
    paradigm: str
    complexity: ComplexitySummary
    analysis_steps: List[Dict[str, Any]]
    diagrams: Dict[str, str]
    validation: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class AgentEventStatus(str, Enum):
    """
    Lifecycle states emitted by agents and workflow nodes.
    
    State transitions:
        PENDING -> STARTED -> COMPLETED
                         -> FAILED
                         -> PENDING (for HITL)
                         -> RESOLVED (after HITL)
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
    
    Attributes:
        session_id: Reference to parent session
        stage: Workflow stage name (e.g., "translation", "analysis")
        agent_name: Name of agent emitting event
        status: Current lifecycle status
        payload: Arbitrary event-specific data
        timestamp: Event occurrence time (UTC)
    
    Example:
        >>> event = AgentEvent(
        ...     session_id="abc123",
        ...     stage="analysis",
        ...     agent_name="AnalyzerAgent",
        ...     status=AgentEventStatus.COMPLETED,
        ...     payload={"complexity": "O(n log n)"}
        ... )
    """

    session_id: str
    stage: str
    agent_name: str
    status: AgentEventStatus
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(  # noqa: F841 - stored for audit trail
        default_factory=datetime.utcnow
    )


class HitlAction(str, Enum):
    """
    Human-in-the-Loop decision actions.
    
    Actions:
        APPROVE: Accept agent output as-is
        DENY: Reject agent output (may trigger retry)
        EDIT: Manually correct agent output
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
    
    Invariants:
        - If action is EDIT, edited_output must be provided
        - resolved_at >= session.created_at
    
    Example:
        >>> response = HitlResponse(
        ...     session_id="abc123",
        ...     stage="classification",
        ...     action=HitlAction.EDIT,
        ...     feedback="Paradigm should be dynamic_programming",
        ...     edited_output={"paradigm": "dynamic_programming"}
        ... )
    """

    session_id: str
    stage: str
    action: HitlAction
    feedback: Optional[str] = None
    edited_output: Optional[Any] = None
    resolved_at: datetime = field(  # noqa: F841 - stored for compliance logs
        default_factory=datetime.utcnow
    )
