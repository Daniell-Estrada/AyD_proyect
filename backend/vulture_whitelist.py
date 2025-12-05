"""Whitelist expressions for symbols that frameworks touch dynamically.

The statements below intentionally reference attributes so Vulture recognizes
them as used even though they're resolved via reflection at runtime.
"""

from app.application.agents import state as agent_state
from app.domain.models import analysis as analysis_models
from app.domain.models.ast import ASTMetadata, ASTNode, BinOp, ForLoop, GraphTraversal, Parameter
from app.domain.repositories import interfaces as repo_interfaces
from app.domain.services.complexity_service import ComplexityAnalysisService
from app.domain.services.diagram_service import DiagramService
from app.infrastructure.llm.llm_service import LLMService
from app.infrastructure.parser.language_parser import LanguageParser
from app.infrastructure.parser.transformer import ASTTransformer
from app.infrastructure.persistence.mongodb_service import MongoDBService
from app.infrastructure.persistence.repositories import MongoAgentEventRepository
from app.presentation.api import app as api_app
from app.shared.config import Settings

for _field in (
    "translated_pseudocode",
    "translation_reasoning",
    "parsed_ast",
    "parsing_success",
    "parsing_errors",
    "paradigm_confidence",
    "paradigm_reasoning",
    "complexity_best_case",
    "complexity_worst_case",
    "complexity_average_case",
    "tight_bounds",
    "analysis_technique",
    "recurrence_relation",
    "validation_errors",
    "detailed_explanation",
    "current_stage",
    "requires_approval",
    "approval_received",
    "user_feedback",
    "total_cost_usd",
    "total_duration_ms",
    "retry_count",
):
    getattr(agent_state.AgentState, _field)

for _field in (
    "current_stage",
    "created_at",
    "updated_at",
    "workflow_state",
    "total_cost_usd",
    "total_duration_ms",
):
    getattr(analysis_models.Session, _field)

analysis_models.ComplexitySummary.tight_bounds
analysis_models.AnalysisResult.created_at
analysis_models.AgentEvent.timestamp

analysis_models.AgentEventStatus.STARTED
analysis_models.AgentEventStatus.RESOLVED
analysis_models.HitlAction.DENY
analysis_models.HitlAction.EDIT
analysis_models.HitlResponse.resolved_at

ASTMetadata.complexity_hints
ASTMetadata.is_recursive
ASTMetadata.pattern_type
ASTNode.set_position
ASTNode.get_position
ForLoop.preserve_counter_value
BinOp.short_circuit
Parameter.param_type
GraphTraversal.end_node

for _attr, _value in vars(ASTTransformer).items():
    if callable(_value) and not _attr.startswith("_"):
        _value

for _method in (
    DiagramService.generate_flowchart,
    DiagramService.generate_recursion_tree,
    DiagramService.generate_call_graph,
    DiagramService.generate_architecture_diagram,
    DiagramService.networkx_to_mermaid,
):
    _method

for _method in (
    ComplexityAnalysisService.detect_paradigm,
    ComplexityAnalysisService.analyze_loops,
    ComplexityAnalysisService.solve_recurrence_relation,
    ComplexityAnalysisService.analyze_dynamic_programming,
    ComplexityAnalysisService.validate_with_simulation,
):
    _method

LLMService.invoke_langchain
LLMService.get_available_models

LanguageParser.parse_file

for _method in (
    MongoDBService.get_analysis_results,
    MongoDBService.get_sessions_by_status,
    MongoDBService.delete_session,
):
    _method

MongoAgentEventRepository.list_for_session
repo_interfaces.AgentEventRepository.list_for_session

api_app.socket_app
api_app.start_analysis
api_app.hitl_respond
api_app.root
api_app.health_check
api_app.get_analysis_results
api_app.list_sessions

Settings.model_config
Settings.secret_key
Settings.max_session_duration
