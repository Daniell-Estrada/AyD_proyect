"""
Analysis Workflow orchestrating multiple agents with Human-in-the-Loop checkpoints.
"""

import logging
from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, StateGraph

from app.application.agents.analyzer import AnalyzerAgent
from app.application.agents.classifier import ClassifierAgent
from app.application.agents.documenter import DocumenterAgent
from app.application.agents.parser_agent import ParserAgent
from app.application.agents.state import AgentState
from app.application.agents.translator import TranslatorAgent
from app.application.agents.validator import ValidatorAgent
from app.application.workflows.handlers.agent_node_handler import \
    AgentNodeHandler
from app.application.workflows.handlers.hitl_node_handler import \
    HitlNodeHandler
from app.domain.services.complexity_service import ComplexityAnalysisService
from app.domain.services.diagram_service import DiagramService
from app.infrastructure.llm.llm_service import LLMService
from app.infrastructure.workflows.state_store import (
    WorkflowStateStore, create_workflow_state_store)
from app.shared.config import settings

logger = logging.getLogger(__name__)


class AnalysisWorkflow:
    """
    Main workflow orchestrating all agents using LangGraph.
    Implements Human-in-the-Loop checkpoints for supervision.
    Uses Handler pattern for node execution management.
    """

    def __init__(
        self,
        hitl_callback: Optional[Callable[[str, str, Any, str], Dict[str, Any]]] = None,
        event_callback: Optional[
            Callable[[str, str, str, Dict[str, Any]], None]
        ] = None,
    ):
        self.llm_service = LLMService()
        self.complexity_service = ComplexityAnalysisService()
        self.diagram_service = DiagramService()
        self.state_store: WorkflowStateStore = create_workflow_state_store()

        self.translator = TranslatorAgent(self.llm_service)
        self.parser = ParserAgent()
        self.classifier = ClassifierAgent(
            llm_service=self.llm_service,
            complexity_service=self.complexity_service,
        )
        self.analyzer = AnalyzerAgent(
            self.llm_service,
            self.complexity_service,
            self.diagram_service,
        )
        self.validator = ValidatorAgent(
            self.llm_service,
            self.complexity_service,
        )
        self.documenter = DocumenterAgent(
            self.llm_service,
            self.diagram_service,
        )

        self._agent_handler = AgentNodeHandler(
            event_callback=event_callback,
            state_recorder=self._record_state_snapshot,
        )
        self._hitl_handler = HitlNodeHandler(
            hitl_callback=hitl_callback,
            event_callback=event_callback,
        )

        self.hitl_callback = hitl_callback
        self.enable_hitl = settings.enable_hitl
        self.event_callback = event_callback

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build LangGraph state graph."""
        workflow = StateGraph(AgentState)

        workflow.add_node("translator", self._translator_node)
        workflow.add_node("parser", self._parser_node)
        workflow.add_node("classifier", self._classifier_node)
        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("validator", self._validator_node)
        workflow.add_node("documenter", self._documenter_node)

        if self.enable_hitl:
            workflow.add_node("hitl_translation", self._hitl_translation_node)
            workflow.add_node("hitl_classification", self._hitl_classification_node)
            workflow.add_node("hitl_analysis", self._hitl_analysis_node)
            workflow.add_node("hitl_validation", self._hitl_validation_node)

        workflow.set_entry_point("translator")

        if self.enable_hitl:
            workflow.add_edge("translator", "hitl_translation")
            workflow.add_conditional_edges(
                "hitl_translation",
                self._check_hitl_approval,
                {
                    "approved": "parser",
                    "denied": "translator",
                    "edited": "parser",
                },
            )

            workflow.add_edge("parser", "classifier")
            workflow.add_edge("classifier", "hitl_classification")
            workflow.add_conditional_edges(
                "hitl_classification",
                self._check_hitl_approval,
                {
                    "approved": "analyzer",
                    "denied": "classifier",
                    "edited": "analyzer",
                },
            )

            workflow.add_edge("analyzer", "hitl_analysis")
            workflow.add_conditional_edges(
                "hitl_analysis",
                self._check_hitl_approval,
                {
                    "approved": "validator",
                    "denied": "analyzer",
                    "edited": "validator",
                },
            )

            workflow.add_edge("validator", "hitl_validation")
            workflow.add_conditional_edges(
                "hitl_validation",
                self._check_hitl_approval,
                {
                    "approved": "documenter",
                    "denied": "validator",
                    "edited": "documenter",
                },
            )
        else:
            workflow.add_edge("translator", "parser")
            workflow.add_edge("parser", "classifier")
            workflow.add_edge("classifier", "analyzer")
            workflow.add_edge("analyzer", "validator")
            workflow.add_edge("validator", "documenter")

        workflow.add_edge("documenter", END)

        return workflow.compile()

    def _translator_node(self, state: AgentState) -> AgentState:
        """Execute Translator agent node."""
        logger.info("Executing Translator Agent")
        return self._agent_handler.execute_agent(
            stage="translation",
            agent_name="TranslatorAgent",
            agent_fn=self.translator.execute,
            state=state,
            payload_builder=lambda updated_state: self._build_stage_payload(
                "translation",
                updated_state,
            ),
            requires_review=self.enable_hitl,
        )

    def _parser_node(self, state: AgentState) -> AgentState:
        """Execute Parser agent node."""
        logger.info("Executing Parser Agent")
        return self._agent_handler.execute_agent(
            stage="parsing",
            agent_name="ParserAgent",
            agent_fn=self.parser.execute,
            state=state,
            payload_builder=lambda updated_state: self._build_stage_payload(
                "parsing",
                updated_state,
            ),
        )

    def _classifier_node(self, state: AgentState) -> AgentState:
        """Execute Classifier agent node."""
        logger.info("Executing Classifier Agent")
        return self._agent_handler.execute_agent(
            stage="classification",
            agent_name="ClassifierAgent",
            agent_fn=self.classifier.execute,
            state=state,
            payload_builder=lambda updated_state: self._build_stage_payload(
                "classification",
                updated_state,
            ),
            requires_review=self.enable_hitl,
        )

    def _analyzer_node(self, state: AgentState) -> AgentState:
        """Execute Analyzer agent node."""
        logger.info("Executing Analyzer Agent")
        return self._agent_handler.execute_agent(
            stage="analysis",
            agent_name="AnalyzerAgent",
            agent_fn=self.analyzer.execute,
            state=state,
            payload_builder=lambda updated_state: self._build_stage_payload(
                "analysis",
                updated_state,
            ),
            requires_review=self.enable_hitl,
        )

    def _validator_node(self, state: AgentState) -> AgentState:
        """Execute Validator agent node."""
        logger.info("Executing Validator Agent")
        return self._agent_handler.execute_agent(
            stage="validation",
            agent_name="ValidatorAgent",
            agent_fn=self.validator.execute,
            state=state,
            payload_builder=lambda updated_state: self._build_stage_payload(
                "validation",
                updated_state,
            ),
            requires_review=self.enable_hitl,
        )

    def _documenter_node(self, state: AgentState) -> AgentState:
        """Execute Documenter agent node."""
        logger.info("Executing Documenter Agent")
        return self._agent_handler.execute_agent(
            stage="documentation",
            agent_name="DocumenterAgent",
            agent_fn=self.documenter.execute,
            state=state,
            payload_builder=lambda updated_state: self._build_stage_payload(
                "documentation",
                updated_state,
            ),
        )

    def _hitl_translation_node(self, state: AgentState) -> AgentState:
        """HITL checkpoint after translation."""
        return self._hitl_handler.request_review(
            stage="translation",
            agent_name="TranslatorAgent",
            state=state,
            output_key="translated_pseudocode",
            reasoning_key="translation_reasoning",
        )

    def _hitl_classification_node(self, state: AgentState) -> AgentState:
        """HITL checkpoint after classification."""
        return self._hitl_handler.request_review(
            stage="classification",
            agent_name="ClassifierAgent",
            state=state,
            output_key="paradigm",
            reasoning_key="paradigm_reasoning",
        )

    def _hitl_analysis_node(self, state: AgentState) -> AgentState:
        """HITL checkpoint after analysis."""
        complexities = {
            "worst_case": state.get("complexity_worst_case"),
            "best_case": state.get("complexity_best_case"),
            "average_case": state.get("complexity_average_case"),
        }
        state["_analysis_summary"] = complexities
        state["_analysis_summary_patterns"] = state.get("metadata", {}).get("patterns")
        state["_analysis_summary_diagrams"] = state.get("diagrams")

        return self._hitl_handler.request_review(
            stage="analysis",
            agent_name="AnalyzerAgent",
            state=state,
            output_key="_analysis_summary",
            reasoning_key="analysis_steps",
        )

    def _hitl_validation_node(self, state: AgentState) -> AgentState:
        """HITL checkpoint after validation."""
        validation_summary = {
            "passed": state.get("validation_passed", False),
            "issues": state.get("validation_issues", []),
            "results": state.get("validation_results", {}),
        }
        state["_validation_summary"] = validation_summary
        state["validation_reasoning"] = validation_summary["results"]

        return self._hitl_handler.request_review(
            stage="validation",
            agent_name="ValidatorAgent",
            state=state,
            output_key="_validation_summary",
            reasoning_key="validation_reasoning",
        )

    def _check_hitl_approval(self, state: AgentState) -> str:
        """
        Check HITL approval status for conditional routing.

        Returns:
            Routing decision: "approved", "denied", or "edited"
        """
        return self._hitl_handler.check_approval(state)

    def _build_stage_payload(self, stage: str, state: AgentState) -> Dict[str, Any]:
        """Assemble rich payload metadata for each agent stage."""
        if stage == "translation":
            pseudocode = state.get("translated_pseudocode") or ""
            return {
                "output": pseudocode,
                "reasoning": state.get("translation_reasoning"),
                "pseudocode_preview": pseudocode[:400],
            }

        if stage == "parsing":
            metadata = state.get("metadata", {}) or {}
            return {
                "output": state.get("parsed_ast"),
                "reasoning": metadata.get("notes") or state.get("parser_warnings"),
                "parsing_success": state.get("parsing_success"),
                "errors": state.get("parsing_errors"),
            }

        if stage == "classification":
            return {
                "output": state.get("paradigm"),
                "reasoning": state.get("paradigm_reasoning"),
                "confidence": state.get("paradigm_confidence"),
            }

        if stage == "analysis":
            summary = {
                "worst_case": state.get("complexity_worst_case"),
                "best_case": state.get("complexity_best_case"),
                "average_case": state.get("complexity_average_case"),
                "tight_bounds": state.get("tight_bounds"),
            }
            return {
                "output": summary,
                "reasoning": state.get("analysis_steps"),
                "paradigm": state.get("paradigm"),
                "patterns": state.get("metadata", {}).get("patterns"),
                "diagrams": state.get("diagrams"),
                "retry_reason": state.get("analysis_retry_reason"),
            }

        if stage == "validation":
            validation_summary = {
                "passed": state.get("validation_passed", False),
                "issues": state.get("validation_issues", []),
                "results": state.get("validation_results", {}),
            }
            return {
                "output": validation_summary,
                "reasoning": state.get("validation_results"),
            }

        if stage == "documentation":
            return {
                "output": state.get("summary"),
                "reasoning": state.get("detailed_explanation"),
                "algorithm_name": state.get("algorithm_name"),
            }

        return {}

    def run(self, user_input: str, session_id: str) -> Dict[str, Any]:
        """
        Run the complete analysis workflow.
        """
        logger.info(f"Starting workflow for session: {session_id}")

        initial_state: AgentState = {
            "session_id": session_id,
            "user_input": user_input,
            "status": "processing",
            "current_stage": "initialization",
            "errors": [],
            "hitl_approvals": [],
            "analysis_steps": [],
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "total_duration_ms": 0.0,
            "agent_metrics": [],
            "requires_approval": False,
            "approval_received": None,
            "metadata": {},
            "translated_pseudocode": None,
            "translation_reasoning": None,
            "parsing_success": False,
            "parsing_errors": None,
            "parser_warnings": [],
            "parsed_ast": None,
            "validation_passed": False,
            "validation_results": {},
            "validation_errors": None,
            "retry_count": 0,
            "max_retries": 3,
            "paradigm": None,
            "paradigm_confidence": None,
            "paradigm_reasoning": None,
            "analysis_technique": None,
            "recurrence_relation": None,
            "complexity_best_case": None,
            "complexity_worst_case": None,
            "complexity_average_case": None,
            "tight_bounds": None,
            "diagrams": {},
            "algorithm_name": None,
            "summary": None,
            "detailed_explanation": None,
            "final_output": None,
        }

        self._record_state_snapshot("initialization", initial_state)

        try:
            final_state = self.graph.invoke(initial_state)
            self._record_state_snapshot("completed", final_state)

            logger.info(
                f"Workflow completed for session: {session_id}. "
                f"Status: {final_state.get('status')}"
            )

            return final_state

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            initial_state["status"] = "failed"
            initial_state["errors"].append(f"Workflow execution failed: {str(e)}")
            self._record_state_snapshot("failed", initial_state)
            return initial_state

    def _record_state_snapshot(self, stage: str, state: AgentState) -> None:
        """Persist workflow snapshots for observability and resume support."""

        if not self.state_store:
            return

        session_id = state.get("session_id")
        if not session_id:
            return

        try:
            self.state_store.save_state(session_id, stage, state)
        except Exception as exc:
            logger.warning(
                "Failed to persist workflow state for %s at stage %s: %s",
                session_id,
                stage,
                exc,
            )
