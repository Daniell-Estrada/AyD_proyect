"""
Documenter Agent for Algorithm Analysis Application.
"""

import json
import logging
from typing import Any, Dict

from app.application.agents.base import BaseAgent
from app.application.agents.state import AgentState
from app.domain.services.diagram_service import DiagramService
from app.infrastructure.llm.llm_service import LLMService

logger = logging.getLogger(__name__)


class DocumenterAgent(BaseAgent):
    """
    Agent responsible for generating final documentation.
    Creates diagrams, summaries, and detailed explanations.
    """

    def __init__(
        self,
        llm_service: LLMService,
        diagram_service: DiagramService,
    ):
        super().__init__(name="DocumenterAgent", llm_service=llm_service)
        self.llm_service = llm_service
        self.diagram_service = diagram_service
        self.name = "DocumenterAgent"

    def execute(self, state: AgentState) -> AgentState:
        """
        Execute documentation generation.
        """
        logger.info(f"{self.name}: Starting documentation")

        try:
            algorithm_name = self._generate_algorithm_name(state)
            state["algorithm_name"] = algorithm_name

            diagrams = self._generate_all_diagrams(state)
            existing_diagrams = state.get("diagrams", {})
            existing_diagrams.update(diagrams)
            state["diagrams"] = existing_diagrams

            summary = self._generate_summary(state)
            state["summary"] = summary

            explanation = self._generate_explanation(state)
            state["detailed_explanation"] = explanation

            final_output = self._compile_final_output(state)
            state["final_output"] = final_output
            state["current_stage"] = "documentation_complete"
            state["status"] = "completed"

            logger.info(f"{self.name}: Documentation completed")

        except Exception as e:
            logger.error(f"{self.name}: Documentation failed: {e}")
            self._append_error(state, f"Documentation failed: {str(e)}")

        return state

    def _generate_algorithm_name(self, state: AgentState) -> str:
        """Generate or extract algorithm name."""
        ast_dict = state.get("parsed_ast", {})
        if ast_dict.get("type") == "Program":
            for stmt in ast_dict.get("statements", []):
                if stmt.get("type") == "SubroutineDef":
                    name = stmt.get("name")
                    if name:
                        return name

        paradigm = state.get("paradigm", "algorithm")
        user_input = state.get("user_input", "")[:200]

        system_prompt = """Generate a concise, descriptive name for an algorithm. 
        Output only the name, no explanation."""

        user_prompt = f"""Algorithm description: 
            {user_input} Paradigm: {paradigm} 
        Generate a name (e.g., "MergeSort", "DijkstraShortestPath")."""

        try:
            name, _ = self._invoke_llm(state, system_prompt, user_prompt)
            return name.strip().replace(" ", "")
        except:
            return "Algorithm"

    def _generate_all_diagrams(self, state: AgentState) -> Dict[str, str]:
        """Generate all relevant diagrams."""
        diagrams = {}

        try:
            ast_dict = state.get("parsed_ast")
            if ast_dict:
                pass

            complexities = {
                "Best Case": state.get("complexity_best_case", "Ω(?)"),
                "Average Case": state.get("complexity_average_case", "Θ(?)"),
                "Worst Case": state.get("complexity_worst_case", "O(?)"),
            }
            diagrams["complexity_comparison"] = (
                self.diagram_service.generate_complexity_comparison(complexities)
            )

        except Exception as e:
            logger.warning(f"Diagram generation failed: {e}")

        return diagrams

    def _generate_summary(self, state: AgentState) -> str:
        """Generate concise summary of analysis."""
        algorithm_name = state.get("algorithm_name", "Algorithm")
        paradigm = state.get("paradigm", "unknown")
        worst_case = state.get("complexity_worst_case", "O(?)")
        best_case = state.get("complexity_best_case", "Ω(?)")

        human_paradigm = paradigm.replace("_", " ").title()
        tail_note = (
            " (tail-recursive style, simula iteración)"
            if paradigm == "tail_recursive_iterative"
            else ""
        )

        summary = f"""**{algorithm_name}** - {human_paradigm} Algorithm{tail_note}

        **Complexity Analysis:**
        - **Best Case:** {best_case}
        - **Worst Case:** {worst_case}
        - **Average Case:** {state.get("complexity_average_case", "Θ(?)")}
        """

        if state.get("tight_bounds"):
            summary += f"- **Tight Bounds:** {state['tight_bounds']}\n"

        return summary

    def _generate_explanation(self, state: AgentState) -> str:
        """Generate detailed explanation using LLM."""
        system_prompt = (
            "You are an expert algorithm educator. Respond concisely for ops review. "
            "Return 3-5 bullet lines covering: algorithm idea, why the worst/average/best "
            "bounds hold, and one key insight. No headings, no code fences, under 120 words. "
            "Use the provided complexity values verbatim; do not invent different bounds."
        )

        analysis_steps = state.get("analysis_steps", [])
        paradigm = state.get("paradigm", "unknown")
        worst_case = state.get("complexity_worst_case", "O(?)")
        best_case = state.get("complexity_best_case", "O(?)")
        average_case = state.get("complexity_average_case", "O(?)")

        user_prompt = (
            "Paradigm: {paradigm}\n"
            "Worst Case: {worst}\n"
            "Average Case: {average}\n"
            "Best Case: {best}\n"
            "Analysis Steps JSON: {steps}\n"
            "Keep it short and focused on complexity rationale only."
        ).format(
            paradigm=paradigm,
            worst=worst_case,
            average=average_case,
            best=best_case,
            steps=json.dumps(analysis_steps, indent=2),
        )

        try:
            explanation, _ = self._invoke_llm(
                state=state,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            return explanation

        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return "Explanation generation failed."

    def _compile_final_output(self, state: AgentState) -> Dict[str, Any]:
        """Compile all results into final output structure."""
        return {
            "algorithm_name": state.get("algorithm_name"),
            "pseudocode": state.get("translated_pseudocode"),
            "paradigm": {
                "name": state.get("paradigm"),
                "confidence": state.get("paradigm_confidence"),
                "reasoning": state.get("paradigm_reasoning"),
            },
            "complexity": {
                "worst_case": state.get("complexity_worst_case"),
                "best_case": state.get("complexity_best_case"),
                "average_case": state.get("complexity_average_case"),
                "tight_bounds": state.get("tight_bounds"),
            },
            "analysis": {
                "technique": state.get("analysis_technique"),
                "steps": state.get("analysis_steps", []),
                "recurrence": state.get("recurrence_relation"),
            },
            "validation": {
                "passed": state.get("validation_passed"),
                "results": state.get("validation_results"),
            },
            "documentation": {
                "summary": state.get("summary"),
                "detailed_explanation": state.get("detailed_explanation"),
            },
            "diagrams": state.get("diagrams", {}),
            "metrics": {
                "total_cost_usd": state.get("total_cost_usd"),
                "total_tokens": state.get("total_tokens"),
                "total_duration_ms": state.get("total_duration_ms"),
                "agent_metrics": [
                    {
                        "provider": m.provider,
                        "model": m.model,
                        "tokens": m.total_tokens,
                        "cost_usd": m.estimated_cost_usd,
                        "duration_ms": m.duration_ms,
                    }
                    for m in state.get("agent_metrics", [])
                ],
            },
        }
