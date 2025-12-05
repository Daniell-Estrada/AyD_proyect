"""Shared dependency singletons for the presentation layer."""

from app.shared.di import (get_agent_event_repository, get_analysis_use_cases,
                           get_hitl_use_cases, get_session_use_cases)

session_use_cases = get_session_use_cases()
analysis_use_cases = get_analysis_use_cases()
hitl_use_cases = get_hitl_use_cases()
agent_event_repository = get_agent_event_repository()

__all__ = [
    "session_use_cases",
    "analysis_use_cases",
    "hitl_use_cases",
    "agent_event_repository",
]
