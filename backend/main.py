"""
Main entry point for Algorithm Complexity Analyzer.
Runs the FastAPI server with Socket.IO support.
"""

import uvicorn

from app.shared.config import settings


def main():
    """Start the API server."""
    print("=" * 60)
    print("  Algorithm Complexity Analyzer - Backend Server")
    print("=" * 60)
    print(f"  Environment: {settings.app_env}")
    print(f"  Host: {settings.api_host}:{settings.api_port}")
    print(f"  Debug: {settings.api_debug}")
    print(f"  HITL Enabled: {settings.enable_hitl}")
    print(f"  Primary LLM: {settings.primary_llm_provider}/{settings.primary_llm_model}")
    print("=" * 60)

    uvicorn.run(
        "app.presentation.api.app:socket_app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

