"""FastAPI entrypoint wiring routers, Socket.IO, and workflow orchestration."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.infrastructure.persistence.mongodb_service import mongodb_service
from app.presentation.api import socket_server
from app.presentation.api.hitl.state import HitlState
from app.presentation.api.realtime.channels import SessionChannelManager
from app.presentation.api.routes.health import create_health_router
from app.presentation.api.routes.sessions import create_sessions_router
from app.presentation.api.services.orchestrator import WorkflowOrchestrator
from app.shared.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""

    logger.info("Starting application...")
    await mongodb_service.connect()
    logger.info("Application started successfully")

    yield

    logger.info("Shutting down application...")
    await mongodb_service.disconnect()
    logger.info("Application shutdown complete")


app = FastAPI(
    title="Algorithm Complexity Analyzer API",
    description="AI-powered algorithm complexity analysis with multi-agent system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

channel_manager = SessionChannelManager()
hitl_state = HitlState()
orchestrator = WorkflowOrchestrator(
    channel_manager=channel_manager,
    hitl_state=hitl_state,
)

app.include_router(create_health_router())
app.include_router(create_sessions_router(orchestrator))

socket_app = socket_server.init_socket_app(app, orchestrator)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.presentation.api.app:socket_app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level=settings.log_level.lower(),
    )
