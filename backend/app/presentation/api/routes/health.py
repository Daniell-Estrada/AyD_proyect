"""Health and metadata endpoints."""

from fastapi import APIRouter

from app.infrastructure.persistence.mongodb_service import mongodb_service


def create_health_router() -> APIRouter:
    router = APIRouter()

    @router.get("/")
    async def root():
        return {
            "name": "Algorithm Complexity Analyzer API",
            "version": "1.0.0",
            "status": "running",
        }

    @router.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "database": "connected" if mongodb_service._connected else "disconnected",
        }

    return router


__all__ = ["create_health_router"]
