"""
MongoDB Service Module for managing sessions, analysis results, and agent logs.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import PyMongoError

from app.shared.config import settings

logger = logging.getLogger(__name__)


class MongoDBService:
    """
    Service for interacting with MongoDB.
    Manages sessions, analysis results, and agent workflow states.
    """

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self._connected = False

    async def connect(self):
        """Establish connection to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(settings.mongodb_uri)
            self.db = self.client[settings.mongodb_db_name]

            await self.client.admin.command("ping")
            self._connected = True

            logger.info(f"Connected to MongoDB: {settings.mongodb_db_name}")

            await self._create_indexes()

        except PyMongoError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("Disconnected from MongoDB")

    async def _create_indexes(self):
        """Create necessary indexes for collections."""
        await self.db.sessions.create_index("session_id", unique=True)
        await self.db.sessions.create_index("created_at")
        await self.db.sessions.create_index("status")
        await self.db.analysis_results.create_index("session_id")
        await self.db.analysis_results.create_index("created_at")
        await self.db.agent_logs.create_index("session_id")
        await self.db.agent_logs.create_index("timestamp")

    async def create_session(
        self,
        session_id: str,
        user_input: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new analysis session.
        """
        session = {
            "session_id": session_id,
            "user_input": user_input,
            "status": "pending",
            "current_stage": "initialization",
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "workflow_state": {},
            "hitl_approvals": [],
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "total_duration_ms": 0,
        }

        await self.db.sessions.insert_one(session)
        logger.info(f"Created session: {session_id}")

        return session

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        return await self.db.sessions.find_one({"session_id": session_id})

    async def update_session(
        self, session_id: str, update_data: Dict[str, Any]
    ) -> bool:
        """
        Update session data.
        """
        update_data["updated_at"] = datetime.utcnow()

        result = await self.db.sessions.update_one(
            {"session_id": session_id}, {"$set": update_data}
        )

        return result.modified_count > 0

    async def update_session_stage(self, session_id: str, stage: str) -> bool:
        """Update current workflow stage of session."""
        return await self.update_session(session_id, {"current_stage": stage})

    async def update_session_status(
        self, session_id: str, status: str, error: Optional[str] = None
    ) -> bool:
        """Update session status (pending, processing, completed, failed)."""
        update_data = {"status": status}
        if error:
            update_data["error"] = error

        return await self.update_session(session_id, update_data)

    async def add_session_metrics(
        self, session_id: str, cost_usd: float, tokens: int, duration_ms: float
    ) -> bool:
        """Add metrics to session totals."""
        result = await self.db.sessions.update_one(
            {"session_id": session_id},
            {
                "$inc": {
                    "total_cost_usd": cost_usd,
                    "total_tokens": tokens,
                    "total_duration_ms": duration_ms,
                },
                "$set": {"updated_at": datetime.utcnow()},
            },
        )

        return result.modified_count > 0

    async def record_hitl_approval(
        self,
        session_id: str,
        stage: str,
        action: str,
        feedback: Optional[str] = None,
    ) -> bool:
        """
        Record Human-in-the-Loop approval/denial.
        """
        approval_record = {
            "stage": stage,
            "action": action,
            "feedback": feedback,
            "timestamp": datetime.utcnow(),
        }

        result = await self.db.sessions.update_one(
            {"session_id": session_id},
            {
                "$push": {"hitl_approvals": approval_record},
                "$set": {"updated_at": datetime.utcnow()},
            },
        )

        return result.modified_count > 0

    async def save_analysis_result(
        self,
        session_id: str,
        algorithm_name: str,
        pseudocode: str,
        ast: Dict[str, Any],
        paradigm: str,
        complexities: Dict[str, str],
        analysis_steps: List[Dict[str, Any]],
        diagrams: Dict[str, str],
        validation_results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save complete analysis result.
        """
        result = {
            "session_id": session_id,
            "algorithm_name": algorithm_name,
            "pseudocode": pseudocode,
            "ast": ast,
            "paradigm": paradigm,
            "complexities": complexities,
            "analysis_steps": analysis_steps,
            "diagrams": diagrams,
            "validation_results": validation_results,
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
        }

        insert_result = await self.db.analysis_results.insert_one(result)
        logger.info(f"Saved analysis result for session: {session_id}")

        return str(insert_result.inserted_id)

    async def get_analysis_results(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all analysis results for a session."""
        cursor = self.db.analysis_results.find({"session_id": session_id})
        return await cursor.to_list(length=None)

    async def get_latest_analysis_result(
        self, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get most recent analysis result for a session."""
        return await self.db.analysis_results.find_one(
            {"session_id": session_id}, sort=[("created_at", -1)]
        )

    async def log_agent_action(
        self,
        *,
        session_id: str,
        agent_name: str,
        stage: str,
        status: str,
        payload: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> str:
        """
        Persist a lifecycle event emitted by an agent stage.
        """

        log_entry: Dict[str, Any] = {
            "session_id": session_id,
            "agent_name": agent_name,
            "stage": stage,
            "status": status,
            "payload": payload or {},
            "timestamp": datetime.utcnow(),
        }

        log_entry["action"] = stage
        log_entry["output_data"] = payload or {}

        if metrics is not None:
            log_entry["metrics"] = metrics
        if error is not None:
            log_entry["error"] = error

        result = await self.db.agent_logs.insert_one(log_entry)
        return str(result.inserted_id)

    async def get_agent_logs(
        self, session_id: str, agent_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get agent logs for a session.
        """
        query = {"session_id": session_id}
        if agent_name:
            query["agent_name"] = agent_name

        cursor = self.db.agent_logs.find(query).sort("timestamp", 1)
        return await cursor.to_list(length=None)

    async def get_sessions_by_status(
        self, status: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get sessions filtered by status."""
        cursor = (
            self.db.sessions.find({"status": status})
            .sort("created_at", -1)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    async def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent sessions."""
        cursor = self.db.sessions.find().sort("created_at", -1).limit(limit)
        return await cursor.to_list(length=limit)

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete session and all related data.
        """
        await self.db.sessions.delete_one({"session_id": session_id})
        await self.db.analysis_results.delete_many({"session_id": session_id})
        await self.db.agent_logs.delete_many({"session_id": session_id})

        logger.info(f"Deleted session and related data: {session_id}")
        return True


mongodb_service = MongoDBService()
