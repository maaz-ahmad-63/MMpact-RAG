import datetime
import logging
from typing import Any, Dict, List, Optional
import certifi
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING, ASCENDING

from ..config import settings

logger = logging.getLogger(__name__)


class MongoDBLogger:
    """Asynchronous logger for MMpact RAG searches using MongoDB + motor."""

    def __init__(self) -> None:
        self._client: Optional[AsyncIOMotorClient] = None
        self._db = None
        self._collection = None

    @property
    def is_configured(self) -> bool:
        return bool(settings.mongodb_url)

    async def startup(self) -> None:
        """Initialize connection pool and create index constraints."""
        if not self.is_configured:
            logger.info("rag: MongoDB logging is disabled (MONGODB_URL is unset).")
            return

        try:
            self._client = AsyncIOMotorClient(
                settings.mongodb_url,
                serverSelectionTimeoutMS=5000,
                tlsCAFile=certifi.where()
            )
            self._db = self._client[settings.mongodb_database]
            self._collection = self._db[settings.mongodb_collection]

            # Trigger connection test and create indexes
            await self._create_indexes()
            logger.info(
                "rag: MongoDB logging ready (database=%s, collection=%s)",
                settings.mongodb_database,
                settings.mongodb_collection,
            )
        except Exception as error:
            logger.error("rag: failed to initialize MongoDB connection (%s)", error)
            # Do not crash the application: degrade gracefully
            self._client = None
            self._db = None
            self._collection = None

    async def shutdown(self) -> None:
        """Close connection pools."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None
            self._collection = None

    async def _create_indexes(self) -> None:
        if self._collection is None:
            return
        # Add index constraints
        await self._collection.create_index([("timestamp", DESCENDING)])
        await self._collection.create_index([("document_ids", ASCENDING)])
        await self._collection.create_index([("found", ASCENDING)])

    async def log_search(self, log_data: Dict[str, Any]) -> None:
        """Write search request metrics to MongoDB. Safe-guarded to never raise errors."""
        if not self.is_configured or self._collection is None:
            return

        try:
            # Ensure timestamp is set
            if "timestamp" not in log_data:
                log_data["timestamp"] = datetime.datetime.now(datetime.timezone.utc)
            await self._collection.insert_one(log_data)
        except Exception as error:
            logger.error("rag: failed to write search log to MongoDB (%s)", error)

    async def log_ask(self, log_data: Dict[str, Any]) -> None:
        """Write /ask request metrics, latencies, tokens and answers to MongoDB. Safe-guarded to never raise."""
        if not self.is_configured or self._collection is None:
            return

        try:
            # Clean copy without secrets
            sanitized = {k: v for k, v in log_data.items() if not any(s in k.lower() for s in ("key", "token", "password", "secret", "auth"))}
            if "timestamp" not in sanitized:
                sanitized["timestamp"] = datetime.datetime.now(datetime.timezone.utc)
            await self._collection.insert_one(sanitized)
        except Exception as error:
            logger.error("rag: failed to write ask log to MongoDB (%s)", error)

    async def get_search_logs(
        self,
        limit: int = 50,
        document_id: Optional[str] = None,
        found: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent search logs from MongoDB."""
        if not self.is_configured or self._collection is None:
            return []

        query: Dict[str, Any] = {}
        if document_id:
            # Matches document_id in document_ids array or single value
            query["document_ids"] = document_id
        if found is not None:
            query["found"] = found

        try:
            cursor = (
                self._collection.find(query)
                .sort("timestamp", DESCENDING)
                .limit(limit)
            )
            logs = []
            async for doc in cursor:
                # Convert ObjectId to string for JSON serialization
                doc["id"] = str(doc.pop("_id"))
                # Convert datetime objects to string format if any
                if isinstance(doc.get("timestamp"), datetime.datetime):
                    doc["timestamp"] = doc["timestamp"].isoformat()
                logs.append(doc)
            return logs
        except Exception as error:
            logger.error("rag: failed to retrieve search logs from MongoDB (%s)", error)
            return []


# Process-wide singleton
mongodb_logger = MongoDBLogger()
