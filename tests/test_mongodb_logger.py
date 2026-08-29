import datetime
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import motor.motor_asyncio
from app.services.mongodb_logger import MongoDBLogger
from app.config import Settings


@pytest.mark.asyncio
async def test_logger_disabled_when_url_unset():
    """Should verify that the logger remains inactive if MONGODB_URL is unset."""
    custom_settings = Settings(mongodb_url="")
    logger = MongoDBLogger()
    # Mock settings reference
    import app.services.mongodb_logger
    orig_settings = app.services.mongodb_logger.settings
    app.services.mongodb_logger.settings = custom_settings

    try:
        assert not logger.is_configured
        await logger.startup()
        # Should not raise errors or construct clients
        assert logger._client is None
    finally:
        app.services.mongodb_logger.settings = orig_settings


@pytest.mark.asyncio
async def test_logger_graceful_on_connection_error(monkeypatch):
    """Should verify that startup errors do not crash the logger startup lifecycle."""
    custom_settings = Settings(mongodb_url="mongodb://localhost:27017")
    logger = MongoDBLogger()
    
    import app.services.mongodb_logger
    orig_settings = app.services.mongodb_logger.settings
    app.services.mongodb_logger.settings = custom_settings

    # Mock AsyncIOMotorClient to raise a connection error
    def mock_client_init(*args, **kwargs):
        raise ConnectionError("Mock connection error")

    monkeypatch.setattr(app.services.mongodb_logger, "AsyncIOMotorClient", mock_client_init)

    try:
        await logger.startup()
        assert logger._client is None
        # Attempt logging - should handle error gracefully without raising exceptions
        await logger.log_search({"query": "test"})
    finally:
        app.services.mongodb_logger.settings = orig_settings


@pytest.mark.asyncio
async def test_logger_successful_write(monkeypatch):
    """Should verify that a successful log write correctly invokes insert_one on the collection."""
    custom_settings = Settings(mongodb_url="mongodb://localhost:27017")
    logger = MongoDBLogger()
    
    import app.services.mongodb_logger
    orig_settings = app.services.mongodb_logger.settings
    app.services.mongodb_logger.settings = custom_settings

    mock_collection = AsyncMock()
    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_client = MagicMock()
    mock_client.__getitem__.return_value = mock_db

    monkeypatch.setattr(app.services.mongodb_logger, "AsyncIOMotorClient", lambda *args, **kwargs: mock_client)
    monkeypatch.setattr(logger, "_create_indexes", AsyncMock())

    try:
        await logger.startup()
        assert logger._client is not None
        
        # Test logging
        log_data = {"query": "vector similarity query", "found": True}
        await logger.log_search(log_data)
        
        mock_collection.insert_one.assert_called_once_with(log_data)
        assert "timestamp" in log_data
    finally:
        app.services.mongodb_logger.settings = orig_settings


@pytest.mark.asyncio
async def test_logger_retrieval(monkeypatch):
    """Should verify that log query retrieval maps cursors and converts ObjectId parameter names."""
    custom_settings = Settings(mongodb_url="mongodb://localhost:27017")
    logger = MongoDBLogger()
    
    import app.services.mongodb_logger
    orig_settings = app.services.mongodb_logger.settings
    app.services.mongodb_logger.settings = custom_settings

    mock_collection = MagicMock()
    
    # Mock find and sort cursors
    mock_cursor = AsyncMock()
    mock_cursor.__aiter__.return_value = [
        {"_id": "mock_id_123", "query": "hello", "found": True, "timestamp": datetime.datetime(2026, 1, 1)}
    ]
    mock_collection.find.return_value.sort.return_value.limit.return_value = mock_cursor
    logger._collection = mock_collection

    try:
        logs = await logger.get_search_logs(limit=10, document_id="doc_abc", found=True)
        assert len(logs) == 1
        assert logs[0]["id"] == "mock_id_123"
        assert "timestamp" in logs[0]
        
        # Verify exact query payload
        mock_collection.find.assert_called_once_with({"document_ids": "doc_abc", "found": True})
    finally:
        app.services.mongodb_logger.settings = orig_settings

