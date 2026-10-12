"""Unit and integration tests for the WhatsApp Cloud API integration layer."""

import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings as app_settings
from app.whatsapp import conversation, router as whatsapp_router, service, webhook
from app.whatsapp.client import WhatsAppAPIError, whatsapp_client
from app.whatsapp.config import WhatsAppSettings
from app.whatsapp.schemas import ParsedWhatsAppMessage


@pytest.fixture(autouse=True)
def clean_memory_state():
    conversation.reset_memory_state()
    yield
    conversation.reset_memory_state()


@pytest.fixture
def whatsapp_config() -> WhatsAppSettings:
    return WhatsAppSettings(
        whatsapp_access_token="mock_token_12345",
        whatsapp_phone_number_id="100200300",
        whatsapp_verify_token="test_secret_verify_token",
        whatsapp_default_tenant_id="tenant_default",
        whatsapp_tenant_mapping_json='{"100200300": "tenant_alpha", "999888777": "tenant_beta"}',
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_settings, "rag_api_token", "")
    application = FastAPI()
    application.include_router(whatsapp_router.router)
    return TestClient(application)


# 1. Webhook Verification (GET /webhook)
def test_webhook_verification_success(client, whatsapp_config):
    with patch("app.whatsapp.router.get_whatsapp_settings", return_value=whatsapp_config):
        response = client.get(
            "/api/whatsapp/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "test_secret_verify_token",
                "hub.challenge": "1158201444",
            },
        )
        assert response.status_code == 200
        assert response.text == "1158201444"


def test_webhook_verification_token_mismatch(client, whatsapp_config):
    with patch("app.whatsapp.router.get_whatsapp_settings", return_value=whatsapp_config):
        response = client.get(
            "/api/whatsapp/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "wrong_token",
                "hub.challenge": "1158201444",
            },
        )
        assert response.status_code == 403


def test_webhook_verification_missing_params(client, whatsapp_config):
    with patch("app.whatsapp.router.get_whatsapp_settings", return_value=whatsapp_config):
        response = client.get("/api/whatsapp/webhook")
        assert response.status_code == 403


# 2. Incoming Webhook Parsing & Non-Text Safety
def test_parse_webhook_payload_extracts_text_message():
    raw_payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WHATSAPP_BUSINESS_ACCOUNT_ID",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15550234567",
                                "phone_number_id": "100200300",
                            },
                            "messages": [
                                {
                                    "from": "16315551234",
                                    "id": "wamid.HBgLMTYzMTU1NTEyMzRV",
                                    "timestamp": "1725184800",
                                    "type": "text",
                                    "text": {"body": "How are drivers alerted about new bookings?"},
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }

    messages = webhook.parse_webhook_payload(raw_payload)
    assert len(messages) == 1
    msg = messages[0]
    assert msg.external_message_id == "wamid.HBgLMTYzMTU1NTEyMzRV"
    assert msg.sender_phone_number == "16315551234"
    assert msg.phone_number_id == "100200300"
    assert msg.message_text == "How are drivers alerted about new bookings?"


def test_parse_webhook_payload_ignores_statuses_and_images():
    status_payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "123",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"phone_number_id": "100200300"},
                            "statuses": [{"id": "wamid.123", "status": "delivered"}],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }
    assert webhook.parse_webhook_payload(status_payload) == []

    image_payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "123",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"phone_number_id": "100200300"},
                            "messages": [{"from": "12345", "id": "wamid.img", "type": "image"}],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }
    assert webhook.parse_webhook_payload(image_payload) == []


# 3. Server-side Tenant Mapping & Message Processing
@pytest.mark.asyncio
async def test_process_incoming_message_flow(whatsapp_config):
    msg_id = f"wamid_flow_{uuid.uuid4().hex}"
    parsed = ParsedWhatsAppMessage(
        external_message_id=msg_id,
        sender_phone_number="+16315551234",
        message_type="text",
        message_text="What payment options do you support?",
        phone_number_id="100200300",
    )

    mock_rag_response = {
        "query": "What payment options do you support?",
        "answer": "We accept credit cards, debit cards, and digital wallets. [SOURCE 1]",
        "sources": [{"source_id": "SOURCE 1", "source": "Payments.pdf"}],
        "latency": {"total_ms": 200},
    }

    with patch("app.rag.service.ask", new_callable=AsyncMock, return_value=mock_rag_response) as mock_ask, \
         patch.object(whatsapp_client, "send_text_message", new_callable=AsyncMock) as mock_send:

        result = await service.process_incoming_whatsapp_message(parsed, settings=whatsapp_config)

        assert result["status"] == "processed"
        assert result["tenant_id"] == "tenant_alpha"  # Mapped from 100200300
        assert "credit cards" in result["answer"]
        assert result["send_status"] == "sent"

        # Verify RAG was invoked with the resolved tenant
        mock_ask.assert_called_once()
        assert mock_ask.call_args.kwargs["tenant_id"] == "tenant_alpha"
        assert mock_ask.call_args.kwargs["query"] == "What payment options do you support?"

        # Verify reply was sent to the customer
        mock_send.assert_called_once()
        assert mock_send.call_args.kwargs["recipient_phone_number"] == "+16315551234"
        assert "credit cards" in mock_send.call_args.kwargs["message"]


# 4. Message Idempotency Check
@pytest.mark.asyncio
async def test_message_idempotency_prevents_duplicate_processing(whatsapp_config):
    msg_id = f"wamid_dup_{uuid.uuid4().hex}"
    parsed = ParsedWhatsAppMessage(
        external_message_id=msg_id,
        sender_phone_number="+16315551234",
        message_type="text",
        message_text="Hello again!",
        phone_number_id="100200300",
    )

    with patch("app.rag.service.ask", new_callable=AsyncMock, return_value={"answer": "Hi!"}) as mock_ask, \
         patch.object(whatsapp_client, "send_text_message", new_callable=AsyncMock) as mock_send:

        # First run: should process
        result1 = await service.process_incoming_whatsapp_message(parsed, settings=whatsapp_config)
        assert result1["status"] == "processed"
        assert mock_ask.call_count == 1
        assert mock_send.call_count == 1

        # Second run with same external_message_id: should skip
        result2 = await service.process_incoming_whatsapp_message(parsed, settings=whatsapp_config)
        assert result2["status"] == "skipped"
        assert result2["reason"] == "already_processed"
        # Verify RAG and WhatsApp send were NOT called again
        assert mock_ask.call_count == 1
        assert mock_send.call_count == 1


# 5. Conversation History Reuse
@pytest.mark.asyncio
async def test_conversation_reuse_for_same_customer(whatsapp_config):
    phone = f"+1999{uuid.uuid4().int % 10000000:07d}"
    msg1 = ParsedWhatsAppMessage(
        external_message_id=f"wamid_seq_1_{uuid.uuid4().hex}",
        sender_phone_number=phone,
        message_text="First question",
        phone_number_id="999888777",
    )
    msg2 = ParsedWhatsAppMessage(
        external_message_id=f"wamid_seq_2_{uuid.uuid4().hex}",
        sender_phone_number=phone,
        message_text="Follow-up question",
        phone_number_id="999888777",
    )

    with patch("app.rag.service.ask", new_callable=AsyncMock, return_value={"answer": "Sample answer"}), \
         patch.object(whatsapp_client, "send_text_message", new_callable=AsyncMock):

        res1 = await service.process_incoming_whatsapp_message(msg1, settings=whatsapp_config)
        res2 = await service.process_incoming_whatsapp_message(msg2, settings=whatsapp_config)

        # Both messages must share the exact same conversation_id
        assert res1["conversation_id"] == res2["conversation_id"]
        assert res1["tenant_id"] == "tenant_beta"
        assert res2["tenant_id"] == "tenant_beta"


# 6. Graceful RAG Failure Fallback
@pytest.mark.asyncio
async def test_rag_failure_sends_fallback_message(whatsapp_config):
    parsed = ParsedWhatsAppMessage(
        external_message_id=f"wamid_fail_{uuid.uuid4().hex}",
        sender_phone_number="+15554443333",
        message_text="A question causing an error",
        phone_number_id="100200300",
    )

    with patch("app.rag.service.ask", new_callable=AsyncMock, side_effect=Exception("Database connection timed out")), \
         patch.object(whatsapp_client, "send_text_message", new_callable=AsyncMock) as mock_send:

        result = await service.process_incoming_whatsapp_message(parsed, settings=whatsapp_config)
        assert result["status"] == "processed"
        assert result["answer"] == whatsapp_config.whatsapp_fallback_message

        # Fallback message was sent to WhatsApp
        mock_send.assert_called_once()
        assert mock_send.call_args.kwargs["message"] == whatsapp_config.whatsapp_fallback_message


# 7. WhatsApp API Failure Handled Gracefully
@pytest.mark.asyncio
async def test_whatsapp_api_failure_handled_without_exception(whatsapp_config):
    parsed = ParsedWhatsAppMessage(
        external_message_id=f"wamid_send_fail_{uuid.uuid4().hex}",
        sender_phone_number="+15554443333",
        message_text="Hello",
        phone_number_id="100200300",
    )

    with patch("app.rag.service.ask", new_callable=AsyncMock, return_value={"answer": "Hello customer!"}), \
         patch.object(whatsapp_client, "send_text_message", new_callable=AsyncMock, side_effect=WhatsAppAPIError("Meta rate limit")):

        result = await service.process_incoming_whatsapp_message(parsed, settings=whatsapp_config)
        assert result["status"] == "processed"
        assert result["send_status"] == "failed"


# 8. Webhook POST Endpoint Acknowledgment
def test_webhook_post_returns_immediate_200(client):
    raw_payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "123",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"phone_number_id": "100200300"},
                            "messages": [
                                {
                                    "from": "16315551234",
                                    "id": "wamid.post_test",
                                    "timestamp": "1725184800",
                                    "type": "text",
                                    "text": {"body": "Testing webhook POST endpoint"},
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }

    with patch("app.whatsapp.service.process_incoming_whatsapp_message", new_callable=AsyncMock):
        response = client.post("/api/whatsapp/webhook", json=raw_payload)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# 9. WhatsApp Health Endpoint
def test_whatsapp_health_endpoint(client, whatsapp_config):
    with patch("app.whatsapp.router.get_whatsapp_settings", return_value=whatsapp_config):
        response = client.get("/api/whatsapp/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["configuration"]["configured"] is True
        assert data["configuration"]["default_tenant_id"] == "tenant_default"

