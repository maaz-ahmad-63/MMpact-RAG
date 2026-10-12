"""End-to-end integration tests for WhatsApp Webhook -> RAG Ask -> WhatsApp Send.

Tests the complete flow:
POST /api/whatsapp/webhook
  → parse incoming WhatsApp text message
  → identify tenant
  → get/create conversation
  → save user message
  → call existing /api/rag/ask
  → receive RAG answer
  → save assistant message
  → call mocked WhatsApp send_text_message()

Only external Meta HTTP communication is mocked.
"""

import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import ASGITransport

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app
from app.rag.config import rag_settings
from app.rag.llm import MockLLMProvider
from app.whatsapp import conversation
from app.whatsapp.client import whatsapp_client
from app.whatsapp.config import WhatsAppSettings, set_whatsapp_settings


@pytest.fixture(autouse=True)
def clean_test_environment():
    conversation.reset_memory_state()
    yield
    conversation.reset_memory_state()


@pytest.fixture
def e2e_whatsapp_config() -> WhatsAppSettings:
    """Config with mock Meta credentials and tenant mapping."""
    return WhatsAppSettings(
        whatsapp_access_token="mock_meta_access_token_abc123",
        whatsapp_phone_number_id="10987654321",
        whatsapp_verify_token="test_webhook_token_2026",
        whatsapp_default_tenant_id="",
        whatsapp_tenant_mapping_json='{"10987654321": "", "20987654321": "tenant_corp"}',
    )


@pytest.fixture
def mock_llm():
    """Mock LLM provider to avoid requiring third-party cloud API keys during test runs."""
    provider = MockLLMProvider()

    async def _generate(messages, **kwargs):
        prompt_text = messages[-1]["content"] if messages else ""
        if "RideOn" in prompt_text or "booking" in prompt_text.lower() or "driver" in prompt_text.lower():
            answer = "Drivers receive instant push alerts for any new booking request via OneSignal. [SOURCE 1]"
        else:
            answer = "Here is the information you requested. [SOURCE 1]"
        return answer, {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}

    provider.generate = AsyncMock(side_effect=_generate)
    return provider


def build_meta_webhook_payload(
    message_id: str,
    from_phone: str,
    text_body: str,
    phone_number_id: str = "10987654321",
    display_phone_number: str = "15550234567",
) -> dict:
    """Helper to generate a realistic Meta WhatsApp Cloud API incoming text webhook event."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WHATSAPP_BUSINESS_ACCOUNT_ID_999",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": display_phone_number,
                                "phone_number_id": phone_number_id,
                            },
                            "contacts": [
                                {
                                    "profile": {"name": "Test Rider"},
                                    "wa_id": from_phone.replace("+", ""),
                                }
                            ],
                            "messages": [
                                {
                                    "from": from_phone.replace("+", ""),
                                    "id": message_id,
                                    "timestamp": "1725184800",
                                    "type": "text",
                                    "text": {
                                        "body": text_body,
                                    },
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }


@pytest.mark.asyncio
async def test_full_e2e_whatsapp_webhook_flow(e2e_whatsapp_config, mock_llm):
    """Complete end-to-end test simulating incoming customer webhook against real RAG database."""
    set_whatsapp_settings(e2e_whatsapp_config)

    unique_msg_id = f"wamid_e2e_{uuid.uuid4().hex}"
    sender_phone = f"+1631{uuid.uuid4().int % 10000000:07d}"
    clean_sender = sender_phone.replace("+", "")
    user_query = "How are drivers alerted about new bookings?"

    webhook_payload = build_meta_webhook_payload(
        message_id=unique_msg_id,
        from_phone=sender_phone,
        text_body=user_query,
        phone_number_id="10987654321",
    )

    mock_send = AsyncMock(
        return_value={"messaging_product": "whatsapp", "messages": [{"id": "wamid_out_1"}]}
    )

    with patch.object(whatsapp_client, "send_text_message", mock_send), \
         patch("app.rag.service.llm.get_llm_provider", return_value=mock_llm):

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Post webhook event to FastAPI endpoint
            response = await client.post("/api/whatsapp/webhook", json=webhook_payload)
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}

        # 2. Verify Conversation Creation & Storage
        conv = await conversation.get_or_create_conversation(
            tenant_id="",
            phone_number=clean_sender,
        )
        assert conv is not None
        conversation_id = conv["id"]

        # 3. Verify User Message and Assistant Message Storage
        history = await conversation.get_conversation_history(conversation_id, limit=10)
        assert len(history) >= 2

        # First message is user query
        assert history[0]["role"] == "user"
        assert history[0]["content"] == user_query

        # Second message is assistant RAG answer
        assert history[1]["role"] == "assistant"
        assistant_answer = history[1]["content"]
        assert len(assistant_answer) > 0
        assert "push alerts" in assistant_answer

        # 4. Verify mocked WhatsApp send_text_message was invoked
        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args.kwargs
        assert call_kwargs["recipient_phone_number"] == clean_sender
        assert call_kwargs["phone_number_id"] == "10987654321"
        assert call_kwargs["message"] == assistant_answer


@pytest.mark.asyncio
async def test_e2e_multi_turn_conversation_reuse(e2e_whatsapp_config, mock_llm):
    """Test that a second message from the same customer reuses the existing conversation thread."""
    set_whatsapp_settings(e2e_whatsapp_config)

    sender_phone = f"+1631{uuid.uuid4().int % 10000000:07d}"
    clean_sender = sender_phone.replace("+", "")
    msg_id_1 = f"wamid_turn1_{uuid.uuid4().hex}"
    msg_id_2 = f"wamid_turn2_{uuid.uuid4().hex}"

    payload_1 = build_meta_webhook_payload(
        message_id=msg_id_1,
        from_phone=sender_phone,
        text_body="What payment options do you support?",
    )
    payload_2 = build_meta_webhook_payload(
        message_id=msg_id_2,
        from_phone=sender_phone,
        text_body="And can I pay with cash?",
    )

    mock_send = AsyncMock(return_value={"messaging_product": "whatsapp"})

    with patch.object(whatsapp_client, "send_text_message", mock_send), \
         patch("app.rag.service.llm.get_llm_provider", return_value=mock_llm):

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Turn 1
            res1 = await client.post("/api/whatsapp/webhook", json=payload_1)
            assert res1.status_code == 200

            # Turn 2
            res2 = await client.post("/api/whatsapp/webhook", json=payload_2)
            assert res2.status_code == 200

        # Check conversation record
        conv = await conversation.get_or_create_conversation(
            tenant_id="",
            phone_number=clean_sender,
        )
        conversation_id = conv["id"]

        # History should now have 4 turns (user1, asst1, user2, asst2)
        history = await conversation.get_conversation_history(conversation_id, limit=10)
        assert len(history) == 4
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "What payment options do you support?"
        assert history[1]["role"] == "assistant"
        assert history[2]["role"] == "user"
        assert history[2]["content"] == "And can I pay with cash?"
        assert history[3]["role"] == "assistant"

        # WhatsApp send was called twice (once per turn)
        assert mock_send.call_count == 2


@pytest.mark.asyncio
async def test_e2e_idempotency_ignores_duplicate_webhook_retries(e2e_whatsapp_config, mock_llm):
    """Test that duplicate Meta webhook deliveries are safely ignored without re-running RAG."""
    set_whatsapp_settings(e2e_whatsapp_config)

    duplicate_msg_id = f"wamid_dup_{uuid.uuid4().hex}"
    sender_phone = f"+1631{uuid.uuid4().int % 10000000:07d}"
    clean_sender = sender_phone.replace("+", "")

    payload = build_meta_webhook_payload(
        message_id=duplicate_msg_id,
        from_phone=sender_phone,
        text_body="Hello, is this booking confirmed?",
    )

    mock_send = AsyncMock(return_value={"messaging_product": "whatsapp"})

    with patch.object(whatsapp_client, "send_text_message", mock_send), \
         patch("app.rag.service.llm.get_llm_provider", return_value=mock_llm):

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # First delivery
            r1 = await client.post("/api/whatsapp/webhook", json=payload)
            assert r1.status_code == 200
            assert mock_send.call_count == 1

            # Second delivery (Meta retry with identical external message ID)
            r2 = await client.post("/api/whatsapp/webhook", json=payload)
            assert r2.status_code == 200

            # Third delivery
            r3 = await client.post("/api/whatsapp/webhook", json=payload)
            assert r3.status_code == 200

        # mock_send should STILL be called only once
        assert mock_send.call_count == 1

        # History should contain only 1 user turn and 1 assistant turn
        conv = await conversation.get_or_create_conversation(
            tenant_id="",
            phone_number=clean_sender,
        )
        history = await conversation.get_conversation_history(conv["id"], limit=10)
        assert len(history) == 2
