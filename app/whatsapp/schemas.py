"""Pydantic schemas for the Meta WhatsApp Cloud API integration layer.

Handles Meta WhatsApp Cloud API webhooks, normalized internal models, and webhook acknowledgments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WhatsAppText(BaseModel):
    body: str


class WhatsAppMessageItem(BaseModel):
    from_: str = Field(alias="from")
    id: str
    timestamp: str
    type: str
    text: Optional[WhatsAppText] = None

    model_config = {"populate_by_name": True, "extra": "ignore"}


class WhatsAppMetadata(BaseModel):
    display_phone_number: Optional[str] = None
    phone_number_id: Optional[str] = None


class WhatsAppValue(BaseModel):
    messaging_product: Optional[str] = None
    metadata: Optional[WhatsAppMetadata] = None
    contacts: Optional[List[Dict[str, Any]]] = None
    messages: Optional[List[WhatsAppMessageItem]] = None
    statuses: Optional[List[Dict[str, Any]]] = None

    model_config = {"extra": "ignore"}


class WhatsAppChange(BaseModel):
    value: WhatsAppValue
    field: str

    model_config = {"extra": "ignore"}


class WhatsAppEntry(BaseModel):
    id: str
    changes: List[WhatsAppChange]

    model_config = {"extra": "ignore"}


class WhatsAppWebhookPayload(BaseModel):
    object: str
    entry: List[WhatsAppEntry]

    model_config = {"extra": "ignore"}


class ParsedWhatsAppMessage(BaseModel):
    """Normalized internal representation of an incoming text message."""

    external_message_id: str
    sender_phone_number: str
    message_type: str = "text"
    message_text: str
    timestamp: Optional[str] = None
    phone_number_id: Optional[str] = None
    display_phone_number: Optional[str] = None
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)


class WhatsAppWebhookAck(BaseModel):
    status: str = "ok"
