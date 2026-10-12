"""Configuration for the WhatsApp integration layer.

Isolated from the RAG layer settings and application core settings.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Dict

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class WhatsAppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Meta WhatsApp Cloud API credentials
    whatsapp_access_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_verify_token: str = ""
    whatsapp_api_version: str = "v21.0"
    whatsapp_api_base_url: str = "https://graph.facebook.com"

    # Multi-tenant routing
    whatsapp_default_tenant_id: str = "default"
    # JSON string mapping phone_number_id -> tenant_id, e.g. '{"1029384756": "tenant_a"}'
    whatsapp_tenant_mapping_json: str = ""

    # Behavior
    whatsapp_send_fallback_on_error: bool = True
    whatsapp_fallback_message: str = (
        "Sorry, I am unable to process your request right now. Please try again shortly."
    )
    whatsapp_timeout_s: float = 15.0

    @property
    def tenant_mapping(self) -> Dict[str, str]:
        """Parse the tenant mapping JSON if provided."""
        if not self.whatsapp_tenant_mapping_json.strip():
            return {}
        try:
            parsed = json.loads(self.whatsapp_tenant_mapping_json)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except Exception as e:
            logger.warning("whatsapp: invalid WHATSAPP_TENANT_MAPPING_JSON: %s", e)
        return {}

    @property
    def is_configured(self) -> bool:
        """Returns True if minimum Meta API credentials are set."""
        return bool(self.whatsapp_access_token and self.whatsapp_phone_number_id)

    def resolve_tenant(self, phone_number_id: str | None) -> str:
        """Resolve tenant_id from incoming WhatsApp phone_number_id."""
        if phone_number_id:
            mapping = self.tenant_mapping
            if phone_number_id in mapping:
                return mapping[phone_number_id]
        return self.whatsapp_default_tenant_id

    def public_state(self) -> dict[str, Any]:
        """Safe public representation of WhatsApp config without exposing secrets."""
        return {
            "configured": self.is_configured,
            "api_version": self.whatsapp_api_version,
            "phone_number_id": self.whatsapp_phone_number_id or None,
            "default_tenant_id": self.whatsapp_default_tenant_id,
            "tenant_mapping_count": len(self.tenant_mapping),
            "verify_token_set": bool(self.whatsapp_verify_token),
        }


_settings_instance: WhatsAppSettings | None = None


def get_whatsapp_settings() -> WhatsAppSettings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = WhatsAppSettings()
    return _settings_instance


def set_whatsapp_settings(settings: WhatsAppSettings) -> None:
    global _settings_instance
    _settings_instance = settings


whatsapp_settings = get_whatsapp_settings()
