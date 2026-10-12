"""WhatsApp Cloud API HTTP client.

Responsible exclusively for communicating with Meta's WhatsApp Cloud API.
Centralizes all external WhatsApp network calls with strict timeout and error handling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from .config import WhatsAppSettings, get_whatsapp_settings

logger = logging.getLogger(__name__)


class WhatsAppAPIError(Exception):
    """Raised when communication with Meta WhatsApp Cloud API fails."""


class WhatsAppClient:
    """Client for Meta WhatsApp Cloud API."""

    def __init__(self, settings: Optional[WhatsAppSettings] = None):
        self._settings = settings

    @property
    def settings(self) -> WhatsAppSettings:
        return self._settings or get_whatsapp_settings()

    async def send_text_message(
        self,
        recipient_phone_number: str,
        message: str,
        phone_number_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a plain text message to a WhatsApp user via Meta Cloud API.

        Args:
            recipient_phone_number: Destination E.164 phone number (e.g. "+1234567890" or "1234567890").
            message: The text content of the message.
            phone_number_id: The sending WhatsApp Phone Number ID (defaults to config).

        Returns:
            Meta API response dictionary.

        Raises:
            WhatsAppAPIError: If the API call fails or credentials are missing.
        """
        cfg = self.settings
        token = cfg.whatsapp_access_token
        pid = (phone_number_id or cfg.whatsapp_phone_number_id or "").strip()

        if not token:
            raise WhatsAppAPIError("WHATSAPP_ACCESS_TOKEN is not configured.")
        if not pid:
            raise WhatsAppAPIError("WHATSAPP_PHONE_NUMBER_ID is not configured.")

        # Clean recipient phone number (remove +, spaces, dashes)
        clean_to = "".join(c for c in recipient_phone_number if c.isdigit())
        if not clean_to:
            raise WhatsAppAPIError(f"Invalid recipient phone number: {recipient_phone_number}")

        url = f"{cfg.whatsapp_api_base_url.rstrip('/')}/{cfg.whatsapp_api_version}/{pid}/messages"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": clean_to,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": message,
            },
        }

        logger.info(
            "whatsapp: sending text message (to=%s, pid=%s, chars=%d)",
            clean_to[-4:].rjust(len(clean_to), "*"), pid, len(message),
        )

        try:
            async with httpx.AsyncClient(timeout=cfg.whatsapp_timeout_s) as client:
                response = await client.post(url, headers=headers, json=payload)

                if response.status_code not in (200, 201):
                    logger.error(
                        "whatsapp: Meta API error (status=%s, pid=%s): %s",
                        response.status_code, pid, response.text,
                    )
                    raise WhatsAppAPIError(
                        f"WhatsApp Cloud API returned status {response.status_code}: {response.text}"
                    )

                data = response.json()
                logger.info("whatsapp: message successfully sent (to=%s)", clean_to[-4:].rjust(len(clean_to), "*"))
                return data

        except WhatsAppAPIError:
            raise
        except httpx.TimeoutException as e:
            logger.error("whatsapp: timeout sending message to Meta API: %s", e)
            raise WhatsAppAPIError(f"Timeout communicating with WhatsApp Cloud API: {e}") from e
        except Exception as e:
            logger.error("whatsapp: unexpected error sending message: %s", e)
            raise WhatsAppAPIError(f"Error sending WhatsApp message: {e}") from e


whatsapp_client = WhatsAppClient()
