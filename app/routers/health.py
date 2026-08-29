from fastapi import APIRouter

from ..config import settings
from ..services import models

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Unauthenticated liveness probe.

    Reports the REGISTRY rather than a configured model, because the service no
    longer has one: every reply resolves `text_llm` at request time.
    """
    payload: dict = {
        "ok": True,
        "service": "ai-agent-service",
        "tools_enabled": settings.tools_available,
        # Names only. Never the values.
        "api_keys_configured": settings.configured_keys,
    }

    try:
        registry = models.registry()
    except models.RegistryError as error:
        payload["ok"] = False
        payload["llm_registry"] = {"loaded": False, "error": str(error)}
        return payload

    payload["llm_registry"] = {
        "loaded": True,
        "source": registry.source,
        "providers": models.provider_names(),
        "models": len(registry.models),
        "default": registry.default,
        "identifiers": models.known_identifiers(),
    }
    return payload
