import logging

from fastapi import APIRouter, Depends, HTTPException, status

from ..config import settings
from ..schemas import ClassifyRequest, ClassifyResponse, ReplyRequest, ReplyResponse
from ..security import require_token
from ..services import agent
from ..services.models import UnknownModel
from ..services.prompts import PromptUnavailable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"], dependencies=[Depends(require_token)])


def _require_api_key(request: ReplyRequest | None = None) -> None:
    """Fail loudly rather than sending an unauthenticated request upstream.

    Only checks that SOME provider key exists. Which provider a given `text_llm`
    needs is known after the resolver runs, and that provider verifies its own
    key on construction - so a deployment with only a Gemini key still serves
    Gemini models while GROQ_API_KEY is blank.
    """
    if not settings.api_key_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No provider API key is configured. Set at least one of "
                "GROQ_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY or ANTHROPIC_API_KEY."
            ),
        )


def _translate(error: Exception) -> HTTPException:
    """Map provider SDK errors onto HTTP status codes.

    The Groq and Anthropic SDKs share the same generated exception surface
    (`status_code`, `message`, and the *Error class names), so this matches
    structurally instead of importing one vendor's classes.
    """
    name = type(error).__name__
    # Groq/Anthropic expose `status_code`; google-genai's APIError uses `code`.
    status_code = getattr(error, "status_code", None) or getattr(error, "code", None)
    message = getattr(error, "message", str(error))

    # A provider that could not be constructed - almost always a missing key or
    # an uninstalled SDK. Its own message already names the variable/package.
    if isinstance(error, RuntimeError) and (
        "API_KEY" in str(error) or "not installed" in str(error)
    ):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        )

    if name in {"ClientError", "ServerError", "APIError"} and isinstance(status_code, int):
        # google-genai's error family.
        if status_code == 429:
            return HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Upstream model rate limit reached",
            )
        if status_code in {401, 403}:
            return HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The model provider rejected the API key",
            )
        code = (
            status.HTTP_502_BAD_GATEWAY
            if status_code >= 500
            else status.HTTP_400_BAD_REQUEST
        )
        return HTTPException(status_code=code, detail=f"Model provider error: {message}")

    if name == "RateLimitError" or status_code == 429:
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Upstream model rate limit reached",
        )
    if name in {"APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout", "TimeoutException"}:
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not reach the model provider",
        )
    if isinstance(status_code, int):
        code = (
            status.HTTP_502_BAD_GATEWAY
            if status_code >= 500
            else status.HTTP_400_BAD_REQUEST
        )
        return HTTPException(status_code=code, detail=f"Model provider error: {message}")

    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to generate a reply",
    )


@router.post("/reply", response_model=ReplyResponse)
async def reply(request: ReplyRequest) -> ReplyResponse:
    """Generate a reply. The only endpoint that produces one.

    Request:
      {
        "client_id": "laravel-prod",
        "channel": "whatsapp",
        "contact_id": "923001234567",
        "prompt_id": "support",
        "message": "where is my order?",
        "context": { "order_id": 4417 }
      }
    Response:
      { "reply": "...", "thread_id": "th_…", "tool_calls_made": [...] }

    Any application can call this: pass your own `client_id` and either a
    `prompt_id` you registered or a `system_prompt` inline. Nothing here branches
    on who the caller is.

    The prompt, tone, language, tools and every business fact in `context` come
    from the calling application. This endpoint applies no policy of its own - if
    that application did not authorize the reply, it never gets here.
    """
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="message must not be empty",
        )

    _require_api_key(request)

    try:
        return await agent.generate_reply(request)
    except HTTPException:
        raise
    except agent.PromptNotFound as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except UnknownModel as error:
        # The caller named a model with no mapping. Answering on the default
        # would silently give them a different model than they asked for.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error
    except PromptUnavailable as error:
        # No prompt, no reply. Answering with a house default is the one thing
        # this service must never do.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error
    except PermissionError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001
        logger.exception("reply generation failed")
        raise _translate(error) from error


@router.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Structured analysis of an inbound message (intent, sentiment, urgency,
    language, escalation need) for Laravel routing and reporting."""
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="message must not be empty",
        )

    _require_api_key()

    try:
        return await agent.classify(request)
    except Exception as error:  # noqa: BLE001
        logger.exception("classification failed")
        raise _translate(error) from error
