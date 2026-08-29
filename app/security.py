import hmac

from fastapi import Depends, HTTPException, status

from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings


security = HTTPBearer(auto_error=False)


async def require_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> None:
    """Shared-secret auth for RAG endpoints.

    If RAG_API_TOKEN is unset the check is skipped, which is only appropriate
    when the service is bound to a private network.
    """

    if not settings.rag_api_token:
        return

    token = credentials.credentials if credentials else ""

    if not hmac.compare_digest(token, settings.rag_api_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
        )
