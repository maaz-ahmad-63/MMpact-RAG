"""Developer dashboard: JSON under /api/v1/dashboard, one HTML page at /ui.

There was no frontend in this repository - no package.json, no templates, no
static directory - so nothing could be "extended". Rather than introduce a
frontend framework, a build step and a node toolchain into a service that has
none, the dashboard is ONE self-contained HTML file served by FastAPI and read
from disk. It is the existing stack: Python, FastAPI, no new dependency.

Auth
----
Every JSON route sits behind the service's existing `require_token`. The HTML
page is served WITHOUT it and holds no data: it is an empty shell that asks for
the token in the browser and fetches everything with an Authorization header.
That way the page can load at all (a browser cannot attach a bearer token to a
top-level navigation), while every byte of agent trace, conversation, tool
result and log stays behind the same auth as the rest of the service.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse

from ..security import require_token
from . import store
from .schemas import OverviewResponse, RunDetail, RunListResponse

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])

#: Data routes only. Applied as a dependency per route rather than on the
#: router, because the HTML page below must stay reachable by a browser.
_auth = [Depends(require_token)]


def _unavailable(error: Exception) -> HTTPException:
    return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error))


@router.get("/overview", response_model=OverviewResponse, dependencies=_auth)
async def overview(hours: int = Query(default=24, ge=1, le=8760)) -> OverviewResponse:
    """Counters for the front page, over a rolling window."""
    try:
        return OverviewResponse.model_validate(await store.overview(hours))
    except store.TraceStoreUnavailable as error:
        raise _unavailable(error) from error


@router.get("/runs", response_model=RunListResponse, dependencies=_auth)
async def runs(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    conversation_id: str | None = None,
    user_id: str | None = None,
    run_status: str | None = Query(default=None, alias="status"),
    action: str | None = None,
) -> RunListResponse:
    try:
        rows, total = await store.list_runs(
            limit=limit, offset=offset, conversation_id=conversation_id,
            user_id=user_id, status=run_status, action=action,
        )
    except store.TraceStoreUnavailable as error:
        raise _unavailable(error) from error
    return RunListResponse(runs=rows, total=total, limit=limit, offset=offset)


@router.get("/runs/{run_id}", response_model=RunDetail, dependencies=_auth)
async def run_detail(run_id: str) -> RunDetail:
    """The whole execution: steps, searches with their chunks, LLM and tool calls."""
    try:
        detail = await store.get_run(run_id)
    except store.TraceStoreUnavailable as error:
        raise _unavailable(error) from error
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No run {run_id}")
    return RunDetail.model_validate(detail)


@router.get("/rag", dependencies=_auth)
async def rag(
    limit: int = Query(default=100, ge=1, le=500),
    hours: int = Query(default=168, ge=1, le=8760),
) -> dict:
    try:
        return {"searches": await store.rag_activity(limit=limit, window_hours=hours)}
    except store.TraceStoreUnavailable as error:
        raise _unavailable(error) from error


@router.get("/tools", dependencies=_auth)
async def tools(
    limit: int = Query(default=100, ge=1, le=500),
    hours: int = Query(default=168, ge=1, le=8760),
) -> dict:
    try:
        return {"calls": await store.tool_activity(limit=limit, window_hours=hours)}
    except store.TraceStoreUnavailable as error:
        raise _unavailable(error) from error


@router.get("/models", dependencies=_auth)
async def model_usage(hours: int = Query(default=168, ge=1, le=8760)) -> dict:
    """Which models actually ran, from the llm_calls rows.

    This is the page that answers "is the model really coming from Laravel?" -
    change `text_llm` there and a new row appears here with no deployment.
    """
    try:
        return {"models": await store.model_usage(window_hours=hours)}
    except store.TraceStoreUnavailable as error:
        raise _unavailable(error) from error


@router.get("/logs", dependencies=_auth)
async def logs(
    limit: int = Query(default=100, ge=1, le=500),
    hours: int = Query(default=168, ge=1, le=8760),
) -> dict:
    """Errors across runs, steps and LLM calls - the error page's data."""
    try:
        return {"errors": await store.list_errors(limit=limit, window_hours=hours)}
    except store.TraceStoreUnavailable as error:
        raise _unavailable(error) from error


@router.get("/status", dependencies=_auth)
async def dashboard_status() -> dict:
    """Whether tracing is configured and migrated. Names only, never secrets."""
    configured = store.storage_configured()
    present = False
    detail = None
    if configured:
        try:
            present = await store.tables_present()
        except Exception as error:  # noqa: BLE001
            detail = str(error)
    return {
        "storage_configured": configured,
        "tables_present": present,
        "detail": detail
        or (
            None
            if present
            else "Run `python -m app.graph.migrate` to create the trace tables."
            if configured
            else "Set RAG_DATABASE_URL (or DATABASE_URL). The agent runs without it; "
            "only the dashboard is affected."
        ),
    }


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def ui() -> HTMLResponse:
    """The dashboard page itself. Carries no data - see the module docstring."""
    page = STATIC_DIR / "dashboard.html"
    try:
        return HTMLResponse(page.read_text(encoding="utf-8"))
    except OSError as error:
        logger.error("dashboard page missing at %s (%s)", page, error)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "Dashboard page is not installed"
        ) from error
