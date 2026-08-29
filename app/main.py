import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from .config import settings
from .routers import agent as agent_router
from .routers import health as health_router
from .routers import openai_compat as openai_compat_router
from .routers import prompt as prompt_router
from .routers import thread as thread_router
from .services import agent as agent_service
from .services import gateway, laravel, models
from .services import tools as tool_registry

# RAG layer (app/rag/). Self-contained: it shares only the settings file and the
# bearer-token dependency, and nothing in the agent imports it. Imported behind a
# guard because the upload route needs `python-multipart`, and a deployment that
# has not reinstalled requirements.txt must lose RAG - not the whole service.
try:
    from .rag import router as rag_router
    from .rag import service as rag_service
except Exception as _rag_import_error:  # noqa: BLE001 - RAG must never block boot
    rag_router = rag_service = None
    logging.getLogger(__name__).warning(
        "RAG layer not loaded (%s). Every existing endpoint is unaffected; "
        "install requirements.txt to enable /api/rag/*.", _rag_import_error,
    )

# LangChain agent + developer dashboard (app/graph/). Guarded for the same
# reason as RAG above: it needs `langchain-core`, and a deployment that has not
# reinstalled requirements.txt must lose these routes, not the service.
# /agent/reply keeps working either way - this is an additional path to a reply,
# never a replacement for the existing one.
try:
    from .graph import dashboard as graph_dashboard
    from .graph import router as graph_router
    from .graph import store as graph_store
except Exception as _graph_import_error:  # noqa: BLE001
    graph_router = graph_dashboard = graph_store = None
    logging.getLogger(__name__).warning(
        "LangChain agent not loaded (%s). Every existing endpoint is unaffected; "
        "`pip install langchain-core` to enable /api/v1/agent/* and the dashboard.",
        _graph_import_error,
    )

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format=_LOG_FORMAT,
)


def _attach_file_log() -> None:
    """Also write the log to a rotating file, when LOG_DIR is set.

    stdout is what `docker compose logs` shows and what Docker rotates; this is
    the copy that outlives that window. RotatingFileHandler is stdlib, so this
    costs no dependency.

    A log sink must never stop the service: an unwritable directory is reported
    on stdout - which is still working - and the service carries on.
    """
    if not settings.log_dir:
        return

    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    try:
        directory = Path(settings.log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            directory / settings.log_file_name,
            maxBytes=settings.log_file_max_mb * 1024 * 1024,
            backupCount=settings.log_file_keep,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logging.getLogger().addHandler(handler)

        # uvicorn configures its own loggers with propagate=False, so their
        # access and error lines never reach the root handler above - and those
        # are exactly the lines showing a request arrive. Attach directly, but
        # ONLY where propagation is off: adding it to a logger that still
        # propagates writes every record twice, once here and once at the root.
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            uvicorn_logger = logging.getLogger(name)
            if not uvicorn_logger.propagate:
                uvicorn_logger.addHandler(handler)
    except Exception as error:  # noqa: BLE001 - logging must not break startup
        logging.getLogger(__name__).error(
            "could not open the log file in %s (%s). Continuing with stdout only.",
            settings.log_dir, error,
        )


_attach_file_log()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await laravel.startup()
    except Exception as error:  # noqa: BLE001
        logger.warning("laravel connector startup error: %s", error)

    try:
        await gateway.startup()
    except Exception as error:  # noqa: BLE001
        logger.warning("gateway back-channel startup error: %s", error)

    try:
        registry = models.registry()
        registry_state = (
            f"{len(registry.models)} models / {len(registry.providers)} providers "
            f"from {registry.source}"
        )
    except models.RegistryError as error:
        # Not fatal at boot: /health reports it, and every request fails with the
        # same message. Crashing here would hide it behind a restart loop.
        registry_state = f"FAILED TO LOAD ({error})"

    logger.info(
        "AI agent ready (llm_registry=%s, tools=%s, data_dir=%s, api_keys=%s)",
        registry_state,
        ",".join(tool_registry.available_tools()) or "none",
        settings.data_dir,
        ",".join(settings.configured_keys) or "NONE SET",
    )
    # Loads the embedding model once, here, rather than inside the first upload.
    # It never raises: an unconfigured or unreachable RAG layer reports 503 on
    # its own routes and changes nothing above.
    if rag_service is not None:
        await rag_service.startup()

    # Report whether agent traces can be persisted. Never raises and never
    # migrates on its own: creating tables is an explicit operator action
    # (`python -m app.graph.migrate`), and a graph run works without them - the
    # trace is logged instead of stored.
    if graph_store is not None:
        try:
            if not graph_store.storage_configured():
                logger.info(
                    "agent tracing: no database configured - runs are logged, not "
                    "stored, and the dashboard will report this"
                )
            elif await graph_store.tables_present():
                logger.info("agent tracing: ready")
            else:
                logger.warning(
                    "agent tracing: database configured but the trace tables are "
                    "missing. Run `python -m app.graph.migrate`"
                )
        except Exception as error:  # noqa: BLE001 - tracing must not block boot
            logger.warning("agent tracing: could not be verified (%s)", error)

    yield
    if rag_service is not None:
        await rag_service.shutdown()
    try:
        await gateway.shutdown()
    except Exception as error:  # noqa: BLE001
        logger.warning("gateway shutdown error: %s", error)

    try:
        await laravel.shutdown()
    except Exception as error:  # noqa: BLE001
        logger.warning("laravel shutdown error: %s", error)

    await agent_service.shutdown()


app = FastAPI(
    title="Client-Agnostic AI Agent Service",
    version="2.0.0",
    description=(
        "LLM reasoning for any client application. No caller is assumed: every "
        "request carries its own client_id, prompt and context. The service holds "
        "no prompts, personas or business rules of its own - only the threads and "
        "prompt records clients explicitly registered."
    ),
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def _log_validation_error(request: Request, error: RequestValidationError):
    """FastAPI's own 422, with a log line saying what the request actually sent.

    A validation 422 never reaches the route, so nothing else logs it and the
    access log shows only the status. The usual cause on /documents/upload is a
    multipart body whose `file` part never arrived - wrong field name, a Text
    field where a File was meant, or a hand-set Content-Type with no boundary.
    """
    arrived = "<not form-encoded>"
    try:
        # Already parsed and cached by the failed validation; not a re-read.
        arrived = ", ".join(sorted((await request.form()).keys())) or "<empty>"
    except Exception:  # noqa: BLE001 - diagnostics must not mask the 422
        pass
    logger.warning(
        "422 %s %s - content-type=%r fields=[%s] errors=%s",
        request.method, request.url.path,
        request.headers.get("content-type"), arrived, error.errors(),
    )
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(error.errors())})


app.include_router(health_router.router)
app.include_router(thread_router.router)
app.include_router(prompt_router.router)
app.include_router(agent_router.router)
# OpenAI-compatible surface, so OpenClaw can use this service as a model provider.
app.include_router(openai_compat_router.router)
# Document ingestion and vector search, at /api/rag/*. Additive: no existing
# route's path, request or response changes because this is mounted.
if rag_router is not None:
    app.include_router(rag_router.router)
# LangChain agent at /api/v1/agent/* and the developer dashboard at
# /api/v1/dashboard/*. Additive: no existing route's path, request or response
# changes because these are mounted.
if graph_router is not None:
    app.include_router(graph_router.router)
    app.include_router(graph_dashboard.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
