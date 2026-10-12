import logging
from contextlib import asynccontextmanager

from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .rag import router as rag_router
from .rag import service as rag_service
from .services.mongodb_logger import mongodb_logger
from .whatsapp import conversation as whatsapp_conversation
from .whatsapp import router as whatsapp_router

FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
FRONTEND_INDEX_FILE = FRONTEND_DIST_DIR / "index.html"
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format=_LOG_FORMAT,
)


def _attach_file_log() -> None:
    """Also write the log to a rotating file, when LOG_DIR is set."""
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
    # Loads the embedding model once, here, rather than inside the first upload.
    # It never raises: an unconfigured or unreachable RAG layer reports 503 on
    # its own routes.
    if rag_service is not None:
        await rag_service.startup()
    await mongodb_logger.startup()
    await whatsapp_conversation.init_whatsapp_tables()

    yield
    await mongodb_logger.shutdown()
    if rag_service is not None:
        await rag_service.shutdown()


app = FastAPI(
    title="MMpact RAG API",
    version="2.0.0",
    description=(
        "Standalone document intelligence, hybrid retrieval (pgvector + FTS + RRF) and answer generation."
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
    arrived = "<not form-encoded>"
    try:
        arrived = ", ".join(sorted((await request.form()).keys())) or "<empty>"
    except Exception:  # noqa: BLE001
        pass
    logger.warning(
        "422 %s %s - content-type=%r fields=[%s] errors=%s",
        request.method, request.url.path,
        request.headers.get("content-type"), arrived, error.errors(),
    )
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(error.errors())})


# Mount static assets if built frontend exists
if FRONTEND_ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_ASSETS_DIR)), name="assets")


@app.get("/favicon.svg", include_in_schema=False)
async def favicon_svg():
    fav = FRONTEND_DIST_DIR / "favicon.svg"
    if fav.is_file():
        return FileResponse(fav)
    raise HTTPException(status_code=404, detail="Not Found")


@app.get("/icons.svg", include_in_schema=False)
async def icons_svg():
    icons = FRONTEND_DIST_DIR / "icons.svg"
    if icons.is_file():
        return FileResponse(icons)
    raise HTTPException(status_code=404, detail="Not Found")


@app.get("/health", tags=["health"])
async def service_health() -> dict:
    """Unauthenticated service health check."""
    return {
        "status": "healthy",
        "service": "MMpact RAG Service"
    }


@app.get("/", tags=["frontend"])
async def root_endpoint():
    """Serves frontend index.html when built, or fallback status JSON."""
    if FRONTEND_INDEX_FILE.is_file():
        return FileResponse(FRONTEND_INDEX_FILE)
    return {
        "status": "healthy",
        "service": "MMpact RAG Service"
    }


if rag_router is not None:
    app.include_router(rag_router.router)
    app.include_router(rag_router.logs_router)

if whatsapp_router is not None:
    app.include_router(whatsapp_router.router)


@app.get("/{full_path:path}", include_in_schema=False)
async def frontend_spa_fallback(full_path: str):
    """Fallback for SPA client-side routes, guarding API endpoints."""
    # Never return HTML for API paths or documentation endpoints
    if (
        full_path == "api"
        or full_path.startswith("api/")
        or full_path == "docs"
        or full_path.startswith("docs/")
        or full_path == "redoc"
        or full_path.startswith("redoc/")
        or full_path == "openapi.json"
    ):
        raise HTTPException(status_code=404, detail="Not Found")

    # If it is a static file that exists in frontend/dist, serve it
    static_candidate = FRONTEND_DIST_DIR / full_path
    if static_candidate.is_file():
        return FileResponse(static_candidate)

    # Return index.html for SPA client-side routes if the build exists
    if FRONTEND_INDEX_FILE.is_file():
        return FileResponse(FRONTEND_INDEX_FILE)

    raise HTTPException(status_code=404, detail="Not Found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )

