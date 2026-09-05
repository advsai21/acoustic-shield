"""
app/main.py

FastAPI application entrypoint.

Run from the backend/ directory:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings


def _configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


_configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice Integrity Verification Framework — Backend",
    description=(
        "Orchestration layer: audio ingestion -> ML adapters -> risk fusion "
        "-> action engine -> privacy-preserving storage."
    ),
    version="0.1.0",
)

app.include_router(router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    # Keep this thin and generic — never leak stack traces or internal
    # model/module details to clients.
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
