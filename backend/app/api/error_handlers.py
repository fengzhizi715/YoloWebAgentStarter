from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.errors import ConflictError, DomainError, NotFoundError, ValidationError

logger = logging.getLogger("ywa")


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def handle_domain_error(request: Request, exc: DomainError) -> JSONResponse:
        status_code = 400
        if isinstance(exc, NotFoundError):
            status_code = 404
        elif isinstance(exc, ConflictError):
            status_code = 409
        elif isinstance(exc, ValidationError):
            status_code = 422
        logger.warning(
            "Request rejected method=%s path=%s code=%s message=%s",
            request.method,
            request.url.path,
            exc.error_code,
            exc.message,
        )
        return JSONResponse(
            status_code=status_code,
            content={"error": {"code": exc.error_code, "message": exc.message, "details": exc.details}},
        )

    @app.exception_handler(Exception)
    async def handle_unknown_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled request error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "An unexpected server error occurred."}},
        )
