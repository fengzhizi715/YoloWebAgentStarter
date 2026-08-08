from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from app.core.config import Settings


def get_session(request: Request) -> Iterator[Session]:
    yield from request.app.state.database.sessions()


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_storage(request: Request):
    return request.app.state.storage

