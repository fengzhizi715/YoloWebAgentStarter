from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.error_handlers import register_error_handlers
from app.api.routes import api_router
from app.core.config import Settings
from app.core.database import Database
from app.core.migrations import upgrade_database
from app.core.storage import Storage


def create_app(settings: Settings | None = None, *, run_migrations: bool = True) -> FastAPI:
    resolved = settings or Settings.from_env()
    resolved.ensure_directories()
    database = Database(resolved.database_url)
    storage = Storage(resolved.data_dir, resolved.import_root)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if run_migrations:
            upgrade_database(resolved.project_root / "backend", resolved.database_url)
        yield
        database.dispose()

    app = FastAPI(title="YoloWebAgentStarter API", version="0.1.0-dev", lifespan=lifespan)
    app.state.settings = resolved
    app.state.database = database
    app.state.storage = storage
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()

