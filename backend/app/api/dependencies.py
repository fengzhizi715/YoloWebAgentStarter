from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.training.service import TrainingService
from app.training.runtime.queue import training_queue
from app.models.evaluation import YoloEvaluationRunner
from app.auto_annotation.queue import auto_annotation_queue
from app.auto_annotation.service import AutoAnnotationService
from app.video_import.queue import video_import_queue
from app.video_import.service import VideoImportService
from app.agent.service import AgentService
from app.models.service import ModelService
from app.agent.read_tools import build_default_registry


def get_session(request: Request) -> Iterator[Session]:
    yield from request.app.state.database.sessions()


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_storage(request: Request):
    return request.app.state.storage


def get_training_service(request: Request) -> TrainingService:
    return TrainingService(request.app.state.database.session_factory, request.app.state.storage, training_queue)


def get_evaluation_runner(request: Request) -> YoloEvaluationRunner:
    return YoloEvaluationRunner(request.app.state.database.session_factory, request.app.state.storage)


def get_auto_annotation_service(request: Request) -> AutoAnnotationService:
    return AutoAnnotationService(request.app.state.database.session_factory, request.app.state.storage, auto_annotation_queue)


def get_video_import_service(request: Request) -> VideoImportService:
    return VideoImportService(
        request.app.state.database.session_factory,
        request.app.state.storage,
        request.app.state.settings,
        video_import_queue,
    )


def get_agent_service(request: Request) -> AgentService:
    storage = request.app.state.storage
    session_factory = request.app.state.database.session_factory
    settings = request.app.state.settings
    training_service = TrainingService(session_factory, storage, training_queue)
    model_service = ModelService(storage)
    auto_annotation_service = AutoAnnotationService(session_factory, storage, auto_annotation_queue)
    evaluation_runner = YoloEvaluationRunner(session_factory, storage)
    registry = build_default_registry(
        storage,
        training_service=training_service,
        model_service=model_service,
        session_factory=session_factory,
    )
    return AgentService(
        session_factory,
        settings,
        registry=registry,
        storage=storage,
        training_service=training_service,
        model_service=model_service,
        auto_annotation_service=auto_annotation_service,
        evaluation_runner=evaluation_runner,
    )
