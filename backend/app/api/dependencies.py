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
