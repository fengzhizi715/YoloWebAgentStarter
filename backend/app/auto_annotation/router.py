from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.dependencies import get_auto_annotation_service, get_session
from app.auto_annotation.schemas import AutoAnnotationCreateRequest, AutoAnnotationLogResponse, AutoAnnotationTaskResponse
from app.auto_annotation.service import AutoAnnotationService, task_response


dataset_router = APIRouter(prefix="/datasets", tags=["auto-annotation"])
task_router = APIRouter(prefix="/auto-annotation", tags=["auto-annotation"])


@dataset_router.post("/{dataset_id}/auto-annotation", response_model=AutoAnnotationTaskResponse, status_code=202)
def create_auto_annotation(
    dataset_id: str,
    payload: AutoAnnotationCreateRequest,
    session: Session = Depends(get_session),
    service: AutoAnnotationService = Depends(get_auto_annotation_service),
) -> AutoAnnotationTaskResponse:
    return task_response(service.create_task(session, dataset_id, payload))


@dataset_router.get("/{dataset_id}/auto-annotation", response_model=list[AutoAnnotationTaskResponse])
def list_auto_annotations(
    dataset_id: str,
    session: Session = Depends(get_session),
    service: AutoAnnotationService = Depends(get_auto_annotation_service),
) -> list[AutoAnnotationTaskResponse]:
    return [task_response(item) for item in service.list_tasks(session, dataset_id)]


@task_router.get("/{task_id}", response_model=AutoAnnotationTaskResponse)
def get_auto_annotation(task_id: str, session: Session = Depends(get_session), service: AutoAnnotationService = Depends(get_auto_annotation_service)) -> AutoAnnotationTaskResponse:
    return task_response(service.get_task(session, task_id))


@task_router.post("/{task_id}/stop", response_model=AutoAnnotationTaskResponse)
def stop_auto_annotation(task_id: str, session: Session = Depends(get_session), service: AutoAnnotationService = Depends(get_auto_annotation_service)) -> AutoAnnotationTaskResponse:
    return task_response(service.stop_task(session, task_id))


@task_router.get("/{task_id}/logs", response_model=AutoAnnotationLogResponse)
def get_auto_annotation_logs(
    task_id: str,
    tail: int | None = Query(default=None, ge=1, le=10000),
    session: Session = Depends(get_session),
    service: AutoAnnotationService = Depends(get_auto_annotation_service),
) -> AutoAnnotationLogResponse:
    return service.logs(session, task_id, tail)
