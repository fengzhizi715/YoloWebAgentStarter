from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.api.dependencies import get_session, get_video_import_service
from app.core.task_types import TaskType
from app.video_import.schemas import SamplingMode, VideoImportTaskResponse
from app.video_import.service import VideoImportService, task_response


router = APIRouter(prefix="/video-imports", tags=["video-imports"])


@router.post("", response_model=VideoImportTaskResponse, status_code=201)
async def create_video_import(
    file: UploadFile = File(...),
    name: str = Form(...),
    task_type: TaskType = Form(...),
    split: str = Form(default="train"),
    sampling_mode: SamplingMode = Form(default="fps"),
    sampling_value: float = Form(default=1),
    start_seconds: float = Form(default=0),
    end_seconds: float | None = Form(default=None),
    split_strategy: str = Form(default="single"),
    time_block_seconds: float = Form(default=30),
    train_ratio: float = Form(default=0.8),
    val_ratio: float = Form(default=0.1),
    test_ratio: float = Form(default=0.1),
    session: Session = Depends(get_session),
    service: VideoImportService = Depends(get_video_import_service),
) -> VideoImportTaskResponse:
    return task_response(
        service.create_task(
            session,
            source=file.file,
            source_file_name=file.filename or "video",
            name=name,
            task_type=task_type.value,
            split=split,
            sampling_mode=sampling_mode,
            sampling_value=sampling_value,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            split_strategy=split_strategy,
            time_block_seconds=time_block_seconds,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    )


@router.get("/{task_id}", response_model=VideoImportTaskResponse)
def get_video_import(
    task_id: str,
    session: Session = Depends(get_session),
    service: VideoImportService = Depends(get_video_import_service),
) -> VideoImportTaskResponse:
    return task_response(service.get_task(session, task_id))


@router.post("/{task_id}/start", response_model=VideoImportTaskResponse, status_code=202)
def start_video_import(
    task_id: str,
    session: Session = Depends(get_session),
    service: VideoImportService = Depends(get_video_import_service),
) -> VideoImportTaskResponse:
    return task_response(service.start_task(session, task_id))


@router.post("/{task_id}/retry", response_model=VideoImportTaskResponse, status_code=202)
def retry_video_import(
    task_id: str,
    session: Session = Depends(get_session),
    service: VideoImportService = Depends(get_video_import_service),
) -> VideoImportTaskResponse:
    return task_response(service.retry_task(session, task_id))
