from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_session, get_settings, get_storage
from app.core.config import Settings
from app.core.errors import NotFoundError, ValidationError
from app.core.ids import new_id
from app.core.models import Annotation, ClassLabel, Dataset, ImageItem
from app.core.schemas import (
    AnnotationResponse,
    AutoSplitRequest,
    BulkImageSplitUpdate,
    ClassLabelCreate,
    ClassLabelResponse,
    DatasetCreate,
    DatasetDetailResponse,
    DatasetResponse,
    DatasetUpdate,
    DuplicateReport,
    ImageItemResponse,
    ImagePage,
    ImageSplitUpdate,
    ReplaceAnnotationsRequest,
    ScanImagesRequest,
    ScanImagesResponse,
    SplitOperationResponse,
    TileDatasetRequest,
    TileDatasetResponse,
    SplitName,
    UploadImagesResponse,
    ValidationReport,
    VideoImportResponse,
    YoloImportResponse,
)
from app.core.storage import Storage
from app.core.task_types import TaskType
from app.dataset.annotation.service import list_annotations, replace_annotations
from app.dataset.exchange.yolo import YoloArchiveLimits, export_dataset, import_dataset
from app.dataset.exchange.coco import export_coco, import_coco
from app.dataset.quality.schemas import DatasetQualityReport
from app.dataset.quality.service import DatasetQualityService
from app.dataset.images import (
    add_uploaded_images,
    auto_split_images,
    delete_image,
    get_image,
    list_images,
    scan_images,
    update_image_split,
    update_image_splits,
)
from app.dataset.preparation.duplicates import DuplicateDetector
from app.dataset.preparation.tiling import DatasetTiler
from app.dataset.video import import_video_frames
from app.dataset.service import (
    create_class,
    create_dataset,
    delete_class,
    delete_dataset,
    get_dataset,
    list_classes,
    list_datasets,
    update_dataset,
)
from app.dataset.validation import validate_dataset


router = APIRouter(prefix="/datasets", tags=["datasets"])
file_router = APIRouter(prefix="/images", tags=["images"])


def _dataset_response(dataset: Dataset, *, annotated_image_count: int = 0) -> DatasetResponse:
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        task_type=dataset.task_type,
        image_count=dataset.image_count,
        annotated_image_count=annotated_image_count,
        class_count=dataset.class_count,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
    )


def _dataset_responses(session: Session, datasets: list[Dataset]) -> list[DatasetResponse]:
    if not datasets:
        return []
    ids = [dataset.id for dataset in datasets]
    annotated = dict(
        session.execute(
            select(Annotation.dataset_id, func.count(func.distinct(Annotation.image_id)))
            .where(Annotation.dataset_id.in_(ids))
            .group_by(Annotation.dataset_id)
        ).all()
    )
    return [_dataset_response(dataset, annotated_image_count=annotated.get(dataset.id, 0)) for dataset in datasets]


def _class_response(item: ClassLabel) -> ClassLabelResponse:
    return ClassLabelResponse.model_validate(item)


def _image_response(item: ImageItem) -> ImageItemResponse:
    return ImageItemResponse(
        id=item.id,
        dataset_id=item.dataset_id,
        file_name=item.file_name,
        width=item.width,
        height=item.height,
        split=item.split,
        status=item.status,
        file_url=f"/api/images/{item.id}/file",
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValidationError("upload_too_large", f"Upload exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    return content


@router.post("/import/yolo", response_model=YoloImportResponse, status_code=201)
async def import_yolo_archive(
    file: UploadFile = File(...),
    name: str = Form(...),
    task_type: TaskType = Form(...),
    description: str | None = Form(default=None),
    default_split: SplitName = Form(default="train"),
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> YoloImportResponse:
    content = await _read_upload(file, settings.max_upload_bytes)
    if not name.strip():
        raise ValidationError("dataset_name_required", "Dataset name is required.")
    dataset, imported_images, imported_annotations = import_dataset(
        session,
        storage,
        content,
        DatasetCreate(name=name, description=description, task_type=task_type),
        default_split,
        limits=YoloArchiveLimits(
            max_members=settings.max_yolo_archive_members,
            max_member_bytes=settings.max_yolo_archive_member_bytes,
            max_total_uncompressed_bytes=settings.max_yolo_archive_uncompressed_bytes,
            max_compression_ratio=settings.max_yolo_archive_compression_ratio,
        ),
    )
    return YoloImportResponse(
        dataset=_dataset_responses(session, [dataset])[0],
        imported_images=imported_images,
        imported_annotations=imported_annotations,
    )


@router.post("/import/coco", response_model=YoloImportResponse, status_code=201)
async def import_coco_archive(
    file: UploadFile = File(...),
    name: str = Form(...),
    task_type: TaskType = Form(...),
    description: str | None = Form(default=None),
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> YoloImportResponse:
    dataset, images, annotations = import_coco(
        session,
        storage,
        await _read_upload(file, settings.max_upload_bytes),
        DatasetCreate(name=name, description=description, task_type=task_type),
        limits=YoloArchiveLimits(
            max_members=settings.max_yolo_archive_members,
            max_member_bytes=settings.max_yolo_archive_member_bytes,
            max_total_uncompressed_bytes=settings.max_yolo_archive_uncompressed_bytes,
            max_compression_ratio=settings.max_yolo_archive_compression_ratio,
        ),
    )
    return YoloImportResponse(dataset=_dataset_responses(session, [dataset])[0], imported_images=images, imported_annotations=annotations)


@router.get("", response_model=list[DatasetResponse])
def get_datasets(session: Session = Depends(get_session)) -> list[DatasetResponse]:
    return _dataset_responses(session, list_datasets(session))


@router.post("", response_model=DatasetResponse, status_code=201)
def post_dataset(payload: DatasetCreate, session: Session = Depends(get_session)) -> DatasetResponse:
    return _dataset_response(create_dataset(session, payload))


@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
def get_dataset_detail(dataset_id: str, session: Session = Depends(get_session)) -> DatasetDetailResponse:
    dataset = get_dataset(session, dataset_id)
    classes = list_classes(session, dataset_id)
    return DatasetDetailResponse(
        **_dataset_responses(session, [dataset])[0].model_dump(),
        classes=[_class_response(item) for item in classes],
        image_total=dataset.image_count,
    )


@router.patch("/{dataset_id}", response_model=DatasetResponse)
def patch_dataset(dataset_id: str, payload: DatasetUpdate, session: Session = Depends(get_session)) -> DatasetResponse:
    return _dataset_response(update_dataset(session, dataset_id, payload))


@router.delete("/{dataset_id}", status_code=204, response_class=Response, response_model=None)
def remove_dataset(dataset_id: str, session: Session = Depends(get_session), storage: Storage = Depends(get_storage)) -> None:
    delete_dataset(session, storage, dataset_id)


@router.get("/{dataset_id}/classes", response_model=list[ClassLabelResponse])
def get_classes(dataset_id: str, session: Session = Depends(get_session)) -> list[ClassLabelResponse]:
    return [_class_response(item) for item in list_classes(session, dataset_id)]


@router.post("/{dataset_id}/classes", response_model=ClassLabelResponse, status_code=201)
def post_class(dataset_id: str, payload: ClassLabelCreate, session: Session = Depends(get_session)) -> ClassLabelResponse:
    return _class_response(create_class(session, dataset_id, payload))


@router.delete("/{dataset_id}/classes/{class_id}", status_code=204, response_class=Response, response_model=None)
def remove_class(dataset_id: str, class_id: str, session: Session = Depends(get_session)) -> None:
    delete_class(session, dataset_id, class_id)


@router.get("/{dataset_id}/images", response_model=ImagePage)
def get_images(
    dataset_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=500),
    session: Session = Depends(get_session),
) -> ImagePage:
    items = list_images(session, dataset_id)
    return ImagePage(items=[_image_response(item) for item in items[offset : offset + limit]], total=len(items))


@router.post("/{dataset_id}/images/upload", response_model=UploadImagesResponse, status_code=201)
async def upload_images(
    dataset_id: str,
    files: list[UploadFile] = File(...),
    split: SplitName = Form(default="train"),
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> UploadImagesResponse:
    uploads: list[tuple[str, bytes]] = []
    total = 0
    for file in files:
        content = await file.read(settings.max_upload_bytes - total + 1)
        total += len(content)
        if total > settings.max_upload_bytes:
            raise ValidationError("upload_too_large", "The combined image upload exceeds the configured limit.")
        uploads.append((file.filename or "image", content))
    items = add_uploaded_images(session, storage, dataset_id, uploads, split)
    return UploadImagesResponse(imported=len(items), items=[_image_response(item) for item in items])


@router.post("/{dataset_id}/video/import", response_model=VideoImportResponse)
async def import_video(
    dataset_id: str,
    file: UploadFile = File(...),
    split: SplitName = Form(default="train"),
    frame_interval: int = Form(default=30),
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> VideoImportResponse:
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi"}:
        raise ValidationError("video_format_unsupported", "Supported video formats are mp4, mov and avi.")
    return import_video_frames(session, storage, dataset_id, await _read_upload(file, settings.max_upload_bytes), suffix, split, frame_interval)


@router.post("/{dataset_id}/images/scan", response_model=ScanImagesResponse)
def scan_dataset_images(
    dataset_id: str,
    payload: ScanImagesRequest,
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
) -> ScanImagesResponse:
    return scan_images(session, storage, dataset_id, payload.path, payload.recursive, payload.split)


@router.patch("/{dataset_id}/images/{image_id}", response_model=ImageItemResponse)
def patch_image(
    dataset_id: str,
    image_id: str,
    payload: ImageSplitUpdate,
    session: Session = Depends(get_session),
) -> ImageItemResponse:
    image = get_image(session, image_id)
    if image.dataset_id != dataset_id:
        raise NotFoundError("image_not_found", "Image was not found in this dataset.")
    return _image_response(update_image_split(session, image_id, payload.split))


@router.post("/{dataset_id}/images/bulk-split", response_model=SplitOperationResponse)
def bulk_update_image_split(dataset_id: str, payload: BulkImageSplitUpdate, session: Session = Depends(get_session)) -> SplitOperationResponse:
    return update_image_splits(session, dataset_id, payload)


@router.post("/{dataset_id}/images/auto-split", response_model=SplitOperationResponse)
def auto_split_dataset_images(dataset_id: str, payload: AutoSplitRequest, session: Session = Depends(get_session)) -> SplitOperationResponse:
    return auto_split_images(session, dataset_id, payload)


@router.delete("/{dataset_id}/images/{image_id}", status_code=204, response_class=Response, response_model=None)
def remove_image(
    dataset_id: str,
    image_id: str,
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
) -> None:
    image = get_image(session, image_id)
    if image.dataset_id != dataset_id:
        raise NotFoundError("image_not_found", "Image was not found in this dataset.")
    delete_image(session, storage, image_id)


@router.get("/{dataset_id}/images/{image_id}/annotations", response_model=list[AnnotationResponse])
def get_image_annotations(dataset_id: str, image_id: str, session: Session = Depends(get_session)) -> list[AnnotationResponse]:
    image = get_image(session, image_id)
    if image.dataset_id != dataset_id:
        raise NotFoundError("image_not_found", "Image was not found in this dataset.")
    return list_annotations(session, image_id)


@router.put("/{dataset_id}/images/{image_id}/annotations", response_model=list[AnnotationResponse])
def put_image_annotations(
    dataset_id: str,
    image_id: str,
    payload: ReplaceAnnotationsRequest,
    session: Session = Depends(get_session),
) -> list[AnnotationResponse]:
    image = get_image(session, image_id)
    if image.dataset_id != dataset_id:
        raise NotFoundError("image_not_found", "Image was not found in this dataset.")
    return replace_annotations(session, image_id, payload.annotations)


@router.post("/{dataset_id}/validate", response_model=ValidationReport)
def validate_dataset_route(
    dataset_id: str,
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
) -> ValidationReport:
    return validate_dataset(session, storage, dataset_id)


@router.get("/{dataset_id}/quality/report", response_model=DatasetQualityReport)
def get_dataset_quality_report(dataset_id: str, session: Session = Depends(get_session)) -> DatasetQualityReport:
    return DatasetQualityService().report(session, dataset_id)


@router.get("/{dataset_id}/duplicates", response_model=DuplicateReport)
def get_duplicate_report(
    dataset_id: str,
    phash_distance: int = Query(default=8, ge=0, le=64),
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
) -> DuplicateReport:
    return DuplicateReport.model_validate(DuplicateDetector().analyze(list_images(session, dataset_id), storage, phash_distance))


@router.post("/{dataset_id}/tile", response_model=TileDatasetResponse, status_code=201)
def tile_dataset(
    dataset_id: str,
    payload: TileDatasetRequest,
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
) -> TileDatasetResponse:
    return DatasetTiler().create(session, storage, dataset_id, payload)


@router.get("/{dataset_id}/export/yolo")
def export_yolo_archive(
    dataset_id: str,
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
) -> FileResponse:
    content, file_name = export_dataset(session, storage, dataset_id)
    output_path = storage.export_path(f"{new_id('yolo')}_{Path(file_name).name}")
    output_path.write_bytes(content)
    return FileResponse(output_path, media_type="application/zip", filename=file_name)


@router.get("/{dataset_id}/export/coco")
def export_coco_archive(dataset_id: str, session: Session = Depends(get_session), storage: Storage = Depends(get_storage)) -> FileResponse:
    content, file_name = export_coco(session, storage, dataset_id)
    output_path = storage.export_path(f"{new_id('coco')}_{Path(file_name).name}")
    output_path.write_bytes(content)
    return FileResponse(output_path, media_type="application/zip", filename=file_name)


@file_router.get("/{image_id}/file", name="get_image_file")
def get_image_file(image_id: str, session: Session = Depends(get_session), storage: Storage = Depends(get_storage)) -> FileResponse:
    image = get_image(session, image_id)
    path = storage.image_path(image.dataset_id, image.storage_name)
    if not path.is_file():
        raise NotFoundError("image_file_not_found", "The managed image file was not found.")
    return FileResponse(path, media_type=_media_type(path), filename=image.file_name)


def _media_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")
