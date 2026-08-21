from __future__ import annotations

import math

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.errors import NotFoundError, ValidationError
from app.core.ids import new_id
from app.core.models import Annotation, AutoAnnotationTask, ClassLabel, ImageItem, ModelVersion
from app.core.storage import Storage
from app.core.task_types import annotation_type_for
from app.core.time import utc_now
from app.dataset.annotation.geometry import validate_bbox, validate_obb, validate_polygon
from app.dataset.service import get_dataset
from app.models.result_parser import Detection
from app.training.observability.log_store import TrainingLogStore
from app.auto_annotation.schemas import AutoAnnotationCreateRequest, AutoAnnotationLogResponse, AutoAnnotationTaskResponse


def task_response(task: AutoAnnotationTask) -> AutoAnnotationTaskResponse:
    return AutoAnnotationTaskResponse.model_validate(task)


class AutoAnnotationService:
    def __init__(self, session_factory: sessionmaker[Session], storage: Storage, queue) -> None:
        self.session_factory = session_factory
        self.storage = storage
        self.queue = queue

    def create_task(self, session: Session, dataset_id: str, payload: AutoAnnotationCreateRequest) -> AutoAnnotationTask:
        dataset = get_dataset(session, dataset_id)
        model = session.get(ModelVersion, payload.model_id)
        if model is None:
            raise NotFoundError("model_not_found", "Model version was not found.")
        if model.status != "active":
            raise ValidationError("model_archived", "Auto annotation requires an active managed model.")
        if model.format != "pt" or model.engine_type != "ultralytics":
            raise ValidationError("unsupported_model_engine", "Auto annotation requires a managed Ultralytics PT model.")
        model_path = self.storage.managed_model_path(model.model_path)
        if not model_path.is_file():
            raise NotFoundError("model_file_missing", "The managed model file is missing.")
        if model.task_type != dataset.task_type:
            raise ValidationError("auto_annotation_task_mismatch", "The model task type must match the dataset task type.")
        image_query = select(ImageItem).where(ImageItem.dataset_id == dataset_id)
        if payload.skip_annotated_images:
            image_query = image_query.where(~ImageItem.annotations.any())
        images = list(session.scalars(image_query.order_by(ImageItem.created_at, ImageItem.id)))
        if not images:
            message = "The dataset has no unannotated images. Disable skip_annotated_images to process existing annotations." if payload.skip_annotated_images else "The dataset has no images to annotate."
            raise ValidationError("auto_annotation_dataset_empty", message)
        target_classes = list(session.scalars(select(ClassLabel).where(ClassLabel.dataset_id == dataset_id).order_by(ClassLabel.class_index)))
        if not target_classes:
            raise ValidationError("auto_annotation_classes_missing", "Add at least one dataset class before running auto annotation.")
        source_classes = list(
            session.scalars(
                select(ClassLabel).where(ClassLabel.dataset_id == model.dataset_id).order_by(ClassLabel.class_index)
            )
        ) if model.dataset_id else []
        mapping = self._resolve_mapping(payload.class_mapping, target_classes, source_classes)
        task_id = new_id("auto")
        log_path = self.storage.auto_annotation_task_dir(task_id) / "auto_annotation.log"
        task = AutoAnnotationTask(
            id=task_id,
            dataset_id=dataset.id,
            model_id=model.id,
            task_type=dataset.task_type,
            status="pending",
            clean_old_annotations=payload.clean_old_annotations,
            skip_annotated_images=payload.skip_annotated_images,
            confidence=payload.confidence,
            iou=payload.iou,
            class_mapping=mapping,
            total_images=len(images),
            logs_path=str(log_path),
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        self.queue.submit(task.id)
        return task

    def list_tasks(self, session: Session, dataset_id: str) -> list[AutoAnnotationTask]:
        get_dataset(session, dataset_id)
        return list(
            session.scalars(
                select(AutoAnnotationTask)
                .where(AutoAnnotationTask.dataset_id == dataset_id)
                .order_by(AutoAnnotationTask.created_at.desc(), AutoAnnotationTask.id.desc())
            )
        )

    def get_task(self, session: Session, task_id: str) -> AutoAnnotationTask:
        task = session.get(AutoAnnotationTask, task_id)
        if task is None:
            raise NotFoundError("auto_annotation_task_not_found", "Auto-annotation task was not found.")
        return task

    def stop_task(self, session: Session, task_id: str) -> AutoAnnotationTask:
        task = self.get_task(session, task_id)
        if task.status in {"completed", "failed", "stopped"}:
            return task
        if task.status == "pending":
            task.status = "stopped"
            task.finished_at = utc_now()
            task.error_message = "Auto annotation stopped before the task started."
        else:
            task.stop_requested = True
        session.commit()
        session.refresh(task)
        self.queue.stop_pending(task_id)
        return task

    def logs(self, session: Session, task_id: str, tail: int | None = None) -> AutoAnnotationLogResponse:
        task = self.get_task(session, task_id)
        store = TrainingLogStore(task.logs_path or "")
        return AutoAnnotationLogResponse(task_id=task.id, logs=store.read(tail), line_count=store.line_count())

    @staticmethod
    def _resolve_mapping(
        requested: dict[str, str],
        target_classes: list[ClassLabel],
        source_classes: list[ClassLabel],
    ) -> dict[str, str]:
        target_by_id = {item.id: item for item in target_classes}
        target_by_name = {item.name.casefold(): item for item in target_classes}
        mapping: dict[str, str] = {}
        for source in source_classes:
            target = target_by_name.get(source.name.casefold())
            if target is not None:
                mapping[str(source.class_index)] = target.id
        for raw_index, target_id in requested.items():
            try:
                source_index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise ValidationError("invalid_class_mapping", "Model class indexes must be integers.") from exc
            if source_index < 0 or target_id not in target_by_id:
                raise ValidationError("invalid_class_mapping", "Class mapping must point to an existing dataset class.")
            if source_classes and source_index not in {item.class_index for item in source_classes}:
                raise ValidationError("invalid_class_mapping", "Class mapping refers to a class not provided by the selected model.")
            mapping[str(source_index)] = target_id
        if not mapping:
            raise ValidationError("class_mapping_empty", "No model classes can be mapped to this dataset.")
        return mapping


def save_auto_annotations(
    session: Session,
    task: AutoAnnotationTask,
    image: ImageItem,
    detections: list[Detection],
) -> tuple[int, int, bool]:
    """Persist model output as editable auto annotations, preserving human labels by default."""

    if task.skip_annotated_images and session.scalar(select(Annotation.id).where(Annotation.image_id == image.id).limit(1)) is not None:
        return 0, 0, True
    if task.clean_old_annotations:
        session.execute(delete(Annotation).where(Annotation.image_id == image.id))
    else:
        session.execute(delete(Annotation).where(Annotation.image_id == image.id, Annotation.source == "auto"))

    created: list[Annotation] = []
    skipped = 0
    expected_type = annotation_type_for(task.task_type)
    for detection in detections:
        class_id = task.class_mapping.get(str(detection.class_index))
        if class_id is None:
            skipped += 1
            continue
        annotation = _annotation_from_detection(task, image, detection, class_id, expected_type)
        if annotation is None:
            skipped += 1
            continue
        created.append(annotation)
    session.add_all(created)
    session.flush()
    annotation_count = session.scalar(select(func.count()).select_from(Annotation).where(Annotation.image_id == image.id)) or 0
    image.status = "annotated" if annotation_count else "unannotated"
    session.commit()
    return len(created), skipped, False


def _annotation_from_detection(
    task: AutoAnnotationTask,
    image: ImageItem,
    detection: Detection,
    class_id: str,
    expected_type: str,
) -> Annotation | None:
    common = {
        "id": new_id("ann"),
        "image_id": image.id,
        "dataset_id": image.dataset_id,
        "class_id": class_id,
        "type": expected_type,
        "source": "auto",
    }
    if expected_type == "classify":
        return Annotation(**common)
    if expected_type == "bbox":
        from app.core.schemas import BBox

        bbox = BBox(x=detection.x, y=detection.y, width=detection.width, height=detection.height)
        validate_bbox(bbox, image.width, image.height)
        return Annotation(**common, x=bbox.x, y=bbox.y, width=bbox.width, height=bbox.height)
    if expected_type == "polygon":
        points = [list(point) for point in detection.polygon or ()]
        if len(points) < 3:
            return None
        polygon = [(float(point[0]), float(point[1])) for point in points]
        validate_polygon(polygon, image.width, image.height)
        return Annotation(**common, polygon=points)
    obb = _obb_from_points(detection.obb_points)
    if obb is None:
        return None
    validate_obb(obb, image.width, image.height)
    return Annotation(**common, obb=obb.model_dump())


def _obb_from_points(points: tuple[tuple[float, float], ...] | None):
    from app.core.schemas import OBB

    if points is None or len(points) != 4:
        return None
    edge_a = math.dist(points[0], points[1])
    edge_b = math.dist(points[1], points[2])
    if edge_a <= 0 or edge_b <= 0:
        return None
    cx = sum(point[0] for point in points) / 4
    cy = sum(point[1] for point in points) / 4
    angle = math.degrees(math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]))
    return OBB(cx=cx, cy=cy, width=edge_a, height=edge_b, angle=angle)
