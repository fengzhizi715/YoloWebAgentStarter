from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.models import Annotation, ClassLabel, Dataset, ImageItem
from app.core.schemas import OBB, ValidationIssue, ValidationReport
from app.core.storage import Storage
from app.core.task_types import annotation_type_for
from app.dataset.annotation.geometry import validate_bbox, validate_obb, validate_polygon


def validate_dataset(session: Session, storage: Storage, dataset_id: str) -> ValidationReport:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        from app.core.errors import NotFoundError

        raise NotFoundError("dataset_not_found", "Dataset was not found.")

    issues: list[ValidationIssue] = []
    classes = list(session.scalars(select(ClassLabel).where(ClassLabel.dataset_id == dataset_id)))
    class_ids = {item.id for item in classes}
    indexes = sorted(item.class_index for item in classes)
    if indexes != list(range(len(indexes))):
        issues.append(
            ValidationIssue(
                level="warning",
                code="class_indexes_not_contiguous",
                message="Class indexes are not contiguous from zero.",
            )
        )

    images = list(session.scalars(select(ImageItem).where(ImageItem.dataset_id == dataset_id)))
    for image in images:
        try:
            path = storage.image_path(dataset_id, image.storage_name)
        except Exception:
            path = None
        if path is None or not path.is_file():
            issues.append(
                ValidationIssue(
                    level="error",
                    code="image_file_missing",
                    message="The managed image file is missing.",
                    image_id=image.id,
                )
            )
        image_annotations = list(
            session.scalars(
                select(Annotation)
                .options(joinedload(Annotation.class_label))
                .where(Annotation.image_id == image.id)
                .order_by(Annotation.created_at, Annotation.id)
            )
        )
        if not image_annotations:
            issues.append(
                ValidationIssue(
                    level="warning",
                    code="image_unannotated",
                    message="The image has no annotations.",
                    image_id=image.id,
                )
            )
        if dataset.task_type == "classify" and len(image_annotations) > 1:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="classify_multiple",
                    message="Classification datasets allow exactly one class per image.",
                    image_id=image.id,
                )
            )
        for annotation in image_annotations:
            if annotation.dataset_id != dataset_id or annotation.class_id not in class_ids:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="annotation_class_mismatch",
                        message="Annotation class does not belong to this dataset.",
                        image_id=image.id,
                        annotation_id=annotation.id,
                    )
                )
                continue
            expected_type = annotation_type_for(dataset.task_type)
            if annotation.type != expected_type:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="annotation_family_mismatch",
                        message=f"{dataset.task_type} datasets require {expected_type} annotations.",
                        image_id=image.id,
                        annotation_id=annotation.id,
                    )
                )
                continue
            try:
                if annotation.type == "bbox":
                    from app.core.schemas import BBox

                    validate_bbox(
                        BBox(
                            x=annotation.x or 0,
                            y=annotation.y or 0,
                            width=annotation.width or 0,
                            height=annotation.height or 0,
                        ),
                        image.width,
                        image.height,
                    )
                elif annotation.type == "polygon":
                    validate_polygon(
                        [tuple(point) for point in (annotation.polygon or [])],
                        image.width,
                        image.height,
                    )
                elif annotation.type == "obb":
                    validate_obb(OBB(**(annotation.obb or {})), image.width, image.height)
            except Exception as exc:
                from app.core.errors import DomainError

                message = exc.message if isinstance(exc, DomainError) else "Annotation geometry is invalid."
                code = exc.error_code if isinstance(exc, DomainError) else "annotation_geometry_invalid"
                issues.append(
                    ValidationIssue(
                        level="error",
                        code=code,
                        message=message,
                        image_id=image.id,
                        annotation_id=annotation.id,
                    )
                )

    error_count = sum(issue.level == "error" for issue in issues)
    warning_count = sum(issue.level == "warning" for issue in issues)
    return ValidationReport(
        dataset_id=dataset_id,
        valid=error_count == 0,
        error_count=error_count,
        warning_count=warning_count,
        issues=issues,
    )
