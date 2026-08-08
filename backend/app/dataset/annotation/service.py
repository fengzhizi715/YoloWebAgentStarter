from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, joinedload

from app.core.errors import ValidationError
from app.core.ids import new_id
from app.core.models import Annotation, ClassLabel
from app.core.schemas import AnnotationInput, AnnotationResponse, BBox
from app.core.task_types import annotation_type_for
from app.dataset.annotation.geometry import validate_bbox, validate_polygon
from app.dataset.images import get_image
from app.dataset.service import get_dataset


def _response(annotation: Annotation) -> AnnotationResponse:
    class_label = annotation.class_label
    bbox = None
    if annotation.type == "bbox":
        bbox = BBox(x=annotation.x or 0, y=annotation.y or 0, width=annotation.width or 0, height=annotation.height or 0)
    polygon = [tuple(point) for point in annotation.polygon] if annotation.polygon else None
    return AnnotationResponse(
        id=annotation.id,
        image_id=annotation.image_id,
        dataset_id=annotation.dataset_id,
        class_id=annotation.class_id,
        class_index=class_label.class_index,
        label=class_label.name,
        color=class_label.color,
        type=annotation.type,
        bbox=bbox,
        polygon=polygon,
        source=annotation.source,
        created_at=annotation.created_at,
        updated_at=annotation.updated_at,
    )


def list_annotations(session: Session, image_id: str) -> list[AnnotationResponse]:
    get_image(session, image_id)
    query = (
        select(Annotation)
        .options(joinedload(Annotation.class_label))
        .where(Annotation.image_id == image_id)
        .order_by(Annotation.created_at, Annotation.id)
    )
    return [_response(item) for item in session.scalars(query)]


def _validate_input(item: AnnotationInput, image_width: int, image_height: int) -> None:
    if item.type == "bbox" and item.bbox is not None:
        validate_bbox(item.bbox, image_width, image_height)
    elif item.type == "polygon" and item.polygon is not None:
        validate_polygon(item.polygon, image_width, image_height)


def replace_annotations(
    session: Session,
    image_id: str,
    items: list[AnnotationInput],
) -> list[AnnotationResponse]:
    image = get_image(session, image_id)
    dataset = get_dataset(session, image.dataset_id)
    expected_type = annotation_type_for(dataset.task_type)
    class_ids = set(
        session.scalars(select(ClassLabel.id).where(ClassLabel.dataset_id == dataset.id))
    )
    created: list[Annotation] = []
    for item in items:
        if item.type != expected_type:
            raise ValidationError(
                "annotation_family_mismatch",
                f"{dataset.task_type} datasets require {expected_type} annotations.",
            )
        if item.class_id not in class_ids:
            raise ValidationError("class_not_in_dataset", "Annotation class does not belong to this dataset.")
        _validate_input(item, image.width, image.height)
        created.append(
            Annotation(
                id=new_id("ann"),
                image_id=image.id,
                dataset_id=dataset.id,
                class_id=item.class_id,
                type=item.type,
                x=item.bbox.x if item.bbox else None,
                y=item.bbox.y if item.bbox else None,
                width=item.bbox.width if item.bbox else None,
                height=item.bbox.height if item.bbox else None,
                polygon=[list(point) for point in item.polygon] if item.polygon else None,
                source=item.source,
            )
        )
    session.execute(delete(Annotation).where(Annotation.image_id == image.id))
    session.add_all(created)
    image.status = "annotated" if created else "unannotated"
    session.commit()
    return list_annotations(session, image.id)

