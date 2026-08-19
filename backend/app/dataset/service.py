from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError
from app.core.ids import new_id
from app.core.models import Annotation, ClassLabel, Dataset, ImageItem
from app.core.schemas import ClassLabelCreate, DatasetCreate, DatasetUpdate
from app.core.storage import Storage


def get_dataset(session: Session, dataset_id: str) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise NotFoundError("dataset_not_found", "Dataset was not found.")
    return dataset


def list_datasets(session: Session) -> list[Dataset]:
    return list(session.scalars(select(Dataset).order_by(Dataset.created_at.desc())))


def create_dataset(session: Session, payload: DatasetCreate) -> Dataset:
    dataset = Dataset(
        id=new_id("ds"),
        name=payload.name,
        description=payload.description,
        task_type=payload.task_type.value,
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    return dataset


def update_dataset(session: Session, dataset_id: str, payload: DatasetUpdate) -> Dataset:
    dataset = get_dataset(session, dataset_id)
    changes = payload.model_dump(exclude_unset=True)
    for field, value in changes.items():
        setattr(dataset, field, value)
    session.commit()
    session.refresh(dataset)
    return dataset


def delete_dataset(session: Session, storage: Storage, dataset_id: str) -> None:
    dataset = get_dataset(session, dataset_id)
    session.delete(dataset)
    session.commit()
    storage.remove_dataset(dataset_id)


def refresh_dataset_counts(session: Session, dataset_id: str, *, commit: bool = False) -> None:
    dataset = get_dataset(session, dataset_id)
    dataset.image_count = session.scalar(
        select(func.count()).select_from(ImageItem).where(ImageItem.dataset_id == dataset_id)
    ) or 0
    dataset.class_count = session.scalar(
        select(func.count()).select_from(ClassLabel).where(ClassLabel.dataset_id == dataset_id)
    ) or 0
    if commit:
        session.commit()


def list_classes(session: Session, dataset_id: str) -> list[ClassLabel]:
    get_dataset(session, dataset_id)
    query = select(ClassLabel).where(ClassLabel.dataset_id == dataset_id).order_by(ClassLabel.class_index)
    return list(session.scalars(query))


def create_class(session: Session, dataset_id: str, payload: ClassLabelCreate) -> ClassLabel:
    get_dataset(session, dataset_id)
    next_index = session.scalar(
        select(func.max(ClassLabel.class_index)).where(ClassLabel.dataset_id == dataset_id)
    )
    class_label = ClassLabel(
        id=new_id("cls"),
        dataset_id=dataset_id,
        class_index=(next_index + 1) if next_index is not None else 0,
        name=payload.name,
        color=payload.color,
    )
    session.add(class_label)
    try:
        session.flush()
        refresh_dataset_counts(session, dataset_id)
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise ConflictError("class_name_exists", "A class with this name already exists.") from exc
    session.refresh(class_label)
    return class_label


def delete_class(session: Session, dataset_id: str, class_id: str) -> None:
    class_label = session.get(ClassLabel, class_id)
    if class_label is None or class_label.dataset_id != dataset_id:
        raise NotFoundError("class_not_found", "Class label was not found.")
    annotation_count = session.scalar(
        select(func.count()).select_from(Annotation).where(Annotation.class_id == class_id)
    ) or 0
    if annotation_count:
        raise ConflictError("class_in_use", "Delete annotations using this class before deleting it.")
    session.delete(class_label)
    session.flush()
    remaining = list(
        session.scalars(
            select(ClassLabel).where(ClassLabel.dataset_id == dataset_id).order_by(ClassLabel.class_index)
        )
    )
    for index, item in enumerate(remaining):
        item.class_index = index
    refresh_dataset_counts(session, dataset_id)
    session.commit()
