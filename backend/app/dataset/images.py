from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError, ValidationError
from app.core.ids import new_id
from app.core.models import ImageItem
from app.core.schemas import ScanImagesResponse, SplitName
from app.core.storage import SUPPORTED_IMAGE_SUFFIXES, Storage
from app.dataset.service import get_dataset, refresh_dataset_counts


def get_image(session: Session, image_id: str) -> ImageItem:
    image = session.get(ImageItem, image_id)
    if image is None:
        raise NotFoundError("image_not_found", "Image was not found.")
    return image


def list_images(session: Session, dataset_id: str) -> list[ImageItem]:
    get_dataset(session, dataset_id)
    query = select(ImageItem).where(ImageItem.dataset_id == dataset_id).order_by(ImageItem.created_at, ImageItem.id)
    return list(session.scalars(query))


def _stage_image(
    session: Session,
    storage: Storage,
    dataset_id: str,
    original_name: str,
    content: bytes,
    split: SplitName,
) -> ImageItem:
    image_id = new_id("img")
    storage_name = storage.safe_storage_name(original_name, image_id)
    width, height = storage.write_image(dataset_id, storage_name, content)
    image = ImageItem(
        id=image_id,
        dataset_id=dataset_id,
        file_name=Path(original_name).name,
        storage_name=storage_name,
        width=width,
        height=height,
        split=split,
        status="unannotated",
    )
    session.add(image)
    return image


def add_uploaded_images(
    session: Session,
    storage: Storage,
    dataset_id: str,
    uploads: list[tuple[str, bytes]],
    split: SplitName,
) -> list[ImageItem]:
    get_dataset(session, dataset_id)
    staged: list[ImageItem] = []
    try:
        for file_name, content in uploads:
            staged.append(_stage_image(session, storage, dataset_id, file_name, content, split))
        session.flush()
        refresh_dataset_counts(session, dataset_id)
        session.commit()
    except Exception:
        session.rollback()
        for image in staged:
            storage.image_path(dataset_id, image.storage_name).unlink(missing_ok=True)
        raise
    for image in staged:
        session.refresh(image)
    return staged


def scan_images(
    session: Session,
    storage: Storage,
    dataset_id: str,
    requested_path: str,
    recursive: bool,
    split: SplitName,
) -> ScanImagesResponse:
    get_dataset(session, dataset_id)
    root = storage.resolve_import_directory(requested_path)
    paths = sorted(path for path in (root.rglob("*") if recursive else root.iterdir()) if path.is_file())
    total_found = len(paths)
    skipped = 0
    invalid = 0
    uploads: list[tuple[str, bytes]] = []
    for path in paths:
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            skipped += 1
            continue
        try:
            uploads.append((path.name, path.read_bytes()))
        except OSError:
            invalid += 1
    imported = 0
    for file_name, content in uploads:
        try:
            add_uploaded_images(session, storage, dataset_id, [(file_name, content)], split)
            imported += 1
        except ValidationError:
            invalid += 1
    return ScanImagesResponse(total_found=total_found, imported=imported, skipped=skipped, invalid=invalid)


def update_image_split(session: Session, image_id: str, split: SplitName) -> ImageItem:
    image = get_image(session, image_id)
    image.split = split
    session.commit()
    session.refresh(image)
    return image


def delete_image(session: Session, storage: Storage, image_id: str) -> None:
    image = get_image(session, image_id)
    dataset_id = image.dataset_id
    path = storage.image_path(dataset_id, image.storage_name)
    session.delete(image)
    session.flush()
    refresh_dataset_counts(session, dataset_id)
    session.commit()
    path.unlink(missing_ok=True)

