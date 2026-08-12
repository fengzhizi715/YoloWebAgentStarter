from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path, PurePosixPath

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ValidationError
from app.core.ids import new_id
from app.core.models import Annotation, ClassLabel, Dataset, ImageItem
from app.core.schemas import BBox, DatasetCreate
from app.core.storage import Storage
from app.dataset.annotation.geometry import validate_bbox, validate_polygon
from app.dataset.exchange.yolo import YoloArchiveLimits, _archive_entries, _read_archive_member
from app.dataset.service import refresh_dataset_counts


def export_coco(session: Session, storage: Storage, dataset_id: str) -> tuple[bytes, str]:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        from app.core.errors import NotFoundError
        raise NotFoundError("dataset_not_found", "Dataset was not found.")
    if dataset.task_type not in {"detect", "segment"}:
        raise ValidationError("coco_task_unsupported", "COCO exchange currently supports detect and segment datasets.")
    classes = list(session.scalars(select(ClassLabel).where(ClassLabel.dataset_id == dataset_id).order_by(ClassLabel.class_index)))
    image_ids = {item.id: index + 1 for index, item in enumerate(session.scalars(select(ImageItem).where(ImageItem.dataset_id == dataset_id).order_by(ImageItem.id)))}
    images = list(session.scalars(select(ImageItem).where(ImageItem.dataset_id == dataset_id).order_by(ImageItem.id)))
    annotations = list(session.scalars(select(Annotation).where(Annotation.dataset_id == dataset_id).order_by(Annotation.id)))
    class_ids = {item.id: item.class_index + 1 for item in classes}
    archive_names = {item.id: f"images/{item.split}/{item.id}_{Path(item.file_name).name}" for item in images}
    payload = {"info": {"description": dataset.name, "version": "1.0"}, "licenses": [], "images": [{"id": image_ids[item.id], "file_name": archive_names[item.id], "starter_original_file_name": item.file_name, "width": item.width, "height": item.height, "split": item.split} for item in images], "annotations": [], "categories": [{"id": item.class_index + 1, "name": item.name, "supercategory": "object"} for item in classes]}
    for index, annotation in enumerate(annotations, 1):
        image = next(item for item in images if item.id == annotation.image_id)
        if annotation.type == "bbox" and annotation.x is not None:
            bbox = [annotation.x, annotation.y, annotation.width, annotation.height]
            segmentation = []
        elif annotation.type == "polygon" and annotation.polygon:
            xs, ys = zip(*annotation.polygon, strict=True)
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            segmentation = [[coordinate for point in annotation.polygon for coordinate in point]]
        else:
            raise ValidationError("coco_annotation_unsupported", "COCO export found an unsupported annotation type.")
        payload["annotations"].append({"id": index, "image_id": image_ids[image.id], "category_id": class_ids[annotation.class_id], "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0, "segmentation": segmentation})
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("annotations.json", json.dumps(payload, ensure_ascii=False, indent=2))
        for image in images:
            archive.writestr(archive_names[image.id], storage.read_image(dataset_id, image.storage_name))
    return output.getvalue(), f"{dataset.name}_coco.zip"


def import_coco(
    session: Session,
    storage: Storage,
    content: bytes,
    dataset_payload: DatasetCreate,
    *,
    limits: YoloArchiveLimits | None = None,
) -> tuple[Dataset, int, int]:
    if dataset_payload.task_type.value not in {"detect", "segment"}:
        raise ValidationError("coco_task_unsupported", "COCO import currently supports detect and segment datasets.")
    try:
        archive, entries = _archive_entries(content, limits or YoloArchiveLimits())
        manifest_name = next(name for name in entries if PurePosixPath(name).name in {"annotations.json", "instances.json"})
        manifest = json.loads(_read_archive_member(archive, entries[manifest_name], 10 * 1024 * 1024))
    except (zipfile.BadZipFile, StopIteration, json.JSONDecodeError) as exc:
        raise ValidationError("invalid_coco_archive", "COCO ZIP must contain a valid annotations.json manifest.") from exc
    dataset: Dataset | None = None
    try:
        dataset = Dataset(
            id=new_id("ds"),
            name=dataset_payload.name,
            description=dataset_payload.description,
            task_type=dataset_payload.task_type.value,
        )
        session.add(dataset)
        category_map: dict[int, str] = {}
        for class_index, category in enumerate(sorted(manifest.get("categories", []), key=lambda item: item.get("id", 0))):
            class_label = ClassLabel(
                id=new_id("cls"),
                dataset_id=dataset.id,
                class_index=class_index,
                name=str(category["name"]),
                color="#22c55e",
            )
            session.add(class_label)
            category_map[int(category["id"])] = class_label.id
        session.flush()
        image_members = [name for name in entries if PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        image_map: dict[int, ImageItem] = {}
        for item in manifest.get("images", []):
            manifest_file_name = str(item["file_name"])
            file_name = PurePosixPath(manifest_file_name).name
            exact = manifest_file_name.lstrip("./")
            source = exact if exact in entries else None
            if source is None:
                basename_matches = [member for member in image_members if PurePosixPath(member).name == file_name]
                source = basename_matches[0] if len(basename_matches) == 1 else None
            if source is None:
                raise ValidationError("coco_image_missing", f"COCO image file is missing: {item['file_name']}")
            image_id = new_id("img")
            name = PurePosixPath(str(item.get("starter_original_file_name") or file_name)).name
            storage_name = storage.safe_storage_name(name, image_id)
            width, height = storage.write_image(dataset.id, storage_name, _read_archive_member(archive, entries[source], (limits or YoloArchiveLimits()).max_member_bytes))
            image = ImageItem(id=image_id, dataset_id=dataset.id, file_name=name, storage_name=storage_name, width=width, height=height, split=str(item.get("split", "train")) if item.get("split", "train") in {"train", "val", "test"} else "train", status="unannotated")
            session.add(image)
            image_map[int(item["id"])] = image
        annotation_count = 0
        annotated_ids: set[str] = set()
        for item in manifest.get("annotations", []):
            image = image_map.get(int(item["image_id"]))
            class_id = category_map.get(int(item["category_id"]))
            if image is None or class_id is None:
                continue
            segmentation = item.get("segmentation") or []
            if dataset.task_type == "segment":
                if not segmentation or not isinstance(segmentation, list) or not segmentation or not isinstance(segmentation[0], list):
                    raise ValidationError("coco_segment_missing_polygon", "Segment COCO annotations must contain a polygon segmentation.")
                raw = segmentation[0] if isinstance(segmentation[0], list) else segmentation
                if len(raw) < 6 or len(raw) % 2:
                    raise ValidationError("coco_segment_invalid_polygon", "Segment COCO polygons require at least three point pairs.")
                polygon = list(zip(raw[::2], raw[1::2], strict=True))
                validate_polygon(polygon, image.width, image.height)
                annotation = Annotation(id=new_id("ann"), image_id=image.id, dataset_id=dataset.id, class_id=class_id, type="polygon", polygon=[list(point) for point in polygon], source="imported")
            else:
                x, y, width, height = item["bbox"]
                validate_bbox(BBox(x=x, y=y, width=width, height=height), image.width, image.height)
                annotation = Annotation(id=new_id("ann"), image_id=image.id, dataset_id=dataset.id, class_id=class_id, type="bbox", x=x, y=y, width=width, height=height, source="imported")
            session.add(annotation); annotation_count += 1; annotated_ids.add(image.id)
        session.flush()
        for image in image_map.values():
            image.status = "annotated" if image.id in annotated_ids else "unannotated"
        refresh_dataset_counts(session, dataset.id)
        session.commit(); session.refresh(dataset)
        return dataset, len(image_map), annotation_count
    except Exception:
        session.rollback()
        if dataset is not None:
            storage.remove_dataset(dataset.id)
        raise
    finally:
        archive.close()
