from __future__ import annotations

import io
import math
import posixpath
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ValidationError
from app.core.ids import new_id
from app.core.models import Annotation, ClassLabel, Dataset, ImageItem
from app.core.schemas import BBox, DatasetCreate, OBB, SplitName
from app.core.storage import SUPPORTED_IMAGE_SUFFIXES, Storage
from app.dataset.annotation.geometry import obb_corners, validate_bbox, validate_obb, validate_polygon
from app.dataset.service import get_dataset, refresh_dataset_counts


_SAFE_MEMBER = re.compile(r"^[^/]+(?:/[^/]+)*$")


def _safe_member_name(name: str) -> str:
    normalized = posixpath.normpath(name.replace("\\", "/"))
    if normalized in {"", "."} or normalized.startswith("../") or normalized == ".." or normalized.startswith("/"):
        raise ValidationError("unsafe_archive_path", "YOLO archive contains an unsafe path.")
    if not _SAFE_MEMBER.fullmatch(normalized):
        raise ValidationError("unsafe_archive_path", "YOLO archive contains an unsafe path.")
    return normalized


def _archive_entries(payload: bytes) -> dict[str, bytes]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise ValidationError("invalid_yolo_archive", "The uploaded file is not a valid ZIP archive.") from exc
    entries: dict[str, bytes] = {}
    with archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = _safe_member_name(info.filename)
            if info.file_size > 100 * 1024 * 1024:
                raise ValidationError("archive_member_too_large", "A YOLO archive member is too large.")
            entries[name] = archive.read(info)
    if not entries:
        raise ValidationError("empty_yolo_archive", "The YOLO archive is empty.")
    return entries


def _names_from_yaml(entries: dict[str, bytes]) -> list[str]:
    yaml_name = next((name for name in entries if PurePosixPath(name).name.lower() in {"data.yaml", "dataset.yaml"}), None)
    if yaml_name is None:
        return []
    try:
        document = yaml.safe_load(entries[yaml_name]) or {}
    except yaml.YAMLError as exc:
        raise ValidationError("invalid_dataset_yaml", "The YOLO data.yaml file is invalid.") from exc
    names = document.get("names", []) if isinstance(document, dict) else []
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list):
        return [str(item) for item in names]
    return []


def _relative_key(name: str, kind: str) -> str:
    path = PurePosixPath(name)
    parts = list(path.parts)
    if kind in parts:
        parts = parts[parts.index(kind) + 1 :]
    if parts and parts[0] in {"train", "val", "test"}:
        return "/".join(parts)
    return "/".join(parts)


def _split_for_image(name: str, default: SplitName) -> SplitName:
    parts = PurePosixPath(name).parts
    for split in ("train", "val", "test"):
        if split in parts:
            return split  # type: ignore[return-value]
    return default


def _label_for_image(image_name: str, labels: dict[str, tuple[str, bytes]]) -> bytes | None:
    key = _relative_key(image_name, "images")
    candidate = str(PurePosixPath(key).with_suffix(".txt"))
    return labels.get(candidate, ("", b""))[1] or None


def _class_for_image(image_name: str) -> str | None:
    """Read a YOLO classification class folder from a managed archive member."""

    parts = PurePosixPath(image_name).parts
    for index, split in enumerate(parts):
        if split in {"train", "val", "test"}:
            if index + 2 != len(parts) - 1:
                return None
            return parts[index + 1]
    return None


def _obb_from_yolo(numbers: list[float], image_width: int, image_height: int) -> OBB:
    if len(numbers) != 8:
        raise ValidationError("invalid_obb_label", "OBB labels require four point pairs.")
    if any(coordinate < 0 or coordinate > 1 for coordinate in numbers):
        raise ValidationError("invalid_obb_label", "OBB point coordinates must be normalized to the [0, 1] range.")
    points = [(numbers[index] * image_width, numbers[index + 1] * image_height) for index in range(0, 8, 2)]
    center_x = sum(point[0] for point in points) / 4
    center_y = sum(point[1] for point in points) / 4
    width = math.dist(points[0], points[1])
    height = math.dist(points[1], points[2])
    angle = math.degrees(math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]))
    obb = OBB(cx=center_x, cy=center_y, width=width, height=height, angle=angle)
    validate_obb(obb, image_width, image_height)
    # Ultralytics OBB labels contain four adjacent rectangle corners.  The
    # persisted Starter contract is centre/width/height/angle, so accepting a
    # general quadrilateral would silently change its geometry on export.
    expected = obb_corners(obb)
    tolerance = max(0.01, max(image_width, image_height) * 1e-7)
    if any(math.dist(actual, rebuilt) > tolerance for actual, rebuilt in zip(points, expected, strict=True)):
        raise ValidationError("invalid_obb_label", "OBB points must describe an ordered rectangle.")
    return obb


def _obb_to_yolo(obb: dict | None, image_width: int, image_height: int) -> str:
    try:
        resolved = OBB(**(obb or {}))
    except Exception as exc:
        raise ValidationError("invalid_obb_annotation", "Oriented bounding-box annotation is invalid.") from exc
    validate_obb(resolved, image_width, image_height)
    return " ".join(
        f"{coordinate:.8f}"
        for point in obb_corners(resolved)
        for coordinate in (point[0] / image_width, point[1] / image_height)
    )


def _classification_for_image(annotations: list[Annotation]) -> Annotation | None:
    classified = [annotation for annotation in annotations if annotation.type == "classify"]
    if len(classified) > 1:
        raise ValidationError("classify_multiple", "Classification datasets allow exactly one class per image.")
    return classified[0] if classified else None


def _parse_yolo_labels(
    content: bytes | None,
    task_type: str,
    image_width: int,
    image_height: int,
    class_ids: dict[int, str],
) -> list[Annotation]:
    if content is None:
        return []
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("invalid_label_file", "A YOLO label file is not UTF-8 text.") from exc
    annotations: list[Annotation] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        values = line.split()
        try:
            class_index = int(values[0])
            numbers = [float(value) for value in values[1:]]
        except (ValueError, IndexError) as exc:
            raise ValidationError("invalid_label_line", f"Invalid YOLO label at line {line_number}.") from exc
        if class_index not in class_ids or not all(math.isfinite(value) for value in numbers):
            raise ValidationError("invalid_label_line", f"Invalid class or coordinate at line {line_number}.")
        if task_type == "detect":
            if len(numbers) != 4:
                raise ValidationError("invalid_bbox_label", f"Detect labels require four values at line {line_number}.")
            center_x, center_y, width, height = numbers
            bbox = BBox(
                x=(center_x - width / 2) * image_width,
                y=(center_y - height / 2) * image_height,
                width=width * image_width,
                height=height * image_height,
            )
            validate_bbox(bbox, image_width, image_height)
            annotations.append(
                Annotation(
                    id=new_id("ann"),
                    class_id=class_ids[class_index],
                    type="bbox",
                    x=bbox.x,
                    y=bbox.y,
                    width=bbox.width,
                    height=bbox.height,
                    source="imported",
                )
            )
        elif task_type == "segment":
            if len(numbers) < 6 or len(numbers) % 2:
                raise ValidationError("invalid_polygon_label", f"Segment labels require point pairs at line {line_number}.")
            polygon = [(numbers[index] * image_width, numbers[index + 1] * image_height) for index in range(0, len(numbers), 2)]
            validate_polygon(polygon, image_width, image_height)
            annotations.append(
                Annotation(
                    id=new_id("ann"),
                    class_id=class_ids[class_index],
                    type="polygon",
                    polygon=[list(point) for point in polygon],
                    source="imported",
                )
            )
        elif task_type == "obb":
            obb = _obb_from_yolo(numbers, image_width, image_height)
            annotations.append(
                Annotation(
                    id=new_id("ann"),
                    class_id=class_ids[class_index],
                    type="obb",
                    obb=obb.model_dump(),
                    source="imported",
                )
            )
        else:
            raise ValidationError("unsupported_yolo_task", f"YOLO label files are not supported for {task_type} datasets.")
    return annotations


def export_dataset(session: Session, storage: Storage, dataset_id: str) -> tuple[bytes, str]:
    dataset = get_dataset(session, dataset_id)
    classes = list(session.scalars(select(ClassLabel).where(ClassLabel.dataset_id == dataset_id).order_by(ClassLabel.class_index)))
    images = list(session.scalars(select(ImageItem).where(ImageItem.dataset_id == dataset_id).order_by(ImageItem.split, ImageItem.id)))
    annotations = list(session.scalars(select(Annotation).where(Annotation.dataset_id == dataset_id).order_by(Annotation.image_id, Annotation.id)))
    annotations_by_image: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        annotations_by_image.setdefault(annotation.image_id, []).append(annotation)
    class_indexes = {item.id: item.class_index for item in classes}
    class_names = {item.id: item.name for item in classes}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if dataset.task_type != "classify":
            archive.writestr(
                "data.yaml",
                yaml.safe_dump(
                    {"path": ".", "train": "images/train", "val": "images/val", "test": "images/test", "names": [item.name for item in classes]},
                    sort_keys=False,
                ),
            )
        for image in images:
            path = storage.image_path(dataset_id, image.storage_name)
            if not path.is_file():
                raise ValidationError("image_file_missing", f"Managed image file is missing: {image.file_name}.")
            suffix = path.suffix.lower()
            export_stem = f"{image.id}_{PurePosixPath(image.file_name).stem}"
            if dataset.task_type == "classify":
                annotation = _classification_for_image(annotations_by_image.get(image.id, []))
                if annotation is None:
                    continue
                class_name = class_names.get(annotation.class_id)
                if class_name is None:
                    raise ValidationError("annotation_class_mismatch", "Annotation class does not belong to this dataset.")
                archive.writestr(f"{image.split}/{class_name}/{export_stem}{suffix}", path.read_bytes())
                continue
            archive.writestr(f"images/{image.split}/{export_stem}{suffix}", path.read_bytes())
            label_lines: list[str] = []
            for annotation in annotations_by_image.get(image.id, []):
                class_index = class_indexes.get(annotation.class_id)
                if class_index is None:
                    raise ValidationError("annotation_class_mismatch", "Annotation class does not belong to this dataset.")
                if annotation.type == "bbox":
                    center_x = ((annotation.x or 0) + (annotation.width or 0) / 2) / image.width
                    center_y = ((annotation.y or 0) + (annotation.height or 0) / 2) / image.height
                    label_lines.append(
                        f"{class_index} {center_x:.8f} {center_y:.8f} {(annotation.width or 0) / image.width:.8f} {(annotation.height or 0) / image.height:.8f}"
                    )
                elif annotation.type == "polygon":
                    points = annotation.polygon or []
                    normalized = " ".join(f"{point[0] / image.width:.8f} {point[1] / image.height:.8f}" for point in points)
                    label_lines.append(f"{class_index} {normalized}")
                elif annotation.type == "obb":
                    label_lines.append(f"{class_index} {_obb_to_yolo(annotation.obb, image.width, image.height)}")
                else:
                    raise ValidationError("unsupported_annotation_type", "Annotation type cannot be exported to YOLO labels.")
            archive.writestr(f"labels/{image.split}/{export_stem}.txt", "\n".join(label_lines) + ("\n" if label_lines else ""))
    return output.getvalue(), f"{dataset.name.replace(' ', '_') or 'dataset'}.zip"


def export_dataset_directory(session: Session, storage: Storage, dataset_id: str, target_dir: Path) -> dict[str, object]:
    """Materialize a training-ready YOLO directory using persisted image splits."""

    dataset = get_dataset(session, dataset_id)
    classes = list(
        session.scalars(
            select(ClassLabel).where(ClassLabel.dataset_id == dataset_id).order_by(ClassLabel.class_index)
        )
    )
    images = list(
        session.scalars(
            select(ImageItem).where(ImageItem.dataset_id == dataset_id).order_by(ImageItem.split, ImageItem.id)
        )
    )
    annotations = list(
        session.scalars(
            select(Annotation).where(Annotation.dataset_id == dataset_id).order_by(Annotation.image_id, Annotation.id)
        )
    )
    target = Path(target_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    class_indexes = {item.id: item.class_index for item in classes}
    class_names = {item.id: item.name for item in classes}
    annotations_by_image: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        annotations_by_image.setdefault(annotation.image_id, []).append(annotation)
    counts = {"train": 0, "val": 0, "test": 0}
    label_count = 0

    for image in images:
        if image.split not in counts:
            raise ValidationError("invalid_image_split", f"Unsupported image split: {image.split}.")
        source = storage.image_path(dataset_id, image.storage_name)
        if not source.is_file():
            raise ValidationError("image_file_missing", f"Managed image file is missing: {image.file_name}.")
        stem = f"{image.id}_{PurePosixPath(image.file_name).stem}"
        if dataset.task_type == "classify":
            annotation = _classification_for_image(annotations_by_image.get(image.id, []))
            if annotation is None:
                continue
            class_name = class_names.get(annotation.class_id)
            if class_name is None:
                raise ValidationError("annotation_class_mismatch", "Annotation class does not belong to this dataset.")
            image_destination = target / image.split / class_name / f"{stem}{source.suffix.lower()}"
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, image_destination)
            counts[image.split] += 1
            label_count += 1
            continue
        image_destination = target / "images" / image.split / f"{stem}{source.suffix.lower()}"
        label_destination = target / "labels" / image.split / f"{stem}.txt"
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_destination)
        lines: list[str] = []
        for annotation in annotations_by_image.get(image.id, []):
            class_index = class_indexes.get(annotation.class_id)
            if class_index is None:
                raise ValidationError("annotation_class_mismatch", "Annotation class does not belong to this dataset.")
            if annotation.type == "bbox":
                center_x = ((annotation.x or 0) + (annotation.width or 0) / 2) / image.width
                center_y = ((annotation.y or 0) + (annotation.height or 0) / 2) / image.height
                lines.append(
                    f"{class_index} {center_x:.8f} {center_y:.8f} {(annotation.width or 0) / image.width:.8f} {(annotation.height or 0) / image.height:.8f}"
                )
            elif annotation.type == "polygon":
                normalized = " ".join(
                    f"{point[0] / image.width:.8f} {point[1] / image.height:.8f}"
                    for point in (annotation.polygon or [])
                )
                lines.append(f"{class_index} {normalized}")
            elif annotation.type == "obb":
                lines.append(f"{class_index} {_obb_to_yolo(annotation.obb, image.width, image.height)}")
            else:
                raise ValidationError("unsupported_annotation_type", "Annotation type cannot be exported to YOLO labels.")
        if lines:
            label_count += len(lines)
        label_destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[image.split] += 1

    if dataset.task_type == "classify":
        return {
            "root": str(target),
            "data_yaml": str(target),
            "counts": counts,
            "label_count": label_count,
        }

    data_yaml = target / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(target),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": [item.name for item in classes],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "root": str(target),
        "data_yaml": str(data_yaml),
        "counts": counts,
        "label_count": label_count,
    }


def import_dataset(
    session: Session,
    storage: Storage,
    payload: bytes,
    dataset_payload: DatasetCreate,
    default_split: SplitName = "train",
) -> tuple[Dataset, int, int]:
    entries = _archive_entries(payload)
    image_entries = {name: data for name, data in entries.items() if PurePosixPath(name).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES}
    if not image_entries:
        raise ValidationError("no_images_in_archive", "The YOLO archive does not contain supported images.")
    labels = {
        _relative_key(name, "labels"): (name, data)
        for name, data in entries.items()
        if PurePosixPath(name).suffix.lower() == ".txt" and ("labels" in PurePosixPath(name).parts or PurePosixPath(name).name != "classes.txt")
    }
    if dataset_payload.task_type.value == "classify":
        classification_classes = {_class_for_image(member_name) for member_name in image_entries}
        if None in classification_classes:
            raise ValidationError(
                "invalid_classification_layout",
                "Classification archives must use train|val|test/<class-name>/<image> paths.",
            )
        names = sorted(str(name) for name in classification_classes)
    else:
        names = _names_from_yaml(entries)
        max_class_index = -1
        if not names:
            for content in labels.values():
                for line in content[1].decode("utf-8", errors="ignore").splitlines():
                    if line.strip():
                        try:
                            max_class_index = max(max_class_index, int(line.split()[0]))
                        except (ValueError, IndexError):
                            pass
            names = [f"class_{index}" for index in range(max_class_index + 1)]
    if not names:
        raise ValidationError("no_classes_in_archive", "The YOLO archive does not define any classes.")

    dataset = Dataset(
        id=new_id("ds"),
        name=dataset_payload.name,
        description=dataset_payload.description,
        task_type=dataset_payload.task_type.value,
    )
    session.add(dataset)
    class_records = [ClassLabel(id=new_id("cls"), dataset_id=dataset.id, class_index=index, name=name, color="#22c55e") for index, name in enumerate(names)]
    session.add_all(class_records)
    class_ids = {item.class_index: item.id for item in class_records}
    class_ids_by_name = {item.name: item.id for item in class_records}
    staged_paths: list[tuple[str, str]] = []
    imported_annotations = 0
    try:
        session.flush()
        for member_name, content in sorted(image_entries.items()):
            image_id = new_id("img")
            storage_name = storage.safe_storage_name(PurePosixPath(member_name).name, image_id)
            width, height = storage.write_image(dataset.id, storage_name, content)
            staged_paths.append((dataset.id, storage_name))
            split = _split_for_image(member_name, default_split)
            image = ImageItem(
                id=image_id,
                dataset_id=dataset.id,
                file_name=PurePosixPath(member_name).name,
                storage_name=storage_name,
                width=width,
                height=height,
                split=split,
                status="unannotated",
            )
            session.add(image)
            session.flush()
            if dataset.task_type == "classify":
                class_name = _class_for_image(member_name)
                class_id = class_ids_by_name.get(class_name or "")
                if class_id is None:
                    raise ValidationError("invalid_classification_layout", "Image class folder is not defined by the archive.")
                parsed = [
                    Annotation(
                        id=new_id("ann"),
                        class_id=class_id,
                        type="classify",
                        source="imported",
                    )
                ]
            else:
                parsed = _parse_yolo_labels(_label_for_image(member_name, labels), dataset.task_type, width, height, class_ids)
            for annotation in parsed:
                annotation.image_id = image.id
                annotation.dataset_id = dataset.id
            session.add_all(parsed)
            image.status = "annotated" if parsed else "unannotated"
            imported_annotations += len(parsed)
        refresh_dataset_counts(session, dataset.id)
        session.commit()
    except Exception:
        session.rollback()
        for stored_dataset_id, storage_name in staged_paths:
            storage.image_path(stored_dataset_id, storage_name).unlink(missing_ok=True)
        raise
    session.refresh(dataset)
    return dataset, len(image_entries), imported_annotations
