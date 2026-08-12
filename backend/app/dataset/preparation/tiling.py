from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ValidationError
from app.core.ids import new_id
from app.core.models import Annotation, ClassLabel, Dataset, ImageItem
from app.core.schemas import TileDatasetRequest, TileDatasetResponse
from app.core.storage import Storage
from app.dataset.service import get_dataset, refresh_dataset_counts


@dataclass(frozen=True)
class TileWindow:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def tile_windows(width: int, height: int, tile_size: int, overlap: float) -> list[TileWindow]:
    stride = max(1, round(tile_size * (1 - overlap)))
    return [
        TileWindow(left=x, top=y, right=min(x + tile_size, width), bottom=min(y + tile_size, height))
        for y in _starts(height, tile_size, stride)
        for x in _starts(width, tile_size, stride)
    ]


def _starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    values = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if values[-1] != last:
        values.append(last)
    return values


def clip_bbox(annotation: Annotation, tile: TileWindow) -> dict | None:
    if None in (annotation.x, annotation.y, annotation.width, annotation.height):
        return None
    left = max(annotation.x, tile.left)
    top = max(annotation.y, tile.top)
    right = min(annotation.x + annotation.width, tile.right)
    bottom = min(annotation.y + annotation.height, tile.bottom)
    if right <= left or bottom <= top:
        return None
    return {"type": "bbox", "x": left - tile.left, "y": top - tile.top, "width": right - left, "height": bottom - top}


def clip_polygon(points: list[list[float]], tile: TileWindow) -> list[list[float]]:
    polygon = [[float(point[0]), float(point[1])] for point in points]
    for boundary, value in (("left", tile.left), ("right", tile.right), ("top", tile.top), ("bottom", tile.bottom)):
        polygon = _clip_boundary(polygon, boundary, value)
        if not polygon:
            return []
    translated = [[round(point[0] - tile.left, 6), round(point[1] - tile.top, 6)] for point in polygon]
    return translated if len(translated) >= 3 and _area(translated) > 1e-6 else []


def _clip_boundary(points: list[list[float]], boundary: str, value: float) -> list[list[float]]:
    result: list[list[float]] = []
    if not points:
        return result
    previous = points[-1]
    for current in points:
        previous_inside = _inside(previous, boundary, value)
        current_inside = _inside(current, boundary, value)
        if previous_inside != current_inside:
            result.append(_intersection(previous, current, boundary, value))
        if current_inside:
            result.append(current)
        previous = current
    return result


def _inside(point: list[float], boundary: str, value: float) -> bool:
    if boundary == "left":
        return point[0] >= value
    if boundary == "right":
        return point[0] <= value
    if boundary == "top":
        return point[1] >= value
    return point[1] <= value


def _intersection(start: list[float], end: list[float], boundary: str, value: float) -> list[float]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    ratio = (value - start[0]) / dx if boundary in {"left", "right"} and dx else (value - start[1]) / dy if dy else 0
    return [start[0] + ratio * dx, start[1] + ratio * dy]


def _area(points: list[list[float]]) -> float:
    return abs(sum(points[index - 1][0] * point[1] - point[0] * points[index - 1][1] for index, point in enumerate(points))) / 2


class DatasetTiler:
    """Starter adaptation of upstream tiling geometry; creates a separate managed dataset."""

    max_tiles = 5000

    def create(self, session: Session, storage: Storage, dataset_id: str, payload: TileDatasetRequest) -> TileDatasetResponse:
        source = get_dataset(session, dataset_id)
        if source.task_type not in {"detect", "segment"}:
            raise ValidationError("tiling_task_unsupported", "Image tiling currently supports detect and segment datasets.")
        images = list(session.scalars(select(ImageItem).where(ImageItem.dataset_id == source.id).order_by(ImageItem.created_at)))
        requested_tiles = sum(len(tile_windows(image.width, image.height, payload.tile_size, payload.overlap)) for image in images)
        if requested_tiles > self.max_tiles:
            raise ValidationError("tiling_output_too_large", f"Tiling would create {requested_tiles} images; the Starter limit is {self.max_tiles}.")
        annotations = list(session.scalars(select(Annotation).where(Annotation.dataset_id == source.id)))
        if any(item.type not in ({"bbox"} if source.task_type == "detect" else {"polygon"}) for item in annotations):
            raise ValidationError("tiling_annotation_type_unsupported", "Tiling requires the dataset's native bbox or polygon annotation type.")

        derived = Dataset(id=new_id("ds"), name=payload.name.strip(), description=payload.description or f"Tiled from {source.name} ({payload.tile_size}px, {payload.overlap:.0%} overlap).", task_type=source.task_type)
        session.add(derived)
        session.flush()
        class_map = self._copy_classes(session, source, derived)
        annotations_by_image: dict[str, list[Annotation]] = {}
        for annotation in annotations:
            annotations_by_image.setdefault(annotation.image_id, []).append(annotation)
        image_count = annotation_count = skipped_empty = 0
        try:
            for image in images:
                with Image.open(storage.image_path(source.id, image.storage_name)) as opened:
                    for index, tile in enumerate(tile_windows(image.width, image.height, payload.tile_size, payload.overlap), start=1):
                        transformed = self._transform(annotations_by_image.get(image.id, []), tile, class_map)
                        if not transformed and not payload.keep_empty_tiles:
                            skipped_empty += 1
                            continue
                        tile_name = f"{image.file_name.rsplit('.', 1)[0]}_tile_{index:04d}.png"
                        image_id = new_id("img")
                        storage_name = storage.safe_storage_name(tile_name, image_id)
                        output = io.BytesIO()
                        crop = opened.crop((tile.left, tile.top, tile.right, tile.bottom))
                        try:
                            if crop.mode not in {"RGB", "RGBA", "L"}:
                                crop = crop.convert("RGB")
                            crop.save(output, format="PNG")
                        finally:
                            crop.close()
                        storage.write_image(derived.id, storage_name, output.getvalue())
                        target = ImageItem(id=image_id, dataset_id=derived.id, file_name=tile_name, storage_name=storage_name, width=tile.width, height=tile.height, split=image.split, status="annotated" if transformed else "unannotated")
                        session.add(target)
                        for item in transformed:
                            session.add(Annotation(id=new_id("ann"), image_id=image_id, dataset_id=derived.id, class_id=item["class_id"], type=item["type"], x=item.get("x"), y=item.get("y"), width=item.get("width"), height=item.get("height"), polygon=item.get("polygon"), source="imported"))
                        image_count += 1
                        annotation_count += len(transformed)
            refresh_dataset_counts(session, derived.id)
            session.commit()
            session.refresh(derived)
        except Exception:
            session.rollback()
            storage.remove_dataset(derived.id)
            raise
        return TileDatasetResponse(dataset_id=derived.id, source_dataset_id=source.id, generated_images=image_count, generated_annotations=annotation_count, skipped_empty_tiles=skipped_empty)

    @staticmethod
    def _copy_classes(session: Session, source: Dataset, derived: Dataset) -> dict[str, str]:
        result: dict[str, str] = {}
        for label in source.classes:
            copied_id = new_id("cls")
            session.add(ClassLabel(id=copied_id, dataset_id=derived.id, class_index=label.class_index, name=label.name, color=label.color))
            result[label.id] = copied_id
        return result

    @staticmethod
    def _transform(annotations: list[Annotation], tile: TileWindow, class_map: dict[str, str]) -> list[dict]:
        results: list[dict] = []
        for annotation in annotations:
            if annotation.type == "bbox":
                clipped = clip_bbox(annotation, tile)
                if clipped:
                    results.append({**clipped, "class_id": class_map[annotation.class_id]})
            elif annotation.type == "polygon" and annotation.polygon:
                polygon = clip_polygon(annotation.polygon, tile)
                if polygon:
                    results.append({"type": "polygon", "class_id": class_map[annotation.class_id], "polygon": polygon})
        return results
