from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from app.core.errors import ValidationError


class ModelCache:
    """Adapted from upstream ModelCache, limited to managed Ultralytics PT models."""

    _cache: dict[str, Any] = {}
    _last_access: dict[str, float] = {}
    _max_size = 3

    @classmethod
    def get_model(cls, model_id: str, model_path: Path) -> Any:
        now = time.time()
        if model_id in cls._cache:
            cls._last_access[model_id] = now
            return cls._cache[model_id]
        if len(cls._cache) >= cls._max_size:
            oldest = min(cls._last_access, key=cls._last_access.get)
            cls._cache.pop(oldest, None)
            cls._last_access.pop(oldest, None)
        model = YOLO(str(model_path))
        cls._cache[model_id] = model
        cls._last_access[model_id] = now
        return model


def run_test_inference(*, model_id: str, model_path: Path, task_type: str, image_bytes: bytes, confidence: float, iou: float, class_names: dict[int, str]) -> dict:
    """Adapted from upstream ModelVersionService.run_test_inference without external model paths or save-test."""
    suffix = ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(image_bytes)
        temporary_path = Path(temporary.name)
    try:
        started = time.perf_counter()
        results = ModelCache.get_model(model_id, model_path).predict(source=str(temporary_path), conf=min(confidence, 0.01) if confidence > 0 else 0.01, iou=iou, verbose=False)
        detections = _parse_results(results, task_type, class_names)
        return {"model_id": model_id, "task_type": task_type, "detections": detections, "inference_time_ms": (time.perf_counter() - started) * 1000}
    except Exception as exc:
        raise ValidationError("inference_failed", f"Local model inference failed: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _parse_results(results: list[Any], task_type: str, class_names: dict[int, str]) -> list[dict]:
    output: list[dict] = []
    for result in results:
        if task_type == "classify" and getattr(result, "probs", None) is not None:
            for index in result.probs.top5:
                output.append({"class_index": int(index), "class_name": class_names.get(int(index), f"class_{index}"), "confidence": float(result.probs.data[index]), "x": 0, "y": 0, "width": 0, "height": 0, "polygon": None, "obb_points": None})
            continue
        if task_type == "obb" and getattr(result, "obb", None) is not None:
            for item in result.obb:
                points = item.xyxyxyxy[0].tolist()
                xs, ys = [point[0] for point in points], [point[1] for point in points]
                index = int(item.cls.item())
                output.append({"class_index": index, "class_name": class_names.get(index, f"class_{index}"), "confidence": float(item.conf.item()), "x": min(xs), "y": min(ys), "width": max(xs) - min(xs), "height": max(ys) - min(ys), "polygon": None, "obb_points": points})
            continue
        boxes = getattr(result, "boxes", None)
        mask_points = getattr(getattr(result, "masks", None), "xy", None)
        if boxes is None:
            continue
        for position, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            index = int(box.cls.item())
            polygon = mask_points[position].tolist() if task_type == "segment" and mask_points is not None and position < len(mask_points) else None
            output.append({"class_index": index, "class_name": class_names.get(index, f"class_{index}"), "confidence": float(box.conf.item()), "x": float(x1), "y": float(y1), "width": float(x2 - x1), "height": float(y2 - y1), "polygon": polygon, "obb_points": None})
    return output
