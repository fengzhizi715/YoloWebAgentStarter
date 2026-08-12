from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from app.core.errors import ValidationError
from app.models.result_parser import YoloResultParser


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
        actual_confidence = min(confidence, 0.01) if confidence > 0 else 0.01
        results = ModelCache.get_model(model_id, model_path).predict(source=str(temporary_path), conf=actual_confidence, iou=iou, verbose=False)
        detections = [
            {
                "class_index": item.class_index,
                "class_name": class_names.get(item.class_index, f"class_{item.class_index}"),
                "confidence": item.confidence,
                "x": item.x,
                "y": item.y,
                "width": item.width,
                "height": item.height,
                "polygon": [list(point) for point in item.polygon] if item.polygon else None,
                "obb_points": [list(point) for point in item.obb_points] if item.obb_points else None,
            }
            for item in YoloResultParser().parse(results, task_type)
        ]
        return {"model_id": model_id, "task_type": task_type, "detections": detections, "inference_time_ms": (time.perf_counter() - started) * 1000}
    except Exception as exc:
        raise ValidationError("inference_failed", f"Local model inference failed: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
