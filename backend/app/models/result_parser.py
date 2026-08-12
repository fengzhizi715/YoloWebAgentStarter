from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    class_index: int
    confidence: float
    x: float
    y: float
    width: float
    height: float
    polygon: tuple[tuple[float, float], ...] | None = None
    obb_points: tuple[tuple[float, float], ...] | None = None


def yolo_xyxy_to_bbox(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    """Adapted from upstream result_parser.py; coordinates remain absolute pixels."""

    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("invalid xyxy bbox")
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    width, height = x2 - x1, y2 - y1
    if width <= 1e-4 or height <= 1e-4:
        raise ValueError("invalid xyxy bbox")
    return x1, y1, width, height


class YoloResultParser:
    """Community port of the upstream parser, excluding pose-only output."""

    def parse(self, results, task_type: str) -> list[Detection]:
        detections: list[Detection] = []
        for result in results:
            if task_type == "classify":
                detections.extend(self._parse_classification(getattr(result, "probs", None)))
                continue
            obb = getattr(result, "obb", None)
            if obb is not None:
                detections.extend(self._parse_obb(obb))
                continue
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            masks = getattr(result, "masks", None)
            mask_xy = getattr(masks, "xy", None) if masks is not None else None
            for index, box in enumerate(boxes):
                try:
                    x, y, width, height = yolo_xyxy_to_bbox(*self._flatten_xyxy_four(box))
                    class_index = int(self._scalar(box.cls))
                    confidence = float(self._scalar(box.conf))
                except (ValueError, TypeError, IndexError):
                    continue
                detections.append(
                    Detection(
                        class_index=class_index,
                        confidence=confidence,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        polygon=self._polygon(mask_xy, index),
                    )
                )
        return detections

    def _parse_classification(self, probs) -> list[Detection]:
        if probs is None:
            return []
        data = getattr(probs, "data", None)
        output: list[Detection] = []
        for index in getattr(probs, "top5", []):
            try:
                confidence = self._scalar(data[index])
            except (TypeError, IndexError):
                continue
            output.append(Detection(int(index), confidence, 0.0, 0.0, 0.0, 0.0))
        return output

    def _parse_obb(self, obb) -> list[Detection]:
        output: list[Detection] = []
        for item in obb:
            points = self._obb_points(item)
            if points is None:
                continue
            xs, ys = [point[0] for point in points], [point[1] for point in points]
            try:
                class_index = int(self._scalar(item.cls))
                confidence = float(self._scalar(item.conf))
            except (ValueError, TypeError, IndexError):
                continue
            output.append(
                Detection(
                    class_index,
                    confidence,
                    min(xs),
                    min(ys),
                    max(xs) - min(xs),
                    max(ys) - min(ys),
                    obb_points=points,
                )
            )
        return output

    def _obb_points(self, item) -> tuple[tuple[float, float], ...] | None:
        raw = getattr(item, "xyxyxyxy", None)
        if raw is None:
            return None
        raw = self._to_list(raw)
        while isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
            raw = raw[0]
        if not isinstance(raw, list):
            return None
        if len(raw) >= 4 and all(isinstance(point, (list, tuple)) for point in raw[:4]):
            points = tuple((float(point[0]), float(point[1])) for point in raw[:4] if len(point) >= 2)
        elif len(raw) >= 8:
            values = [float(value) for value in raw[:8]]
            points = tuple((values[index], values[index + 1]) for index in range(0, 8, 2))
        else:
            return None
        return points if len(points) == 4 else None

    def _polygon(self, mask_xy, index: int) -> tuple[tuple[float, float], ...] | None:
        if mask_xy is None:
            return None
        try:
            raw = self._to_list(mask_xy[index])
        except (IndexError, TypeError):
            return None
        if not isinstance(raw, list):
            return None
        points = tuple((float(point[0]), float(point[1])) for point in raw if isinstance(point, (list, tuple)) and len(point) >= 2)
        return points if len(points) >= 3 else None

    def _flatten_xyxy_four(self, box) -> tuple[float, float, float, float]:
        raw = getattr(box, "xyxy", None)
        if raw is None:
            raise ValueError("missing xyxy")
        values = self._to_list(raw)
        while isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
            values = values[0]
        if not isinstance(values, list) or len(values) < 4:
            raise ValueError("xyxy too short")
        return tuple(float(value) for value in values[:4])  # type: ignore[return-value]

    @staticmethod
    def _to_list(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        return value.tolist() if hasattr(value, "tolist") else value

    @staticmethod
    def _scalar(value) -> float:
        if hasattr(value, "item"):
            return float(value.item())
        if hasattr(value, "__getitem__") and not isinstance(value, (str, bytes)):
            return float(value[0])
        return float(value)
