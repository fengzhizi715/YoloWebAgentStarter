from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as mask_utils


@dataclass(frozen=True)
class EvalInstance:
    image_file: str
    class_index: int
    bbox: tuple[float, float, float, float]
    confidence: float | None = None
    polygon: tuple[tuple[float, float], ...] | None = None


class ErrorSampleAnalyzer:
    """Community subset of upstream error analysis for detect, segment and OBB."""

    def collect(
        self,
        run_dir: str | Path,
        confidence_threshold: float,
        export_path: str | Path | None = None,
        split: str = "val",
    ) -> list[dict]:
        self._split = split if split in {"train", "val", "test"} else "val"
        run_path = Path(run_dir)
        export_root = Path(export_path) if export_path else run_path.parent / "yolo"
        task_type = self._task_type(export_root)
        image_sizes = self._image_sizes(export_root)
        ground_truth = self._read_ground_truth(export_root, task_type)
        predictions = self._read_predictions(run_path, image_sizes)
        samples: list[dict] = []
        samples.extend(self._low_confidence_samples(predictions, confidence_threshold))
        if ground_truth:
            samples.extend(self._missed_detection_samples(ground_truth, predictions, confidence_threshold))
            samples.extend(self._false_positive_samples(ground_truth, predictions, confidence_threshold))
        return samples[:200]

    @staticmethod
    def _task_type(export_root: Path) -> str:
        try:
            payload = yaml.safe_load((export_root / "data.yaml").read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            return "detect"
        return str(payload.get("task") or "detect")

    def _read_ground_truth(self, export_root: Path, task_type: str) -> list[EvalInstance]:
        labels_dir = export_root / "labels" / self._split
        if not labels_dir.exists():
            return []
        image_files_by_stem = self._image_files_by_stem(export_root)
        instances: list[EvalInstance] = []
        for label_file in labels_dir.glob("*.txt"):
            image_file = image_files_by_stem.get(label_file.stem, f"{label_file.stem}.jpg")
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    class_index = int(float(parts[0]))
                    values = [float(value) for value in parts[1:]]
                except ValueError:
                    continue
                polygon = self._ground_truth_polygon(values, task_type)
                bbox = self._polygon_to_xyxy(polygon) if polygon else self._center_xywh_to_xyxy(values[:4])
                instances.append(EvalInstance(image_file=image_file, class_index=class_index, bbox=bbox, polygon=polygon))
        return instances

    def _ground_truth_polygon(self, values: list[float], task_type: str) -> tuple[tuple[float, float], ...] | None:
        if task_type not in {"segment", "obb"}:
            return None
        raw = values[:8] if task_type == "obb" else values
        if len(raw) < 6 or len(raw) % 2:
            return None
        return tuple((raw[index], raw[index + 1]) for index in range(0, len(raw), 2))

    def _read_predictions(self, run_path: Path, image_sizes: dict[str, tuple[int, int]]) -> list[EvalInstance]:
        path = run_path / "predictions.json"
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(data, list):
            return []
        return self._nested_predictions(data, image_sizes) or self._flat_predictions(data, image_sizes)

    def _nested_predictions(self, data: list, image_sizes: dict[str, tuple[int, int]]) -> list[EvalInstance]:
        instances: list[EvalInstance] = []
        for item in data:
            if not isinstance(item, dict) or "predictions" not in item:
                continue
            image_file = self._canonical_image_file(item.get("image_id") or item.get("image") or item.get("path"), image_sizes)
            for prediction in item.get("predictions", []):
                instance = self._prediction_instance(image_file, prediction, image_sizes, flat_ultralytics=False)
                if instance:
                    instances.append(instance)
        return instances

    def _flat_predictions(self, data: list, image_sizes: dict[str, tuple[int, int]]) -> list[EvalInstance]:
        instances: list[EvalInstance] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            image_file = self._canonical_image_file(item.get("image_id") or item.get("image") or item.get("path"), image_sizes)
            instance = self._prediction_instance(image_file, item, image_sizes, flat_ultralytics=True)
            if instance:
                instances.append(instance)
        return instances

    @staticmethod
    def _canonical_image_file(raw_name, image_sizes: dict[str, tuple[int, int]]) -> str:
        image_file = Path(str(raw_name or "")).name
        if image_file in image_sizes:
            return image_file
        matches = [name for name in image_sizes if Path(name).stem == Path(image_file).stem]
        return matches[0] if len(matches) == 1 else image_file

    def _prediction_instance(
        self,
        image_file: str,
        prediction: dict,
        image_sizes: dict[str, tuple[int, int]],
        *,
        flat_ultralytics: bool,
    ) -> EvalInstance | None:
        if not image_file:
            return None
        image_size = image_sizes.get(image_file)
        try:
            class_index = int(prediction.get("class_index", prediction.get("category_id", prediction.get("class", 0))))
            confidence = float(prediction.get("confidence", prediction.get("score", 0.0)))
        except (TypeError, ValueError):
            return None
        polygon = self._prediction_polygon(prediction, image_size)
        raw_bbox = prediction.get("bbox") or prediction.get("box")
        if raw_bbox and len(raw_bbox) >= 4:
            try:
                values = [float(value) for value in raw_bbox[:4]]
            except (TypeError, ValueError):
                return None
            bbox = self._top_left_xywh_to_xyxy(values, image_size) if flat_ultralytics else self._normalized_or_center_bbox(values, image_size)
        elif polygon:
            bbox = self._polygon_to_xyxy(polygon)
        else:
            return None
        return EvalInstance(image_file=image_file, class_index=class_index, bbox=bbox, confidence=confidence, polygon=polygon)

    def _prediction_polygon(self, prediction: dict, image_size: tuple[int, int] | None) -> tuple[tuple[float, float], ...] | None:
        raw = prediction.get("polygon") or prediction.get("points") or prediction.get("obb") or prediction.get("poly")
        polygon = self._normalize_polygon(raw, image_size)
        if polygon:
            return polygon
        segmentation = prediction.get("segmentation")
        if not isinstance(segmentation, dict):
            return None
        try:
            decoded = mask_utils.decode(segmentation)
            contours, _ = cv2.findContours(np.asarray(decoded, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            return None
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        if len(contour) < 3:
            return None
        height, width = decoded.shape[:2]
        return tuple((float(x) / width, float(y) / height) for x, y in contour)

    @staticmethod
    def _normalize_polygon(raw, image_size: tuple[int, int] | None) -> tuple[tuple[float, float], ...] | None:
        if not isinstance(raw, list):
            return None
        if len(raw) >= 6 and all(isinstance(value, (int, float)) for value in raw):
            raw = [[raw[index], raw[index + 1]] for index in range(0, len(raw) - 1, 2)]
        if len(raw) < 3:
            return None
        points: list[tuple[float, float]] = []
        for point in raw:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError):
                continue
            if image_size and max(abs(x), abs(y)) > 1.0:
                x, y = x / image_size[0], y / image_size[1]
            points.append((x, y))
        return tuple(points) if len(points) >= 3 else None

    @staticmethod
    def _normalized_or_center_bbox(values: list[float], image_size: tuple[int, int] | None) -> tuple[float, float, float, float]:
        if max(abs(value) for value in values) <= 1.0:
            return ErrorSampleAnalyzer._center_xywh_to_xyxy(values)
        if not image_size:
            return ErrorSampleAnalyzer._center_xywh_to_xyxy(values)
        x, y, width, height = values
        return ErrorSampleAnalyzer._center_xywh_to_xyxy([x / image_size[0], y / image_size[1], width / image_size[0], height / image_size[1]])

    @staticmethod
    def _top_left_xywh_to_xyxy(values: list[float], image_size: tuple[int, int] | None) -> tuple[float, float, float, float]:
        x, y, width, height = values
        if image_size:
            image_width, image_height = image_size
            return x / image_width, y / image_height, (x + width) / image_width, (y + height) / image_height
        return x, y, x + width, y + height

    @staticmethod
    def _center_xywh_to_xyxy(values: list[float]) -> tuple[float, float, float, float]:
        x, y, width, height = values
        return x - width / 2, y - height / 2, x + width / 2, y + height / 2

    @staticmethod
    def _polygon_to_xyxy(polygon: tuple[tuple[float, float], ...]) -> tuple[float, float, float, float]:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _low_confidence_samples(predictions: list[EvalInstance], threshold: float) -> list[dict]:
        return [
            {"image_file": item.image_file, "type": "low_confidence", "confidence": item.confidence, "class_index": item.class_index, "message": "Prediction confidence is below the evaluation threshold."}
            for item in predictions
            if item.confidence is not None and item.confidence < threshold
        ]

    def _missed_detection_samples(self, ground_truth: list[EvalInstance], predictions: list[EvalInstance], threshold: float) -> list[dict]:
        confident = [item for item in predictions if (item.confidence or 0.0) >= threshold]
        return [
            {"image_file": item.image_file, "type": "missed_detection", "class_index": item.class_index, "message": "Ground-truth object was not matched by a confident prediction."}
            for item in ground_truth
            if not self._has_match(item, confident)
        ]

    def _false_positive_samples(self, ground_truth: list[EvalInstance], predictions: list[EvalInstance], threshold: float) -> list[dict]:
        return [
            {"image_file": item.image_file, "type": "false_positive", "confidence": item.confidence, "class_index": item.class_index, "message": "Prediction did not match any ground-truth object."}
            for item in predictions
            if (item.confidence or 0.0) >= threshold and not self._has_match(item, ground_truth)
        ]

    def _has_match(self, target: EvalInstance, candidates: list[EvalInstance], threshold: float = 0.5) -> bool:
        return any(
            target.image_file == candidate.image_file
            and target.class_index == candidate.class_index
            and self._instance_iou(target, candidate) >= threshold
            for candidate in candidates
        )

    def _instance_iou(self, left: EvalInstance, right: EvalInstance) -> float:
        if left.polygon and right.polygon:
            return self._polygon_iou(left.polygon, right.polygon)
        return self._bbox_iou(left.bbox, right.bbox)

    @staticmethod
    def _polygon_iou(left: tuple[tuple[float, float], ...], right: tuple[tuple[float, float], ...]) -> float:
        size = 512
        masks = []
        for polygon in (left, right):
            mask = np.zeros((size, size), dtype=np.uint8)
            points = np.array([[min(max(round(x * (size - 1)), 0), size - 1), min(max(round(y * (size - 1)), 0), size - 1)] for x, y in polygon], dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
            masks.append(mask)
        intersection = int(np.logical_and(masks[0], masks[1]).sum())
        union = int(np.logical_or(masks[0], masks[1]).sum())
        return intersection / union if union else 0.0

    @staticmethod
    def _bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
        intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        union = left_area + right_area - intersection
        return intersection / union if union else 0.0

    def _image_sizes(self, export_root: Path) -> dict[str, tuple[int, int]]:
        images_dir = export_root / "images" / self._split
        if not images_dir.is_dir():
            return {}
        sizes: dict[str, tuple[int, int]] = {}
        for image_path in images_dir.iterdir():
            try:
                with Image.open(image_path) as image:
                    sizes[image_path.name] = image.size
            except OSError:
                continue
        return sizes

    def _image_files_by_stem(self, export_root: Path) -> dict[str, str]:
        images_dir = export_root / "images" / self._split
        if not images_dir.is_dir():
            return {}
        return {path.stem: path.name for path in images_dir.iterdir() if path.is_file()}
