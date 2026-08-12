from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.ids import new_id
from app.core.models import Annotation, ClassLabel, ImageItem, ModelEvaluationRecord, ModelTestRecord, ModelVersion, TrainingTask
from app.core.storage import Storage
from app.core.time import utc_now
from app.models import onnx
from app.models.inference import run_test_inference
from app.models.schemas import ModelVersionList, ModelVersionResponse, ModelVersionUpdate


class ModelService:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def register_training_artifacts(self, session: Session, task: TrainingTask) -> list[ModelVersion]:
        if task.status != "completed":
            return []
        registered: list[ModelVersion] = []
        created_ids: list[str] = []
        try:
            for artifact_type, source in (("best", task.best_model_path), ("last", task.last_model_path)):
                if not source:
                    continue
                source_path = Path(source).expanduser().resolve()
                if not source_path.is_file():
                    continue
                model = session.scalar(
                    select(ModelVersion).where(
                        ModelVersion.training_task_id == task.id,
                        ModelVersion.artifact_type == artifact_type,
                    )
                )
                if model is None:
                    model_id = f"model_{task.id}_{artifact_type}"
                    model = ModelVersion(
                        id=model_id,
                        name=f"{task.name} ({artifact_type})",
                        version=f"{task.id}-{artifact_type}",
                        dataset_id=task.dataset_id,
                        training_task_id=task.id,
                        source="training_task",
                        artifact_type=artifact_type,
                        format="pt",
                        task_type=task.task_type,
                        engine_type="ultralytics",
                        model_path="",
                        base_model=task.model_name,
                        status="active",
                        metrics_json={},
                        notes="",
                    )
                    session.add(model)
                    session.flush()
                    created_ids.append(model_id)
                destination = self.storage.copy_model_artifact(source_path, model.id, f"{artifact_type}.pt")
                model.name = model.name or f"{task.name} ({artifact_type})"
                model.dataset_id = task.dataset_id
                model.task_type = task.task_type
                model.format = "pt"
                model.artifact_type = artifact_type
                model.model_path = str(destination)
                model.base_model = task.model_name
                self._apply_metrics(model, task.metrics_json or {})
                registered.append(model)
            session.commit()
        except Exception:
            session.rollback()
            for model_id in created_ids:
                self.storage.remove_model_version(model_id)
            raise
        for model in registered:
            session.refresh(model)
        return registered

    def list_models(self, session: Session, dataset_id: str | None = None, include_archived: bool = False) -> ModelVersionList:
        query = select(ModelVersion)
        if dataset_id:
            query = query.where(ModelVersion.dataset_id == dataset_id)
        if not include_archived:
            query = query.where(ModelVersion.status == "active")
        query = query.order_by(ModelVersion.created_at.desc(), ModelVersion.id.desc())
        items = list(session.scalars(query))
        total_query = select(func.count()).select_from(ModelVersion)
        if dataset_id:
            total_query = total_query.where(ModelVersion.dataset_id == dataset_id)
        if not include_archived:
            total_query = total_query.where(ModelVersion.status == "active")
        total = session.scalar(total_query) or 0
        return ModelVersionList(items=[model_response(item) for item in items], total=total)

    def get_model(self, session: Session, model_id: str) -> ModelVersion:
        model = session.get(ModelVersion, model_id)
        if model is None:
            raise NotFoundError("model_not_found", "Model version was not found.")
        return model

    def get_model_response(self, session: Session, model_id: str) -> ModelVersionResponse:
        return model_response(self.get_model(session, model_id))

    def update_model(self, session: Session, model_id: str, payload: ModelVersionUpdate) -> ModelVersionResponse:
        model = self.get_model(session, model_id)
        changes = payload.model_dump(exclude_unset=True)
        for field, value in changes.items():
            setattr(model, field, value.strip() if isinstance(value, str) else value)
        session.commit()
        session.refresh(model)
        return model_response(model)

    def archive_model(self, session: Session, model_id: str) -> ModelVersionResponse:
        model = self.get_model(session, model_id)
        model.status = "archived"
        model.archived_at = utc_now()
        session.commit()
        session.refresh(model)
        return model_response(model)

    def restore_model(self, session: Session, model_id: str) -> ModelVersionResponse:
        model = self.get_model(session, model_id)
        model.status = "active"
        model.archived_at = None
        session.commit()
        session.refresh(model)
        return model_response(model)

    def delete_model(self, session: Session, model_id: str) -> None:
        model = self.get_model(session, model_id)
        self.storage.managed_model_path(model.model_path)
        session.delete(model)
        try:
            session.commit()
        except Exception as exc:
            session.rollback()
            raise ConflictError("model_delete_failed", "Model version could not be deleted.") from exc
        self.storage.remove_model_version(model.id)

    def download_path(self, session: Session, model_id: str) -> Path:
        model = self.get_model(session, model_id)
        path = self.storage.managed_model_path(model.model_path)
        if path.suffix.lower() not in {".pt", ".onnx"}:
            raise ValidationError("unsupported_model_format", "Only PT and ONNX model files can be downloaded.")
        if not path.is_file():
            raise NotFoundError("model_file_missing", "The managed model file is missing.")
        return path

    def export_onnx(self, session: Session, model_id: str) -> ModelVersionResponse:
        source_model = self.get_model(session, model_id)
        if source_model.format == "onnx":
            return model_response(source_model)
        if source_model.format != "pt":
            raise ValidationError("unsupported_model_format", "Only PT models can be exported to ONNX.")
        if source_model.engine_type != "ultralytics":
            raise ValidationError("unsupported_model_engine", "ONNX export requires the Ultralytics engine.")
        source_path = self.download_path(session, model_id)
        existing = session.scalar(
            select(ModelVersion).where(
                ModelVersion.source_model_id == source_model.id,
                ModelVersion.format == "onnx",
            )
        )
        if existing is not None and Path(existing.model_path).is_file():
            return model_response(existing)

        export_model = existing
        created = False
        if export_model is None:
            export_model = ModelVersion(
                id=f"model_{source_model.id}_onnx",
                name=f"{Path(source_model.name).stem}.onnx",
                version=f"{source_model.version}-onnx",
                dataset_id=source_model.dataset_id,
                source_model_id=source_model.id,
                source="exported",
                artifact_type="onnx",
                format="onnx",
                task_type=source_model.task_type,
                engine_type=source_model.engine_type,
                model_path="",
                base_model=source_model.name,
                status="active",
                metrics_json=dict(source_model.metrics_json or {}),
                notes=f"Exported from {source_model.name}.",
            )
            session.add(export_model)
            session.flush()
            created = True
        output_path = self.storage.model_version_dir(export_model.id) / "model.onnx"
        try:
            exported_path = onnx.export_fp32_onnx(source_path, output_path)
            exported_path = self.storage.managed_model_path(exported_path)
            if not exported_path.is_file():
                raise RuntimeError("ONNX exporter did not produce a managed file.")
            export_model.model_path = str(exported_path)
            export_model.metrics_json = dict(source_model.metrics_json or {})
            self._apply_metrics(export_model, export_model.metrics_json)
            session.commit()
            session.refresh(export_model)
            return model_response(export_model)
        except Exception as exc:
            session.rollback()
            if created:
                self.storage.remove_model_version(export_model.id)
            raise ValidationError("onnx_export_failed", f"ONNX FP32 export failed: {exc}") from exc

    def run_test_inference(self, session: Session, model_id: str, image_bytes: bytes, confidence: float = 0.25, iou: float = 0.45) -> dict:
        model = self.get_model(session, model_id)
        if model.format != "pt" or model.engine_type != "ultralytics":
            raise ValidationError("unsupported_model_engine", "Quick test currently requires a managed Ultralytics PT model.")
        path = self.download_path(session, model_id)
        classes = list(session.scalars(select(ClassLabel).where(ClassLabel.dataset_id == model.dataset_id))) if model.dataset_id else []
        return run_test_inference(model_id=model.id, model_path=path, task_type=model.task_type, image_bytes=image_bytes, confidence=confidence, iou=iou, class_names={item.class_index: item.name for item in classes})

    def save_test_record(self, session: Session, model_id: str, file_name: str, image_bytes: bytes, result: dict) -> ModelTestRecord:
        self.get_model(session, model_id)
        record = ModelTestRecord(id=new_id("test"), model_id=model_id, file_name=Path(file_name).name or "test.jpg", image_path="", result_json=result)
        path = self.storage.write_model_test_image(model_id, record.id, record.file_name, image_bytes)
        record.image_path = str(path)
        session.add(record)
        session.commit()
        session.refresh(record)
        return record

    def list_test_records(self, session: Session, model_id: str) -> list[ModelTestRecord]:
        self.get_model(session, model_id)
        return list(session.scalars(select(ModelTestRecord).where(ModelTestRecord.model_id == model_id).order_by(ModelTestRecord.created_at.desc())))

    def evaluate(self, session: Session, model_id: str, split: str, confidence: float, iou: float) -> ModelEvaluationRecord:
        model = self.get_model(session, model_id)
        if not model.dataset_id:
            raise ValidationError("evaluation_dataset_missing", "This managed model is not attached to a dataset.")
        if model.format != "pt" or model.engine_type != "ultralytics":
            raise ValidationError("unsupported_model_engine", "Local evaluation requires a managed Ultralytics PT model.")
        images = list(session.scalars(select(ImageItem).where(ImageItem.dataset_id == model.dataset_id, ImageItem.split == split).order_by(ImageItem.created_at).limit(500)))
        if not images:
            raise ValidationError("evaluation_split_empty", f"The {split} split has no images to evaluate.")
        classes = list(session.scalars(select(ClassLabel).where(ClassLabel.dataset_id == model.dataset_id)))
        class_names = {item.class_index: item.name for item in classes}
        annotations_by_image: dict[str, list[Annotation]] = {}
        for annotation in session.scalars(select(Annotation).where(Annotation.dataset_id == model.dataset_id)):
            annotations_by_image.setdefault(annotation.image_id, []).append(annotation)
        path = self.download_path(session, model_id)
        true_positive = false_positive = false_negative = 0
        errors: list[dict] = []
        for image in images:
            inference = run_test_inference(model_id=model.id, model_path=path, task_type=model.task_type, image_bytes=self.storage.read_image(model.dataset_id, image.storage_name), confidence=confidence, iou=iou, class_names=class_names)
            expected = annotations_by_image.get(image.id, [])
            if model.task_type == "classify":
                actual = inference["detections"][0]["class_index"] if inference["detections"] else None
                target = expected[0].class_label.class_index if expected else None
                if actual == target and target is not None:
                    true_positive += 1
                elif target is not None:
                    false_negative += 1
                    errors.append({"image_id": image.id, "image_file": image.file_name, "type": "misclassified", "expected_class_index": target, "predicted_class_index": actual, "message": "Top-1 prediction does not match the saved class."})
                continue
            ground_truth: list[dict] = []
            for annotation in expected:
                box = self._annotation_box(annotation)
                if box is not None:
                    ground_truth.append({**box, "class_index": annotation.class_label.class_index})
            predictions = list(inference["detections"])
            matched_predictions: set[int] = set()
            for truth in ground_truth:
                candidate = next((index for index, prediction in enumerate(predictions) if index not in matched_predictions and prediction["class_index"] == truth["class_index"] and self._iou(truth, prediction) >= iou), None)
                if candidate is None:
                    false_negative += 1
                    errors.append({"image_id": image.id, "image_file": image.file_name, "type": "missed_detection", "class_index": truth["class_index"], "gt_bbox": self._box_list(truth), "message": "No matching prediction above the evaluation IoU threshold."})
                else:
                    matched_predictions.add(candidate)
                    true_positive += 1
            for index, prediction in enumerate(predictions):
                if index not in matched_predictions:
                    false_positive += 1
                    errors.append({"image_id": image.id, "image_file": image.file_name, "type": "false_positive", "class_index": prediction["class_index"], "confidence": prediction["confidence"], "pred_bbox": self._box_list(prediction), "message": "Prediction has no matching saved annotation."})
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        result = {"images_evaluated": len(images), "split": split, "metrics": {"true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(2 * precision * recall / max(precision + recall, 1e-12), 6)}, "error_samples": errors[:200], "error_sample_count": len(errors)}
        record = ModelEvaluationRecord(id=new_id("eval"), model_id=model.id, dataset_id=model.dataset_id, split=split, confidence=confidence, iou=iou, result_json=result)
        session.add(record)
        session.commit()
        session.refresh(record)
        return record

    def list_evaluations(self, session: Session, model_id: str) -> list[ModelEvaluationRecord]:
        self.get_model(session, model_id)
        return list(session.scalars(select(ModelEvaluationRecord).where(ModelEvaluationRecord.model_id == model_id).order_by(ModelEvaluationRecord.created_at.desc())))

    @staticmethod
    def _annotation_box(annotation: Annotation) -> dict | None:
        if annotation.type == "bbox" and None not in (annotation.x, annotation.y, annotation.width, annotation.height):
            return {"x": annotation.x, "y": annotation.y, "width": annotation.width, "height": annotation.height}
        if annotation.type == "polygon" and annotation.polygon:
            xs, ys = [point[0] for point in annotation.polygon], [point[1] for point in annotation.polygon]
            return {"x": min(xs), "y": min(ys), "width": max(xs) - min(xs), "height": max(ys) - min(ys)}
        if annotation.type == "obb" and annotation.obb:
            return {"x": annotation.obb["cx"] - annotation.obb["width"] / 2, "y": annotation.obb["cy"] - annotation.obb["height"] / 2, "width": annotation.obb["width"], "height": annotation.obb["height"]}
        return None

    @staticmethod
    def _iou(left: dict, right: dict) -> float:
        x1, y1 = max(left["x"], right["x"]), max(left["y"], right["y"])
        x2, y2 = min(left["x"] + left["width"], right["x"] + right["width"]), min(left["y"] + left["height"], right["y"] + right["height"])
        overlap = max(0, x2 - x1) * max(0, y2 - y1)
        return overlap / max(left["width"] * left["height"] + right["width"] * right["height"] - overlap, 1e-12)

    @staticmethod
    def _box_list(value: dict) -> list[float]:
        return [round(float(value[key]), 3) for key in ("x", "y", "width", "height")]

    def preannotate(self, session: Session, model_id: str, dataset_id: str, image_ids: list[str], confidence: float, iou: float) -> list[dict]:
        model = self.get_model(session, model_id)
        if model.dataset_id != dataset_id:
            raise ValidationError("preannotation_dataset_mismatch", "Pre-annotation requires a model trained from this dataset.")
        if model.format != "pt" or model.engine_type != "ultralytics":
            raise ValidationError("unsupported_model_engine", "Pre-annotation requires a managed Ultralytics PT model.")
        target_classes = {item.class_index: item.id for item in session.scalars(select(ClassLabel).where(ClassLabel.dataset_id == dataset_id))}
        ids = list(dict.fromkeys(image_ids))
        images = list(session.scalars(select(ImageItem).where(ImageItem.dataset_id == dataset_id, ImageItem.id.in_(ids))))
        if len(images) != len(ids):
            raise ValidationError("image_not_found", "One or more selected images were not found in this dataset.")
        path = self.download_path(session, model_id)
        names = {index: str(index) for index in target_classes}
        results: list[dict] = []
        for image in images:
            inference = run_test_inference(model_id=model.id, model_path=path, task_type=model.task_type, image_bytes=self.storage.read_image(dataset_id, image.storage_name), confidence=confidence, iou=iou, class_names=names)
            annotations: list[dict] = []
            for item in inference["detections"]:
                class_id = target_classes.get(item["class_index"])
                if class_id is None:
                    continue
                if model.task_type == "segment" and item["polygon"]:
                    annotations.append({"type": "polygon", "class_id": class_id, "polygon": item["polygon"], "source": "manual"})
                elif model.task_type == "obb" and item["obb_points"]:
                    # Starter persists canonical center/size/angle OBB; keep proposal review in canvas by bounding box fallback.
                    annotations.append({"type": "obb", "class_id": class_id, "obb": {"cx": item["x"] + item["width"] / 2, "cy": item["y"] + item["height"] / 2, "width": item["width"], "height": item["height"], "angle": 0}, "source": "manual"})
                elif model.task_type == "detect":
                    annotations.append({"type": "bbox", "class_id": class_id, "bbox": {"x": item["x"], "y": item["y"], "width": item["width"], "height": item["height"]}, "source": "manual"})
                elif model.task_type == "classify":
                    annotations = [{"type": "classify", "class_id": class_id, "source": "manual"}]
                    break
            results.append({"image_id": image.id, "annotations": annotations})
        return results

    def compare(self, session: Session, baseline_id: str, candidate_id: str) -> dict:
        baseline, candidate = self.get_model(session, baseline_id), self.get_model(session, candidate_id)
        if baseline.id == candidate.id or not baseline.dataset_id or baseline.dataset_id != candidate.dataset_id or baseline.task_type != candidate.task_type:
            raise ValidationError("invalid_model_comparison", "Choose two different models from the same dataset and task type.")
        keys = ("precision", "recall", "map50", "map50_95")
        deltas = {key: (getattr(candidate, key) - getattr(baseline, key)) if getattr(candidate, key) is not None and getattr(baseline, key) is not None else None for key in keys}
        suggestions = [f"Candidate improves {key} by {value:+.3f}." for key, value in deltas.items() if value is not None and value > 0.001]
        if not suggestions:
            suggestions.append("Review validation metrics and quick-test results before replacing the baseline.")
        return {"dataset_id": baseline.dataset_id, "baseline": {"id": baseline.id, "name": baseline.name, "metrics": baseline.metrics_json}, "candidate": {"id": candidate.id, "name": candidate.name, "metrics": candidate.metrics_json}, "deltas": deltas, "suggestions": suggestions}

    @staticmethod
    def _apply_metrics(model: ModelVersion, metrics: dict) -> None:
        model.metrics_json = metrics or {}
        model.precision = metrics.get("precision")
        model.recall = metrics.get("recall")
        model.map50 = metrics.get("map50")
        model.map50_95 = metrics.get("map50_95")


def model_response(model: ModelVersion) -> ModelVersionResponse:
    return ModelVersionResponse.model_validate(model)
