from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.ids import new_id
from app.core.models import AutoAnnotationTask, ClassLabel, ImageItem, ModelEvaluationRecord, ModelTestRecord, ModelVersion, TrainingTask
from app.core.storage import Storage
from app.core.time import utc_now
from app.models import onnx
from app.models.evaluation import run_evaluation_process
from app.models.evaluation_artifacts import EvaluationArtifactManager
from app.models.error_samples import ErrorSampleAnalyzer
from app.models.inference import run_test_inference
from app.models.schemas import ModelVersionList, ModelVersionResponse, ModelVersionUpdate
from app.dataset.exchange.yolo import export_dataset_directory
from app.training.observability.log_store import TrainingLogStore


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
        auto_annotation_task = session.scalar(
            select(AutoAnnotationTask.id)
            .where(AutoAnnotationTask.model_id == model.id)
            .limit(1)
        )
        if auto_annotation_task is not None:
            raise ConflictError("model_in_use", "Archive this model instead; it is retained as auto-annotation task history.")
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
        path = self.download_path(session, model.id)
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

    def create_evaluation(self, session: Session, model_id: str, split: str, confidence: float, iou: float) -> ModelEvaluationRecord:
        model = self.get_model(session, model_id)
        if not model.dataset_id:
            raise ValidationError("evaluation_dataset_missing", "This managed model is not attached to a dataset.")
        if model.format != "pt" or model.engine_type != "ultralytics":
            raise ValidationError("unsupported_model_engine", "Local evaluation requires a managed Ultralytics PT model.")
        self.download_path(session, model.id)
        image_count = session.scalar(select(func.count()).select_from(ImageItem).where(ImageItem.dataset_id == model.dataset_id, ImageItem.split == split)) or 0
        if not image_count:
            raise ValidationError("evaluation_split_empty", f"The {split} split has no images to evaluate.")
        evaluation_id = new_id("eval")
        task_root = self.storage.evaluation_task_dir(evaluation_id)
        try:
            export = export_dataset_directory(session, self.storage, model.dataset_id, task_root / "dataset")
            if not export["annotated_image_counts"].get(split):
                raise ValidationError("evaluation_split_empty", f"The exported {split} split has no annotated images to evaluate.")
            record = ModelEvaluationRecord(
                id=evaluation_id,
                model_id=model.id,
                dataset_id=model.dataset_id,
                split=split,
                status="pending",
                confidence=confidence,
                iou=iou,
                result_json={},
                export_path=str(export["root"]),
                data_path=str(export["data_yaml"]),
                run_dir=str(task_root / "run"),
                logs_path=str(task_root / "evaluation.log"),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record
        except Exception:
            session.rollback()
            self.storage.remove_evaluation_task(evaluation_id)
            raise

    def run_evaluation(self, session: Session, evaluation_id: str) -> None:
        record = session.get(ModelEvaluationRecord, evaluation_id)
        if record is None or record.status != "running":
            return
        model = self.get_model(session, record.model_id)
        try:
            return_code, _logs, metrics = run_evaluation_process(record, model)
            run_dir = Path(record.run_dir or "")
            artifacts = EvaluationArtifactManager().find_artifacts(run_dir)
            record.result_json = {
                "split": record.split,
                "task_type": model.task_type,
                "metrics": metrics,
                "artifacts": artifacts,
                "error_samples": ErrorSampleAnalyzer().collect(record.run_dir or "", record.confidence, record.export_path, record.split)
                if model.task_type != "classify"
                else [],
            }
            record.status = "completed" if return_code == 0 else "failed"
            record.error_message = None if return_code == 0 else f"yolo val exited with code {return_code}"
            record.finished_at = utc_now()
            session.commit()
        except Exception as exc:
            record.status = "failed"
            record.error_message = str(exc)
            record.finished_at = utc_now()
            session.commit()

    def list_evaluations(self, session: Session, model_id: str) -> list[ModelEvaluationRecord]:
        self.get_model(session, model_id)
        return list(session.scalars(select(ModelEvaluationRecord).where(ModelEvaluationRecord.model_id == model_id).order_by(ModelEvaluationRecord.created_at.desc())))

    def get_evaluation(self, session: Session, model_id: str, evaluation_id: str) -> ModelEvaluationRecord:
        self.get_model(session, model_id)
        record = session.get(ModelEvaluationRecord, evaluation_id)
        if record is None or record.model_id != model_id:
            raise NotFoundError("evaluation_not_found", "Model evaluation was not found.")
        return record

    def evaluation_logs(self, session: Session, model_id: str, evaluation_id: str, tail: int | None = None) -> dict:
        record = self.get_evaluation(session, model_id, evaluation_id)
        if not record.logs_path:
            return {"evaluation_id": record.id, "logs": "", "line_count": 0}
        path = self.storage.managed_evaluation_path(record.logs_path)
        store = TrainingLogStore(path)
        return {"evaluation_id": record.id, "logs": store.read(tail), "line_count": store.line_count()}

    def evaluation_artifact_path(self, session: Session, model_id: str, evaluation_id: str, artifact: str) -> Path:
        record = self.get_evaluation(session, model_id, evaluation_id)
        path_value = (record.result_json or {}).get("artifacts", {}).get(artifact)
        if artifact not in {"confusion_matrix", "pr_curve", "box_pr_curve", "mask_pr_curve", "predictions"} or not path_value:
            raise NotFoundError("evaluation_artifact_not_found", "Evaluation artifact was not found.")
        path = self.storage.managed_evaluation_path(path_value)
        if not path.is_file():
            raise NotFoundError("evaluation_artifact_not_found", "Evaluation artifact was not found.")
        return path

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
    def _obb_from_points(points: list[list[float]]) -> dict:
        import math

        if len(points) != 4:
            raise ValidationError("invalid_obb_prediction", "The model returned an invalid oriented bounding box.")
        cx = sum(point[0] for point in points) / 4
        cy = sum(point[1] for point in points) / 4
        width = math.dist(points[0], points[1])
        height = math.dist(points[1], points[2])
        angle = math.degrees(math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]))
        return {"cx": cx, "cy": cy, "width": width, "height": height, "angle": angle}

    @staticmethod
    def _apply_metrics(model: ModelVersion, metrics: dict) -> None:
        model.metrics_json = metrics or {}
        model.precision = metrics.get("precision")
        model.recall = metrics.get("recall")
        model.map50 = metrics.get("map50")
        model.map50_95 = metrics.get("map50_95")


def model_response(model: ModelVersion) -> ModelVersionResponse:
    return ModelVersionResponse.model_validate(model)
