from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.models import ModelVersion, TrainingTask
from app.core.storage import Storage
from app.core.time import utc_now
from app.models import onnx
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

    @staticmethod
    def _apply_metrics(model: ModelVersion, metrics: dict) -> None:
        model.metrics_json = metrics or {}
        model.precision = metrics.get("precision")
        model.recall = metrics.get("recall")
        model.map50 = metrics.get("map50")
        model.map50_95 = metrics.get("map50_95")


def model_response(model: ModelVersion) -> ModelVersionResponse:
    return ModelVersionResponse.model_validate(model)
