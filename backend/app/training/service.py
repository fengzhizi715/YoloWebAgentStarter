from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.ids import new_id
from app.core.models import Annotation, ClassLabel, Dataset, ImageItem, TrainingProfile, TrainingTask
from app.core.storage import Storage
from app.core.task_types import TaskType
from app.dataset.exchange.yolo import export_dataset_directory
from app.dataset.service import get_dataset
from app.dataset.validation import validate_dataset
from app.training.artifacts.checkpoints import checkpoint_paths
from app.training.config import build_training_command, resolve_model_reference, validate_model_family
from app.training.observability.log_store import TrainingLogStore
from app.training.observability.summary import write_training_summary
from app.training.observability.telemetry import TrainingTelemetryService
from app.training.runtime.process_registry import process_registry
from app.training.runtime.queue import TrainingQueue
from app.training.runtime.device_service import DeviceService
from app.training.schemas import (
    TrainingLogResponse,
    TrainingProfileCreate,
    TrainingProfileUpdate,
    TrainingProfileResponse,
    TrainingSummaryResponse,
    TrainingTaskCreate,
    TrainingTaskList,
    TrainingTaskResponse,
    TrainingTaskResumeRequest,
)


class TrainingService:
    def __init__(self, session_factory: sessionmaker[Session], storage: Storage, queue: TrainingQueue) -> None:
        self.session_factory = session_factory
        self.storage = storage
        self.queue = queue
        self.queue.configure(session_factory, storage)

    def create_task(self, session: Session, payload: TrainingTaskCreate) -> TrainingTaskResponse:
        return self._create_task(session, payload)

    def devices(self) -> list[dict[str, object]]:
        return [device.as_dict() for device in DeviceService().list_devices()]

    def resume_task(self, session: Session, task_id: str, payload: TrainingTaskResumeRequest) -> TrainingTaskResponse:
        previous = self.get_task(session, task_id)
        if previous.status not in {"completed", "failed", "stopped"}:
            raise ValidationError("resume_task_not_finished", "Only completed, failed, or stopped training tasks can be resumed.")
        if payload.resume_epoch and previous.status == "completed":
            raise ValidationError(
                "resume_completed_task",
                "A completed task cannot restore epoch state; start a new task from last.pt with resume_epoch=false.",
            )
        if payload.resume_epoch and payload.epochs is not None and payload.epochs != previous.epochs:
            raise ValidationError(
                "resume_epochs_fixed",
                "Epoch-state resume keeps the source task's total epochs; use resume_epoch=false to train additional epochs.",
            )
        checkpoint = checkpoint_paths(previous.run_dir or "").get("last")
        if checkpoint is None:
            raise NotFoundError("resume_checkpoint_not_found", "This training task has no managed last.pt checkpoint.")
        source_run_dir = self.storage.managed_training_path(previous.run_dir or "")
        source_checkpoint = self.storage.managed_training_path(checkpoint)
        if not source_run_dir.is_dir() or not source_checkpoint.is_file():
            raise NotFoundError("resume_checkpoint_not_found", "This training task has no managed last.pt checkpoint.")
        source_config = dict(previous.config_json or {})
        source_config["name"] = (payload.name or f"{previous.name} (resume)").strip()
        if payload.epochs is not None and not payload.resume_epoch:
            source_config["epochs"] = payload.epochs
        task_payload = TrainingTaskCreate.model_validate(source_config)
        return self._create_task(
            session,
            task_payload,
            resume_checkpoint=source_checkpoint,
            resume_run_dir=source_run_dir if payload.resume_epoch else None,
            resume_epoch=payload.resume_epoch,
            resume_source_task_id=previous.id,
        )

    def _create_task(
        self,
        session: Session,
        payload: TrainingTaskCreate,
        *,
        resume_checkpoint: Path | None = None,
        resume_run_dir: Path | None = None,
        resume_epoch: bool = False,
        resume_source_task_id: str | None = None,
    ) -> TrainingTaskResponse:
        payload.device = DeviceService().normalize_selector(payload.device)
        dataset = get_dataset(session, payload.dataset_id)
        task_type = payload.task_type or TaskType(dataset.task_type)
        if task_type.value != dataset.task_type:
            raise ValidationError("task_type_mismatch", "Training task type must match the dataset task type.")
        validate_model_family(task_type, payload.model)
        model_reference = resolve_model_reference(payload.model, self.storage)
        self._validate_dataset_ready(session, dataset)

        task_id = new_id("train")
        task_root = self.storage.training_task_dir(task_id)
        export_root = task_root / "dataset"
        run_dir = resume_run_dir if resume_run_dir is not None else task_root / "run"
        try:
            if resume_checkpoint is not None:
                model_reference = str(resume_checkpoint)
            export = export_dataset_directory(session, self.storage, dataset.id, export_root)
            data_yaml = str(export["data_yaml"])
            command = build_training_command(
                task_type=task_type,
                model=model_reference,
                data_yaml=data_yaml,
                run_dir=str(run_dir),
                epochs=payload.epochs,
                img_size=payload.img_size,
                batch_size=payload.batch_size,
                device=payload.device,
                workers=payload.workers,
                seed=payload.seed,
                optimizer=payload.optimizer,
                lr0=payload.lr0,
                patience=payload.patience,
                resume=resume_epoch,
            )
            task = TrainingTask(
                id=task_id,
                dataset_id=dataset.id,
                name=payload.name.strip(),
                status="pending",
                task_type=task_type.value,
                model_name=payload.model,
                model_path=model_reference,
                epochs=payload.epochs,
                img_size=payload.img_size,
                batch_size=payload.batch_size,
                device=payload.device,
                workers=payload.workers,
                val_ratio=payload.val_ratio,
                seed=payload.seed,
                optimizer=payload.optimizer,
                lr0=payload.lr0,
                patience=payload.patience,
                config_json={
                    **payload.model_dump(),
                    **({"resume_source_task_id": resume_source_task_id, "resume_epoch": resume_epoch} if resume_source_task_id else {}),
                },
                command_args_json=command.args,
                command_preview=command.readable,
                export_path=str(export["root"]),
                data_yaml_path=data_yaml,
                run_dir=str(run_dir),
                logs_path=str(task_root / "train.log"),
                summary_path=str(task_root / "training_summary.json"),
                progress_total_epochs=payload.epochs,
            )
            session.add(task)
            session.commit()
            session.refresh(task)
        except Exception:
            session.rollback()
            self.storage.remove_training_task(task_id)
            raise
        self.queue.submit(task_id)
        return task_response(task)

    def list_tasks(self, session: Session, dataset_id: str | None = None) -> TrainingTaskList:
        query = select(TrainingTask).order_by(TrainingTask.created_at.desc(), TrainingTask.id.desc())
        if dataset_id:
            query = query.where(TrainingTask.dataset_id == dataset_id)
        tasks = list(session.scalars(query))
        telemetry = TrainingTelemetryService()
        for task in tasks:
            telemetry.refresh_task(session, task)
        return TrainingTaskList(items=[task_response(task) for task in tasks])

    def get_task(self, session: Session, task_id: str) -> TrainingTask:
        task = session.get(TrainingTask, task_id)
        if task is None:
            raise NotFoundError("training_task_not_found", "Training task was not found.")
        return task

    def get_task_response(self, session: Session, task_id: str) -> TrainingTaskResponse:
        task = self.get_task(session, task_id)
        return task_response(TrainingTelemetryService().refresh_task(session, task))

    def stop_task(self, session: Session, task_id: str) -> TrainingTaskResponse:
        task = self.get_task(session, task_id)
        if task.status in {"completed", "failed", "stopped"}:
            return task_response(task)
        if task.status == "pending":
            task.status = "stopped"
            task.error_message = "Training stopped before it started."
            from app.core.time import utc_now

            task.finished_at = utc_now()
            session.commit()
            self.queue.stop_pending(task_id)
            return task_response(task)
        task.stop_requested = True
        session.commit()
        process_registry.request_stop(task_id)
        session.refresh(task)
        return task_response(task)

    def logs(self, session: Session, task_id: str, tail: int = 200) -> TrainingLogResponse:
        task = self.get_task(session, task_id)
        store = TrainingLogStore(task.logs_path or "train.log")
        return TrainingLogResponse(task_id=task_id, logs=store.read(tail), line_count=store.line_count())

    def summary(self, session: Session, task_id: str) -> TrainingSummaryResponse:
        task = TrainingTelemetryService().refresh_task(session, self.get_task(session, task_id))
        store = TrainingLogStore(task.logs_path or "train.log")
        if task.status in {"completed", "failed", "stopped"} and task.summary_path and Path(task.summary_path).is_file():
            import json

            try:
                data = json.loads(Path(task.summary_path).read_text(encoding="utf-8"))
                return TrainingSummaryResponse.model_validate(data)
            except (OSError, ValueError):
                pass
        data = write_training_summary(task, task.dataset)
        data["log_summary"]["line_count"] = store.line_count()
        return TrainingSummaryResponse.model_validate(data)

    def checkpoint(self, session: Session, task_id: str, name: str) -> Path:
        task = self.get_task(session, task_id)
        if name not in {"best", "last"}:
            raise ValidationError("invalid_checkpoint", "Checkpoint must be best or last.")
        paths = checkpoint_paths(task.run_dir or "")
        path = paths[name]
        if path is None:
            raise NotFoundError("checkpoint_not_found", f"The {name}.pt checkpoint is not available.")
        return Path(path)

    def create_profile(self, session: Session, payload: TrainingProfileCreate) -> TrainingProfileResponse:
        dataset = get_dataset(session, payload.dataset_id)
        if dataset.task_type != payload.task_type.value:
            raise ValidationError("task_type_mismatch", "Training profile type must match the dataset task type.")
        validate_model_family(payload.task_type, payload.model)
        profile = TrainingProfile(
            id=new_id("profile"),
            dataset_id=dataset.id,
            name=payload.name.strip(),
            description=payload.description,
            model_name=payload.model,
            task_type=payload.task_type.value,
            device=DeviceService().normalize_selector(payload.device),
            **payload.model_dump(exclude={"dataset_id", "name", "description", "model", "task_type", "device"}),
        )
        session.add(profile)
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise ConflictError("training_profile_create_failed", "Training profile could not be saved.") from exc
        session.refresh(profile)
        return TrainingProfileResponse.model_validate(profile)

    def list_profiles(self, session: Session, dataset_id: str | None = None) -> list[TrainingProfileResponse]:
        query = select(TrainingProfile).order_by(TrainingProfile.created_at.desc())
        if dataset_id:
            query = query.where(TrainingProfile.dataset_id == dataset_id)
        return [TrainingProfileResponse.model_validate(item) for item in session.scalars(query)]

    def update_profile(self, session: Session, profile_id: str, payload: TrainingProfileUpdate) -> TrainingProfileResponse:
        profile = session.get(TrainingProfile, profile_id)
        if profile is None:
            raise NotFoundError("training_profile_not_found", "Training profile was not found.")
        changes = payload.model_dump(exclude_unset=True)
        if "device" in changes and changes["device"] is not None:
            changes["device"] = DeviceService().normalize_selector(changes["device"])
        if "task_type" in changes and changes["task_type"].value != profile.dataset.task_type:
            raise ValidationError("task_type_mismatch", "Training profile type must match the dataset task type.")
        if "model" in changes:
            validate_model_family(payload.task_type or TaskType(profile.task_type), changes["model"])
            changes["model_name"] = changes.pop("model")
        for field, value in changes.items():
            setattr(profile, field, value.value if isinstance(value, TaskType) else value)
        session.commit()
        session.refresh(profile)
        return TrainingProfileResponse.model_validate(profile)

    def _validate_dataset_ready(self, session: Session, dataset: Dataset) -> None:
        report = validate_dataset(session, self.storage, dataset.id)
        if not report.valid:
            raise ValidationError(
                "dataset_not_ready",
                "Dataset validation must pass before training starts.",
                details={"issues": [item.model_dump() for item in report.issues if item.level == "error"]},
            )
        split_counts = dict(
            session.execute(
                select(ImageItem.split, func.count(ImageItem.id)).where(ImageItem.dataset_id == dataset.id).group_by(ImageItem.split)
            ).all()
        )
        if not split_counts.get("train") or not split_counts.get("val"):
            raise ValidationError("training_split_missing", "Training requires at least one train image and one val image.")
        if dataset.task_type == "classify":
            annotated_splits = set(
                session.scalars(
                    select(ImageItem.split)
                    .join(Annotation, Annotation.image_id == ImageItem.id)
                    .where(
                        ImageItem.dataset_id == dataset.id,
                        Annotation.dataset_id == dataset.id,
                        Annotation.type == "classify",
                    )
                )
            )
            missing = {"train", "val"} - annotated_splits
            if missing:
                raise ValidationError(
                    "training_classification_split_unannotated",
                    "Classification training requires at least one annotated image in both train and val splits.",
                    details={"missing_splits": sorted(missing)},
                )
        class_count = session.scalar(select(func.count()).select_from(ClassLabel).where(ClassLabel.dataset_id == dataset.id)) or 0
        if not class_count:
            raise ValidationError("training_classes_missing", "Training requires at least one class label.")
        annotation_count = session.scalar(select(func.count()).select_from(Annotation).where(Annotation.dataset_id == dataset.id)) or 0
        if not annotation_count:
            raise ValidationError("training_annotations_missing", "Training requires at least one annotation.")


def task_response(task: TrainingTask) -> TrainingTaskResponse:
    return TrainingTaskResponse.model_validate(task)
