from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.core.time import UTCDateTime, utc_now


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, onupdate=utc_now, nullable=False)


class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"
    __table_args__ = (CheckConstraint("task_type IN ('detect', 'segment')", name="ck_dataset_task_type"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_type: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    image_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    class_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    classes: Mapped[list["ClassLabel"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    images: Mapped[list["ImageItem"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    annotations: Mapped[list["Annotation"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    training_profiles: Mapped[list["TrainingProfile"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    training_tasks: Mapped[list["TrainingTask"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    model_versions: Mapped[list["ModelVersion"]] = relationship(back_populates="dataset")


class ClassLabel(Base, TimestampMixin):
    __tablename__ = "class_labels"
    __table_args__ = (
        UniqueConstraint("dataset_id", "class_index", name="uq_class_label_dataset_index"),
        UniqueConstraint("dataset_id", "name", name="uq_class_label_dataset_name"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=False)
    class_index: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    color: Mapped[str] = mapped_column(String(16), default="#22c55e", nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="classes")
    annotations: Mapped[list["Annotation"]] = relationship(back_populates="class_label")


class ImageItem(Base, TimestampMixin):
    __tablename__ = "image_items"
    __table_args__ = (
        UniqueConstraint("dataset_id", "storage_name", name="uq_image_item_dataset_storage"),
        CheckConstraint("split IN ('train', 'val', 'test')", name="ck_image_item_split"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=False)
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_name: Mapped[str] = mapped_column(String(512), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    split: Mapped[str] = mapped_column(String(16), default="train", index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="unannotated", index=True, nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="images")
    annotations: Mapped[list["Annotation"]] = relationship(back_populates="image", cascade="all, delete-orphan")


class Annotation(Base, TimestampMixin):
    __tablename__ = "annotations"
    __table_args__ = (CheckConstraint("type IN ('bbox', 'polygon')", name="ck_annotation_type"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("image_items.id", ondelete="CASCADE"), index=True, nullable=False)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=False)
    class_id: Mapped[str] = mapped_column(ForeignKey("class_labels.id", ondelete="RESTRICT"), index=True, nullable=False)
    type: Mapped[str] = mapped_column(String(16), nullable=False)
    x: Mapped[float | None] = mapped_column(Float, nullable=True)
    y: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[float | None] = mapped_column(Float, nullable=True)
    height: Mapped[float | None] = mapped_column(Float, nullable=True)
    polygon: Mapped[list[list[float]] | None] = mapped_column(JSON, nullable=True)
    source: Mapped[str] = mapped_column(String(16), default="manual", nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="annotations")
    image: Mapped[ImageItem] = relationship(back_populates="annotations")
    class_label: Mapped[ClassLabel] = relationship(back_populates="annotations")


class TrainingProfile(Base, TimestampMixin):
    __tablename__ = "training_profiles"
    __table_args__ = (CheckConstraint("task_type IN ('detect', 'segment')", name="ck_training_profile_task_type"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), default="yolo11n.pt", nullable=False)
    task_type: Mapped[str] = mapped_column(String(16), nullable=False)
    epochs: Mapped[int] = mapped_column(Integer, default=50, nullable=False)
    img_size: Mapped[int] = mapped_column(Integer, default=640, nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, default=16, nullable=False)
    device: Mapped[str] = mapped_column(String(64), default="auto", nullable=False)
    workers: Mapped[int] = mapped_column(Integer, default=2, nullable=False)
    val_ratio: Mapped[float] = mapped_column(Float, default=0.2, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, default=42, nullable=False)
    optimizer: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lr0: Mapped[float | None] = mapped_column(Float, nullable=True)
    patience: Mapped[int | None] = mapped_column(Integer, nullable=True)

    dataset: Mapped[Dataset] = relationship(back_populates="training_profiles")


class TrainingTask(Base, TimestampMixin):
    __tablename__ = "training_tasks"
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'stopped')", name="ck_training_task_status"),
        CheckConstraint("task_type IN ('detect', 'segment')", name="ck_training_task_type"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=False)
    profile_id: Mapped[str | None] = mapped_column(ForeignKey("training_profiles.id", ondelete="SET NULL"), index=True, nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True, nullable=False)
    task_type: Mapped[str] = mapped_column(String(16), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_path: Mapped[str] = mapped_column(Text, nullable=False)
    epochs: Mapped[int] = mapped_column(Integer, nullable=False)
    img_size: Mapped[int] = mapped_column(Integer, nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    device: Mapped[str] = mapped_column(String(64), nullable=False)
    workers: Mapped[int] = mapped_column(Integer, default=2, nullable=False)
    val_ratio: Mapped[float] = mapped_column(Float, default=0.2, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, default=42, nullable=False)
    optimizer: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lr0: Mapped[float | None] = mapped_column(Float, nullable=True)
    patience: Mapped[int | None] = mapped_column(Integer, nullable=True)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    command_args_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    command_preview: Mapped[str | None] = mapped_column(Text, nullable=True)
    export_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    data_yaml_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    logs_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    best_model_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_model_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress_epoch: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    progress_total_epochs: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    progress_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    stop_requested: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    dataset: Mapped[Dataset] = relationship(back_populates="training_tasks")
    profile: Mapped[TrainingProfile | None] = relationship()
    model_versions: Mapped[list["ModelVersion"]] = relationship(back_populates="training_task")


class ModelVersion(Base, TimestampMixin):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("training_task_id", "artifact_type", name="uq_model_version_training_artifact"),
        CheckConstraint("source IN ('training_task', 'exported')", name="ck_model_version_source"),
        CheckConstraint("artifact_type IN ('best', 'last', 'onnx')", name="ck_model_version_artifact_type"),
        CheckConstraint("format IN ('pt', 'onnx')", name="ck_model_version_format"),
        CheckConstraint("status IN ('active', 'archived')", name="ck_model_version_status"),
        CheckConstraint("task_type IN ('detect', 'segment')", name="ck_model_version_task_type"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(64), default="v1", nullable=False)
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("datasets.id", ondelete="SET NULL"), index=True, nullable=True)
    training_task_id: Mapped[str | None] = mapped_column(ForeignKey("training_tasks.id", ondelete="SET NULL"), index=True, nullable=True)
    source_model_id: Mapped[str | None] = mapped_column(ForeignKey("model_versions.id", ondelete="SET NULL"), index=True, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="training_task", nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(32), default="best", nullable=False)
    format: Mapped[str] = mapped_column(String(16), default="pt", nullable=False)
    task_type: Mapped[str] = mapped_column(String(16), default="detect", nullable=False)
    engine_type: Mapped[str] = mapped_column(String(32), default="ultralytics", nullable=False)
    model_path: Mapped[str] = mapped_column(Text, nullable=False)
    base_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="active", index=True, nullable=False)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    map50: Mapped[float | None] = mapped_column(Float, nullable=True)
    map50_95: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    archived_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    dataset: Mapped[Dataset | None] = relationship(back_populates="model_versions")
    training_task: Mapped[TrainingTask | None] = relationship(back_populates="model_versions", foreign_keys=[training_task_id])
    source_model: Mapped["ModelVersion | None"] = relationship(remote_side=[id], foreign_keys=[source_model_id])
