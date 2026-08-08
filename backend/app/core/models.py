from __future__ import annotations

from datetime import datetime

from sqlalchemy import CheckConstraint, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
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

