"""Starter baseline: datasets, classes, images, annotations.

Revision ID: 0001_starter_baseline
Revises:
Create Date: 2026-08-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_starter_baseline"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("task_type", sa.String(length=16), nullable=False),
        sa.Column("image_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("class_count", sa.Integer(), nullable=False, server_default="0"),
        *_timestamps(),
        sa.CheckConstraint("task_type IN ('detect', 'segment')", name="ck_dataset_task_type"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_datasets_task_type", "datasets", ["task_type"])

    op.create_table(
        "class_labels",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("class_index", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("color", sa.String(length=16), nullable=False, server_default="#22c55e"),
        *_timestamps(),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dataset_id", "class_index", name="uq_class_label_dataset_index"),
        sa.UniqueConstraint("dataset_id", "name", name="uq_class_label_dataset_name"),
    )
    op.create_index("ix_class_labels_dataset_id", "class_labels", ["dataset_id"])

    op.create_table(
        "image_items",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("file_name", sa.String(length=512), nullable=False),
        sa.Column("storage_name", sa.String(length=512), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("split", sa.String(length=16), nullable=False, server_default="train"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="unannotated"),
        *_timestamps(),
        sa.CheckConstraint("split IN ('train', 'val', 'test')", name="ck_image_item_split"),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dataset_id", "storage_name", name="uq_image_item_dataset_storage"),
    )
    op.create_index("ix_image_items_dataset_id", "image_items", ["dataset_id"])
    op.create_index("ix_image_items_split", "image_items", ["split"])
    op.create_index("ix_image_items_status", "image_items", ["status"])

    op.create_table(
        "annotations",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("image_id", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("class_id", sa.String(length=64), nullable=False),
        sa.Column("type", sa.String(length=16), nullable=False),
        sa.Column("x", sa.Float(), nullable=True),
        sa.Column("y", sa.Float(), nullable=True),
        sa.Column("width", sa.Float(), nullable=True),
        sa.Column("height", sa.Float(), nullable=True),
        sa.Column("polygon", sa.JSON(), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False, server_default="manual"),
        *_timestamps(),
        sa.CheckConstraint("type IN ('bbox', 'polygon')", name="ck_annotation_type"),
        sa.ForeignKeyConstraint(["class_id"], ["class_labels.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["image_id"], ["image_items.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_annotations_class_id", "annotations", ["class_id"])
    op.create_index("ix_annotations_dataset_id", "annotations", ["dataset_id"])
    op.create_index("ix_annotations_image_id", "annotations", ["image_id"])


def downgrade() -> None:
    op.drop_table("annotations")
    op.drop_table("image_items")
    op.drop_table("class_labels")
    op.drop_table("datasets")

