"""Add local auto-annotation task records."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0009_auto_annotation_tasks"
down_revision: str | None = "0008_evaluation_artifacts"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "auto_annotation_tasks",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("model_id", sa.String(length=64), nullable=False),
        sa.Column("task_type", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("clean_old_annotations", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.25"),
        sa.Column("iou", sa.Float(), nullable=False, server_default="0.45"),
        sa.Column("class_mapping", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("total_images", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processed_images", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_annotations", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("skipped_images", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("progress_percent", sa.Float(), nullable=False, server_default="0"),
        sa.Column("logs_path", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("stop_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'stopped')", name="ck_auto_annotation_task_status"),
        sa.CheckConstraint("task_type IN ('detect', 'segment', 'obb', 'classify')", name="ck_auto_annotation_task_type"),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_id"], ["model_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_auto_annotation_tasks_dataset_id", "auto_annotation_tasks", ["dataset_id"])
    op.create_index("ix_auto_annotation_tasks_model_id", "auto_annotation_tasks", ["model_id"])
    op.create_index("ix_auto_annotation_tasks_status", "auto_annotation_tasks", ["status"])


def downgrade() -> None:
    op.drop_index("ix_auto_annotation_tasks_status", table_name="auto_annotation_tasks")
    op.drop_index("ix_auto_annotation_tasks_model_id", table_name="auto_annotation_tasks")
    op.drop_index("ix_auto_annotation_tasks_dataset_id", table_name="auto_annotation_tasks")
    op.drop_table("auto_annotation_tasks")
