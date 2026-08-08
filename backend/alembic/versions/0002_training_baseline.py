"""Add the Starter local training tables.

Revision ID: 0002_training_baseline
Revises: 0001_starter_baseline
Create Date: 2026-08-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_training_baseline"
down_revision: str | None = "0001_starter_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "training_profiles",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("model_name", sa.String(length=255), nullable=False, server_default="yolo11n.pt"),
        sa.Column("task_type", sa.String(length=16), nullable=False),
        sa.Column("epochs", sa.Integer(), nullable=False, server_default="50"),
        sa.Column("img_size", sa.Integer(), nullable=False, server_default="640"),
        sa.Column("batch_size", sa.Integer(), nullable=False, server_default="16"),
        sa.Column("device", sa.String(length=64), nullable=False, server_default="auto"),
        sa.Column("workers", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("val_ratio", sa.Float(), nullable=False, server_default="0.2"),
        sa.Column("seed", sa.Integer(), nullable=False, server_default="42"),
        sa.Column("optimizer", sa.String(length=64), nullable=True),
        sa.Column("lr0", sa.Float(), nullable=True),
        sa.Column("patience", sa.Integer(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint("task_type IN ('detect', 'segment')", name="ck_training_profile_task_type"),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_training_profiles_dataset_id", "training_profiles", ["dataset_id"])

    op.create_table(
        "training_tasks",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("profile_id", sa.String(length=64), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("task_type", sa.String(length=16), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_path", sa.Text(), nullable=False),
        sa.Column("epochs", sa.Integer(), nullable=False),
        sa.Column("img_size", sa.Integer(), nullable=False),
        sa.Column("batch_size", sa.Integer(), nullable=False),
        sa.Column("device", sa.String(length=64), nullable=False),
        sa.Column("workers", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("val_ratio", sa.Float(), nullable=False, server_default="0.2"),
        sa.Column("seed", sa.Integer(), nullable=False, server_default="42"),
        sa.Column("optimizer", sa.String(length=64), nullable=True),
        sa.Column("lr0", sa.Float(), nullable=True),
        sa.Column("patience", sa.Integer(), nullable=True),
        sa.Column("config_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("command_args_json", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("command_preview", sa.Text(), nullable=True),
        sa.Column("export_path", sa.Text(), nullable=True),
        sa.Column("data_yaml_path", sa.Text(), nullable=True),
        sa.Column("run_dir", sa.Text(), nullable=True),
        sa.Column("logs_path", sa.Text(), nullable=True),
        sa.Column("summary_path", sa.Text(), nullable=True),
        sa.Column("best_model_path", sa.Text(), nullable=True),
        sa.Column("last_model_path", sa.Text(), nullable=True),
        sa.Column("progress_epoch", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("progress_total_epochs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("progress_percent", sa.Float(), nullable=False, server_default="0"),
        sa.Column("metrics_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("stop_requested", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'stopped')", name="ck_training_task_status"),
        sa.CheckConstraint("task_type IN ('detect', 'segment')", name="ck_training_task_type"),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["profile_id"], ["training_profiles.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_training_tasks_dataset_id", "training_tasks", ["dataset_id"])
    op.create_index("ix_training_tasks_profile_id", "training_tasks", ["profile_id"])
    op.create_index("ix_training_tasks_status", "training_tasks", ["status"])


def downgrade() -> None:
    op.drop_table("training_tasks")
    op.drop_table("training_profiles")
