"""Add the Starter managed model tables.

Revision ID: 0003_models_baseline
Revises: 0002_training_baseline
Create Date: 2026-08-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_models_baseline"
down_revision: str | None = "0002_training_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False, server_default="v1"),
        sa.Column("dataset_id", sa.String(length=64), nullable=True),
        sa.Column("training_task_id", sa.String(length=64), nullable=True),
        sa.Column("source_model_id", sa.String(length=64), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False, server_default="training_task"),
        sa.Column("artifact_type", sa.String(length=32), nullable=False, server_default="best"),
        sa.Column("format", sa.String(length=16), nullable=False, server_default="pt"),
        sa.Column("task_type", sa.String(length=16), nullable=False, server_default="detect"),
        sa.Column("engine_type", sa.String(length=32), nullable=False, server_default="ultralytics"),
        sa.Column("model_path", sa.Text(), nullable=False),
        sa.Column("base_model", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="active"),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("map50", sa.Float(), nullable=True),
        sa.Column("map50_95", sa.Float(), nullable=True),
        sa.Column("metrics_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("archived_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("source IN ('training_task', 'exported')", name="ck_model_version_source"),
        sa.CheckConstraint("artifact_type IN ('best', 'last', 'onnx')", name="ck_model_version_artifact_type"),
        sa.CheckConstraint("format IN ('pt', 'onnx')", name="ck_model_version_format"),
        sa.CheckConstraint("status IN ('active', 'archived')", name="ck_model_version_status"),
        sa.CheckConstraint("task_type IN ('detect', 'segment')", name="ck_model_version_task_type"),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["training_task_id"], ["training_tasks.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["source_model_id"], ["model_versions.id"], ondelete="SET NULL"),
        sa.UniqueConstraint("training_task_id", "artifact_type", name="uq_model_version_training_artifact"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_versions_dataset_id", "model_versions", ["dataset_id"])
    op.create_index("ix_model_versions_training_task_id", "model_versions", ["training_task_id"])
    op.create_index("ix_model_versions_source_model_id", "model_versions", ["source_model_id"])
    op.create_index("ix_model_versions_status", "model_versions", ["status"])


def downgrade() -> None:
    op.drop_table("model_versions")
