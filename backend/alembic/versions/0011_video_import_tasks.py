"""Add Starter-owned video import tasks and frame provenance."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0011_video_import_tasks"
down_revision: str | None = "0010_auto_annotation_skip_annotated_images"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("image_items") as batch:
        batch.add_column(sa.Column("source_type", sa.String(length=32), nullable=False, server_default="image"))
        batch.add_column(sa.Column("source_file", sa.String(length=512), nullable=True))
        batch.add_column(sa.Column("source_group_id", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("source_video_task_id", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("source_checksum", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("frame_index", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("timestamp", sa.Float(), nullable=True))
    op.create_index("ix_image_items_source_group_id", "image_items", ["source_group_id"])
    op.create_index("ix_image_items_source_video_task_id", "image_items", ["source_video_task_id"])

    op.create_table(
        "video_import_tasks",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("task_type", sa.String(length=16), nullable=False),
        sa.Column("split", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("config_json", sa.JSON(), nullable=False),
        sa.Column("source_file_name", sa.String(length=512), nullable=False),
        sa.Column("source_storage_name", sa.String(length=512), nullable=False),
        sa.Column("source_checksum", sa.String(length=64), nullable=False),
        sa.Column("video_info_json", sa.JSON(), nullable=False),
        sa.Column("total_images", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("generated_images", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("progress_percent", sa.Float(), nullable=False, server_default="0"),
        sa.Column("logs_path", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed')", name="ck_video_import_task_status"),
        sa.CheckConstraint("task_type IN ('detect', 'segment', 'obb', 'classify')", name="ck_video_import_task_type"),
        sa.CheckConstraint("split IN ('train', 'val', 'test')", name="ck_video_import_task_split"),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_video_import_tasks_dataset_id", "video_import_tasks", ["dataset_id"])
    op.create_index("ix_video_import_tasks_status", "video_import_tasks", ["status"])


def downgrade() -> None:
    op.drop_index("ix_video_import_tasks_status", table_name="video_import_tasks")
    op.drop_index("ix_video_import_tasks_dataset_id", table_name="video_import_tasks")
    op.drop_table("video_import_tasks")
    op.drop_index("ix_image_items_source_video_task_id", table_name="image_items")
    op.drop_index("ix_image_items_source_group_id", table_name="image_items")
    with op.batch_alter_table("image_items") as batch:
        batch.drop_column("timestamp")
        batch.drop_column("frame_index")
        batch.drop_column("source_checksum")
        batch.drop_column("source_video_task_id")
        batch.drop_column("source_group_id")
        batch.drop_column("source_file")
        batch.drop_column("source_type")
