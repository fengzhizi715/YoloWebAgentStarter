"""Add managed paths and lifecycle timestamps to evaluation records.

Revision ID: 0008_evaluation_artifacts
Revises: 0007_model_evaluation_status
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_evaluation_artifacts"
down_revision: str | None = "0007_model_evaluation_status"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("model_evaluation_records", sa.Column("export_path", sa.Text(), nullable=True))
    op.add_column("model_evaluation_records", sa.Column("data_path", sa.Text(), nullable=True))
    op.add_column("model_evaluation_records", sa.Column("run_dir", sa.Text(), nullable=True))
    op.add_column("model_evaluation_records", sa.Column("logs_path", sa.Text(), nullable=True))
    op.add_column("model_evaluation_records", sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("model_evaluation_records", sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("model_evaluation_records", "finished_at")
    op.drop_column("model_evaluation_records", "started_at")
    op.drop_column("model_evaluation_records", "logs_path")
    op.drop_column("model_evaluation_records", "run_dir")
    op.drop_column("model_evaluation_records", "data_path")
    op.drop_column("model_evaluation_records", "export_path")
