"""Add persisted local evaluation reports and error samples.

Revision ID: 0006_model_evaluation_records
Revises: 0005_model_test_records
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_model_evaluation_records"
down_revision: str | None = "0005_model_test_records"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_evaluation_records",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("model_id", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("split", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("iou", sa.Float(), nullable=False),
        sa.Column("result_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_id"], ["model_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_evaluation_records_model_id", "model_evaluation_records", ["model_id"])
    op.create_index("ix_model_evaluation_records_dataset_id", "model_evaluation_records", ["dataset_id"])


def downgrade() -> None:
    op.drop_index("ix_model_evaluation_records_dataset_id", table_name="model_evaluation_records")
    op.drop_index("ix_model_evaluation_records_model_id", table_name="model_evaluation_records")
    op.drop_table("model_evaluation_records")
