"""Add managed local model quick-test records.

Revision ID: 0005_model_test_records
Revises: 0004_community_v2_tasks
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_model_test_records"
down_revision: str | None = "0004_community_v2_tasks"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_test_records",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("model_id", sa.String(length=64), nullable=False),
        sa.Column("file_name", sa.String(length=512), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column("result_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["model_id"], ["model_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_test_records_model_id", "model_test_records", ["model_id"])


def downgrade() -> None:
    op.drop_index("ix_model_test_records_model_id", table_name="model_test_records")
    op.drop_table("model_test_records")
