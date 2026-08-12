"""Add lifecycle state to local evaluation records.

Revision ID: 0007_model_evaluation_status
Revises: 0006_model_evaluation_records
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_model_evaluation_status"
down_revision: str | None = "0006_model_evaluation_records"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("model_evaluation_records", sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"))
    op.add_column("model_evaluation_records", sa.Column("error_message", sa.Text(), nullable=True))
    op.create_index("ix_model_evaluation_records_status", "model_evaluation_records", ["status"])


def downgrade() -> None:
    op.drop_index("ix_model_evaluation_records_status", table_name="model_evaluation_records")
    op.drop_column("model_evaluation_records", "error_message")
    op.drop_column("model_evaluation_records", "status")
