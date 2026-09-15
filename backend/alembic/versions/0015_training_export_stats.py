"""Persist per-task YOLO export statistics for the training summary.

Revision ID: 0015_training_export_stats
Revises: 0014_agent_approval_execution_state
Create Date: 2026-09-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0015_training_export_stats"
down_revision: str | None = "0014_agent_approval_execution_state"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "training_tasks",
        sa.Column("export_stats_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )


def downgrade() -> None:
    op.drop_column("training_tasks", "export_stats_json")
