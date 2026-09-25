"""Persist Agent context and execution mode for stable retries."""

import sqlalchemy as sa
from alembic import op

revision = "0016_agent_run_context"
down_revision = "0015_training_export_stats"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Historical runs did not record their mode; retry them conservatively.
    op.add_column("agent_runs", sa.Column("read_only", sa.Boolean(), nullable=False, server_default=sa.true()))
    op.add_column("agent_runs", sa.Column("context_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")))


def downgrade() -> None:
    op.drop_column("agent_runs", "context_json")
    op.drop_column("agent_runs", "read_only")
