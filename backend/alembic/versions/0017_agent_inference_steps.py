"""Persist actual per-round inference provenance and timing."""

import sqlalchemy as sa
from alembic import op

revision = "0017_agent_inference_steps"
down_revision = "0016_agent_run_context"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_runs", sa.Column("inference_steps_json", sa.JSON(), nullable=False, server_default=sa.text("'[]'")))


def downgrade() -> None:
    op.drop_column("agent_runs", "inference_steps_json")
