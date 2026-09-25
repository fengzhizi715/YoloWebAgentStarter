"""Bind sessions and runs to versioned built-in assistant profiles."""

from alembic import op
import sqlalchemy as sa

revision = "0018_agent_profiles"
down_revision = "0017_agent_inference_steps"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ("agent_sessions", "agent_runs"):
        # Never infer tighter permissions for historical runs/approvals.
        op.add_column(table, sa.Column("profile_id", sa.String(32), nullable=False, server_default="global"))
        op.add_column(table, sa.Column("profile_version", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("agent_sessions", sa.Column("context_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")))


def downgrade() -> None:
    op.drop_column("agent_sessions", "context_json")
    for table in ("agent_runs", "agent_sessions"):
        op.drop_column(table, "profile_version")
        op.drop_column(table, "profile_id")
