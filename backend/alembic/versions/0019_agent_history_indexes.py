"""Index paginated Agent history without changing existing records."""

from alembic import op

revision = "0019_agent_history_indexes"
down_revision = "0018_agent_profiles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_agent_sessions_updated_id", "agent_sessions", ["updated_at", "id"])
    op.create_index("ix_agent_runs_session_created_id", "agent_runs", ["session_id", "created_at", "id"])
    op.create_index("ix_agent_messages_session_sequence", "agent_messages", ["session_id", "sequence"])


def downgrade() -> None:
    op.drop_index("ix_agent_messages_session_sequence", table_name="agent_messages")
    op.drop_index("ix_agent_runs_session_created_id", table_name="agent_runs")
    op.drop_index("ix_agent_sessions_updated_id", table_name="agent_sessions")
