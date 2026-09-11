"""Add an explicit execution claim state for Agent approvals."""

from collections.abc import Sequence

from alembic import op


revision: str = "0014_agent_approval_execution_state"
down_revision: str | None = "0013_agent_mvp"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


_STATUS_CHECK = "status IN ('pending', 'approved', 'executing', 'rejected', 'expired', 'executed')"
_PREVIOUS_STATUS_CHECK = "status IN ('pending', 'approved', 'rejected', 'expired', 'executed')"


def upgrade() -> None:
    with op.batch_alter_table("agent_approvals") as batch:
        batch.drop_constraint("ck_agent_approval_status", type_="check")
        batch.create_check_constraint("ck_agent_approval_status", _STATUS_CHECK)


def downgrade() -> None:
    with op.batch_alter_table("agent_approvals") as batch:
        batch.drop_constraint("ck_agent_approval_status", type_="check")
        batch.create_check_constraint("ck_agent_approval_status", _PREVIOUS_STATUS_CHECK)
