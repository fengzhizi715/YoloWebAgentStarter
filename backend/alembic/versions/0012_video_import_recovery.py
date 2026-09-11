"""Add recoverable checkpoints and explicit start confirmation to video imports."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0012_video_import_recovery"
down_revision: str | None = "0011_video_import_tasks"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("video_import_tasks") as batch:
        batch.add_column(sa.Column("start_requested", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch.add_column(sa.Column("checkpoint_next_frame_index", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("checkpoint_next_sample_at", sa.Float(), nullable=True))
        batch.add_column(sa.Column("output_bytes", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    with op.batch_alter_table("video_import_tasks") as batch:
        batch.drop_column("output_bytes")
        batch.drop_column("checkpoint_next_sample_at")
        batch.drop_column("checkpoint_next_frame_index")
        batch.drop_column("start_requested")
