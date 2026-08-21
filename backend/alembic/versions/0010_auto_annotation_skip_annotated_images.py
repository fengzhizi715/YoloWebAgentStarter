"""Skip existing annotations by default for local auto annotation."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0010_auto_annotation_skip_annotated_images"
down_revision: str | None = "0009_auto_annotation_tasks"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "auto_annotation_tasks",
        sa.Column("skip_annotated_images", sa.Boolean(), nullable=False, server_default=sa.true()),
    )


def downgrade() -> None:
    op.drop_column("auto_annotation_tasks", "skip_annotated_images")
