"""Add YoloWebAgent-compatible OBB and classification task families.

Revision ID: 0004_community_v2_tasks
Revises: 0003_models_baseline
Create Date: 2026-08-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_community_v2_tasks"
down_revision: str | None = "0003_models_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TASKS = "'detect', 'segment', 'obb', 'classify'"
_BASE_TASKS = "'detect', 'segment'"


def _replace_task_constraint(table: str, constraint: str, values: str) -> None:
    with op.batch_alter_table(table, recreate="always") as batch:
        batch.drop_constraint(constraint, type_="check")
        batch.create_check_constraint(constraint, f"task_type IN ({values})")


def upgrade() -> None:
    _replace_task_constraint("datasets", "ck_dataset_task_type", _TASKS)
    _replace_task_constraint("training_profiles", "ck_training_profile_task_type", _TASKS)
    _replace_task_constraint("training_tasks", "ck_training_task_type", _TASKS)
    _replace_task_constraint("model_versions", "ck_model_version_task_type", _TASKS)
    with op.batch_alter_table("annotations", recreate="always") as batch:
        batch.add_column(sa.Column("obb", sa.JSON(), nullable=True))
        batch.drop_constraint("ck_annotation_type", type_="check")
        batch.create_check_constraint("ck_annotation_type", "type IN ('bbox', 'polygon', 'obb', 'classify')")


def downgrade() -> None:
    with op.batch_alter_table("annotations", recreate="always") as batch:
        batch.drop_constraint("ck_annotation_type", type_="check")
        batch.drop_column("obb")
        batch.create_check_constraint("ck_annotation_type", "type IN ('bbox', 'polygon')")
    _replace_task_constraint("model_versions", "ck_model_version_task_type", _BASE_TASKS)
    _replace_task_constraint("training_tasks", "ck_training_task_type", _BASE_TASKS)
    _replace_task_constraint("training_profiles", "ck_training_profile_task_type", _BASE_TASKS)
    _replace_task_constraint("datasets", "ck_dataset_task_type", _BASE_TASKS)
