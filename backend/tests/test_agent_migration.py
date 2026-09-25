from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text


def test_existing_runs_migrate_to_safe_retry_defaults(tmp_path):
    backend = Path(__file__).resolve().parents[1]
    config = Config(str(backend / "alembic.ini"))
    config.set_main_option("script_location", str(backend / "alembic"))
    url = f"sqlite:///{tmp_path / 'legacy.db'}"
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "0015_training_export_stats")
    engine = create_engine(url)
    try:
        with engine.begin() as connection:
            connection.execute(text("""INSERT INTO agent_sessions (id,title,created_at,updated_at)
                VALUES ('asess_old','old','2026-01-01','2026-01-01')"""))
            connection.execute(text("""INSERT INTO agent_runs
                (id,session_id,status,provider,model,created_at,updated_at)
                VALUES ('arun_old','asess_old','failed','mock','mock','2026-01-01','2026-01-01')"""))
        command.upgrade(config, "head")
        with engine.connect() as connection:
            row = connection.execute(text("SELECT read_only, context_json, inference_steps_json FROM agent_runs WHERE id='arun_old'")).one()
            assert row.read_only == 1
            assert row.context_json == "{}"
            assert row.inference_steps_json == "[]"
            for table in ("agent_runs", "agent_sessions"):
                binding = connection.execute(text(f"SELECT profile_id, profile_version FROM {table}")).one()
                assert binding.profile_id == "global" and binding.profile_version == 1
        command.downgrade(config, "0015_training_export_stats")
        command.upgrade(config, "head")
        with engine.connect() as connection:
            assert connection.execute(text("SELECT count(*) FROM agent_runs")).scalar_one() == 1
    finally:
        engine.dispose()
