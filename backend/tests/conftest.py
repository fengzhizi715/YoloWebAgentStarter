from __future__ import annotations

import threading
from pathlib import Path
from time import monotonic

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def wait_for_local_workers(timeout: float = 10.0) -> None:
    """Keep worker mocks and their database alive until queued work has drained."""
    deadline = monotonic() + timeout
    while True:
        workers = [
            thread for thread in threading.enumerate()
            if thread.name.startswith(("ywa-training-", "ywa-auto-annotation-"))
        ]
        if not workers:
            return
        for worker in workers:
            worker.join(timeout=max(0.0, deadline - monotonic()))
        if monotonic() >= deadline:
            alive = [worker.name for worker in workers if worker.is_alive()]
            assert not alive, f"Local task workers did not finish: {alive}"


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    # The dependency keeps environment/inference patches installed through
    # worker cleanup; otherwise an approved Agent task can start real YOLO
    # after its test has already restored YWA_YOLO_EXECUTABLE.
    settings = Settings(
        project_root=Path(__file__).resolve().parents[2],
        data_dir=tmp_path / "data",
        import_root=tmp_path / "imports",
        database_url=f"sqlite:///{tmp_path / 'starter.db'}",
    )
    with TestClient(create_app(settings)) as test_client:
        try:
            yield test_client
        finally:
            wait_for_local_workers()
