from __future__ import annotations

import threading

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.models import TrainingTask


class TrainingQueue:
    """A single-process FIFO queue for local training subprocesses."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session_factory: sessionmaker[Session] | None = None
        self._active_task_id: str | None = None

    def configure(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def submit(self, task_id: str) -> None:
        self._pump()

    def stop_pending(self, task_id: str) -> None:
        self._pump()

    def on_finished(self, task_id: str) -> None:
        with self._lock:
            if self._active_task_id == task_id:
                self._active_task_id = None
        self._pump()

    def recover_orphaned(self) -> None:
        if self._session_factory is None:
            return
        from app.core.time import utc_now

        with self._session_factory() as session:
            tasks = list(session.scalars(select(TrainingTask).where(TrainingTask.status == "running")))
            for task in tasks:
                task.status = "failed"
                task.error_message = "Training process was interrupted by a service restart."
                task.finished_at = utc_now()
            if tasks:
                session.commit()
        self._pump()

    def _pump(self) -> None:
        with self._lock:
            if self._active_task_id is not None or self._session_factory is None:
                return
            session_factory = self._session_factory
            with session_factory() as session:
                next_id = session.scalar(
                    select(TrainingTask.id)
                    .where(TrainingTask.status == "pending")
                    .order_by(TrainingTask.created_at.asc(), TrainingTask.id.asc())
                    .limit(1)
                )
            if next_id is None:
                return
            self._active_task_id = next_id
        from app.training.runtime.runner import TrainingRunner

        threading.Thread(target=TrainingRunner(session_factory, self).run, args=(next_id,), daemon=True).start()


training_queue = TrainingQueue()
