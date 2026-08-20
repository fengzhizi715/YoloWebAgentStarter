from __future__ import annotations

import threading

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.models import AutoAnnotationTask
from app.core.storage import Storage
from app.core.time import utc_now


class AutoAnnotationQueue:
    """Single-process FIFO queue matching the upstream task/progress workflow."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session_factory: sessionmaker[Session] | None = None
        self._storage: Storage | None = None
        self._active_task_id: str | None = None

    def configure(self, session_factory: sessionmaker[Session], storage: Storage) -> None:
        self._session_factory = session_factory
        self._storage = storage

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
        with self._session_factory() as session:
            tasks = list(session.scalars(select(AutoAnnotationTask).where(AutoAnnotationTask.status == "running")))
            for task in tasks:
                task.status = "failed"
                task.error_message = "Auto annotation was interrupted by a service restart."
                task.finished_at = utc_now()
            if tasks:
                session.commit()
        self._pump()

    def _pump(self) -> None:
        with self._lock:
            if self._active_task_id is not None or self._session_factory is None or self._storage is None:
                return
            session_factory = self._session_factory
            with session_factory() as session:
                next_id = session.scalar(
                    select(AutoAnnotationTask.id)
                    .where(AutoAnnotationTask.status == "pending")
                    .order_by(AutoAnnotationTask.created_at.asc(), AutoAnnotationTask.id.asc())
                    .limit(1)
                )
            if next_id is None:
                return
            self._active_task_id = next_id
        from app.auto_annotation.runner import AutoAnnotationRunner

        threading.Thread(
            target=AutoAnnotationRunner(session_factory, self, self._storage).run,
            args=(next_id,),
            daemon=True,
        ).start()


auto_annotation_queue = AutoAnnotationQueue()
