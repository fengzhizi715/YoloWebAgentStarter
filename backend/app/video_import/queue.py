from __future__ import annotations

import threading

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.models import VideoImportTask
from app.core.storage import Storage
from app.training.observability.log_store import TrainingLogStore


class VideoImportQueue:
    """One local worker keeps disk-heavy frame extraction predictable."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session_factory: sessionmaker[Session] | None = None
        self._storage: Storage | None = None
        self._settings: Settings | None = None
        self._active_task_id: str | None = None

    def configure(self, session_factory: sessionmaker[Session], storage: Storage, settings: Settings) -> None:
        self._session_factory = session_factory
        self._storage = storage
        self._settings = settings

    def submit(self, task_id: str) -> None:
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
            tasks = list(session.scalars(select(VideoImportTask).where(VideoImportTask.status == "running")))
            for task in tasks:
                task.status = "pending"
                task.start_requested = True
                task.error_message = None
                TrainingLogStore(task.logs_path or "").append(
                    f"Service restart detected; resuming from frame {task.checkpoint_next_frame_index}."
                )
            if tasks:
                session.commit()
        self._pump()

    def _pump(self) -> None:
        with self._lock:
            if self._active_task_id is not None or self._session_factory is None or self._storage is None or self._settings is None:
                return
            session_factory = self._session_factory
            with session_factory() as session:
                task_id = session.scalar(
                    select(VideoImportTask.id)
                    .where(VideoImportTask.status == "pending", VideoImportTask.start_requested.is_(True))
                    .order_by(VideoImportTask.created_at.asc(), VideoImportTask.id.asc())
                    .limit(1)
                )
            if task_id is None:
                return
            self._active_task_id = task_id
        from app.video_import.runner import VideoImportRunner

        threading.Thread(
            target=VideoImportRunner(session_factory, self, self._storage, self._settings).run,
            args=(task_id,),
            daemon=True,
        ).start()


video_import_queue = VideoImportQueue()
