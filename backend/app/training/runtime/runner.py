from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from app.core.models import TrainingTask
from app.core.time import utc_now
from app.core.storage import Storage
from app.training.artifacts.checkpoints import checkpoint_paths
from app.training.observability.log_store import TrainingLogStore
from app.training.observability.metrics import TrainingMetricsParser
from app.training.observability.summary import write_training_summary
from app.training.runtime.process_registry import process_registry


def resolve_yolo_command_prefix() -> list[str]:
    override = os.getenv("YWA_YOLO_EXECUTABLE", "").strip()
    if override:
        return [override]
    repository_root = Path(__file__).resolve().parents[4]
    candidate = repository_root / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("yolo.exe" if os.name == "nt" else "yolo")
    if candidate.is_file():
        return [str(candidate)]
    found = shutil.which("yolo")
    if found:
        return [found]
    return [sys.executable, "-c", "from ultralytics.cfg import entrypoint; raise SystemExit(entrypoint())"]


class TrainingRunner:
    def __init__(self, session_factory: sessionmaker[Session], queue, storage: Storage | None = None) -> None:
        self.session_factory = session_factory
        self.queue = queue
        self.storage = storage
        self.metrics_parser = TrainingMetricsParser()

    def run(self, task_id: str) -> None:
        process: subprocess.Popen[str] | None = None
        try:
            task = self._claim(task_id)
            if task is None:
                return
            log_store = TrainingLogStore(task.logs_path or "train.log")
            log_store.append(f"Training started for task {task.id}.")
            command = self._command(task)
            log_store.append("Command: " + " ".join(command))
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    shell=False,
                    start_new_session=os.name == "posix",
                )
                process_registry.register(task_id, process)
                if self._stop_requested(task_id):
                    process_registry.request_stop(task_id)
                assert process.stdout is not None
                for line in process.stdout:
                    clean_line = line.rstrip("\n")
                    log_store.append(clean_line)
                    self._update_progress(task_id, clean_line)
                return_code = process.wait()
            except Exception as exc:
                log_store.append(f"Training process failed to start or read output: {exc}")
                self._finish(task_id, -1, log_store, process_error=str(exc))
                return
            self._finish(task_id, return_code, log_store)
        finally:
            process_registry.unregister(task_id)
            self.queue.on_finished(task_id)

    def _claim(self, task_id: str) -> TrainingTask | None:
        with self.session_factory() as session:
            task = session.get(TrainingTask, task_id)
            if task is None or task.status != "pending":
                return None
            if task.stop_requested:
                task.status = "stopped"
                task.finished_at = utc_now()
                task.error_message = "Training stopped before the process started."
                session.commit()
                return None
            task.status = "running"
            task.started_at = utc_now()
            task.progress_total_epochs = task.epochs
            task.progress_epoch = 0
            task.progress_percent = 0.0
            session.commit()
            session.refresh(task)
            return task

    def _command(self, task: TrainingTask) -> list[str]:
        args = [str(item) for item in (task.command_args_json or [])]
        if not args:
            raise RuntimeError("Training command is missing.")
        if args[0] == "yolo":
            return [*resolve_yolo_command_prefix(), *args[1:]]
        return args

    def _stop_requested(self, task_id: str) -> bool:
        if process_registry.stop_requested(task_id):
            return True
        with self.session_factory() as session:
            task = session.get(TrainingTask, task_id)
            return bool(task and task.stop_requested)

    def _update_progress(self, task_id: str, line: str) -> None:
        progress = self.metrics_parser.parse_progress_line(line)
        if not progress:
            return
        with self.session_factory() as session:
            task = session.get(TrainingTask, task_id)
            if task is None or task.status != "running":
                return
            total = int(progress.get("total_epochs", task.epochs))
            epoch = min(int(progress.get("epoch", 0)), total)
            task.progress_total_epochs = max(total, 1)
            task.progress_epoch = max(epoch, 0)
            task.progress_percent = round(task.progress_epoch / task.progress_total_epochs * 100, 2)
            session.commit()

    def _finish(self, task_id: str, return_code: int, log_store: TrainingLogStore, process_error: str | None = None) -> None:
        with self.session_factory() as session:
            task = session.get(TrainingTask, task_id)
            if task is None:
                return
            stopped = self._stop_requested(task_id)
            artifacts = checkpoint_paths(task.run_dir or "")
            task.best_model_path = artifacts["best"]
            task.last_model_path = artifacts["last"]
            if task.run_dir:
                task.metrics_json = self.metrics_parser.parse_results(Path(task.run_dir) / "results.csv")
            task.finished_at = utc_now()
            if stopped:
                task.status = "stopped"
                task.error_message = "Training stopped by user."
                log_store.append(task.error_message)
            elif return_code == 0 and artifacts["best"] and artifacts["last"]:
                task.status = "completed"
                task.progress_epoch = task.progress_total_epochs or task.epochs
                task.progress_total_epochs = task.progress_total_epochs or task.epochs
                task.progress_percent = 100.0
                task.error_message = None
                log_store.append("Training completed with best.pt and last.pt.")
            else:
                task.status = "failed"
                task.error_message = process_error or (
                    "Training completed without both best.pt and last.pt."
                    if return_code == 0
                    else f"YOLO training exited with code {return_code}."
                )
                log_store.append(task.error_message)
            session.commit()
            dataset = task.dataset
            if task.status == "completed" and self.storage is not None:
                try:
                    from app.models.service import ModelService

                    registered = ModelService(self.storage).register_training_artifacts(session, task)
                    log_store.append(f"Registered {len(registered)} model artifacts.")
                except Exception as exc:
                    log_store.append(f"Model artifact registration failed: {exc}")
            write_training_summary(task, dataset)
            session.commit()
