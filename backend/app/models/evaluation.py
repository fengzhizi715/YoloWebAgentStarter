from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from sqlalchemy import select

from app.core.models import ModelEvaluationRecord, ModelVersion
from app.core.time import utc_now
from app.core.storage import Storage
from app.training.observability.log_store import TrainingLogStore
from app.training.observability.metrics import TrainingMetricsParser
from app.training.runtime.runner import resolve_yolo_command_prefix


def build_evaluation_command(record: ModelEvaluationRecord, model: ModelVersion) -> list[str]:
    if not record.data_path or not record.run_dir:
        raise ValueError("Evaluation export or run directory is missing.")
    run_path = Path(record.run_dir)
    return [
        *resolve_yolo_command_prefix(),
        model.task_type,
        "val",
        f"model={model.model_path}",
        f"data={record.data_path}",
        f"split={record.split}",
        # Keep predictions below the review threshold so the upstream error
        # sample pass can identify low-confidence results after validation.
        f"conf={min(record.confidence, 0.001) if record.confidence > 0 else 0.001}",
        f"iou={record.iou}",
        f"project={run_path.parent}",
        f"name={run_path.name}",
        "plots=True",
        "save_json=True",
        "exist_ok=True",
    ]


def run_evaluation_process(record: ModelEvaluationRecord, model: ModelVersion) -> tuple[int, str, dict[str, float]]:
    command = build_evaluation_command(record, model)
    log_store = TrainingLogStore(record.logs_path or Path(record.run_dir or ".") / "evaluation.log")
    log_store.append("Command: " + " ".join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=False, bufsize=1)
    assert process.stdout is not None
    lines: list[str] = []
    for line in process.stdout:
        clean = line.rstrip("\n")
        lines.append(clean)
        log_store.append(clean)
    return_code = process.wait()
    return return_code, log_store.read(), parse_validation_metrics("\n".join(lines), model.task_type)


def parse_validation_metrics(text: str, task_type: str) -> dict[str, float]:
    return TrainingMetricsParser().parse_validation_text(text, task_type)


class YoloEvaluationRunner:
    """Community port of upstream app/evaluation/runner.py."""

    def __init__(self, session_factory, storage: Storage) -> None:
        self.session_factory = session_factory
        self.storage = storage

    def start_task(self, task_id: str) -> None:
        threading.Thread(target=self.run, args=(task_id,), daemon=True).start()

    def run(self, task_id: str) -> None:
        from app.models.service import ModelService

        with self.session_factory() as session:
            record = session.get(ModelEvaluationRecord, task_id)
            if record is None or record.status != "pending":
                return
            record.status = "running"
            record.started_at = utc_now()
            session.commit()
            ModelService(self.storage).run_evaluation(session, task_id)

    def recover_orphaned(self) -> None:
        """Fail interrupted work and restart pending tasks after a local restart."""

        with self.session_factory() as session:
            running = list(session.scalars(select(ModelEvaluationRecord).where(ModelEvaluationRecord.status == "running")))
            pending_ids = list(session.scalars(select(ModelEvaluationRecord.id).where(ModelEvaluationRecord.status == "pending")))
            for record in running:
                record.status = "failed"
                record.error_message = "Evaluation process was interrupted by a service restart."
                record.finished_at = utc_now()
            if running:
                session.commit()
        for task_id in pending_ids:
            self.start_task(task_id)
