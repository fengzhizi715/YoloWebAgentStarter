from __future__ import annotations

import json
from pathlib import Path

from app.core.models import Dataset, TrainingTask
from app.training.artifacts.checkpoints import checkpoint_paths
from app.training.observability.log_store import TrainingLogStore
from app.training.observability.metrics import TrainingMetricsParser


def write_training_summary(task: TrainingTask, dataset: Dataset) -> dict:
    checkpoint_data = checkpoint_paths(task.run_dir or "")
    log_store = TrainingLogStore(task.logs_path or "train.log")
    text = log_store.read()
    metrics = dict(task.metrics_json or {})
    history: list[dict[str, float]] = []
    if task.run_dir:
        parser = TrainingMetricsParser()
        results_path = Path(task.run_dir) / "results.csv"
        metrics.update(parser.parse_results(results_path))
        history = parser.parse_history(results_path)
    risks: list[str] = []
    if task.status != "completed":
        risks.append(f"training_status_{task.status}")
    if checkpoint_data["best"] is None:
        risks.append("best_checkpoint_missing")
    if checkpoint_data["last"] is None:
        risks.append("last_checkpoint_missing")
    if not metrics:
        risks.append("metrics_missing")
    summary = {
        "task_id": task.id,
        "status": task.status,
        "training_config": task.config_json or {},
        "dataset": {"id": dataset.id, "name": dataset.name, "task_type": dataset.task_type},
        "progress": {"epoch": task.progress_epoch, "total_epochs": task.progress_total_epochs, "percent": task.progress_percent},
        "metrics": {**metrics, "history": history},
        "checkpoints": checkpoint_data,
        "log_summary": {"line_count": len(text.splitlines()), "tail": text.splitlines()[-20:]},
        "risks": risks,
        "next_steps": ["Review the validation metrics and keep the best.pt checkpoint."] if task.status == "completed" else ["Inspect the training log and configuration before retrying."],
    }
    if task.summary_path:
        path = Path(task.summary_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
