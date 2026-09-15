from __future__ import annotations

import json
from pathlib import Path

from app.core.models import Dataset, TrainingTask
from app.training.artifacts.checkpoints import checkpoint_paths
from app.training.observability.log_store import TrainingLogStore
from app.training.observability.metrics import TrainingMetricsParser

_MIN_VALIDATION_IMAGES = 5
_MIN_VALIDATION_RATIO = 0.1


def _validation_split_too_small(export_stats: dict) -> bool:
    """Flag a validation split that cannot produce a meaningful mAP.

    The persisted split is reused as-is, so a partially annotated dataset can
    leave only a handful of labelled validation images.  Surface it instead of
    presenting the resulting metric as trustworthy.
    """

    annotated = export_stats.get("annotated_image_counts")
    if not isinstance(annotated, dict):
        return False
    total = sum(value for value in annotated.values() if isinstance(value, (int, float)))
    if total <= 0:
        return False
    val_count = annotated.get("val") or 0
    return val_count < _MIN_VALIDATION_IMAGES or val_count < _MIN_VALIDATION_RATIO * total


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
    timing = {
        key: metrics[key]
        for key in ("elapsed_seconds", "epoch_time_seconds", "batch_time_seconds", "speed_it_per_sec")
        if isinstance(metrics.get(key), (int, float))
    }
    export_stats = dict(task.export_stats_json or {})
    risks: list[str] = []
    if task.status != "completed":
        risks.append(f"training_status_{task.status}")
    if checkpoint_data["best"] is None:
        risks.append("best_checkpoint_missing")
    if checkpoint_data["last"] is None:
        risks.append("last_checkpoint_missing")
    if not metrics:
        risks.append("metrics_missing")
    if _validation_split_too_small(export_stats):
        risks.append("val_split_too_small")
    summary = {
        "task_id": task.id,
        "status": task.status,
        "training_config": task.config_json or {},
        "dataset": {"id": dataset.id, "name": dataset.name, "task_type": dataset.task_type},
        "export_stats": export_stats,
        "progress": {"epoch": task.progress_epoch, "total_epochs": task.progress_total_epochs, "percent": task.progress_percent},
        "metrics": {**metrics, "history": history},
        "timing": timing,
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
