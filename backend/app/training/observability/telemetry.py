from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from app.core.models import TrainingTask
from app.training.observability.metrics import TrainingMetricsParser


class TrainingTelemetryService:
    """Refresh an active task from Ultralytics' append-only results.csv.

    Adapted from the upstream training observability flow: task reads refresh
    persisted telemetry while a process is running, instead of waiting for the
    runner's terminal-state metrics parse.
    """

    def __init__(self) -> None:
        self.metrics_parser = TrainingMetricsParser()

    def refresh_task(self, session: Session, task: TrainingTask) -> TrainingTask:
        if task.status != "running" or not task.run_dir:
            return task
        try:
            results_path = Path(task.run_dir) / "results.csv"
            history = self.metrics_parser.parse_history(results_path)
            latest_metrics = self.metrics_parser.parse_results(results_path)
        except (OSError, UnicodeError, ValueError):
            # Ultralytics may be between writes while this request polls it.
            return task
        if not history and not latest_metrics:
            return task

        changed = False
        metrics = dict(task.metrics_json or {})
        for key, value in latest_metrics.items():
            if metrics.get(key) != value:
                metrics[key] = value
                changed = True
        if changed:
            task.metrics_json = metrics

        if history:
            latest_epoch = int(history[-1].get("epoch", -1)) + 1
            total = max(task.epochs, 1)
            epoch = min(max(latest_epoch, 0), total)
            percent = round(epoch / total * 100, 2)
            if task.progress_epoch < epoch:
                task.progress_epoch = epoch
                changed = True
            if task.progress_total_epochs != total:
                task.progress_total_epochs = total
                changed = True
            if task.progress_percent != percent:
                task.progress_percent = percent
                changed = True
        if changed:
            session.commit()
            session.refresh(task)
        return task
