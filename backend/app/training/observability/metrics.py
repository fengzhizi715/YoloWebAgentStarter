from __future__ import annotations

import csv
import re
from pathlib import Path


_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "precision": ("metrics/precision(b)", "metrics/precision(p)", "precision"),
    "recall": ("metrics/recall(b)", "metrics/recall(p)", "recall"),
    "map50": ("metrics/map50(b)", "metrics/map50(p)", "map50"),
    "map50_95": ("metrics/map50-95(b)", "metrics/map50-95(p)", "map50_95"),
}


def _read_rows(path: str | Path) -> list[dict[str, str | None]]:
    csv_path = Path(path)
    if not csv_path.is_file():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalise_row(row: dict[str, str | None]) -> dict[str, str]:
    return {
        key.strip().lower(): value.strip()
        for key, value in row.items()
        if isinstance(key, str) and isinstance(value, str) and value.strip()
    }


def _number(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


class TrainingMetricsParser:
    def parse_progress_line(self, line: str) -> dict[str, float]:
        match = re.search(r"(\d+)\s*/\s*(\d+)", line)
        progress: dict[str, float] = {}
        if match:
            progress.update({"epoch": float(match.group(1)), "total_epochs": float(match.group(2))})

        # Ultralytics' tqdm output reports the current batch throughput. Keep
        # this derived value as the live batch duration; results.csv remains
        # the source of truth for epoch metrics and cumulative training time.
        speed = re.search(r"([\d.]+)\s*(?:it|iter)/s", line, re.IGNORECASE)
        if speed:
            it_per_second = _number(speed.group(1))
            if it_per_second and it_per_second > 0:
                progress["speed_it_per_sec"] = it_per_second
                progress["batch_time_seconds"] = 1.0 / it_per_second
        else:
            seconds_per_batch = re.search(r"([\d.]+)\s*s/(?:it|iter)", line, re.IGNORECASE)
            if seconds_per_batch:
                batch_time = _number(seconds_per_batch.group(1))
                if batch_time and batch_time > 0:
                    progress["batch_time_seconds"] = batch_time
                    progress["speed_it_per_sec"] = 1.0 / batch_time
        return progress

    def _parse_row(self, row: dict[str, str | None], previous_elapsed: float | None = None) -> dict[str, float]:
        values = _normalise_row(row)
        point: dict[str, float] = {}

        for output_name, aliases in _METRIC_ALIASES.items():
            for alias in aliases:
                value = _number(values.get(alias))
                if value is not None:
                    point[output_name] = value
                    break

        epoch = _number(values.get("epoch"))
        if epoch is not None:
            point["epoch"] = epoch

        elapsed = _number(values.get("time"))
        if elapsed is not None:
            point["elapsed_seconds"] = elapsed
            if previous_elapsed is not None:
                point["epoch_time_seconds"] = max(0.0, elapsed - previous_elapsed)

        train_losses: list[float] = []
        for key, raw_value in values.items():
            value = _number(raw_value)
            if value is None:
                continue
            if key.startswith(("train/", "val/")) and key.endswith("_loss"):
                canonical = key.replace("/", "_")
                point[canonical] = value
                if key.startswith("train/"):
                    train_losses.append(value)
            elif key.startswith("lr/"):
                point[key.replace("/", "_")] = value

        # Ultralytics exposes component losses in results.csv. A compact
        # aggregate makes the current-loss card useful for every task type,
        # including classification where the component name differs.
        direct_loss = _number(values.get("train/loss"))
        if direct_loss is None:
            direct_loss = _number(values.get("loss"))
        if direct_loss is not None:
            point["train_loss"] = direct_loss
            point["loss"] = direct_loss
        elif train_losses:
            point["loss"] = round(sum(train_losses), 6)
        return point

    def parse_results(self, path: str | Path) -> dict[str, float]:
        rows = _read_rows(path)
        if not rows:
            return {}
        previous_elapsed = None
        for row in rows[:-1]:
            previous_elapsed = _number(_normalise_row(row).get("time"))
        return self._parse_row(rows[-1], previous_elapsed)

    def parse_history(self, path: str | Path) -> list[dict[str, float]]:
        rows = _read_rows(path)
        history: list[dict[str, float]] = []
        previous_elapsed = None
        for row in rows:
            point = self._parse_row(row, previous_elapsed)
            if point:
                history.append(point)
            previous_elapsed = _number(_normalise_row(row).get("time"))
        return history

    def parse_validation_text(self, text: str, task_type: str) -> dict[str, float]:
        """Community extension of the upstream metrics parser for standalone val."""

        clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
        number = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        if task_type == "classify":
            matches = list(re.finditer(rf"^\s*all\s+({number})\s+({number})\s*$", clean, re.MULTILINE))
            if not matches:
                return {}
            top1, top5 = (float(value) for value in matches[-1].groups())
            return {"top1": top1, "top5": top5}
        metric_count = 8 if task_type == "segment" else 4
        values = r"\s+".join([rf"({number})"] * metric_count)
        matches = list(re.finditer(rf"^\s*all\s+\d+\s+\d+\s+{values}\s*$", clean, re.MULTILINE))
        if not matches:
            return {}
        parsed = [float(value) for value in matches[-1].groups()]
        metrics = dict(zip(("precision", "recall", "map50", "map50_95"), parsed[:4], strict=True))
        if task_type == "segment":
            metrics.update(dict(zip(("mask_precision", "mask_recall", "mask_map50", "mask_map50_95"), parsed[4:], strict=True)))
        return metrics
