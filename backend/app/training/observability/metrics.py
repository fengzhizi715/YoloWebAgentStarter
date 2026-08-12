from __future__ import annotations

import csv
import re
from pathlib import Path


class TrainingMetricsParser:
    def parse_progress_line(self, line: str) -> dict[str, float]:
        match = re.search(r"(\d+)\s*/\s*(\d+)", line)
        if not match:
            return {}
        return {"epoch": float(match.group(1)), "total_epochs": float(match.group(2))}

    def parse_results(self, path: str | Path) -> dict[str, float]:
        csv_path = Path(path)
        if not csv_path.is_file():
            return {}
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return {}
        row = {key.strip().lower(): value for key, value in rows[-1].items()}
        aliases = {
            "precision": ("metrics/precision(b)", "metrics/precision(p)", "precision"),
            "recall": ("metrics/recall(b)", "metrics/recall(p)", "recall"),
            "map50": ("metrics/map50(b)", "metrics/map50(p)", "map50"),
            "map50_95": ("metrics/map50-95(b)", "metrics/map50-95(p)", "map50_95"),
        }
        result: dict[str, float] = {}
        for output_name, names in aliases.items():
            for name in names:
                value = row.get(name)
                if value not in (None, ""):
                    try:
                        result[output_name] = float(value)
                    except ValueError:
                        pass
                    break
        return result

    def parse_history(self, path: str | Path) -> list[dict[str, float]]:
        csv_path = Path(path)
        if not csv_path.is_file():
            return []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        history: list[dict[str, float]] = []
        for row in rows:
            values = {key.strip().lower(): value for key, value in row.items()}
            point: dict[str, float] = {}
            for output, aliases in {"epoch": ("epoch",), "precision": ("metrics/precision(b)", "metrics/precision(p)", "precision"), "recall": ("metrics/recall(b)", "metrics/recall(p)", "recall"), "map50": ("metrics/map50(b)", "metrics/map50(p)", "map50"), "map50_95": ("metrics/map50-95(b)", "metrics/map50-95(p)", "map50_95")}.items():
                for alias in aliases:
                    try:
                        if values.get(alias) not in (None, ""):
                            point[output] = float(values[alias])
                            break
                    except ValueError:
                        continue
            if point:
                history.append(point)
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
