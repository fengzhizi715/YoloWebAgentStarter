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
