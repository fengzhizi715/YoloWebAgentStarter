from __future__ import annotations

from pathlib import Path


def checkpoint_paths(run_dir: str | Path) -> dict[str, str | None]:
    root = Path(run_dir).expanduser().resolve()
    values: dict[str, str | None] = {}
    for name in ("best", "last"):
        path = root / "weights" / f"{name}.pt"
        values[name] = str(path) if path.is_file() else None
    return values
