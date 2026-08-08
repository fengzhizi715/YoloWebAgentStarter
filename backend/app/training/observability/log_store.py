from __future__ import annotations

from pathlib import Path


class TrainingLogStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text)
            if text and not text.endswith("\n"):
                handle.write("\n")

    def read(self, tail: int | None = None) -> str:
        if not self.path.is_file():
            return ""
        lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        if tail is not None and tail > 0:
            lines = lines[-tail:]
        return "\n".join(lines)

    def line_count(self) -> int:
        if not self.path.is_file():
            return 0
        return len(self.path.read_text(encoding="utf-8", errors="replace").splitlines())
