from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _resolve_path(raw: str | None, default: Path, project_root: Path) -> Path:
    value = Path(raw).expanduser() if raw else default
    if not value.is_absolute():
        value = project_root / value
    return value.resolve()


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    import_root: Path
    database_url: str
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: tuple[str, ...] = ("http://127.0.0.1:5173", "http://localhost:5173")
    max_upload_bytes: int = 50 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        backend_root = Path(__file__).resolve().parents[2]
        project_root = backend_root.parent
        data_dir = _resolve_path(os.getenv("YWA_DATA_DIR"), project_root / "data", project_root)
        import_root = _resolve_path(os.getenv("YWA_IMPORT_ROOT"), data_dir / "imports", project_root)
        database_url = os.getenv("YWA_DATABASE_URL") or f"sqlite:///{data_dir / 'yolowebagent-starter.db'}"
        origins = tuple(
            item.strip()
            for item in os.getenv("YWA_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",")
            if item.strip()
        )
        return cls(
            project_root=project_root,
            data_dir=data_dir,
            import_root=import_root,
            database_url=database_url,
            host=os.getenv("YWA_HOST", "127.0.0.1"),
            port=int(os.getenv("YWA_PORT", "8000")),
            cors_origins=origins,
            max_upload_bytes=int(os.getenv("YWA_MAX_UPLOAD_MB", "50")) * 1024 * 1024,
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.import_root.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "datasets").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "exports").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "runs" / "training").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "tmp").mkdir(parents=True, exist_ok=True)
