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
    max_video_upload_bytes: int = 2 * 1024 * 1024 * 1024
    max_video_duration_seconds: int = 4 * 60 * 60
    max_video_generated_images: int = 100_000
    max_video_output_bytes: int = 50 * 1024 * 1024 * 1024
    min_video_free_bytes: int = 2 * 1024 * 1024 * 1024
    max_yolo_archive_members: int = 2_000
    max_yolo_archive_member_bytes: int = 100 * 1024 * 1024
    max_yolo_archive_uncompressed_bytes: int = 250 * 1024 * 1024
    max_yolo_archive_compression_ratio: float = 100.0
    sam_model: str | None = None
    sam_device: str = "auto"
    sam_img_size: int = 1024
    agent_provider: str = "mock"
    agent_model: str = "mock-model"
    agent_api_key: str | None = None
    agent_base_url: str | None = None
    agent_max_tool_rounds: int = 8
    agent_approval_ttl_seconds: int = 3600
    agent_timeout_seconds: float = 60.0
    agent_temperature: float = 0.2
    agent_auth_scheme: str = "bearer"
    agent_auth_header_name: str | None = None

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
            max_video_upload_bytes=int(os.getenv("YWA_MAX_VIDEO_UPLOAD_MB", "2048")) * 1024 * 1024,
            max_video_duration_seconds=int(os.getenv("YWA_MAX_VIDEO_DURATION_SECONDS", str(4 * 60 * 60))),
            max_video_generated_images=int(os.getenv("YWA_MAX_VIDEO_GENERATED_IMAGES", "100000")),
            max_video_output_bytes=int(os.getenv("YWA_MAX_VIDEO_OUTPUT_MB", str(50 * 1024))) * 1024 * 1024,
            min_video_free_bytes=int(os.getenv("YWA_MIN_VIDEO_FREE_MB", "2048")) * 1024 * 1024,
            max_yolo_archive_members=int(os.getenv("YWA_MAX_YOLO_ARCHIVE_MEMBERS", "2000")),
            max_yolo_archive_member_bytes=int(os.getenv("YWA_MAX_YOLO_ARCHIVE_MEMBER_MB", "100")) * 1024 * 1024,
            max_yolo_archive_uncompressed_bytes=int(os.getenv("YWA_MAX_YOLO_ARCHIVE_UNCOMPRESSED_MB", "250")) * 1024 * 1024,
            max_yolo_archive_compression_ratio=float(os.getenv("YWA_MAX_YOLO_ARCHIVE_COMPRESSION_RATIO", "100")),
            sam_model=os.getenv("YWA_SAM_MODEL") or None,
            sam_device=os.getenv("YWA_SAM_DEVICE", "auto"),
            sam_img_size=int(os.getenv("YWA_SAM_IMGSZ", "1024")),
            agent_provider=(os.getenv("YWA_AGENT_PROVIDER") or "mock").strip().lower() or "mock",
            agent_model=(os.getenv("YWA_AGENT_MODEL") or "mock-model").strip() or "mock-model",
            agent_api_key=(os.getenv("YWA_AGENT_API_KEY") or "").strip() or None,
            agent_base_url=(os.getenv("YWA_AGENT_BASE_URL") or "").strip() or None,
            agent_max_tool_rounds=max(1, min(int(os.getenv("YWA_AGENT_MAX_TOOL_ROUNDS", "8")), 32)),
            agent_approval_ttl_seconds=max(60, min(int(os.getenv("YWA_AGENT_APPROVAL_TTL_SECONDS", "3600")), 86_400)),
            agent_timeout_seconds=max(5.0, min(float(os.getenv("YWA_AGENT_TIMEOUT_SECONDS", "60")), 300.0)),
            agent_temperature=max(0.0, min(float(os.getenv("YWA_AGENT_TEMPERATURE", "0.2")), 2.0)),
            agent_auth_scheme=(os.getenv("YWA_AGENT_AUTH_SCHEME") or "bearer").strip().lower() or "bearer",
            agent_auth_header_name=(os.getenv("YWA_AGENT_AUTH_HEADER_NAME") or "").strip() or None,
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.import_root.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "datasets").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "exports").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "runs" / "training").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "runs" / "evaluation").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "tmp").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "video-imports").mkdir(parents=True, exist_ok=True)
