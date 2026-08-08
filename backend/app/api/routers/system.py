from fastapi import APIRouter, Depends

from app.api.dependencies import get_settings
from app.core.config import Settings

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/info")
def info(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    return {
        "name": "YoloWebAgentStarter",
        "edition": "community",
        "version": "0.1.0-dev",
        "task_types": ["detect", "segment"],
        "data_dir": str(settings.data_dir),
        "import_root": str(settings.import_root),
        "auth_enabled": False,
    }

