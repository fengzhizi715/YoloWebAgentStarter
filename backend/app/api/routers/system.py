from fastapi import APIRouter, Depends

from app.api.dependencies import get_settings
from app.core.config import Settings

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/info")
def info(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    sam_configured = bool(settings.sam_model)
    return {
        "name": "YoloWebAgentStarter",
        "edition": "community",
        "version": "0.1.0-dev",
        "task_types": ["detect", "segment", "obb", "classify"],
        "data_dir": str(settings.data_dir),
        "import_root": str(settings.import_root),
        "auth_enabled": False,
        "sam": {
            "model_configured": sam_configured,
            "box_prompt_available": True,
            "point_prompt_available": sam_configured,
            "box_backend": "ultralytics_sam" if sam_configured else "box_stub",
        },
    }
