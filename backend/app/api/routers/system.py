from fastapi import APIRouter, Depends

from app.api.dependencies import get_settings
from app.core.config import Settings
from app.settings.service import SamSettingsService
from app.training.runtime.device_service import DeviceService

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/info")
def info(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    sam = SamSettingsService(settings).get()
    return {
        "name": "YoloWebAgentStarter",
        "edition": "community",
        "version": "0.1.0-dev",
        "task_types": ["detect", "segment", "obb", "classify"],
        "data_dir": str(settings.data_dir),
        "import_root": str(settings.import_root),
        "auth_enabled": False,
        "sam": {
            "model_configured": sam.model_configured,
            "box_prompt_available": sam.enabled,
            "point_prompt_available": sam.enabled and sam.model_configured,
            "box_backend": "ultralytics_sam" if sam.model_configured else "box_stub",
        },
        "training_devices": [device.as_dict() for device in DeviceService().list_devices()],
    }
