from __future__ import annotations

import json
import re

from app.core.config import Settings
from app.core.errors import ValidationError
from app.settings.schemas import SamSettingsResponse, SamSettingsUpdate


class SamSettingsService:
    """Persist SAM workspace settings in the Starter data directory."""

    _DEVICE = re.compile(r"^(?:auto|cpu|mps|cuda(?::\d+)?|\d+(?:,\d+)*)$", re.IGNORECASE)

    def __init__(self, settings: Settings) -> None:
        self.path = (settings.data_dir / "settings.json").resolve()
        self.env_model = (settings.sam_model or "").strip()
        self.env_device = settings.sam_device.strip().lower()
        self.env_img_size = settings.sam_img_size

    def get(self) -> SamSettingsResponse:
        section = self._read().get("sam", {})
        model = str(section.get("model", self.env_model)).strip()
        device = str(section.get("device", self.env_device or "auto")).strip().lower()
        try:
            img_size = int(section.get("img_size", self.env_img_size))
        except (TypeError, ValueError):
            img_size = self.env_img_size
        return SamSettingsResponse(
            enabled=bool(section.get("enabled", True)),
            model=model,
            device=device,
            img_size=max(64, min(img_size, 4096)),
            fallback_mode="disabled" if section.get("fallback_mode") == "disabled" else "box",
            model_configured=bool(model),
        )

    def update(self, payload: SamSettingsUpdate) -> SamSettingsResponse:
        device = payload.device.strip().lower()
        if not self._DEVICE.fullmatch(device):
            raise ValidationError("invalid_sam_device", "SAM 设备必须是 auto、cpu、mps、cuda[:id] 或 CUDA id 列表。")
        data = self._read()
        data["sam"] = {
            "enabled": payload.enabled,
            "model": payload.model.strip(),
            "device": device,
            "img_size": payload.img_size,
            "fallback_mode": payload.fallback_mode,
        }
        self._write(data)
        # Import lazily to keep the settings and SAM domains acyclic at
        # module-load time. Dropping the old model is important on CUDA hosts
        # where its weights may otherwise remain resident after a setting edit.
        from app.sam.service import clear_model_cache

        clear_model_cache()
        return self.get()

    def _read(self) -> dict:
        if not self.path.is_file():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValidationError("settings_read_failed", "本地设置文件无法读取。") from exc
        return value if isinstance(value, dict) else {}

    def _write(self, data: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.path)
        except OSError as exc:
            raise ValidationError("settings_write_failed", "本地设置无法保存。") from exc
