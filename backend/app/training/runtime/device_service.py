from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.errors import ValidationError


@dataclass(frozen=True)
class TrainingDevice:
    id: str
    type: str
    name: str
    index: int | None = None
    memory_total_mb: int | None = None
    memory_free_mb: int | None = None
    status: str = "unknown"

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "index": self.index,
            "memory_total_mb": self.memory_total_mb,
            "memory_free_mb": self.memory_free_mb,
            "status": self.status,
        }


class DeviceService:
    """Detect local devices and validate explicit training selectors."""

    _CUDA_SELECTOR = re.compile(r"^(?:cuda:)?(\d+)(?:,(?:cuda:)?\d+)*$")

    def list_devices(self) -> list[TrainingDevice]:
        devices = [TrainingDevice(id="cpu", type="cpu", name="CPU", status="available")]
        torch = self._torch()
        if torch is None:
            return devices
        try:
            mps = bool(torch.backends.mps.is_available())
        except (AttributeError, RuntimeError):
            mps = False
        if mps:
            devices.append(TrainingDevice(id="mps", type="mps", name="Apple GPU (MPS)", status="available"))
        try:
            cuda_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        except (AttributeError, RuntimeError):
            cuda_count = 0
        for index in range(cuda_count):
            name = f"CUDA GPU {index}"
            total_mb: int | None = None
            free_mb: int | None = None
            try:
                name = str(torch.cuda.get_device_name(index))
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                free_mb = int(free_bytes // (1024 * 1024))
                total_mb = int(total_bytes // (1024 * 1024))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            devices.append(
                TrainingDevice(
                    id=f"cuda:{index}", type="cuda", index=index, name=name,
                    memory_total_mb=total_mb, memory_free_mb=free_mb, status="available",
                )
            )
        return devices

    def normalize_selector(self, selector: str | None) -> str:
        raw = (selector or "auto").strip().lower()
        if raw == "auto":
            devices = self.list_devices()
            cuda_ids = sorted(
                device.index
                for device in devices
                if device.type == "cuda" and device.index is not None
            )
            if cuda_ids:
                return str(cuda_ids[0])
            if any(device.type == "mps" for device in devices):
                return "mps"
            return "cpu"
        if raw in {"cpu", "mps"}:
            if raw == "mps":
                self._require_type(raw)
            return raw
        if raw == "cuda":
            cuda_ids = sorted(
                device.index
                for device in self.list_devices()
                if device.type == "cuda" and device.index is not None
            )
            if not cuda_ids:
                raise ValidationError("training_device_unavailable", "当前环境没有可用的 CUDA 设备。")
            return str(cuda_ids[0])
        if not self._CUDA_SELECTOR.fullmatch(raw):
            raise ValidationError("invalid_training_device", "训练设备必须是 auto、cpu、mps、cuda 或逗号分隔的 CUDA id。")
        ids = [int(part.removeprefix("cuda:")) for part in raw.split(",")]
        if len(set(ids)) != len(ids):
            raise ValidationError("invalid_training_device", "CUDA 设备不能重复选择。")
        cuda_ids = {device.index for device in self.list_devices() if device.type == "cuda" and device.index is not None}
        missing = [index for index in ids if index not in cuda_ids]
        if missing:
            raise ValidationError(
                "training_device_unavailable", f"选择的 CUDA 设备不可用：{missing}。",
                details={"available_gpu_ids": sorted(cuda_ids)},
            )
        return ",".join(str(index) for index in ids)

    def _require_type(self, device_type: str) -> None:
        if not any(device.type == device_type for device in self.list_devices()):
            raise ValidationError("training_device_unavailable", f"当前环境没有可用的 {device_type.upper()} 设备。")

    @staticmethod
    def _torch():
        try:
            import torch

            return torch
        except (ImportError, OSError):
            return None
