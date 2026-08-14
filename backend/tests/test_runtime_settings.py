from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler

from app.core.errors import ValidationError
from app.logs.service import RUNTIME_LOG_BACKUP_COUNT, RUNTIME_LOG_MAX_BYTES, runtime_log_path
from app.training.config import build_training_command
from app.training.runtime.device_service import DeviceService, TrainingDevice
from app.core.task_types import TaskType


def test_sam_settings_are_persisted_in_starter_data_dir(client):
    initial = client.get("/api/settings/sam")
    assert initial.status_code == 200
    assert initial.json()["device"] == "auto"
    assert initial.json()["model_configured"] is False

    updated = client.put(
        "/api/settings/sam",
        json={
            "enabled": True,
            "model": "sam_b.pt",
            "device": "cpu",
            "img_size": 768,
            "fallback_mode": "disabled",
        },
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["model_configured"] is True
    assert updated.json()["img_size"] == 768

    reloaded = client.get("/api/settings/sam").json()
    assert reloaded["model"] == "sam_b.pt"
    assert reloaded["fallback_mode"] == "disabled"
    settings_path = client.app.state.settings.data_dir / "settings.json"
    assert json.loads(settings_path.read_text(encoding="utf-8"))["sam"]["model"] == "sam_b.pt"


def test_sam_settings_update_clears_cached_model(monkeypatch, client):
    from app.sam import service as sam_service

    cleared = 0

    def clear() -> None:
        nonlocal cleared
        cleared += 1

    monkeypatch.setattr(sam_service, "clear_model_cache", clear)
    response = client.put(
        "/api/settings/sam",
        json={"enabled": True, "model": "sam_b.pt", "device": "cpu", "img_size": 1024, "fallback_mode": "box"},
    )
    assert response.status_code == 200, response.text
    assert cleared == 1


def test_training_devices_and_multi_gpu_command(monkeypatch, client):
    devices = client.get("/api/training/devices")
    assert devices.status_code == 200
    assert devices.json()["items"][0]["type"] == "cpu"

    gpu_devices = [
        TrainingDevice(id="cpu", type="cpu", name="CPU", status="available"),
        TrainingDevice(id="cuda:0", type="cuda", name="GPU 0", index=0, status="available"),
        TrainingDevice(id="cuda:1", type="cuda", name="GPU 1", index=1, status="available"),
    ]
    monkeypatch.setattr(DeviceService, "list_devices", lambda _self: gpu_devices)
    assert DeviceService().normalize_selector("auto") == "0"
    assert DeviceService().normalize_selector("cuda:0,cuda:1") == "0,1"
    assert DeviceService().normalize_selector("cuda") == "0"
    try:
        DeviceService().normalize_selector("0,0")
    except ValidationError as exc:
        assert exc.error_code == "invalid_training_device"
    else:
        raise AssertionError("duplicate CUDA ids should be rejected")

    monkeypatch.setattr(DeviceService, "list_devices", lambda _self: [gpu_devices[0], TrainingDevice(id="mps", type="mps", name="Apple GPU (MPS)", status="available")])
    assert DeviceService().normalize_selector("auto") == "mps"

    command = build_training_command(
        task_type=TaskType.DETECT,
        model="yolo11n.pt",
        data_yaml="/tmp/data.yaml",
        run_dir="/tmp/run",
        epochs=1,
        img_size=640,
        batch_size=2,
        device="0,1",
        workers=0,
        seed=42,
    )
    assert "device=0,1" in command.args


def test_runtime_logs_endpoint_returns_startup_log(client):
    response = client.get("/api/logs/runtime?lines=20")
    assert response.status_code == 200
    payload = response.json()
    assert payload["path"].endswith("logs/backend.log")
    assert any("runtime initialized" in line for line in payload["lines"])

    filtered = client.get("/api/logs/runtime?lines=20&level=INFO")
    assert filtered.status_code == 200
    assert all(" INFO " in line for line in filtered.json()["lines"])

    handler = next(handler for handler in logging.getLogger().handlers if getattr(handler, "_ywa_runtime_log", False))
    assert isinstance(handler, RotatingFileHandler)
    assert handler.maxBytes == RUNTIME_LOG_MAX_BYTES
    assert handler.backupCount == RUNTIME_LOG_BACKUP_COUNT


def test_runtime_logs_include_rotated_history_in_chronological_order(client):
    path = runtime_log_path(client.app.state.settings)
    path.with_name("backend.log.2").write_text("oldest-1\noldest-2\n", encoding="utf-8")
    path.with_name("backend.log.1").write_text("middle-1\nmiddle-2\n", encoding="utf-8")
    path.write_text("current-1\ncurrent-2\n", encoding="utf-8")

    response = client.get("/api/logs/runtime?lines=5")
    assert response.status_code == 200
    assert response.json()["lines"] == ["oldest-2", "middle-1", "middle-2", "current-1", "current-2"]
