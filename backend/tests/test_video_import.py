from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from app.core.config import Settings
from app.core.ids import new_id
from app.core.models import Dataset, ImageItem, VideoImportTask
from app.main import create_app


def video_bytes(path: Path) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (32, 24))
    assert writer.isOpened()
    try:
        for index in range(8):
            writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()
    return path.read_bytes()


def cross_platform_video(path: Path) -> Path:
    """Use the first encoder OpenCV exposes on the current OS/CI image."""

    for codec, suffix in (("mp4v", ".mp4"), ("avc1", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi")):
        candidate = path.with_suffix(suffix)
        writer = cv2.VideoWriter(str(candidate), cv2.VideoWriter_fourcc(*codec), 4, (32, 24))
        if not writer.isOpened():
            writer.release()
            continue
        try:
            writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
            writer.write(np.full((24, 32, 3), 255, dtype=np.uint8))
        finally:
            writer.release()
        reader = cv2.VideoCapture(str(candidate))
        try:
            ok, frame = reader.read()
            if reader.isOpened() and ok and frame.shape[:2] == (24, 32):
                return candidate
        finally:
            reader.release()
    raise AssertionError("No supported OpenCV encoder/decoder pair is available on this platform.")


def wait_for_task(client, task_id: str) -> dict:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        task = client.get(f"/api/video-imports/{task_id}").json()
        if task["status"] in {"completed", "failed"}:
            return task
        time.sleep(0.05)
    raise AssertionError("video import task did not finish")


def test_video_import_preflight_then_creates_dataset_with_frame_provenance(client, tmp_path):
    response = client.post(
        "/api/video-imports",
        data={
            "name": "from-video",
            "task_type": "detect",
            "split": "val",
            "sampling_mode": "frame_interval",
            "sampling_value": "2",
            "start_seconds": "0",
        },
        files={"file": ("street.mp4", video_bytes(tmp_path / "street.mp4"), "video/mp4")},
    )
    assert response.status_code == 201, response.text
    preflight = response.json()
    assert preflight["status"] == "pending"
    assert preflight["start_requested"] is False
    assert preflight["video_info_json"]["width"] == 32
    assert preflight["total_images"] == 4
    assert preflight["video_info_json"]["estimated_output_bytes"] > 0
    assert preflight["video_info_json"]["estimate_sampled_frames"] > 0

    started = client.post(f"/api/video-imports/{preflight['id']}/start")
    assert started.status_code == 202, started.text
    assert started.json()["start_requested"] is True
    task = wait_for_task(client, preflight["id"])
    assert task["status"] == "completed", task
    assert task["generated_images"] == 4
    assert task["dataset_id"]

    datasets = client.get("/api/datasets").json()
    dataset = next(item for item in datasets if item["id"] == task["dataset_id"])
    assert dataset["name"] == "from-video"
    assert dataset["image_count"] == 4
    images = client.get(f"/api/datasets/{dataset['id']}/images").json()["items"]
    assert [item["frame_index"] for item in images] == [0, 2, 4, 6]
    assert {item["split"] for item in images} == {"val"}
    assert {item["source_type"] for item in images} == {"video_frame"}
    assert {item["source_file"] for item in images} == {"street.mp4"}
    assert {item["source_video_task_id"] for item in images} == {preflight["id"]}


def test_video_import_fps_one_selects_one_frame_per_second(client, tmp_path):
    """Keep the default FPS=1 behavior aligned with the upstream importer."""

    response = client.post(
        "/api/video-imports",
        data={
            "name": "one-per-second",
            "task_type": "detect",
            "sampling_mode": "fps",
            "sampling_value": "1",
            "start_seconds": "0",
        },
        files={"file": ("one-per-second.mp4", video_bytes(tmp_path / "one-per-second.mp4"), "video/mp4")},
    )
    assert response.status_code == 201, response.text
    preflight = response.json()
    # video_bytes produces 8 frames at 4 FPS: samples are frame 0 (0s) and 4 (1s).
    assert preflight["total_images"] == 2

    assert client.post(f"/api/video-imports/{preflight['id']}/start").status_code == 202
    task = wait_for_task(client, preflight["id"])
    assert task["status"] == "completed", task
    assert task["generated_images"] == 2
    images = client.get(f"/api/datasets/{task['dataset_id']}/images").json()["items"]
    assert [item["frame_index"] for item in images] == [0, 4]
    assert [item["timestamp"] for item in images] == [0.0, 1.0]


def test_preflight_does_not_start_until_that_task_is_confirmed(client, tmp_path):
    first = client.post(
        "/api/video-imports",
        data={"name": "wait", "task_type": "detect", "sampling_mode": "frame_interval", "sampling_value": "2"},
        files={"file": ("first.mp4", video_bytes(tmp_path / "first.mp4"), "video/mp4")},
    ).json()
    second = client.post(
        "/api/video-imports",
        data={"name": "start", "task_type": "detect", "sampling_mode": "frame_interval", "sampling_value": "2"},
        files={"file": ("second.mp4", video_bytes(tmp_path / "second.mp4"), "video/mp4")},
    ).json()

    started = client.post(f"/api/video-imports/{second['id']}/start")
    assert started.status_code == 202, started.text
    assert wait_for_task(client, second["id"])["status"] == "completed"
    still_pending = client.get(f"/api/video-imports/{first['id']}").json()
    assert still_pending["status"] == "pending"
    assert still_pending["start_requested"] is False
    assert still_pending["dataset_id"] is None


def test_video_import_time_blocks_keep_each_time_window_in_one_split(client, tmp_path):
    response = client.post(
        "/api/video-imports",
        data={
            "name": "time-blocks",
            "task_type": "detect",
            "sampling_mode": "frame_interval",
            "sampling_value": "2",
            "split_strategy": "time_blocks",
            "time_block_seconds": "0.5",
            "train_ratio": "0.5",
            "val_ratio": "0.25",
            "test_ratio": "0.25",
        },
        files={"file": ("blocks.mp4", video_bytes(tmp_path / "blocks.mp4"), "video/mp4")},
    )
    assert response.status_code == 201, response.text
    preflight = response.json()
    assert preflight["video_info_json"]["estimated_split_counts"] == {"train": 2, "val": 1, "test": 1}
    client.post(f"/api/video-imports/{preflight['id']}/start")
    task = wait_for_task(client, preflight["id"])
    assert task["status"] == "completed", task
    images = client.get(f"/api/datasets/{task['dataset_id']}/images").json()["items"]
    assert [(item["frame_index"], item["split"]) for item in images] == [(0, "train"), (2, "train"), (4, "val"), (6, "test")]


def test_application_restart_resumes_from_persisted_frame_checkpoint(tmp_path):
    settings = Settings(
        project_root=Path(__file__).resolve().parents[2],
        data_dir=tmp_path / "data",
        import_root=tmp_path / "imports",
        database_url=f"sqlite:///{tmp_path / 'restart.db'}",
    )
    with TestClient(create_app(settings)) as first_client:
        preflight = first_client.post(
            "/api/video-imports",
            data={"name": "resume", "task_type": "detect", "sampling_mode": "frame_interval", "sampling_value": "2"},
            files={"file": ("resume.mp4", video_bytes(tmp_path / "resume.mp4"), "video/mp4")},
        ).json()
        task_id = preflight["id"]
        storage = first_client.app.state.storage
        session_factory = first_client.app.state.database.session_factory
        first_frame = np.zeros((24, 32, 3), dtype=np.uint8)
        assert cv2.imencode(".jpg", first_frame)[0]
        content = cv2.imencode(".jpg", first_frame)[1].tobytes()
        image_id = f"img_{task_id[4:]}_000000000"
        dataset_id = new_id("ds")
        storage_name = storage.safe_storage_name("frame_000000000.jpg", image_id)
        storage.write_image(dataset_id, storage_name, content)
        with session_factory() as session:
            task = session.get(VideoImportTask, task_id)
            assert task is not None
            session.add(Dataset(id=dataset_id, name="resume", description=None, task_type="detect", image_count=1))
            session.add(ImageItem(id=image_id, dataset_id=dataset_id, file_name="frame_000000000.jpg", storage_name=storage_name, width=32, height=24, split="train", source_type="video_frame", source_file="resume.mp4", source_group_id=f"video:{task_id}", source_video_task_id=task_id, source_checksum=task.source_checksum, frame_index=0, timestamp=0.0))
            task.dataset_id = dataset_id
            task.status = "running"
            task.start_requested = True
            task.generated_images = 1
            task.checkpoint_next_frame_index = 2
            task.output_bytes = len(content)
            session.commit()

    with TestClient(create_app(settings)) as restarted_client:
        task = wait_for_task(restarted_client, task_id)
        assert task["status"] == "completed", task
        assert task["generated_images"] == 4
        images = restarted_client.get(f"/api/datasets/{dataset_id}/images").json()["items"]
        assert [item["frame_index"] for item in images] == [0, 2, 4, 6]
        assert len({item["id"] for item in images}) == 4


def test_platform_codec_round_trip_is_accepted_by_video_preflight(client, tmp_path):
    source = cross_platform_video(tmp_path / "codec")
    response = client.post(
        "/api/video-imports",
        data={"name": "codec", "task_type": "detect", "sampling_mode": "frame_interval", "sampling_value": "1"},
        files={"file": (source.name, source.read_bytes(), "video/mp4" if source.suffix == ".mp4" else "video/x-msvideo")},
    )
    assert response.status_code == 201, response.text
    assert response.json()["video_info_json"]["frame_count"] >= 2


def test_video_import_rejects_unsupported_files_and_invalid_sampling(client):
    unsupported = client.post(
        "/api/video-imports",
        data={"name": "bad", "task_type": "detect"},
        files={"file": ("not-video.txt", b"not a video", "text/plain")},
    )
    assert unsupported.status_code == 422
    assert unsupported.json()["error"]["code"] == "video_format_unsupported"

    invalid_sampling = client.post(
        "/api/video-imports",
        data={"name": "bad", "task_type": "detect", "sampling_mode": "frame_interval", "sampling_value": "1.5"},
        files={"file": ("video.mp4", b"not examined", "video/mp4")},
    )
    assert invalid_sampling.status_code == 422
    assert invalid_sampling.json()["error"]["code"] == "video_frame_interval_invalid"


def test_legacy_dataset_scoped_video_route_is_not_exposed(client):
    dataset = client.post("/api/datasets", json={"name": "legacy", "task_type": "detect"}).json()
    response = client.post(
        f"/api/datasets/{dataset['id']}/video/import",
        files={"file": ("legacy.mp4", b"not used", "video/mp4")},
    )
    assert response.status_code == 404
