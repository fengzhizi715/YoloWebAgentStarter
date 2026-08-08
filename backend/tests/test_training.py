from __future__ import annotations

import io
import time
from pathlib import Path

from PIL import Image


def image_bytes() -> bytes:
    image = Image.new("RGB", (80, 60), "#6688aa")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def fake_yolo(path: Path) -> None:
    path.write_text(
        """#!/bin/sh
project=""
name="run"
for arg in "$@"; do
  case "$arg" in
    project=*) project="${arg#project=}" ;;
    name=*) name="${arg#name=}" ;;
  esac
done
mkdir -p "$project/$name/weights"
printf 'epoch,metrics/mAP50(B)\\n0,0.81\\n' > "$project/$name/results.csv"
printf 'fake best' > "$project/$name/weights/best.pt"
printf 'fake last' > "$project/$name/weights/last.pt"
printf '1/1\\n'
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def slow_yolo(path: Path) -> None:
    path.write_text(
        """#!/bin/sh
project=""
name="run"
for arg in "$@"; do
  case "$arg" in
    project=*) project="${arg#project=}" ;;
    name=*) name="${arg#name=}" ;;
  esac
done
mkdir -p "$project/$name/weights"
printf '1/20\\n'
sleep 5
printf 'fake best' > "$project/$name/weights/best.pt"
printf 'fake last' > "$project/$name/weights/last.pt"
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def prepared_dataset(client, task_type: str = "detect") -> tuple[str, str]:
    dataset = client.post("/api/datasets", json={"name": f"train-{task_type}", "task_type": task_type}).json()
    dataset_id = dataset["id"]
    label = client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "object"}).json()["id"]
    images = []
    for split in ("train", "val"):
        response = client.post(
            f"/api/datasets/{dataset_id}/images/upload",
            data={"split": split},
            files={"files": (f"{split}.png", image_bytes(), "image/png")},
        )
        images.append(response.json()["items"][0])
    for image in images:
        annotation = {"type": "bbox", "class_id": label, "bbox": {"x": 5, "y": 5, "width": 25, "height": 20}}
        if task_type == "segment":
            annotation = {"type": "polygon", "class_id": label, "polygon": [[5, 5], [30, 5], [30, 25], [5, 25]]}
        response = client.put(
            f"/api/datasets/{dataset_id}/images/{image['id']}/annotations",
            json={"annotations": [annotation]},
        )
        assert response.status_code == 200, response.text
    return dataset_id, label


def wait_for_terminal(client, task_id: str) -> dict:
    for _ in range(30):
        task = client.get(f"/api/training/tasks/{task_id}").json()
        if task["status"] in {"completed", "failed", "stopped"}:
            return task
        time.sleep(0.05)
    raise AssertionError("training task did not reach a terminal state")


def test_training_task_runs_and_persists_checkpoints(client, tmp_path, monkeypatch):
    executable = tmp_path / "fake-yolo"
    fake_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))
    dataset_id, _ = prepared_dataset(client)

    response = client.post(
        "/api/training/tasks",
        json={"dataset_id": dataset_id, "name": "smoke", "model": "yolo11n.pt", "epochs": 1, "batch_size": 1},
    )
    assert response.status_code == 201, response.text
    task = wait_for_terminal(client, response.json()["id"])
    assert task["status"] == "completed", task
    assert task["best_model_path"].endswith("best.pt")
    assert task["last_model_path"].endswith("last.pt")
    assert client.get(f"/api/training/tasks/{task['id']}/checkpoints/best").status_code == 200
    assert client.get(f"/api/training/tasks/{task['id']}/logs").json()["line_count"] >= 3
    summary = client.get(f"/api/training/tasks/{task['id']}/summary")
    assert summary.status_code == 200
    assert summary.json()["status"] == "completed"


def test_training_rejects_wrong_weight_family(client):
    dataset = client.post("/api/datasets", json={"name": "not-ready", "task_type": "segment"}).json()
    response = client.post(
        "/api/training/tasks",
        json={"dataset_id": dataset["id"], "model": "yolo11n.pt"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "model_task_mismatch"


def test_training_rejects_missing_validation_split(client):
    dataset_id, _ = prepared_dataset(client)
    images = client.get(f"/api/datasets/{dataset_id}/images").json()["items"]
    for image in images:
        client.patch(f"/api/datasets/{dataset_id}/images/{image['id']}", json={"split": "train"})
    response = client.post(
        "/api/training/tasks",
        json={"dataset_id": dataset_id, "model": "yolo11n.pt"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "training_split_missing"


def test_segment_training_uses_segment_weight_family(client, tmp_path, monkeypatch):
    executable = tmp_path / "fake-yolo-segment"
    fake_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))
    dataset_id, _ = prepared_dataset(client, "segment")
    response = client.post(
        "/api/training/tasks",
        json={"dataset_id": dataset_id, "name": "segment-smoke", "model": "yolo11n-seg.pt", "epochs": 1, "batch_size": 1},
    )
    assert response.status_code == 201, response.text
    task = wait_for_terminal(client, response.json()["id"])
    assert task["status"] == "completed", task


def test_running_training_can_be_stopped(client, tmp_path, monkeypatch):
    executable = tmp_path / "slow-yolo"
    slow_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))
    dataset_id, _ = prepared_dataset(client)
    response = client.post(
        "/api/training/tasks",
        json={"dataset_id": dataset_id, "name": "stop-me", "model": "yolo11n.pt", "epochs": 20, "batch_size": 1},
    )
    assert response.status_code == 201, response.text
    task_id = response.json()["id"]
    time.sleep(0.1)
    stopped = client.post(f"/api/training/tasks/{task_id}/stop")
    assert stopped.status_code == 200, stopped.text
    terminal = wait_for_terminal(client, task_id)
    assert terminal["status"] == "stopped", terminal
