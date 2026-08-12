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
[ -z "$YWA_ARGS_CAPTURE" ] || printf '%s\n' "$@" > "$YWA_ARGS_CAPTURE"
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
        elif task_type == "obb":
            annotation = {"type": "obb", "class_id": label, "obb": {"cx": 25, "cy": 20, "width": 25, "height": 20, "angle": 15}}
        elif task_type == "classify":
            annotation = {"type": "classify", "class_id": label}
        response = client.put(
            f"/api/datasets/{dataset_id}/images/{image['id']}/annotations",
            json={"annotations": [annotation]},
        )
        assert response.status_code == 200, response.text
    return dataset_id, label


def wait_for_terminal(client, task_id: str) -> dict:
    for _ in range(100):
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
    models = client.get(f"/api/models?dataset_id={task['dataset_id']}").json()
    assert models["total"] == 2
    best = next(item for item in models["items"] if item["artifact_type"] == "best")
    assert client.get(f"/api/models/{best['id']}/download").content == b"fake best"


def test_completed_task_starts_new_training_from_selected_checkpoint(client, tmp_path, monkeypatch):
    executable = tmp_path / "fake-yolo"
    capture = tmp_path / "yolo-args.txt"
    fake_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))
    monkeypatch.setenv("YWA_ARGS_CAPTURE", str(capture))
    dataset_id, _ = prepared_dataset(client)

    source_response = client.post(
        "/api/training/tasks",
        json={"dataset_id": dataset_id, "name": "source", "model": "yolo11n.pt", "epochs": 1, "batch_size": 1},
    )
    source = wait_for_terminal(client, source_response.json()["id"])
    resumed_response = client.post(
        f"/api/training/tasks/{source['id']}/resume",
        json={"name": "continued", "epochs": 2, "resume_epoch": False},
    )
    assert resumed_response.status_code == 201, resumed_response.text
    resumed = resumed_response.json()
    assert resumed["model_path"] == source["last_model_path"]
    assert resumed["run_dir"] != source["run_dir"]
    assert resumed["epochs"] == 2
    terminal = wait_for_terminal(client, resumed["id"])
    assert terminal["status"] == "completed", terminal
    args = capture.read_text(encoding="utf-8").splitlines()
    assert f"model={source['last_model_path']}" in args
    assert "resume=True" not in args
    assert f"project={Path(resumed['run_dir']).parent}" in args
    assert f"name={Path(resumed['run_dir']).name}" in args


def test_interrupted_task_restores_epoch_state_in_source_run_directory(client, tmp_path, monkeypatch):
    executable = tmp_path / "fake-yolo"
    capture = tmp_path / "yolo-args.txt"
    fake_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))
    monkeypatch.setenv("YWA_ARGS_CAPTURE", str(capture))
    dataset_id, _ = prepared_dataset(client)
    source_response = client.post(
        "/api/training/tasks",
        json={"dataset_id": dataset_id, "name": "interrupted", "model": "yolo11n.pt", "epochs": 4, "batch_size": 1},
    )
    source = wait_for_terminal(client, source_response.json()["id"])
    with client.app.state.database.session_factory() as session:
        from app.core.models import TrainingTask

        source_task = session.get(TrainingTask, source["id"])
        assert source_task is not None
        source_task.status = "failed"
        session.commit()

    resumed_response = client.post(
        f"/api/training/tasks/{source['id']}/resume",
        json={"name": "restored", "resume_epoch": True},
    )
    assert resumed_response.status_code == 201, resumed_response.text
    resumed = resumed_response.json()
    assert resumed["model_path"] == source["last_model_path"]
    assert resumed["run_dir"] == source["run_dir"]
    assert resumed["epochs"] == source["epochs"]
    terminal = wait_for_terminal(client, resumed["id"])
    assert terminal["status"] == "completed", terminal
    args = capture.read_text(encoding="utf-8").splitlines()
    assert f"model={source['last_model_path']}" in args
    assert "resume=True" in args
    assert f"project={Path(source['run_dir']).parent}" in args
    assert f"name={Path(source['run_dir']).name}" in args


def test_completed_task_rejects_epoch_state_resume(client, tmp_path, monkeypatch):
    executable = tmp_path / "fake-yolo"
    fake_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))
    dataset_id, _ = prepared_dataset(client)
    source_response = client.post(
        "/api/training/tasks",
        json={"dataset_id": dataset_id, "name": "finished", "model": "yolo11n.pt", "epochs": 1, "batch_size": 1},
    )
    source = wait_for_terminal(client, source_response.json()["id"])

    response = client.post(f"/api/training/tasks/{source['id']}/resume", json={"resume_epoch": True})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "resume_completed_task"


def test_model_metadata_and_onnx_lifecycle(client, tmp_path, monkeypatch):
    executable = tmp_path / "fake-yolo"
    fake_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))
    dataset_id, _ = prepared_dataset(client)
    response = client.post(
        "/api/training/tasks",
        json={"dataset_id": dataset_id, "name": "model-lifecycle", "model": "yolo11n.pt", "epochs": 1, "batch_size": 1},
    )
    wait_for_terminal(client, response.json()["id"])
    models = client.get(f"/api/models?dataset_id={dataset_id}").json()["items"]
    best = next(item for item in models if item["artifact_type"] == "best")
    calls: list[tuple[str, str]] = []

    def fake_export(source, destination):
        calls.append((str(source), str(destination)))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake onnx")
        return destination

    monkeypatch.setattr("app.models.onnx.export_fp32_onnx", fake_export)
    exported = client.post(f"/api/models/{best['id']}/export-onnx")
    assert exported.status_code == 200, exported.text
    exported_model = exported.json()
    assert exported_model["format"] == "onnx"
    assert client.get(f"/api/models/{exported_model['id']}/download").content == b"fake onnx"
    again = client.post(f"/api/models/{best['id']}/export-onnx")
    assert again.status_code == 200
    assert again.json()["id"] == exported_model["id"]
    assert len(calls) == 1

    updated = client.patch(f"/api/models/{best['id']}", json={"name": "Renamed best", "notes": "smoke"})
    assert updated.status_code == 200
    assert updated.json()["name"] == "Renamed best"
    archived = client.post(f"/api/models/{best['id']}/archive")
    assert archived.status_code == 200
    assert client.get(f"/api/models?dataset_id={dataset_id}").json()["total"] == 2
    restored = client.post(f"/api/models/{best['id']}/restore")
    assert restored.status_code == 200
    assert restored.json()["status"] == "active"
    assert client.delete(f"/api/models/{best['id']}").status_code == 204


def test_onnx_export_failure_does_not_register_an_orphan(client, tmp_path, monkeypatch):
    executable = tmp_path / "fake-yolo"
    fake_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))
    dataset_id, _ = prepared_dataset(client)
    response = client.post(
        "/api/training/tasks",
        json={"dataset_id": dataset_id, "name": "failed-export", "model": "yolo11n.pt", "epochs": 1, "batch_size": 1},
    )
    wait_for_terminal(client, response.json()["id"])
    best = next(item for item in client.get(f"/api/models?dataset_id={dataset_id}").json()["items"] if item["artifact_type"] == "best")

    def fail_export(source, destination):
        raise RuntimeError("converter unavailable")

    monkeypatch.setattr("app.models.onnx.export_fp32_onnx", fail_export)
    exported = client.post(f"/api/models/{best['id']}/export-onnx")
    assert exported.status_code == 422
    assert exported.json()["error"]["code"] == "onnx_export_failed"
    assert client.get(f"/api/models?dataset_id={dataset_id}").json()["total"] == 2


def test_training_rejects_wrong_weight_family(client):
    dataset = client.post("/api/datasets", json={"name": "not-ready", "task_type": "segment"}).json()
    response = client.post(
        "/api/training/tasks",
        json={"dataset_id": dataset["id"], "model": "yolo11n.pt"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "model_task_mismatch"


def test_training_rejects_local_weight_outside_managed_models(client, tmp_path):
    dataset_id, _ = prepared_dataset(client)
    external_weight = tmp_path / "external.pt"
    external_weight.write_bytes(b"not a managed weight")

    response = client.post(
        "/api/training/tasks",
        json={"dataset_id": dataset_id, "model": str(external_weight)},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "model_path_outside_managed_dir"


def test_artifact_registration_failure_retains_training_result(client, tmp_path, monkeypatch):
    executable = tmp_path / "fake-yolo"
    fake_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))

    def fail_registration(self, session, task):
        raise RuntimeError("model registry unavailable")

    monkeypatch.setattr("app.models.service.ModelService.register_training_artifacts", fail_registration)
    dataset_id, _ = prepared_dataset(client)
    response = client.post(
        "/api/training/tasks",
        json={"dataset_id": dataset_id, "name": "artifact-registration-failure", "model": "yolo11n.pt", "epochs": 1, "batch_size": 1},
    )
    assert response.status_code == 201, response.text

    task = wait_for_terminal(client, response.json()["id"])
    assert task["status"] == "failed"
    assert task["finished_at"] is not None
    assert task["best_model_path"].endswith("best.pt")
    assert task["last_model_path"].endswith("last.pt")
    assert task["metrics_json"]
    assert "model registry unavailable" in task["error_message"]
    summary = client.get(f"/api/training/tasks/{task['id']}/summary").json()
    assert summary["status"] == "failed"
    assert summary["checkpoints"]["best"].endswith("best.pt")
    assert client.get(f"/api/models?dataset_id={dataset_id}").json()["total"] == 0


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


def test_obb_and_classify_training_use_their_weight_families(client, tmp_path, monkeypatch):
    executable = tmp_path / "fake-yolo-extra-tasks"
    fake_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))
    for task_type, model in (("obb", "yolo11n-obb.pt"), ("classify", "yolo11n-cls.pt")):
        dataset_id, _ = prepared_dataset(client, task_type)
        response = client.post(
            "/api/training/tasks",
            json={"dataset_id": dataset_id, "name": f"{task_type}-smoke", "model": model, "epochs": 1, "batch_size": 1},
        )
        assert response.status_code == 201, response.text
        task = wait_for_terminal(client, response.json()["id"])
        assert task["status"] == "completed", task


def test_classify_training_requires_annotated_train_and_val_images(client):
    dataset_id, _ = prepared_dataset(client, "classify")
    train_image = next(image for image in client.get(f"/api/datasets/{dataset_id}/images").json()["items"] if image["split"] == "train")
    cleared = client.put(f"/api/datasets/{dataset_id}/images/{train_image['id']}/annotations", json={"annotations": []})
    assert cleared.status_code == 200

    response = client.post(
        "/api/training/tasks",
        json={"dataset_id": dataset_id, "model": "yolo11n-cls.pt"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "training_classification_split_unannotated"


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
