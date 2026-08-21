from __future__ import annotations

import io
import time

from PIL import Image
import pytest

from app.core.models import Annotation, AutoAnnotationTask, ModelVersion
from app.models.result_parser import Detection


def test_auto_annotation_runs_as_a_reviewable_dataset_task(client, monkeypatch):
    dataset = client.post("/api/datasets", json={"name": "auto-demo", "task_type": "detect"}).json()
    dataset_id = dataset["id"]
    class_id = client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "cat"}).json()["id"]
    image_data = io.BytesIO()
    Image.new("RGB", (100, 80), "#4477aa").save(image_data, format="PNG")
    image = client.post(
        f"/api/datasets/{dataset_id}/images/upload",
        files={"files": ("cat.png", image_data.getvalue(), "image/png")},
    )
    assert image.status_code == 201, image.text
    image_id = image.json()["items"][0]["id"]
    model_id = "model_auto_test"
    model_path = client.app.state.storage.model_version_dir(model_id) / "best.pt"
    model_path.write_bytes(b"fake-managed-model")
    with client.app.state.database.session_factory() as session:
        session.add(
            ModelVersion(
                id=model_id,
                name="auto-test-model",
                version="test",
                dataset_id=dataset_id,
                source="training_task",
                artifact_type="best",
                format="pt",
                task_type="detect",
                engine_type="ultralytics",
                model_path=str(model_path),
                status="active",
                metrics_json={},
                notes="",
            )
        )
        session.commit()

    monkeypatch.setattr(
        "app.auto_annotation.runner.run_managed_inference",
        lambda **_: [Detection(class_index=0, confidence=0.9, x=10, y=12, width=30, height=25)],
    )
    created = client.post(
        f"/api/datasets/{dataset_id}/auto-annotation",
        json={"model_id": model_id, "confidence": 0.25, "iou": 0.45},
    )
    assert created.status_code == 202, created.text
    task_id = created.json()["id"]

    task = None
    for _ in range(40):
        task = client.get(f"/api/auto-annotation/{task_id}").json()
        if task["status"] in {"completed", "failed", "stopped"}:
            break
        time.sleep(0.025)
    assert task is not None
    assert task["status"] == "completed", task
    assert task["created_annotations"] == 1
    annotations = client.get(f"/api/datasets/{dataset_id}/images/{image_id}/annotations")
    assert annotations.status_code == 200
    assert annotations.json()[0]["source"] == "auto"
    assert annotations.json()[0]["class_id"] == class_id
    logs = client.get(f"/api/auto-annotation/{task_id}/logs")
    assert logs.status_code == 200
    assert "Auto annotation completed" in logs.json()["logs"]
    deleted = client.delete(f"/api/models/{model_id}")
    assert deleted.status_code == 409, deleted.text
    assert deleted.json()["error"]["code"] == "model_in_use"
    history = client.get(f"/api/auto-annotation/{task_id}").json()
    assert history["model_id"] == model_id


def test_auto_annotation_requires_explicit_mapping_for_different_model_classes(client, monkeypatch):
    target = client.post("/api/datasets", json={"name": "target", "task_type": "detect"}).json()
    target_class_id = client.post(f"/api/datasets/{target['id']}/classes", json={"name": "cat"}).json()["id"]
    source = client.post("/api/datasets", json={"name": "source", "task_type": "detect"}).json()
    client.post(f"/api/datasets/{source['id']}/classes", json={"name": "dog"})
    image_data = io.BytesIO()
    Image.new("RGB", (100, 80), "#4477aa").save(image_data, format="PNG")
    uploaded = client.post(f"/api/datasets/{target['id']}/images/upload", files={"files": ("cat.png", image_data.getvalue(), "image/png")})
    assert uploaded.status_code == 201, uploaded.text
    model_id = "model_cross_dataset"
    model_path = client.app.state.storage.model_version_dir(model_id) / "best.pt"
    model_path.write_bytes(b"fake-managed-model")
    with client.app.state.database.session_factory() as session:
        session.add(ModelVersion(id=model_id, name="dog-model", version="v1", dataset_id=source["id"], source="training_task", artifact_type="best", format="pt", task_type="detect", engine_type="ultralytics", model_path=str(model_path), status="active", metrics_json={}, notes=""))
        session.commit()

    rejected = client.post(f"/api/datasets/{target['id']}/auto-annotation", json={"model_id": model_id})
    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "class_mapping_empty"

    monkeypatch.setattr("app.auto_annotation.runner.run_managed_inference", lambda **_: [Detection(class_index=0, confidence=0.9, x=10, y=12, width=30, height=25)])
    created = client.post(f"/api/datasets/{target['id']}/auto-annotation", json={"model_id": model_id, "class_mapping": {"0": target_class_id}})
    assert created.status_code == 202, created.text
    task_id = created.json()["id"]
    task = _wait_for_terminal_task(client, task_id)
    assert task["status"] == "completed", task
    image_id = uploaded.json()["items"][0]["id"]
    annotations = client.get(f"/api/datasets/{target['id']}/images/{image_id}/annotations").json()
    assert annotations[0]["class_id"] == target_class_id


def test_model_with_active_auto_annotation_task_cannot_be_deleted(client):
    dataset = client.post("/api/datasets", json={"name": "pending", "task_type": "detect"}).json()
    model_id = "model_in_use"
    model_path = client.app.state.storage.model_version_dir(model_id) / "best.pt"
    model_path.write_bytes(b"fake-managed-model")
    with client.app.state.database.session_factory() as session:
        session.add(ModelVersion(id=model_id, name="in-use", version="v1", dataset_id=dataset["id"], source="training_task", artifact_type="best", format="pt", task_type="detect", engine_type="ultralytics", model_path=str(model_path), status="active", metrics_json={}, notes=""))
        session.add(AutoAnnotationTask(id="auto_pending", dataset_id=dataset["id"], model_id=model_id, task_type="detect", status="pending", class_mapping={}))
        session.commit()

    response = client.delete(f"/api/models/{model_id}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "model_in_use"


def test_auto_annotation_skips_images_that_already_have_annotations(client, monkeypatch):
    dataset = client.post("/api/datasets", json={"name": "skip-existing", "task_type": "detect"}).json()
    dataset_id = dataset["id"]
    class_id = client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "cat"}).json()["id"]
    image_data = io.BytesIO()
    Image.new("RGB", (100, 80), "#4477aa").save(image_data, format="PNG")
    first = client.post(f"/api/datasets/{dataset_id}/images/upload", files={"files": ("existing.png", image_data.getvalue(), "image/png")}).json()["items"][0]
    second = client.post(f"/api/datasets/{dataset_id}/images/upload", files={"files": ("new.png", image_data.getvalue(), "image/png")}).json()["items"][0]
    model_id = "model_skip_existing"
    model_path = client.app.state.storage.model_version_dir(model_id) / "best.pt"
    model_path.write_bytes(b"fake-managed-model")
    with client.app.state.database.session_factory() as session:
        session.add(Annotation(id="ann_existing", image_id=first["id"], dataset_id=dataset_id, class_id=class_id, type="bbox", x=10, y=12, width=30, height=25, source="manual"))
        session.add(ModelVersion(id=model_id, name="skip-model", version="v1", dataset_id=dataset_id, source="training_task", artifact_type="best", format="pt", task_type="detect", engine_type="ultralytics", model_path=str(model_path), status="active", metrics_json={}, notes=""))
        session.commit()
    calls: list[str] = []

    def infer(**kwargs):
        calls.append(kwargs["image_path"].name)
        return [Detection(class_index=0, confidence=0.9, x=10, y=12, width=30, height=25)]

    monkeypatch.setattr("app.auto_annotation.runner.run_managed_inference", infer)
    created = client.post(f"/api/datasets/{dataset_id}/auto-annotation", json={"model_id": model_id})
    assert created.status_code == 202, created.text
    task = _wait_for_terminal_task(client, created.json()["id"])
    assert task["status"] == "completed", task
    assert task["total_images"] == 1
    assert len(calls) == 1
    existing_annotations = client.get(f"/api/datasets/{dataset_id}/images/{first['id']}/annotations").json()
    new_annotations = client.get(f"/api/datasets/{dataset_id}/images/{second['id']}/annotations").json()
    assert [item["source"] for item in existing_annotations] == ["manual"]
    assert [item["source"] for item in new_annotations] == ["auto"]


@pytest.mark.parametrize(
    ("task_type", "detection", "annotation_type"),
    [
        ("segment", Detection(class_index=0, confidence=0.9, x=10, y=12, width=30, height=25, polygon=((10, 12), (40, 12), (40, 37), (10, 37))), "polygon"),
        ("obb", Detection(class_index=0, confidence=0.9, x=10, y=12, width=30, height=25, obb_points=((10, 12), (40, 12), (40, 37), (10, 37))), "obb"),
        ("classify", Detection(class_index=0, confidence=0.9, x=0, y=0, width=0, height=0), "classify"),
    ],
)
def test_auto_annotation_persists_every_supported_task_type(client, monkeypatch, task_type, detection, annotation_type):
    dataset = client.post("/api/datasets", json={"name": f"auto-{task_type}", "task_type": task_type}).json()
    dataset_id = dataset["id"]
    client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "target"})
    image_data = io.BytesIO()
    Image.new("RGB", (100, 80), "#4477aa").save(image_data, format="PNG")
    uploaded = client.post(f"/api/datasets/{dataset_id}/images/upload", files={"files": ("sample.png", image_data.getvalue(), "image/png")})
    assert uploaded.status_code == 201, uploaded.text
    model_id = f"model_auto_{task_type}"
    model_path = client.app.state.storage.model_version_dir(model_id) / "best.pt"
    model_path.write_bytes(b"fake-managed-model")
    with client.app.state.database.session_factory() as session:
        session.add(ModelVersion(id=model_id, name=f"{task_type}-model", version="v1", dataset_id=dataset_id, source="training_task", artifact_type="best", format="pt", task_type=task_type, engine_type="ultralytics", model_path=str(model_path), status="active", metrics_json={}, notes=""))
        session.commit()
    monkeypatch.setattr("app.auto_annotation.runner.run_managed_inference", lambda **_: [detection])

    created = client.post(f"/api/datasets/{dataset_id}/auto-annotation", json={"model_id": model_id})
    assert created.status_code == 202, created.text
    task = _wait_for_terminal_task(client, created.json()["id"])
    assert task["status"] == "completed", task
    image_id = uploaded.json()["items"][0]["id"]
    annotations = client.get(f"/api/datasets/{dataset_id}/images/{image_id}/annotations").json()
    assert annotations[0]["type"] == annotation_type


def _wait_for_terminal_task(client, task_id: str) -> dict:
    task = None
    for _ in range(40):
        task = client.get(f"/api/auto-annotation/{task_id}").json()
        if task["status"] in {"completed", "failed", "stopped"}:
            break
        time.sleep(0.025)
    assert task is not None
    return task
