from __future__ import annotations

import io
from datetime import timedelta
from time import monotonic, sleep

import pytest
from PIL import Image

from app.agent.providers.mock import _plan_tools
from app.core.models import AgentApproval
from app.core.time import utc_now


def image_bytes() -> bytes:
    image = Image.new("RGB", (80, 60), "#6688aa")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def fake_yolo(path) -> None:
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


def prepared_dataset(client, task_type: str = "detect") -> tuple[str, str]:
    dataset = client.post("/api/datasets", json={"name": f"agent-train-{task_type}", "task_type": task_type}).json()
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
        response = client.put(
            f"/api/datasets/{dataset_id}/images/{image['id']}/annotations",
            json={"annotations": [{"type": "bbox", "class_id": label, "bbox": {"x": 5, "y": 5, "width": 25, "height": 20}}]},
        )
        assert response.status_code == 200, response.text
    return dataset_id, label


def test_mock_plans_write_tools():
    available = {
        "create_training_task",
        "create_model_evaluation",
        "create_auto_annotation_task",
        "list_training_tasks",
        "list_model_evaluations",
    }
    training = _plan_tools("请为 ds_abc123def456 创建训练任务", available)
    assert [item.name for item in training] == ["create_training_task"]
    assert training[0].arguments["dataset_id"] == "ds_abc123def456"

    evaluation = _plan_tools("对 model_train_1_best 创建评估", available)
    assert [item.name for item in evaluation] == ["create_model_evaluation"]

    auto = _plan_tools("用 model_train_1_best 对 ds_abc123def456 自动标注", available)
    assert [item.name for item in auto] == ["create_auto_annotation_task"]


@pytest.mark.parametrize("wait_for_completion", [True, False])
def test_write_tool_pauses_for_approval_without_creating_task(client, tmp_path, monkeypatch, wait_for_completion):
    executable = tmp_path / "fake-yolo"
    fake_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))
    dataset_id, _ = prepared_dataset(client)

    session_id = client.post("/api/agent/sessions", json={"title": "write"}).json()["id"]
    run = client.post(
        f"/api/agent/sessions/{session_id}/messages?wait_for_completion={str(wait_for_completion).lower()}",
        json={"content": f"请为 {dataset_id} 创建训练任务"},
    )
    assert run.status_code == (200 if wait_for_completion else 202), run.text
    body = run.json()
    deadline = monotonic() + 5
    while body["status"] in {"pending", "running"} and monotonic() < deadline:
        sleep(0.01)
        body = client.get(f"/api/agent/runs/{body['id']}").json()
    assert body["status"] == "awaiting_approval"
    assert body["approvals"]
    approval = body["approvals"][0]
    assert approval["status"] == "pending"
    assert approval["tool_name"] == "create_training_task"
    assert approval["payload_json"]["dataset_id"] == dataset_id
    assert body["tool_calls"][0]["status"] == "awaiting_approval"
    assert body["tool_calls"][0]["result_json"]["requires_approval"] is True

    listed = client.get(f"/api/training/tasks?dataset_id={dataset_id}").json()
    assert listed["items"] == []

    blocked = client.post(f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true", json={"content": "next"})
    assert blocked.status_code == 409
    cancelled = client.post(f"/api/agent/runs/{body['id']}/cancel").json()
    assert cancelled["status"] == "cancelled"
    assert cancelled["approvals"][0]["status"] == "rejected"
    assert client.post(f"/api/agent/approvals/{approval['id']}/approve").status_code == 409
    assert client.get(f"/api/training/tasks?dataset_id={dataset_id}").json()["items"] == []


def test_approve_creates_training_task_once(client, tmp_path, monkeypatch):
    executable = tmp_path / "fake-yolo"
    fake_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))
    dataset_id, _ = prepared_dataset(client)

    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    run = client.post(
        f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true",
        json={"content": f"请为 {dataset_id} 开始训练"},
    ).json()
    approval_id = run["approvals"][0]["id"]

    first = client.post(f"/api/agent/approvals/{approval_id}/approve")
    assert first.status_code == 200, first.text
    payload = first.json()
    assert payload["approval"]["status"] == "executed"
    task_id = payload["approval"]["result_task_id"]
    assert task_id and task_id.startswith("train_")
    assert payload["run"]["status"] == "completed"
    assert any(item["role"] == "assistant" and task_id in item["content"] for item in payload["run"]["messages"])

    listed = client.get(f"/api/training/tasks?dataset_id={dataset_id}").json()["items"]
    assert len(listed) == 1
    assert listed[0]["id"] == task_id

    second = client.post(f"/api/agent/approvals/{approval_id}/approve")
    assert second.status_code == 200, second.text
    assert second.json()["approval"]["result_task_id"] == task_id
    listed_again = client.get(f"/api/training/tasks?dataset_id={dataset_id}").json()["items"]
    assert len(listed_again) == 1


def test_approve_accepts_edited_payload(client, tmp_path, monkeypatch):
    executable = tmp_path / "fake-yolo"
    fake_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))
    dataset_id, _ = prepared_dataset(client)

    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    run = client.post(
        f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true",
        json={"content": f"请为 {dataset_id} 创建训练任务"},
    ).json()
    approval_id = run["approvals"][0]["id"]

    approved = client.post(
        f"/api/agent/approvals/{approval_id}/approve",
        json={"payload": {"epochs": 7, "model": "yolo11n.pt", "shell": "rm -rf /"}},
    )
    assert approved.status_code == 200, approved.text
    stored = approved.json()["approval"]["payload_json"]
    assert stored["epochs"] == 7
    assert stored["model"] == "yolo11n.pt"
    assert "shell" not in stored
    task_id = approved.json()["approval"]["result_task_id"]
    task = client.get(f"/api/training/tasks/{task_id}").json()
    assert task["epochs"] == 7


def test_reject_approval_cancels_run_without_task(client, tmp_path, monkeypatch):
    executable = tmp_path / "fake-yolo"
    fake_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))
    dataset_id, _ = prepared_dataset(client)

    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    run = client.post(
        f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true",
        json={"content": f"请为 {dataset_id} 提交训练"},
    ).json()
    approval_id = run["approvals"][0]["id"]

    rejected = client.post(f"/api/agent/approvals/{approval_id}/reject")
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["approval"]["status"] == "rejected"
    assert rejected.json()["run"]["status"] == "cancelled"
    assert client.get(f"/api/training/tasks?dataset_id={dataset_id}").json()["items"] == []

    again = client.post(f"/api/agent/approvals/{approval_id}/reject")
    assert again.status_code == 200
    assert again.json()["approval"]["status"] == "rejected"


def test_expired_approval_cannot_execute(client, tmp_path, monkeypatch):
    executable = tmp_path / "fake-yolo"
    fake_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))
    dataset_id, _ = prepared_dataset(client)

    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    run = client.post(
        f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true",
        json={"content": f"请为 {dataset_id} 创建训练任务"},
    ).json()
    approval_id = run["approvals"][0]["id"]

    with client.app.state.database.session_factory() as db:
        approval = db.get(AgentApproval, approval_id)
        assert approval is not None
        approval.expires_at = utc_now() - timedelta(seconds=5)
        db.commit()

    expired = client.post(f"/api/agent/approvals/{approval_id}/approve")
    assert expired.status_code == 409
    assert expired.json()["error"]["code"] == "agent_approval_expired"
    assert client.get(f"/api/training/tasks?dataset_id={dataset_id}").json()["items"] == []


def test_approve_auto_annotation_after_confirmation(client, monkeypatch):
    from app.core.models import ModelVersion
    from app.models.result_parser import Detection

    dataset = client.post("/api/datasets", json={"name": "agent-auto", "task_type": "detect"}).json()
    dataset_id = dataset["id"]
    client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "cat"})
    from PIL import Image
    import io

    image_data = io.BytesIO()
    Image.new("RGB", (100, 80), "#4477aa").save(image_data, format="PNG")
    client.post(
        f"/api/datasets/{dataset_id}/images/upload",
        files={"files": ("cat.png", image_data.getvalue(), "image/png")},
    )
    model_id = "model_agent_auto"
    model_path = client.app.state.storage.model_version_dir(model_id) / "best.pt"
    model_path.write_bytes(b"fake-managed-model")
    with client.app.state.database.session_factory() as session:
        session.add(
            ModelVersion(
                id=model_id,
                name="agent-auto-model",
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

    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    run = client.post(
        f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true",
        json={"content": f"用 {model_id} 对 {dataset_id} 自动标注"},
    ).json()
    assert run["status"] == "awaiting_approval"
    assert run["approvals"][0]["tool_name"] == "create_auto_annotation_task"

    approved = client.post(f"/api/agent/approvals/{run['approvals'][0]['id']}/approve")
    assert approved.status_code == 200, approved.text
    task_id = approved.json()["approval"]["result_task_id"]
    assert task_id and task_id.startswith("auto_")
    assert approved.json()["run"]["status"] == "completed"


def test_rejected_approval_cannot_be_approved_later(client, tmp_path, monkeypatch):
    executable = tmp_path / "fake-yolo"
    fake_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))
    dataset_id, _ = prepared_dataset(client)

    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    run = client.post(
        f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true",
        json={"content": f"请为 {dataset_id} 创建训练任务"},
    ).json()
    approval_id = run["approvals"][0]["id"]
    assert client.post(f"/api/agent/approvals/{approval_id}/reject").status_code == 200
    blocked = client.post(f"/api/agent/approvals/{approval_id}/approve")
    assert blocked.status_code == 409
    assert blocked.json()["error"]["code"] == "agent_approval_rejected"
    assert client.get(f"/api/training/tasks?dataset_id={dataset_id}").json()["items"] == []


def test_write_tools_are_registered_as_write_kind(client):
    from app.agent.read_tools import build_default_registry

    registry = build_default_registry(
        client.app.state.storage,
        session_factory=client.app.state.database.session_factory,
    )
    for name in ("create_training_task", "create_model_evaluation", "create_auto_annotation_task"):
        spec = registry.get(name)
        assert spec is not None
        assert spec.kind == "write"


def test_write_propose_does_not_call_training_create(client, monkeypatch):
    dataset_id, _ = prepared_dataset(client)
    called = {"create": 0}

    def boom(*_args, **_kwargs):
        called["create"] += 1
        raise AssertionError("create_task must not run before approval")

    monkeypatch.setattr("app.training.service.TrainingService.create_task", boom)
    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    run = client.post(
        f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true",
        json={"content": f"请为 {dataset_id} 创建训练任务"},
    )
    assert run.status_code == 200, run.text
    assert run.json()["status"] == "awaiting_approval"
    assert called["create"] == 0


def test_approve_resumes_reserved_task_id_without_duplicate(client, tmp_path, monkeypatch):
    executable = tmp_path / "fake-yolo"
    fake_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))
    dataset_id, _ = prepared_dataset(client)

    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    run = client.post(
        f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true",
        json={"content": f"请为 {dataset_id} 创建训练任务"},
    ).json()
    approval_id = run["approvals"][0]["id"]

    # Simulate crash after claim + reserve + create, before executed finalize.
    first = client.post(f"/api/agent/approvals/{approval_id}/approve")
    assert first.status_code == 200, first.text
    task_id = first.json()["approval"]["result_task_id"]
    with client.app.state.database.session_factory() as db:
        from app.core.models import AgentApproval, AgentRun

        approval = db.get(AgentApproval, approval_id)
        run_row = db.get(AgentRun, run["id"])
        assert approval is not None and run_row is not None
        approval.status = "approved"
        approval.result_task_id = task_id
        run_row.status = "awaiting_approval"
        run_row.finished_at = None
        db.commit()

    second = client.post(f"/api/agent/approvals/{approval_id}/approve")
    assert second.status_code == 200, second.text
    assert second.json()["approval"]["status"] == "executed"
    assert second.json()["approval"]["result_task_id"] == task_id
    listed = client.get(f"/api/training/tasks?dataset_id={dataset_id}").json()["items"]
    assert len(listed) == 1
    assert listed[0]["id"] == task_id


def test_approve_strips_tampered_payload_keys(client, tmp_path, monkeypatch):
    executable = tmp_path / "fake-yolo"
    fake_yolo(executable)
    monkeypatch.setenv("YWA_YOLO_EXECUTABLE", str(executable))
    dataset_id, _ = prepared_dataset(client)
    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    run = client.post(
        f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true",
        json={"content": f"请为 {dataset_id} 创建训练任务"},
    ).json()
    approval_id = run["approvals"][0]["id"]
    with client.app.state.database.session_factory() as db:
        from app.core.models import AgentApproval

        approval = db.get(AgentApproval, approval_id)
        assert approval is not None
        payload = dict(approval.payload_json)
        payload["shell"] = "rm -rf /"
        payload["model_path"] = "/tmp/evil.pt"
        approval.payload_json = payload
        db.commit()

    approved = client.post(f"/api/agent/approvals/{approval_id}/approve")
    assert approved.status_code == 200, approved.text
    stored = approved.json()["approval"]["payload_json"]
    assert "shell" not in stored
    assert "model_path" not in stored
    assert stored["dataset_id"] == dataset_id
