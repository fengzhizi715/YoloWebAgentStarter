from __future__ import annotations

from threading import Event
from time import monotonic, sleep

from app.agent.provider import ProviderResponse, ProviderToolCall
from app.agent.providers.mock import MockAgentProvider
from app.agent.tools import AgentToolRegistry, AgentToolSpec
from app.core.models import TrainingTask


def wait_for_run(client, run_id):
    deadline = monotonic() + 5
    while monotonic() < deadline:
        run = client.get(f"/api/agent/runs/{run_id}").json()
        if run["status"] not in {"pending", "running"}:
            return run
        sleep(0.01)
    raise AssertionError("Agent did not finish")


def seed_task(client, dataset_id):
    with client.app.state.database.session_factory() as session:
        session.add(TrainingTask(
            id="train_aabb", dataset_id=dataset_id, name="context-task", status="failed",
            task_type="detect", model_name="yolo11n.pt", model_path="managed.pt",
            epochs=1, img_size=640, batch_size=1, device="cpu", error_message="test failure",
        ))
        session.commit()
    return "train_aabb"


def test_context_is_validated_persisted_and_sent_to_provider(client, monkeypatch):
    dataset_id = client.post("/api/datasets", json={"name": "context", "task_type": "detect"}).json()["id"]
    task_id = seed_task(client, dataset_id)
    captured = []

    def complete(_self, request):
        captured.append(request)
        return ProviderResponse(content="context received")

    monkeypatch.setattr(MockAgentProvider, "complete", complete)
    chat = client.post("/api/agent/sessions", json={}).json()["id"]
    response = client.post(f"/api/agent/sessions/{chat}/messages", json={
        "content": "这个训练为什么失败？", "read_only": True, "context": {"training_task_id": task_id},
    })
    assert response.status_code == 202
    run = wait_for_run(client, response.json()["id"])
    assert run["status"] == "completed"
    assert run["read_only"] is True
    assert run["context"]["dataset_id"] == dataset_id
    assert captured[0].context == {"training_task_id": task_id, "dataset_id": dataset_id}
    assert any(item.role == "system" and task_id in item.content for item in captured[0].messages)
    assert next(item.content for item in captured[0].messages if item.role == "user") == "这个训练为什么失败？"
    # A late cancel must not rewrite a successfully finalized result.
    cancelled = client.post(f"/api/agent/runs/{run['id']}/cancel").json()
    assert cancelled["status"] == "completed" and cancelled["stop_requested"] is False

    other_id = client.post("/api/datasets", json={"name": "other", "task_type": "detect"}).json()["id"]
    for context, code in [
        ({"training_task_id": "train_missing"}, "agent_context_not_found"),
        ({"training_task_id": task_id, "dataset_id": other_id}, "agent_context_mismatch"),
    ]:
        invalid = client.post(f"/api/agent/sessions/{chat}/messages", json={"content": "为什么？", "context": context})
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == code
    assert len(captured) == 1


def test_mock_uses_context_for_question_without_id(client):
    dataset_id = client.post("/api/datasets", json={"name": "context", "task_type": "detect"}).json()["id"]
    task_id = seed_task(client, dataset_id)
    chat = client.post("/api/agent/sessions", json={}).json()["id"]
    response = client.post(f"/api/agent/sessions/{chat}/messages", json={
        "content": "这个训练为什么失败？", "read_only": True, "context": {"training_task_id": task_id},
    })
    run = wait_for_run(client, response.json()["id"])
    assert run["status"] == "completed"
    assert run["tool_calls"][0]["arguments_json"]["task_id"] == task_id


def test_submit_returns_before_inference_and_cancel_discards_late_response(client, monkeypatch):
    entered, release = Event(), Event()

    def complete(_self, request):
        entered.set()
        assert release.wait(5)
        return ProviderResponse(content="late answer", tool_calls=[ProviderToolCall(
            name="create_training_task", arguments={}, call_id="late_write",
        )])

    monkeypatch.setattr(MockAgentProvider, "complete", complete)
    chat = client.post("/api/agent/sessions", json={}).json()["id"]
    try:
        response = client.post(f"/api/agent/sessions/{chat}/messages", json={"content": "start"})
        assert response.status_code == 202
        run_id = response.json()["id"]
        assert response.json()["status"] == "pending"
        assert entered.wait(3) and not release.is_set()
        assert client.get(f"/api/agent/runs/{run_id}").json()["status"] == "running"
        assert client.delete(f"/api/agent/sessions/{chat}").status_code == 409
        blocked = client.post(f"/api/agent/sessions/{chat}/messages", json={"content": "duplicate"})
        assert blocked.status_code == 409
        cancelled = client.post(f"/api/agent/runs/{run_id}/cancel")
        assert cancelled.json()["status"] == "cancelled"
    finally:
        release.set()
        client.app.state.agent_executor.shutdown()
    run = client.get(f"/api/agent/runs/{run_id}").json()
    assert run["status"] == "cancelled"
    assert not run["tool_calls"] and not run["approvals"]
    assert run["inference_steps"][0]["outcome"] == "discarded"
    assert [item["role"] for item in run["messages"]] == ["user"]


def test_cancel_during_tool_stops_remaining_calls(client, monkeypatch):
    entered, release = Event(), Event()
    writes = []

    def slow_read(_session, _arguments):
        entered.set()
        assert release.wait(5)
        return {"ok": True}

    registry = AgentToolRegistry()
    registry.register(AgentToolSpec(name="slow_read", kind="read", description="slow", handler=slow_read, canonical_name="global_summary"))
    registry.register(AgentToolSpec(name="create_training_task", kind="write", description="write",
                                    handler=lambda *_: writes.append(True)))
    monkeypatch.setattr("app.api.dependencies.build_default_registry", lambda *args, **kwargs: registry)
    monkeypatch.setattr(MockAgentProvider, "complete", lambda *_: ProviderResponse(content="", tool_calls=[
        ProviderToolCall(name="slow_read", arguments={}),
        ProviderToolCall(name="create_training_task", arguments={}),
    ]))
    chat = client.post("/api/agent/sessions", json={}).json()["id"]
    try:
        response = client.post(f"/api/agent/sessions/{chat}/messages", json={"content": "start"})
        run_id = response.json()["id"]
        assert entered.wait(3)
        assert client.get(f"/api/agent/runs/{run_id}").json()["tool_calls"][0]["status"] == "running"
        assert client.post(f"/api/agent/runs/{run_id}/cancel").json()["status"] == "cancelled"
    finally:
        release.set()
        client.app.state.agent_executor.shutdown()
    run = client.get(f"/api/agent/runs/{run_id}").json()
    assert run["status"] == "cancelled" and not run["approvals"]
    assert len(run["tool_calls"]) == 1 and run["tool_calls"][0]["status"] == "failed"
    assert writes == []


def test_retry_uses_saved_mode_and_context_and_rejects_changed_context(client, monkeypatch):
    dataset_id = client.post("/api/datasets", json={"name": "original", "task_type": "detect"}).json()["id"]
    chat = client.post("/api/agent/sessions", json={}).json()["id"]
    monkeypatch.setattr(MockAgentProvider, "complete", lambda *_: (_ for _ in ()).throw(RuntimeError("offline")))
    first = client.post(f"/api/agent/sessions/{chat}/messages", json={
        "content": "看看这个数据集", "read_only": True, "context": {"dataset_id": dataset_id},
    }).json()
    assert wait_for_run(client, first["id"])["status"] == "failed"
    observed = []

    def complete(_self, request):
        observed.append(request)
        return ProviderResponse(content="recovered")

    monkeypatch.setattr(MockAgentProvider, "complete", complete)
    response = client.post(f"/api/agent/runs/{first['id']}/retry", json={"read_only": False, "context": {}})
    assert response.status_code == 202
    retried = wait_for_run(client, response.json()["id"])
    assert retried["status"] == "completed"
    assert retried["read_only"] is True and retried["context"]["dataset_id"] == dataset_id
    assert observed[0].context == {"dataset_id": dataset_id}
    assert not any(tool["function"]["name"].startswith("create_") for tool in observed[0].tools)
    assert client.post(f"/api/agent/runs/{retried['id']}/retry").status_code == 409
    assert client.delete(f"/api/datasets/{dataset_id}").status_code == 204
    assert client.post(f"/api/agent/runs/{first['id']}/retry").status_code == 422
