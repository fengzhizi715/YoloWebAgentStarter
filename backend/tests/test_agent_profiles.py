from time import monotonic, sleep

import pytest

from app.agent.provider import ProviderResponse, ProviderToolCall
from app.agent.providers.mock import MockAgentProvider
from app.agent.profiles import PROFILES, get_profile
from app.agent.read_tools import build_default_registry
from app.core.errors import ValidationError
from app.core.models import AgentRun, AgentSession


def send(client, chat, **payload):
    return client.post(f"/api/agent/sessions/{chat}/messages?wait_for_completion=true", json={"content": "hello", **payload})


def test_session_infers_and_inherits_validated_binding(client):
    dataset = client.post("/api/datasets", json={"name": "bound", "task_type": "detect"}).json()["id"]
    response = client.post("/api/agent/sessions", json={"context": {"dataset_id": dataset}})
    chat = response.json()
    assert chat["profile_id"] == "dataset" and chat["profile_version"] == 1
    run = send(client, chat["id"]).json()
    assert run["context"]["dataset_id"] == dataset
    assert run["profile_id"] == "dataset"
    assert client.get(f"/api/agent/sessions/{chat['id']}").json()["context"]["dataset_id"] == dataset
    assert [item["id"] for item in client.get("/api/agent/profiles").json()] == list(PROFILES)
    assert client.post("/api/agent/sessions", json={"profile_id": "workflow"}).status_code == 422
    assert client.post("/api/agent/sessions", json={"context": {"dataset_id": "ds_missing"}}).status_code == 422


def test_context_or_profile_switch_requires_explicit_confirmation(client):
    dataset = client.post("/api/datasets", json={"name": "bound", "task_type": "detect"}).json()["id"]
    chat = client.post("/api/agent/sessions", json={}).json()["id"]
    first = send(client, chat).json()
    change = {"profile_id": "dataset", "context": {"dataset_id": dataset}}
    assert send(client, chat, **change).status_code == 409
    changed = send(client, chat, **change, allow_context_change=True).json()
    assert changed["profile_id"] == "dataset"
    assert client.get(f"/api/agent/runs/{first['id']}").json()["profile_id"] == "global"
    assert send(client, chat, profile_id="training").status_code == 409
    assert send(client, chat, profile_id="training", allow_context_change=True).json()["profile_id"] == "training"


def test_explicit_context_does_not_replace_the_session_mode(client):
    dataset = client.post("/api/datasets", json={"name": "bound", "task_type": "detect"}).json()["id"]
    context = {"dataset_id": dataset}
    chat = client.post("/api/agent/sessions", json={"profile_id": "model", "context": context}).json()["id"]
    for _ in range(2):
        response = send(client, chat, context=context)
        assert response.status_code == 200
        assert response.json()["profile_id"] == "model"
    cleared = send(client, chat, context={}, allow_context_change=True).json()
    assert cleared["profile_id"] == "model" and cleared["context"]["dataset_id"] is None


@pytest.mark.parametrize("explicit_profile", [False, True])
def test_message_cannot_silently_reset_a_saved_profile_version(client, explicit_profile):
    chat = client.post("/api/agent/sessions", json={"profile_id": "model"}).json()["id"]
    with client.app.state.database.session_factory() as db:
        db.get(AgentSession, chat).profile_version = 999
        db.commit()
    response = send(client, chat, **({"profile_id": "model"} if explicit_profile else {}))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "agent_profile_unavailable"
    assert client.get(f"/api/agent/sessions/{chat}").json()["message_count"] == 0


@pytest.mark.parametrize("tool_name", ["training_logs", "create_training_task"])
def test_profile_filters_schema_and_blocks_out_of_scope_model_tool_calls(client, monkeypatch, tool_name):
    seen = []

    def complete(_self, request):
        seen.append(request)
        if request.messages[-1].role == "tool":
            return ProviderResponse(content="denied")
        return ProviderResponse(content="", tool_calls=[ProviderToolCall(name=tool_name, arguments={})])

    monkeypatch.setattr(MockAgentProvider, "complete", complete)
    chat = client.post("/api/agent/sessions", json={"profile_id": "model"}).json()["id"]
    run = send(client, chat).json()
    assert run["status"] == "completed" and not run["approvals"]
    assert run["tool_calls"][0]["result_json"]["error"] == "agent_profile_tool_not_allowed"
    assert all(tool_name not in [tool["function"]["name"] for tool in req.tools] for req in seen)
    assert any("model@1" in message.content for message in seen[0].messages)


def test_legacy_alias_obeys_canonical_profile_allowlist(client):
    registry = build_default_registry(client.app.state.storage)
    from app.agent.profiles import permits
    from app.agent.tools import AgentToolSpec
    assert permits(get_profile("model"), registry.get("get_training_task"))
    assert not permits(get_profile("model"), AgentToolSpec(name="old_logs", kind="read", description="", canonical_name="training_logs"))


def test_retry_restores_original_profile_context_and_read_only(client, monkeypatch):
    chat = client.post("/api/agent/sessions", json={"profile_id": "training"}).json()["id"]
    monkeypatch.setattr(MockAgentProvider, "complete", lambda *_: (_ for _ in ()).throw(RuntimeError("offline")))
    failed = send(client, chat, read_only=True).json()
    monkeypatch.setattr(MockAgentProvider, "complete", lambda *_: ProviderResponse(content="ok"))
    assert send(client, chat, profile_id="global", allow_context_change=True).json()["status"] == "completed"
    retry = client.post(f"/api/agent/runs/{failed['id']}/retry", json={"profile_id": "model"}).json()
    assert retry["profile_id"] == "training" and retry["profile_version"] == 1 and retry["read_only"] is True
    deadline = monotonic() + 3
    while monotonic() < deadline:
        if client.get(f"/api/agent/runs/{retry['id']}").json()["status"] == "completed":
            break
        sleep(0.01)
    assert client.get(f"/api/agent/sessions/{chat}").json()["profile_id"] == "training"
    with client.app.state.database.session_factory() as db:
        assert db.get(AgentSession, chat).profile_id == "training"
        old = db.get(AgentRun, failed["id"])
        old.profile_version = 999
        db.commit()
    assert client.post(f"/api/agent/runs/{failed['id']}/retry").status_code == 422


def test_unknown_profile_versions_fail_closed():
    with pytest.raises(ValidationError):
        get_profile("dataset", 999)


def test_approval_rechecks_saved_profile_before_dispatch(client):
    from datetime import timedelta
    from app.core.models import AgentApproval
    from app.core.time import utc_now

    chat = client.post("/api/agent/sessions", json={"profile_id": "model"}).json()["id"]
    with client.app.state.database.session_factory() as db:
        db.add(AgentRun(id="arun_recheck", session_id=chat, status="awaiting_approval", provider="mock",
                        model="mock", profile_id="model", profile_version=1))
        db.flush()
        db.add(AgentApproval(id="approval_recheck", run_id="arun_recheck", tool_name="create_training_task",
            payload_json={}, status="pending", idempotency_key="recheck", expires_at=utc_now() + timedelta(hours=1)))
        db.commit()
    response = client.post("/api/agent/approvals/approval_recheck/approve")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "agent_profile_tool_not_allowed"
    assert client.get("/api/training/tasks").json()["items"] == []
