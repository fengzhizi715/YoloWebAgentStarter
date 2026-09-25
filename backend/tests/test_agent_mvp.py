from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.agent.providers.mock import MockAgentProvider
from app.agent.provider import ProviderResponse, ProviderToolCall
from app.agent.service import AgentService
from app.core.config import Settings
from app.core.models import AgentApproval, AgentRun
from app.core.errors import ConflictError
from app.core.time import utc_now
from app.main import create_app


def test_agent_status_exposes_provider_without_api_key(client):
    response = client.get("/api/agent/status")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["provider"] == "mock"
    assert payload["model"] == "mock-model"
    assert payload["configured"] is True
    assert payload["api_key_configured"] is False
    assert "api_key" not in payload


def test_agent_session_message_run_persists_with_mock_provider(client):
    created = client.post("/api/agent/sessions", json={"title": "Dataset questions"})
    assert created.status_code == 201, created.text
    session_id = created.json()["id"]
    assert created.json()["title"] == "Dataset questions"

    run = client.post(f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true", json={"content": "hello agent"})
    assert run.status_code == 200, run.text
    body = run.json()
    assert body["status"] == "completed"
    assert body["provider"] == "mock"
    assert body["model"] == "mock-model"
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][1]["role"] == "assistant"
    assert "数据集" in body["messages"][1]["content"] or "Agent" in body["messages"][1]["content"]

    detail = client.get(f"/api/agent/sessions/{session_id}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["message_count"] == 2
    assert detail.json()["latest_run_status"] == "completed"
    assert len(detail.json()["runs"]) == 1

    listed = client.get("/api/agent/sessions")
    assert listed.status_code == 200
    assert any(item["id"] == session_id for item in listed.json())

    fetched = client.get(f"/api/agent/runs/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "completed"


def test_agent_multi_turn_uses_current_question_and_valid_tool_protocol(client, monkeypatch):
    observed = []

    def complete(_self, request):
        observed.append(request.messages)
        return ProviderResponse(content="first answer")

    monkeypatch.setattr(MockAgentProvider, "complete", complete)
    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    first = client.post(f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true", json={"content": "first"})
    assert first.json()["status"] == "completed"
    # Use the real read tool result in the second turn; the callback detects it by role.
    def second_complete(_self, request):
        observed.append(request.messages)
        latest_user = next(item.content for item in reversed(request.messages) if item.role == "user")
        if request.messages[-1].role == "tool":
            assert request.messages[-2].role == "assistant"
            assert request.messages[-2].tool_calls[0].call_id == request.messages[-1].tool_call_id
            return ProviderResponse(content=f"answered {latest_user}")
        return ProviderResponse(content="", tool_calls=[ProviderToolCall(name="global_summary", arguments={}, call_id="call_new")])

    monkeypatch.setattr(MockAgentProvider, "complete", second_complete)
    second = client.post(f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true", json={"content": "second"})
    assert second.json()["status"] == "completed", second.text
    assert second.json()["messages"][-1]["content"] == "answered second"
    assert len(second.json()["tool_calls"]) == 1
    assert observed[-1][-1].role == "tool"


def test_read_only_run_rejects_provider_requested_write(client, monkeypatch):
    observed_tools = []

    def malicious_complete(_self, request):
        observed_tools.append([item["function"]["name"] for item in request.tools])
        if request.messages[-1].role == "tool":
            return ProviderResponse(content="Write was denied")
        return ProviderResponse(content="", tool_calls=[ProviderToolCall(
            name="create_training_task", arguments={"dataset_id": "ds_abc"}, call_id="call_write",
        )])

    monkeypatch.setattr(MockAgentProvider, "complete", malicious_complete)
    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    response = client.post(f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true", json={
        "content": "Read-only question", "read_only": True,
    })
    assert response.json()["status"] == "completed", response.text
    assert all("create_training_task" not in names for names in observed_tools)
    assert response.json()["tool_calls"][0]["result_json"]["error"] == "read_only_run"
    assert response.json()["approvals"] == []


def test_agent_rejects_second_message_while_run_awaiting_approval(tmp_path, monkeypatch):
    settings = Settings(
        project_root=Path(__file__).resolve().parents[2],
        data_dir=tmp_path / "data",
        import_root=tmp_path / "imports",
        database_url=f"sqlite:///{tmp_path / 'agent.db'}",
    )
    with TestClient(create_app(settings)) as client:
        session_id = client.post("/api/agent/sessions", json={}).json()["id"]

        def hang(_request):
            raise AssertionError("provider should not be called for this fixture")

        monkeypatch.setattr(MockAgentProvider, "complete", hang)
        # Seed an awaiting_approval run directly so Week 1 can cover the lock.
        with client.app.state.database.session_factory() as db:
            from app.core.ids import new_id
            from app.core.models import AgentRun as RunModel

            db.add(
                RunModel(
                    id=new_id("arun"),
                    session_id=session_id,
                    status="awaiting_approval",
                    provider="mock",
                    model="mock-model",
                )
            )
            db.commit()

        blocked = client.post(f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true", json={"content": "next"})
        assert blocked.status_code == 409
        assert blocked.json()["error"]["code"] == "agent_run_in_progress"


def test_agent_cancel_and_restart_recovery(tmp_path):
    settings = Settings(
        project_root=Path(__file__).resolve().parents[2],
        data_dir=tmp_path / "data",
        import_root=tmp_path / "imports",
        database_url=f"sqlite:///{tmp_path / 'agent-recover.db'}",
    )
    with TestClient(create_app(settings)) as client:
        session_id = client.post("/api/agent/sessions", json={}).json()["id"]
        with client.app.state.database.session_factory() as db:
            from app.core.ids import new_id

            running = AgentRun(
                id=new_id("arun"),
                session_id=session_id,
                status="running",
                provider="mock",
                model="mock-model",
                started_at=utc_now(),
            )
            awaiting = AgentRun(
                id=new_id("arun"),
                session_id=session_id,
                status="awaiting_approval",
                provider="mock",
                model="mock-model",
                started_at=utc_now(),
            )
            db.add_all([running, awaiting])
            db.commit()
            running_id = running.id
            awaiting_id = awaiting.id

        cancelled = client.post(f"/api/agent/runs/{awaiting_id}/cancel")
        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json()["status"] == "cancelled"

        AgentService(client.app.state.database.session_factory, settings).recover_orphaned()
        with client.app.state.database.session_factory() as db:
            recovered = db.get(AgentRun, running_id)
            kept = db.get(AgentRun, awaiting_id)
            assert recovered is not None and recovered.status == "failed"
            assert "restart" in (recovered.error_message or "")
            assert kept is not None and kept.status == "cancelled"


def test_agent_approval_idempotency_key_is_unique(client):
    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    service = AgentService(client.app.state.database.session_factory, client.app.state.settings)
    with client.app.state.database.session_factory() as db:
        from app.core.ids import new_id

        run = AgentRun(
            id=new_id("arun"),
            session_id=session_id,
            status="awaiting_approval",
            provider="mock",
            model="mock-model",
        )
        db.add(run)
        db.flush()
        first = service.create_approval_record(
            db,
            run=run,
            tool_name="create_training_task",
            payload_json={"dataset_id": "ds_demo"},
            idempotency_key="same-key",
        )
        second = service.create_approval_record(
            db,
            run=run,
            tool_name="create_training_task",
            payload_json={"dataset_id": "ds_demo"},
            idempotency_key="same-key",
        )
        db.commit()
        assert first.id == second.id
        rows = list(db.scalars(select(AgentApproval).where(AgentApproval.idempotency_key == "same-key")))
        assert len(rows) == 1


def test_agent_approval_allows_exactly_one_concurrent_executor(client, monkeypatch):
    from datetime import timedelta

    from app.core.ids import new_id

    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    with client.app.state.database.session_factory() as db:
        run = AgentRun(
            id=new_id("arun"),
            session_id=session_id,
            status="awaiting_approval",
            provider="mock",
            model="mock-model",
        )
        db.add(run)
        db.flush()
        approval = AgentApproval(
            id=new_id("aappr"),
            run_id=run.id,
            tool_name="create_training_task",
            payload_json={"dataset_id": "ds_demo", "name": "demo"},
            status="pending",
            idempotency_key=f"{run.id}:concurrent",
            expires_at=utc_now() + timedelta(hours=1),
        )
        db.add(approval)
        db.commit()
        approval_id = approval.id

    from app.agent.read_tools import build_default_registry
    service = AgentService(client.app.state.database.session_factory, client.app.state.settings,
                           registry=build_default_registry(client.app.state.storage))
    executing = Event()
    release = Event()
    calls = {"count": 0}
    calls_lock = Lock()

    def execute_once(_db, current, *, reserved_task_id=None):
        with calls_lock:
            calls["count"] += 1
        executing.set()
        assert release.wait(timeout=5)
        return reserved_task_id or current.result_task_id, "created by test"

    monkeypatch.setattr(service, "_execute_approved_payload", execute_once)

    def approve() -> str:
        with client.app.state.database.session_factory() as db:
            try:
                return service.approve_approval(db, approval_id).approval.status
            except ConflictError as exc:
                return exc.error_code

    with ThreadPoolExecutor(max_workers=2) as executor:
        winner = executor.submit(approve)
        assert executing.wait(timeout=5)
        loser = executor.submit(approve)
        assert loser.result(timeout=5) == "agent_approval_execution_in_progress"
        release.set()
        assert winner.result(timeout=5) == "executed"

    assert calls["count"] == 1


def test_agent_api_key_stays_out_of_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("YWA_AGENT_API_KEY", "sk-test-secret")
    monkeypatch.setenv("YWA_AGENT_PROVIDER", "mock")
    settings = Settings.from_env()
    # Rebuild with tmp paths while keeping env-derived agent secrets.
    settings = Settings(
        project_root=settings.project_root,
        data_dir=tmp_path / "data",
        import_root=tmp_path / "imports",
        database_url=f"sqlite:///{tmp_path / 'secret.db'}",
        agent_provider=settings.agent_provider,
        agent_model=settings.agent_model,
        agent_api_key=settings.agent_api_key,
    )
    assert settings.agent_api_key == "sk-test-secret"
    with TestClient(create_app(settings)) as client:
        status = client.get("/api/agent/status").json()
        assert status["api_key_configured"] is True
        session_id = client.post("/api/agent/sessions", json={}).json()["id"]
        client.post(f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true", json={"content": "hello"})
        db_bytes = Path(str(settings.database_url).removeprefix("sqlite:///")).read_bytes()
        assert b"sk-test-secret" not in db_bytes


def test_agent_session_survives_app_restart(tmp_path):
    db_path = tmp_path / "persist.db"
    settings = Settings(
        project_root=Path(__file__).resolve().parents[2],
        data_dir=tmp_path / "data",
        import_root=tmp_path / "imports",
        database_url=f"sqlite:///{db_path}",
    )
    with TestClient(create_app(settings)) as client:
        session_id = client.post("/api/agent/sessions", json={"title": "persist-me"}).json()["id"]
        dataset_id = client.post("/api/datasets", json={"name": "context", "task_type": "detect"}).json()["id"]
        run = client.post(f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true", json={
            "content": "hello agent", "read_only": True, "context": {"dataset_id": dataset_id}, "profile_id": "dataset",
        }).json()
        run_id = run["id"]
        assert run["status"] == "completed"

    with TestClient(create_app(settings)) as client:
        detail = client.get(f"/api/agent/sessions/{session_id}")
        assert detail.status_code == 200, detail.text
        body = detail.json()
        assert body["title"] == "persist-me"
        assert body["message_count"] >= 2
        assert any(item["id"] == run_id for item in body["runs"])
        fetched = client.get(f"/api/agent/runs/{run_id}")
        assert fetched.status_code == 200
        assert fetched.json()["status"] == "completed"
        assert fetched.json()["read_only"] is True
        assert fetched.json()["context"]["dataset_id"] == dataset_id
        assert fetched.json()["actual_source"] == "mock"
        assert fetched.json()["inference_steps"] == run["inference_steps"]
        assert fetched.json()["profile_id"] == "dataset"
        assert body["profile_id"] == "dataset" and body["context"]["dataset_id"] == dataset_id


def test_awaiting_approval_survives_restart_and_remains_approvable(tmp_path):
    from datetime import timedelta

    from app.core.ids import new_id
    from app.core.models import AgentApproval, AgentRun
    from app.core.time import utc_now

    db_path = tmp_path / "await.db"
    settings = Settings(
        project_root=Path(__file__).resolve().parents[2],
        data_dir=tmp_path / "data",
        import_root=tmp_path / "imports",
        database_url=f"sqlite:///{db_path}",
    )
    with TestClient(create_app(settings)) as client:
        session_id = client.post("/api/agent/sessions", json={}).json()["id"]
        with client.app.state.database.session_factory() as db:
            run = AgentRun(
                id=new_id("arun"),
                session_id=session_id,
                status="awaiting_approval",
                provider="mock",
                model="mock-model",
                started_at=utc_now(),
            )
            db.add(run)
            db.flush()
            approval = AgentApproval(
                id=new_id("aappr"),
                run_id=run.id,
                tool_name="create_training_task",
                payload_json={"dataset_id": "ds_missing", "name": "x", "action": "create_training_task"},
                status="pending",
                idempotency_key=f"{run.id}:seed",
                expires_at=utc_now() + timedelta(hours=1),
            )
            db.add(approval)
            db.commit()
            run_id = run.id
            approval_id = approval.id

    with TestClient(create_app(settings)) as client:
        AgentService(client.app.state.database.session_factory, settings).recover_orphaned()
        detail = client.get(f"/api/agent/sessions/{session_id}").json()
        assert any(item["id"] == run_id and item["status"] == "awaiting_approval" for item in detail["runs"])
        # Approve remains reachable after restart; claim may advance to approved, but never executed.
        response = client.post(f"/api/agent/approvals/{approval_id}/approve")
        assert response.status_code in {404, 409, 422}
        with client.app.state.database.session_factory() as db:
            kept = db.get(AgentApproval, approval_id)
            assert kept is not None
            assert kept.status in {"pending", "approved"}
            assert kept.status != "executed"
