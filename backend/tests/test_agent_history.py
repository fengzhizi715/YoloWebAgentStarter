from datetime import timedelta

import pytest
from sqlalchemy import event, inspect

from app.core.models import AgentApproval, AgentMessage, AgentRun, AgentSession, AgentToolCall
from app.core.time import utc_now


def seed_sessions(client, count=65):
    stamp = utc_now()
    with client.app.state.database.session_factory() as db:
        db.add_all([AgentSession(id=f"asess_{i:03d}", title=f"Chat {i}", created_at=stamp, updated_at=stamp)
                    for i in range(count)])
        db.commit()


def test_session_pages_have_stable_tie_order_and_constant_query_count(client):
    seed_sessions(client)
    queries = []
    engine = client.app.state.database.engine

    def observe(_connection, _cursor, statement, *_args):
        if statement.lstrip().upper().startswith("SELECT"):
            queries.append(statement)

    event.listen(engine, "before_cursor_execute", observe)
    try:
        page = client.get("/api/agent/sessions/page").json()
    finally:
        event.remove(engine, "before_cursor_execute", observe)
    assert len(page["items"]) == 30
    assert len(queries) == 3  # one page, batched counts, batched latest statuses
    ids = [item["id"] for item in page["items"]]
    while page["next_cursor"]:
        page = client.get("/api/agent/sessions/page", params={"cursor": page["next_cursor"]}).json()
        ids.extend(item["id"] for item in page["items"])
    assert ids == [f"asess_{i:03d}" for i in reversed(range(65))]


def test_cursor_survives_deleted_anchor_and_search_is_literal(client):
    seed_sessions(client, 4)
    first = client.get("/api/agent/sessions/page", params={"limit": 2}).json()
    assert client.delete(f"/api/agent/sessions/{first['items'][-1]['id']}").status_code == 204
    second = client.get("/api/agent/sessions/page", params={"cursor": first["next_cursor"]}).json()
    assert [item["id"] for item in second["items"]] == ["asess_001", "asess_000"]
    with client.app.state.database.session_factory() as db:
        row = db.get(AgentSession, "asess_000")
        row.title = "Quality 100%_ready"
        row.profile_id = "training"
        row.context_json = {"dataset_id": "ds_parent", "training_task_id": "train_old"}
        db.commit()
    for query in ("QUALITY", "%_", "train_old"):
        items = client.get("/api/agent/sessions/page", params={"q": query}).json()["items"]
        assert [item["id"] for item in items] == ["asess_000"]
    assert client.get("/api/agent/sessions/page", params={"q": "absent"}).json()["items"] == []
    assert client.get("/api/agent/sessions/page", params={"q": "training_task_id"}).json()["items"] == []
    # Child-only navigation matches a server-derived parent but not a conflicting parent or extra child.
    params = {"profile_id": "training", "match_context": True, "training_task_id": "train_old"}
    assert client.get("/api/agent/sessions/page", params=params).json()["items"][0]["id"] == "asess_000"
    for extra in ({"dataset_id": "ds_wrong"}, {"model_id": "model_extra"}, {"profile_id": "global"}):
        assert client.get("/api/agent/sessions/page", params={**params, **extra}).json()["items"] == []


@pytest.mark.parametrize("params", [{"cursor": "broken"}, {"cursor": "W251bGwsMV0="},
                                    {"limit": 0}, {"limit": 101}, {"dataset_id": ""}, {"profile_id": "invalid"}])
def test_page_rejects_invalid_parameters(client, params):
    assert client.get("/api/agent/sessions/page", params=params).status_code == 422


def test_large_timeline_is_bounded_and_keeps_historical_evidence_on_demand(client):
    seed_sessions(client, 1)
    stamp = utc_now()
    with client.app.state.database.session_factory() as db:
        old = AgentRun(id="arun_old", session_id="asess_000", status="completed", provider="mock", model="mock",
                       created_at=stamp - timedelta(days=1))
        latest = AgentRun(id="arun_latest", session_id="asess_000", status="awaiting_approval", provider="mock", model="mock",
                          created_at=stamp)
        db.add_all([old, latest])
        db.flush()
        db.add_all([AgentMessage(id=f"amsg_{i}", session_id="asess_000", run_id="arun_old" if i <= 1000 else "arun_latest",
                                role="user" if i % 2 else "assistant", sequence=i, content=f"message {i}")
                    for i in range(1, 1003)])
        db.add(AgentMessage(id="amsg_tool", session_id="asess_000", run_id="arun_old", role="tool", sequence=1003,
                            content="historical tool payload"))
        db.add(AgentToolCall(id="atool_old", run_id="arun_old", name="global_summary", sequence=1,
                             status="completed", result_json={"evidence": "old-evidence" * 10000}))
        db.add(AgentApproval(id="aappr_latest", run_id="arun_latest", tool_name="create_training_task",
                             status="pending", idempotency_key="history-test", expires_at=stamp + timedelta(hours=1)))
        db.commit()
    url = "/api/agent/sessions/asess_000/timeline"
    first = client.get(url)
    assert first.status_code == 200, first.text
    page = first.json()
    assert len(page["messages"]) == 50
    assert page["messages"][0]["sequence"] == 953
    assert page["message_count"] == 1003
    assert page["latest_run_status"] == "awaiting_approval"
    assert [run["id"] for run in page["runs"]] == ["arun_latest"]
    assert page["runs"][0]["approvals"][0]["id"] == "aappr_latest"
    assert "old-evidence" not in first.text
    assert len(first.content) < 30000
    # A new append cannot shift the sequence cursor or duplicate older messages.
    with client.app.state.database.session_factory() as db:
        db.add(AgentMessage(id="amsg_append", session_id="asess_000", role="assistant", sequence=1004, content="new reply"))
        db.commit()
    seen = [message["sequence"] for message in page["messages"]]
    while page["next_before_sequence"]:
        page = client.get(url, params={"before_sequence": page["next_before_sequence"]}).json()
        assert page["runs"] == []
        assert len(page["messages"]) <= 50
        assert all(message["role"] != "tool" for message in page["messages"])
        seen = [message["sequence"] for message in page["messages"]] + seen
    assert seen == list(range(1, 1003))
    evidence = client.get("/api/agent/runs/arun_old").json()
    assert evidence["tool_calls"][0]["result_json"]["evidence"].startswith("old-evidence")
    assert client.get(url, params={"before_sequence": 0}).status_code == 422
    assert client.get("/api/agent/sessions/absent/timeline").status_code == 404


def test_history_indexes_are_installed_by_migrations(client):
    inspector = inspect(client.app.state.database.engine)
    for table, name in (("agent_sessions", "ix_agent_sessions_updated_id"),
                        ("agent_runs", "ix_agent_runs_session_created_id"),
                        ("agent_messages", "ix_agent_messages_session_sequence")):
        assert name in {index["name"] for index in inspector.get_indexes(table)}
