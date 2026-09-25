from __future__ import annotations

import io
import json

from PIL import Image

from app.agent.bounds import bound_log_text, untrusted_text
from app.agent.providers.mock import MockAgentProvider, _plan_tools
from app.agent.provider import ProviderRequest
from app.core.models import ModelVersion


def png_bytes(color: str = "#4477aa", size: tuple[int, int] = (100, 80)) -> bytes:
    image = Image.new("RGB", size, color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _seed_dataset(client) -> str:
    dataset_id = client.post("/api/datasets", json={"name": "agent-demo\x00", "task_type": "detect"}).json()["id"]
    client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "cat", "color": "#ef4444"})
    client.post(
        f"/api/datasets/{dataset_id}/images/upload",
        files={"files": ("cat.png", png_bytes(), "image/png")},
    )
    return dataset_id


def test_untrusted_text_strips_controls_and_truncates():
    assert "\x00" not in untrusted_text("a\x00b")
    assert untrusted_text("x" * 1000, limit=20).endswith("...")


def test_bound_log_text_keeps_tail_only():
    text = "\n".join(f"line-{index}" for index in range(100))
    clipped = bound_log_text(text, max_chars=500, max_lines=5)
    assert "line-99" in clipped
    assert "line-0" not in clipped


def test_mock_plans_quality_and_training_tools():
    available = {
        "global_summary",
        "dataset_summary",
        "dataset_quality_report",
        "dataset_validate",
        "training_list",
        "training_latest_result",
        "model_list",
        "model_compare",
        "evaluation_list",
    }
    quality = _plan_tools("请看一下 ds_abc123def456 的质量报告", available)
    assert [item.name for item in quality] == ["dataset_quality_report"]
    assert quality[0].arguments["dataset_id"] == "ds_abc123def456"

    training = _plan_tools("查看 train_abc123def456 训练摘要", available)
    assert training[0].name == "training_latest_result"

    compare = _plan_tools("对比 model_train_1_best 和 model_train_1_last", available)
    assert compare[0].name == "model_compare"


def test_agent_lists_datasets_via_read_tool(client):
    dataset_id = _seed_dataset(client)
    session_id = client.post("/api/agent/sessions", json={"title": "tools"}).json()["id"]
    run = client.post(
        f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true",
        json={"content": "列出有哪些数据集"},
    )
    assert run.status_code == 200, run.text
    body = run.json()
    assert body["status"] == "completed"
    assert any(item["role"] == "tool" for item in body["messages"])
    assert body["tool_calls"]
    assert body["tool_calls"][0]["name"] == "global_summary"
    assert body["tool_calls"][0]["status"] == "completed"
    result = body["tool_calls"][0]["result_json"]
    assert any(item["id"] == dataset_id for item in result["items"])
    assert all("\x00" not in item["name"] for item in result["items"])
    assistant = next(item for item in body["messages"] if item["role"] == "assistant")
    assert "数据集列表" in assistant["content"]
    assert dataset_id in assistant["content"] or "agent-demo" in assistant["content"]


def test_agent_dataset_quality_and_validate_tools(client):
    dataset_id = _seed_dataset(client)
    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    run = client.post(
        f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true",
        json={"content": f"请给出 {dataset_id} 的质量报告并校验"},
    )
    assert run.status_code == 200, run.text
    body = run.json()
    names = {item["name"] for item in body["tool_calls"]}
    assert "dataset_quality_report" in names
    assert "dataset_validate" in names
    assert all(item["status"] == "completed" for item in body["tool_calls"])
    quality = next(item for item in body["tool_calls"] if item["name"] == "dataset_quality_report")
    assert quality["result_json"]["dataset_id"] == dataset_id
    assert "summary" in quality["result_json"]
    assert "class_distribution" in quality["result_json"]


def test_agent_model_tools_omit_filesystem_paths(client):
    dataset_id = _seed_dataset(client)
    with client.app.state.database.session_factory() as db:
        model = ModelVersion(
            id="model_train_demo_best",
            name="demo-best",
            version="v1",
            dataset_id=dataset_id,
            training_task_id=None,
            source="training_task",
            artifact_type="best",
            format="pt",
            task_type="detect",
            engine_type="ultralytics",
            model_path="/tmp/should-not-leak/best.pt",
            base_model="yolo11n.pt",
            status="active",
            precision=0.8,
            recall=0.7,
            map50=0.75,
            map50_95=0.5,
            metrics_json={"map50": 0.75},
            notes="secret-note\x01",
        )
        db.add(model)
        db.commit()

    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    run = client.post(
        f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true",
        json={"content": f"查看模型 model_train_demo_best 详情"},
    )
    assert run.status_code == 200, run.text
    body = run.json()
    assert body["tool_calls"][0]["name"] == "model_get"
    result = body["tool_calls"][0]["result_json"]
    assert result["id"] == "model_train_demo_best"
    assert "model_path" not in result
    assert "\x01" not in result["notes"]
    serialized = json.dumps(body)
    assert "/tmp/should-not-leak" not in serialized


def test_agent_denies_unknown_tools(client, monkeypatch):
    session_id = client.post("/api/agent/sessions", json={}).json()["id"]

    original = MockAgentProvider.complete

    def force_unknown(self, request: ProviderRequest):
        from app.agent.provider import ProviderResponse, ProviderToolCall

        if any(message.role == "tool" for message in request.messages):
            return original(self, request)
        return ProviderResponse(
            content="",
            tool_calls=[ProviderToolCall(name="rm_rf", arguments={"path": "/"}, call_id="bad")],
        )

    monkeypatch.setattr(MockAgentProvider, "complete", force_unknown)
    run = client.post(f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true", json={"content": "hack"})
    assert run.status_code == 200, run.text
    body = run.json()
    assert body["status"] == "completed"
    assert body["tool_calls"][0]["status"] == "failed"
    assert body["tool_calls"][0]["result_json"]["error"] == "tool_not_allowed"
