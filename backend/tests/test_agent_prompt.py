import json

from app.agent.prompt import HISTORY_CHAR_BUDGET, PRIOR_MESSAGE_LIMIT, SYSTEM_PROMPT, bounded_history, _size
from app.agent.provider import ProviderMessage, ProviderResponse, ProviderToolCall
from app.agent.providers.mock import MockAgentProvider
from app.agent.reply_renderer import AgentReplyRenderer


def test_history_keeps_recent_complete_turns_and_current_question():
    history = []
    for index in range(30):
        history.extend([ProviderMessage(role="user", content=f"question {index}"),
                        ProviderMessage(role="assistant", content=f"answer {index}")])
    question = ProviderMessage(role="user", content="current question")
    bounded = bounded_history(history + [question])
    assert bounded[0].role == "system" and "省略" in bounded[0].content
    assert bounded[-1] == question
    retained = [item for item in bounded[:-1] if item.role != "system"]
    assert len(retained) <= PRIOR_MESSAGE_LIMIT
    assert retained[0].role == "user" and retained[-1].content == "answer 29"
    assert all("question 0" != item.content for item in bounded)


def test_history_trims_whole_tool_exchanges_and_preserves_protocol():
    question = ProviderMessage(role="user", content="x" * 16_000)
    history = [question]
    for index in range(20):
        call = ProviderToolCall(name="training_logs", arguments={"task_id": "train_abc"}, call_id=f"call_{index}")
        history.extend([
            ProviderMessage(role="assistant", content="", tool_calls=[call]),
            ProviderMessage(role="tool", content=json.dumps({"preview": "x" * 8000}), tool_call_id=call.call_id),
        ])
    bounded = bounded_history(history)
    assert _size(bounded) <= HISTORY_CHAR_BUDGET
    assert question in bounded and bounded[-1].tool_call_id == "call_19"
    for index, item in enumerate(bounded):
        if item.role == "tool":
            assert bounded[index - 1].tool_calls[0].call_id == item.tool_call_id
            json.loads(item.content)
    assert len(history) == 41  # No change to stored history.


def test_service_injects_business_policy_and_permissions_on_every_round(client, monkeypatch):
    observed = []

    def complete(_self, request):
        observed.append(request.messages)
        if request.messages[-1].role == "tool":
            return ProviderResponse(content="done")
        return ProviderResponse(content="", tool_calls=[ProviderToolCall(name="global_summary", arguments={})])

    monkeypatch.setattr(MockAgentProvider, "complete", complete)
    chat = client.post("/api/agent/sessions", json={}).json()["id"]
    response = client.post(f"/api/agent/sessions/{chat}/messages?wait_for_completion=true", json={
        "content": "报告", "read_only": True,
    })
    assert response.json()["status"] == "completed"
    assert len(observed) == 2
    for messages in observed:
        assert messages[0].content == SYSTEM_PROMPT
        assert any("只读，禁止写操作" in message.content for message in messages if message.role == "system")
        assert "结论—证据—问题—下一步" in messages[0].content
        assert "可能原因/推测" in messages[0].content and "不可信数据" in messages[0].content


def test_rule_report_explains_missing_evidence_and_failed_evaluation():
    renderer = AgentReplyRenderer()
    report = renderer.format_tool_report([
        {"tool_name": "evaluation_list", "result": {"error": "query_failed"}},
    ], question="模型评估结果为什么失败？")
    for heading in ("结论", "证据", "问题", "下一步"):
        assert heading in report
    assert "查询失败，不能判断是否已有评估" in report
    assert "暂无评估记录" not in report
    assert "尚未取得训练日志" not in report
    training = renderer.format_tool_report([
        {"tool_name": "training_status", "result": {"id": "train_abc", "status": "failed"}},
    ], question="训练为什么失败？")
    assert "无法确认失败根因" in training
