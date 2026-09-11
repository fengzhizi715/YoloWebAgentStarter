from __future__ import annotations

from app.agent.planner import RuleBasedPlanner
from app.agent.reply_renderer import AgentReplyRenderer


def test_rule_planner_plans_read_and_write_tools():
    planner = RuleBasedPlanner()
    available = {
        "global_summary",
        "dataset_validate",
        "create_training_task",
        "create_model_evaluation",
        "create_auto_annotation_task",
        "dataset_quality_report",
        "model_compare",
    }
    listed = planner.plan_tool_calls("列出有哪些数据集", available)
    assert [item.name for item in listed] == ["global_summary"]

    training = planner.structured_plan("请为 ds_abc123def456 用 yolo11s.pt 训练 30 个 epoch", available=available)
    assert training.needs_confirmation
    assert training.steps[-1].tool_name == "create_training_task"
    assert training.steps[-1].arguments["epochs"] == 30
    assert training.steps[-1].arguments["model"] == "yolo11s.pt"

    validate_then = planner.structured_plan(
        "先校验 ds_abc123def456，如果没问题再创建训练",
        available=available,
    )
    assert [step.tool_name for step in validate_then.steps] == ["dataset_validate", "create_training_task"]
    assert validate_then.needs_confirmation


def test_rule_planner_rejects_out_of_scope_automation():
    planner = RuleBasedPlanner()
    plan = planner.structured_plan("帮我跑一个 workflow 闭环并主动学习", available={"global_summary"})
    assert plan.intent == "out_of_scope_hint"
    assert plan.risks
    assert [step.tool_name for step in plan.steps] == ["global_summary"]


def test_rule_planner_training_read_vs_write():
    planner = RuleBasedPlanner()
    available = {
        "training_list",
        "create_training_task",
    }
    read_plan = planner.structured_plan("ds_abc123def456 这个数据集训练得怎么样", available=available)
    assert read_plan.steps
    assert read_plan.steps[0].tool_name == "training_list"
    assert not read_plan.needs_confirmation

    write_plan = planner.structured_plan("请为 ds_abc123def456 创建训练任务", available=available)
    assert write_plan.steps[0].tool_name == "create_training_task"
    assert write_plan.needs_confirmation


def test_reply_renderer_confirmation_and_report():
    renderer = AgentReplyRenderer()
    from app.agent.planner import AgentPlanStep, AgentStructuredPlan

    plan = AgentStructuredPlan(
        intent="create_training_task",
        steps=[
            AgentPlanStep(
                tool_name="create_training_task",
                arguments={"dataset_id": "ds_1"},
                description="创建本地 YOLO 训练任务（需确认）",
                requires_confirmation=True,
            )
        ],
        needs_confirmation=True,
        risks=["训练会占用本机资源。"],
    )
    text = renderer.confirmation_reply(plan)
    assert "需要你确认" in text
    assert "不会自动串联" in text

    report = renderer.format_tool_report(
        [
            {
                "tool_name": "global_summary",
                "result": {
                    "items": [{"id": "ds_1", "name": "demo", "task_type": "detect", "image_count": 1}],
                    "total": 1,
                },
            }
        ]
    )
    assert "数据集列表" in report
