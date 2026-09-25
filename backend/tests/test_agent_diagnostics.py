import pytest

from app.agent.diagnostics import diagnostic_handlers
from app.agent.planner import RuleBasedPlanner
from app.core.models import ModelVersion, ModelEvaluationRecord
from app.models.service import ModelService


def handlers_for(values):
    return {name: (name, {}, lambda _session, _args, data=value: data) for name, value in values.items()}


def test_readiness_blocks_empty_splits_and_explains_missing_evidence():
    handlers = handlers_for({
        "dataset_summary": {"image_count": 10, "class_count": 1, "split_counts": {"train": 10, "val": 0}},
        "dataset_quality_report": {"summary": {"coverage": 0.5}},
        "dataset_validate": {"valid": True, "error_count": 0},
        "training_status": {},
    })
    result = diagnostic_handlers(handlers)["dataset_readiness"][2](None, {"dataset_id": "ds_abc"})
    assert result["status"] == "blocked"
    assert any("val" in issue for issue in result["issues"])
    assert len(result["evidence"]) == 3
    handlers["dataset_summary"] = ("", {}, lambda *_: {"error": "not_found"})
    result = diagnostic_handlers(handlers)["dataset_readiness"][2](None, {})
    assert result["status"] == "unknown"


def test_failure_reports_possible_cause_not_proven_root_cause():
    handlers = handlers_for({
        "dataset_summary": {}, "training_status": {"id": "train_abc", "status": "failed", "error_message": "CUDA out of memory"},
        "training_logs": {"preview": "CUDA out of memory", "line_count": 30},
    })
    result = diagnostic_handlers(handlers)["training_diagnose_failure"][2](None, {"task_id": "train_abc"})
    assert result["hypotheses"][0]["code"] == "memory_pressure"
    assert result["hypotheses"][0]["confidence"] == "possible"
    assert any("不排除其他" in item for item in result["issues"])
    handlers["training_status"][2](None, {})["status"] = "completed"
    assert not diagnostic_handlers(handlers)["training_diagnose_failure"][2](None, {})["hypotheses"]


@pytest.mark.parametrize("summary", [
    {"truncated": True, "preview": "partial JSON"},
    {"image_count": 10, "class_count": 1, "annotated_image_count": 10},
    {"image_count": 10, "class_count": 1, "annotated_image_count": 10, "split_counts": {"train": 8}},
])
def test_readiness_does_not_treat_missing_split_evidence_as_empty(summary):
    handlers = handlers_for({
        "dataset_summary": summary,
        "dataset_quality_report": {"summary": {"coverage": 1}},
        "dataset_validate": {"error_count": 0}, "training_status": {},
    })
    result = diagnostic_handlers(handlers)["dataset_readiness"][2](None, {})
    assert result["status"] == "unknown"
    assert not any("split 为空" in issue for issue in result["issues"])


def test_readiness_requires_quality_evidence_and_never_claims_automatic_readiness():
    values = {
        "dataset_summary": {"image_count": 10, "class_count": 1, "annotated_image_count": 10,
                            "split_counts": {"train": 8, "val": 2}},
        "dataset_quality_report": {"summary": {"coverage": 1}},
        "dataset_validate": {"error_count": 0}, "training_status": {},
    }
    result = diagnostic_handlers(handlers_for(values))["dataset_readiness"][2](None, {})
    assert result["status"] == "review_required"
    values["dataset_quality_report"] = {"truncated": True, "preview": "partial JSON"}
    result = diagnostic_handlers(handlers_for(values))["dataset_readiness"][2](None, {})
    assert result["status"] == "unknown"


def test_planner_uses_diagnostics_and_asks_for_second_model():
    planner = RuleBasedPlanner()
    tools = {"dataset_readiness", "training_diagnose_failure", "model_comparability"}
    assert planner.plan_tool_calls("ds_abc 训练准备报告", tools)[0].name == "dataset_readiness"
    assert planner.plan_tool_calls("train_abc 训练为什么失败", tools)[0].name == "training_diagnose_failure"
    assert planner.plan_tool_calls("比较 model_a 和 model_b", tools)[0].name == "model_comparability"
    assert planner.structured_plan("比较 model_a", available=tools).risks


def test_model_comparison_requires_split_and_snapshot_evidence(client):
    dataset = client.post("/api/datasets", json={"name": "models", "task_type": "detect"}).json()["id"]
    with client.app.state.database.session_factory() as db:
        for index in (1, 2):
            db.add(ModelVersion(id=f"model_{index}", name=f"m{index}", dataset_id=dataset, model_path="managed.pt", map50=0.5 + index / 10))
        db.commit()
        service = ModelService(client.app.state.storage)
        missing = service.assess_comparability(db, "model_1", "model_2")
        assert missing["status"] == "unknown" and missing["deltas"] == {}
        for index, split in ((1, "val"), (2, "test")):
            db.add(ModelEvaluationRecord(id=f"eval_{index}", model_id=f"model_{index}", dataset_id=dataset,
                split=split, status="completed", confidence=0.25, iou=0.5, result_json={"metrics": {"map50": 0.8}}))
        db.commit()
        mismatch = service.assess_comparability(db, "model_1", "model_2")
        assert mismatch["status"] == "incomparable" and not mismatch["comparable"]
        db.get(ModelEvaluationRecord, "eval_2").split = "val"
        db.commit()
        unknown = service.assess_comparability(db, "model_1", "model_2")
        assert unknown["status"] == "unknown" and not unknown["comparable"]
        assert any("快照" in issue for issue in unknown["issues"])
