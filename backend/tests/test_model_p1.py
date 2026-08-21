from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from app.core.models import ModelVersion
from app.core.models import ModelEvaluationRecord
from app.models.evaluation import YoloEvaluationRunner, build_evaluation_command, parse_validation_metrics
from app.models.evaluation_artifacts import EvaluationArtifactManager
from app.models.inference import ModelCache, run_test_inference
from app.models.service import ModelService
from app.models.error_samples import ErrorSampleAnalyzer


def test_inference_uses_upstream_low_threshold_for_client_filtering(monkeypatch, tmp_path: Path):
    captured: dict[str, float] = {}

    class StubModel:
        def predict(self, **kwargs):
            captured.update(kwargs)
            return []

    monkeypatch.setattr(ModelCache, "get_model", lambda *_args: StubModel())
    run_test_inference(model_id="model", model_path=tmp_path / "model.pt", task_type="detect", image_bytes=b"image", confidence=0.73, iou=0.45, class_names={})
    assert captured["conf"] == 0.01


def test_obb_proposal_preserves_predicted_rotation():
    proposal = ModelService._obb_from_points([[10, 10], [10, 30], [30, 30], [30, 10]])
    assert proposal["cx"] == 20
    assert proposal["cy"] == 20
    assert proposal["width"] == 20
    assert proposal["height"] == 20
    assert proposal["angle"] == 90


def test_native_evaluation_command_uses_selected_split(tmp_path: Path):
    model = ModelVersion(id="model", name="model", version="v1", source="training_task", artifact_type="best", format="pt", task_type="obb", engine_type="ultralytics", model_path=str(tmp_path / "best.pt"), status="active", metrics_json={}, notes="")
    record = ModelEvaluationRecord(id="eval", model_id="model", dataset_id="dataset", split="test", status="running", confidence=0.2, iou=0.6, result_json={}, data_path=str(tmp_path / "data.yaml"), run_dir=str(tmp_path / "run"))
    command = build_evaluation_command(record, model)
    assert command[1:3] == ["obb", "val"]
    assert "split=test" in command
    assert f"data={tmp_path / 'data.yaml'}" in command


def test_native_validation_metrics_parser_supports_all_community_tasks():
    detect = parse_validation_metrics("all 12 20 0.8 0.7 0.6 0.5", "detect")
    segment = parse_validation_metrics("all 12 20 0.8 0.7 0.6 0.5 0.75 0.65 0.55 0.45", "segment")
    classify = parse_validation_metrics("all 0.91 0.99", "classify")
    assert detect == {"precision": 0.8, "recall": 0.7, "map50": 0.6, "map50_95": 0.5}
    assert segment == {
        "precision": 0.8,
        "recall": 0.7,
        "map50": 0.6,
        "map50_95": 0.5,
        "mask_precision": 0.75,
        "mask_recall": 0.65,
        "mask_map50": 0.55,
        "mask_map50_95": 0.45,
    }
    assert classify == {"top1": 0.91, "top5": 0.99}


def test_evaluation_command_keeps_low_confidence_predictions(tmp_path: Path):
    model = ModelVersion(id="model", name="model", version="v1", source="training_task", artifact_type="best", format="pt", task_type="detect", engine_type="ultralytics", model_path=str(tmp_path / "best.pt"), status="active", metrics_json={}, notes="")
    record = ModelEvaluationRecord(id="eval", model_id="model", dataset_id="dataset", split="val", status="running", confidence=0.25, iou=0.5, result_json={}, data_path=str(tmp_path / "data.yaml"), run_dir=str(tmp_path / "run"))

    assert "conf=0.001" in build_evaluation_command(record, model)


def test_error_sample_analyzer_reads_ultralytics_detect_json(tmp_path: Path):
    """Use the pinned Ultralytics runtime itself to produce the detect JSON contract."""
    from ultralytics.models.yolo.detect.val import DetectionValidator

    export = _evaluation_export(tmp_path, "detect", "0 0.5 0.5 0.2 0.2\n")
    run = tmp_path / "run"
    run.mkdir()
    validator = object.__new__(DetectionValidator)
    validator.jdict = []
    validator.class_map = [0]
    validator.is_lvis = False
    validator.pred_to_json(
        {"bboxes": torch.tensor([[40.0, 40.0, 60.0, 60.0]]), "conf": torch.tensor([0.9]), "cls": torch.tensor([0.0])},
        {"im_file": Path("sample.png")},
    )
    (run / "predictions.json").write_text(json.dumps(validator.jdict), encoding="utf-8")

    assert set(validator.jdict[0]) == {"image_id", "file_name", "category_id", "bbox", "score"}
    assert ErrorSampleAnalyzer().collect(run, 0.25, export, "val") == []


def test_error_sample_analyzer_reads_ultralytics_obb_json(tmp_path: Path):
    """Use the pinned Ultralytics runtime itself to produce the OBB rbox/poly contract."""
    from ultralytics.models.yolo.obb.val import OBBValidator

    export = _evaluation_export(tmp_path, "obb", "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n")
    run = tmp_path / "run"
    run.mkdir()
    validator = object.__new__(OBBValidator)
    validator.jdict = []
    validator.class_map = [0]
    validator.pred_to_json(
        {"bboxes": torch.tensor([[50.0, 50.0, 20.0, 20.0, 0.0]]), "conf": torch.tensor([0.9]), "cls": torch.tensor([0.0])},
        {"im_file": Path("sample.png")},
    )
    (run / "predictions.json").write_text(json.dumps(validator.jdict), encoding="utf-8")

    assert set(validator.jdict[0]) == {"image_id", "file_name", "category_id", "score", "rbox", "poly"}
    assert ErrorSampleAnalyzer().collect(run, 0.25, export, "val") == []


def test_error_sample_analyzer_reads_ultralytics_segment_rle(tmp_path: Path):
    """Exercise the real pycocotools path used by Ultralytics segment val."""
    from ultralytics.models.yolo.segment.val import SegmentationValidator

    export = _evaluation_export(tmp_path, "segment", "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n")
    run = tmp_path / "run"
    run.mkdir()
    validator = object.__new__(SegmentationValidator)
    validator.jdict = []
    validator.class_map = [0]
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:61, 40:61] = 1
    validator.pred_to_json(
        {
            "bboxes": torch.tensor([[40.0, 40.0, 60.0, 60.0]]),
            "conf": torch.tensor([0.9]),
            "cls": torch.tensor([0.0]),
            "masks": torch.from_numpy(mask[None, :, :]),
        },
        {"im_file": Path("sample.png")},
    )
    (run / "predictions.json").write_text(json.dumps(validator.jdict), encoding="utf-8")

    assert set(validator.jdict[0]) == {"image_id", "file_name", "category_id", "bbox", "score", "segmentation"}
    assert validator.jdict[0]["segmentation"]["size"] == [100, 100]
    assert ErrorSampleAnalyzer().collect(run, 0.25, export, "val") == []


def _evaluation_export(tmp_path: Path, task_type: str, label: str) -> Path:
    export = tmp_path / "dataset"
    (export / "images" / "val").mkdir(parents=True)
    (export / "labels" / "val").mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(export / "images" / "val" / "sample.png")
    (export / "labels" / "val" / "sample.txt").write_text(label, encoding="utf-8")
    (export / "data.yaml").write_text(f"task: {task_type}\n", encoding="utf-8")
    return export


def test_evaluation_runner_recovers_running_record(client):
    dataset = client.post("/api/datasets", json={"name": "recover", "task_type": "detect"}).json()
    source = Path(client.app.state.settings.data_dir) / "source.pt"
    source.write_bytes(b"model")
    with client.app.state.database.session_factory() as session:
        managed = client.app.state.storage.copy_model_artifact(source, "recover-model", "best.pt")
        session.add(ModelVersion(id="recover-model", name="recover", version="v1", dataset_id=dataset["id"], source="training_task", artifact_type="best", format="pt", task_type="detect", engine_type="ultralytics", model_path=str(managed), status="active", metrics_json={}, notes=""))
        session.add(ModelEvaluationRecord(id="recover-eval", model_id="recover-model", dataset_id=dataset["id"], split="val", status="running", confidence=0.25, iou=0.5, result_json={}))
        session.commit()
    runner = YoloEvaluationRunner(client.app.state.database.session_factory, client.app.state.storage)
    runner.recover_orphaned()
    with client.app.state.database.session_factory() as session:
        record = session.get(ModelEvaluationRecord, "recover-eval")
        assert record is not None
        assert record.status == "failed"
        assert "restart" in (record.error_message or "")


def test_evaluation_artifact_manager_uses_upstream_artifact_names(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    confusion = run / "confusion_matrix.png"
    pr_curve = run / "PR_curve.png"
    predictions = run / "predictions.json"
    for path in (confusion, pr_curve, predictions):
        path.write_bytes(b"artifact")

    assert EvaluationArtifactManager().find_artifacts(run) == {
        "confusion_matrix": str(confusion),
        "pr_curve": str(pr_curve),
        "box_pr_curve": None,
        "mask_pr_curve": None,
        "predictions": str(predictions),
    }


def test_evaluation_artifact_manager_reads_ultralytics_segment_curve_names(tmp_path: Path):
    box_curve = tmp_path / "BoxPR_curve.png"
    mask_curve = tmp_path / "MaskPR_curve.png"
    box_curve.write_bytes(b"box")
    mask_curve.write_bytes(b"mask")

    artifacts = EvaluationArtifactManager().find_artifacts(tmp_path)

    assert artifacts["pr_curve"] == str(box_curve)
    assert artifacts["box_pr_curve"] == str(box_curve)
    assert artifacts["mask_pr_curve"] == str(mask_curve)


def test_evaluation_creation_exports_segment_for_native_yolo_val(client, tmp_path, monkeypatch):
    dataset = client.post("/api/datasets", json={"name": "segment", "task_type": "segment"}).json()
    label = client.post(f"/api/datasets/{dataset['id']}/classes", json={"name": "object"}).json()
    image = Image.new("RGB", (64, 48), "white")
    output = io.BytesIO()
    image.save(output, format="PNG")
    uploaded = client.post(
        f"/api/datasets/{dataset['id']}/images/upload",
        data={"split": "val"},
        files={"files": ("segment.png", output.getvalue(), "image/png")},
    ).json()["items"][0]
    saved = client.put(
        f"/api/datasets/{dataset['id']}/images/{uploaded['id']}/annotations",
        json={"annotations": [{"type": "polygon", "class_id": label["id"], "polygon": [[5, 5], [40, 5], [40, 30], [5, 30]]}]},
    )
    assert saved.status_code == 200
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    with client.app.state.database.session_factory() as session:
        managed = client.app.state.storage.copy_model_artifact(model_path, "model-segment", "best.pt")
        session.add(ModelVersion(id="model-segment", name="segment", version="v1", dataset_id=dataset["id"], source="training_task", artifact_type="best", format="pt", task_type="segment", engine_type="ultralytics", model_path=str(managed), status="active", metrics_json={}, notes=""))
        session.commit()
    monkeypatch.setattr("app.models.service.run_evaluation_process", lambda *_args: (0, "all 1 1 0.8 0.7 0.6 0.5", {"precision": 0.8, "recall": 0.7, "map50": 0.6, "map50_95": 0.5}))
    response = client.post("/api/models/model-segment/evaluate", json={"split": "val"})
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["data_path"].endswith("data.yaml")
    assert Path(payload["export_path"], "labels", "val").is_dir()


def test_evaluation_rejects_split_with_only_unannotated_images(client, tmp_path):
    dataset = client.post("/api/datasets", json={"name": "empty-val", "task_type": "detect"}).json()
    client.post(f"/api/datasets/{dataset['id']}/classes", json={"name": "object"})
    image = Image.new("RGB", (64, 48), "white")
    output = io.BytesIO()
    image.save(output, format="PNG")
    uploaded = client.post(
        f"/api/datasets/{dataset['id']}/images/upload",
        data={"split": "val"},
        files={"files": ("unannotated.png", output.getvalue(), "image/png")},
    )
    assert uploaded.status_code == 201
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    with client.app.state.database.session_factory() as session:
        managed = client.app.state.storage.copy_model_artifact(model_path, "model-empty-val", "best.pt")
        session.add(ModelVersion(id="model-empty-val", name="empty-val", version="v1", dataset_id=dataset["id"], source="training_task", artifact_type="best", format="pt", task_type="detect", engine_type="ultralytics", model_path=str(managed), status="active", metrics_json={}, notes=""))
        session.commit()

    response = client.post("/api/models/model-empty-val/evaluate", json={"split": "val"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "evaluation_split_empty"


def test_evaluation_logs_and_artifacts_are_served_from_managed_storage(client, tmp_path):
    dataset = client.post("/api/datasets", json={"name": "artifacts", "task_type": "detect"}).json()
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    with client.app.state.database.session_factory() as session:
        managed = client.app.state.storage.copy_model_artifact(model_path, "artifact-model", "best.pt")
        session.add(ModelVersion(id="artifact-model", name="artifacts", version="v1", dataset_id=dataset["id"], source="training_task", artifact_type="best", format="pt", task_type="detect", engine_type="ultralytics", model_path=str(managed), status="active", metrics_json={}, notes=""))
        task_root = client.app.state.storage.evaluation_task_dir("artifact-eval")
        run_dir = task_root / "run"
        run_dir.mkdir()
        confusion = run_dir / "confusion_matrix.png"
        confusion.write_bytes(b"png")
        logs = task_root / "evaluation.log"
        logs.write_text("first\nsecond\n", encoding="utf-8")
        session.add(ModelEvaluationRecord(id="artifact-eval", model_id="artifact-model", dataset_id=dataset["id"], split="val", status="completed", confidence=0.25, iou=0.5, result_json={"artifacts": {"confusion_matrix": str(confusion)}}, run_dir=str(run_dir), logs_path=str(logs)))
        session.commit()

    log_response = client.get("/api/models/artifact-model/evaluations/artifact-eval/logs?tail=1")
    artifact_response = client.get("/api/models/artifact-model/evaluations/artifact-eval/artifacts/confusion_matrix")

    assert log_response.status_code == 200
    assert log_response.json() == {"evaluation_id": "artifact-eval", "logs": "second", "line_count": 2}
    assert artifact_response.status_code == 200
    assert artifact_response.content == b"png"


def test_model_test_rejects_oversized_upload_with_domain_error(client):
    client.app.state.settings = replace(client.app.state.settings, max_upload_bytes=1)

    response = client.post("/api/models/missing/test", files={"file": ("image.png", b"too large", "image/png")})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "upload_too_large"
