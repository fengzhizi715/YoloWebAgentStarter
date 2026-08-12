from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort
from PIL import Image, ImageDraw

from app.models.onnx import export_fp32_onnx
from app.models.evaluation_artifacts import EvaluationArtifactManager
from create_tiny_demo import write_example


def _write_demo_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 64), "#111827")
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 48, 48), fill=color)
    image.save(path)


def _write_obb_demo(root: Path) -> None:
    for split, color, count in (("train", "#2563eb", 2), ("val", "#16a34a", 1)):
        for index in range(count):
            stem = f"{split}-{index + 1}"
            _write_demo_image(root / "images" / split / f"{stem}.png", color)
            label = root / "labels" / split / f"{stem}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: square\n",
        encoding="utf-8",
    )


def _write_segment_demo(root: Path) -> None:
    for split, color, count in (("train", "#7c3aed", 2), ("val", "#ea580c", 1)):
        for index in range(count):
            stem = f"{split}-{index + 1}"
            _write_demo_image(root / "images" / split / f"{stem}.png", color)
            label = root / "labels" / split / f"{stem}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: square\n",
        encoding="utf-8",
    )


def _write_classification_demo(root: Path) -> None:
    for split in ("train", "val"):
        _write_demo_image(root / split / "blue-square" / f"{split}-blue.png", "#2563eb")
        _write_demo_image(root / split / "green-square" / f"{split}-green.png", "#16a34a")


def _train_cpu(model, data: str, run_root: Path, name: str) -> Path:
    model.train(
        data=data,
        epochs=1,
        imgsz=64,
        batch=2,
        project=str(run_root),
        name=name,
        exist_ok=True,
        workers=0,
        device="cpu",
        optimizer="SGD",
        patience=0,
        plots=False,
    )
    checkpoint = run_root / name / "weights" / "best.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"{name} CPU smoke did not produce best.pt.")
    return checkpoint


def _validate_segment_cpu(checkpoint: Path, data_yaml: Path, run_root: Path) -> None:
    """Run the same native Ultralytics val contract as the upstream evaluation engine."""

    from ultralytics import YOLO

    result = YOLO(str(checkpoint)).val(
        data=str(data_yaml),
        split="val",
        imgsz=64,
        batch=1,
        workers=0,
        device="cpu",
        conf=0.00001,
        project=str(run_root),
        name="segment-val",
        exist_ok=True,
        plots=True,
        save_json=True,
    )
    metrics = result.results_dict
    required_metrics = {
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(M)",
        "metrics/recall(M)",
        "metrics/mAP50(M)",
        "metrics/mAP50-95(M)",
    }
    missing_metrics = required_metrics.difference(metrics)
    if missing_metrics:
        raise RuntimeError(f"Segment val did not return box/mask metrics: {sorted(missing_metrics)}")

    artifacts = EvaluationArtifactManager().find_artifacts(run_root / "segment-val")
    # A one-epoch randomly initialized offline model may have no true positives,
    # in which case Ultralytics intentionally omits PR curves. The artifact-name
    # contract for BoxPR/MaskPR is covered by focused tests.
    missing_artifacts = [name for name in ("confusion_matrix", "predictions") if not artifacts[name]]
    if missing_artifacts:
        raise RuntimeError(f"Segment val did not produce managed artifacts: {missing_artifacts}")
    predictions = json.loads(Path(artifacts["predictions"]).read_text(encoding="utf-8"))
    if not predictions or not isinstance(predictions[0].get("segmentation"), dict):
        raise RuntimeError("Segment val predictions.json does not contain Ultralytics RLE masks.")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ywa-cpu-smoke-") as temporary_directory:
        root = Path(temporary_directory)
        os.environ["MPLCONFIGDIR"] = str(root / "matplotlib")
        os.environ["YOLO_CONFIG_DIR"] = str(root / "ultralytics")

        dataset_root = root / "dataset"
        write_example(dataset_root)

        from ultralytics import YOLO

        run_root = root / "runs"
        checkpoint = _train_cpu(YOLO("yolo11n.yaml"), str(dataset_root / "data.yaml"), run_root, "detect")

        obb_root = root / "obb-dataset"
        _write_obb_demo(obb_root)
        _train_cpu(YOLO("yolo11n-obb.yaml"), str(obb_root / "data.yaml"), run_root, "obb")

        segment_root = root / "segment-dataset"
        _write_segment_demo(segment_root)
        segment_checkpoint = _train_cpu(YOLO("yolo11n-seg.yaml"), str(segment_root / "data.yaml"), run_root, "segment")
        _validate_segment_cpu(segment_checkpoint, segment_root / "data.yaml", run_root)

        classify_root = root / "classify-dataset"
        _write_classification_demo(classify_root)
        _train_cpu(YOLO("yolo11n-cls.yaml"), str(classify_root), run_root, "classify")

        exported = export_fp32_onnx(checkpoint, root / "model.onnx")
        onnx.checker.check_model(exported)
        ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
        print("CPU detect, segment train/val, OBB, classify, and ONNX smoke passed.")


if __name__ == "__main__":
    main()
