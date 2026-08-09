from __future__ import annotations

import os
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort

from app.models.onnx import export_fp32_onnx
from create_tiny_demo import write_example


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ywa-cpu-smoke-") as temporary_directory:
        root = Path(temporary_directory)
        os.environ["MPLCONFIGDIR"] = str(root / "matplotlib")
        os.environ["YOLO_CONFIG_DIR"] = str(root / "ultralytics")

        dataset_root = root / "dataset"
        write_example(dataset_root)

        from ultralytics import YOLO

        run_root = root / "runs"
        YOLO("yolo11n.yaml").train(
            data=str(dataset_root / "data.yaml"),
            epochs=1,
            imgsz=64,
            batch=2,
            project=str(run_root),
            name="detect",
            exist_ok=True,
            workers=0,
            device="cpu",
            optimizer="SGD",
            patience=0,
            plots=False,
        )
        checkpoint = run_root / "detect" / "weights" / "best.pt"
        if not checkpoint.is_file():
            raise RuntimeError("CPU smoke did not produce best.pt.")

        exported = export_fp32_onnx(checkpoint, root / "model.onnx")
        onnx.checker.check_model(exported)
        ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
        print("CPU YOLO and ONNX smoke passed.")


if __name__ == "__main__":
    main()
