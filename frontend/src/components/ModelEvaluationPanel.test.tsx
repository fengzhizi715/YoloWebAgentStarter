// @vitest-environment jsdom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ModelEvaluationPanel } from "./ModelEvaluationPanel";
import type { ModelEvaluationRecord, ModelVersion } from "../types";

vi.mock("../api/client", () => ({
  api: {
    modelEvaluationLogs: vi.fn().mockResolvedValue({ evaluation_id: "eval-1", logs: "validation complete", line_count: 1 }),
    modelEvaluationArtifactUrl: (_modelId: string, evaluationId: string, artifact: string) => `/artifacts/${evaluationId}/${artifact}`,
  },
}));

const model: ModelVersion = {
  id: "model-1", name: "best", version: "v1", dataset_id: "dataset-1", training_task_id: "train-1", source_model_id: null,
  source: "training_task", artifact_type: "best", format: "pt", task_type: "detect", engine_type: "ultralytics", model_path: "/managed/best.pt",
  base_model: "yolo11n.pt", status: "active", precision: null, recall: null, map50: null, map50_95: null, metrics_json: {}, notes: "",
  archived_at: null, created_at: "2026-08-12T10:00:00", updated_at: "2026-08-12T10:00:00",
};

const evaluation: ModelEvaluationRecord = {
  id: "eval-1", model_id: model.id, dataset_id: "dataset-1", split: "val", status: "completed", confidence: 0.25, iou: 0.5,
  result_json: {
    metrics: { precision: 0.8, map50: 0.7 },
    artifacts: { pr_curve: "/managed/BoxPR_curve.png", box_pr_curve: "/managed/BoxPR_curve.png", mask_pr_curve: "/managed/MaskPR_curve.png", confusion_matrix: null },
    error_samples: [{ image_file: "sample.png", type: "false_positive", message: "Prediction did not match ground truth." }],
  },
  error_message: null, export_path: "/managed/dataset", data_path: "/managed/data.yaml", run_dir: "/managed/run", logs_path: "/managed/evaluation.log",
  created_at: "2026-08-12T10:00:00", started_at: "2026-08-12T10:00:01", finished_at: "2026-08-12T10:00:02",
};

let root: Root | undefined;
let container: HTMLDivElement | undefined;

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
});

describe("ModelEvaluationPanel", () => {
  it("renders upstream-style metrics, artifacts, samples and logs", async () => {
    await act(async () => {
      root?.render(<ModelEvaluationPanel model={model} evaluations={[evaluation]} busy={false} onEvaluate={vi.fn()} />);
    });

    expect(container?.textContent).toContain("precision");
    expect(container?.textContent).toContain("0.800");
    const falsePositiveCard = Array.from(container?.querySelectorAll(".evaluation-summary article") ?? []).find(
      (element) => element.textContent?.includes("false_positive"),
    );
    expect(falsePositiveCard?.textContent).toContain("1");
    expect(container?.textContent).toContain("validation complete");
    const images = Array.from(container?.querySelectorAll("img") ?? []);
    expect(images.map((image) => image.getAttribute("src"))).toEqual([
      "/artifacts/eval-1/box_pr_curve",
      "/artifacts/eval-1/mask_pr_curve",
    ]);
  });
});
