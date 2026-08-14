// @vitest-environment jsdom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TrainingView } from "./TrainingView";
import type { Dataset } from "../types";

vi.mock("../api/client", () => ({
  api: {
    listTrainingTasks: vi.fn().mockResolvedValue({ items: [] }),
    listTrainingDevices: vi.fn().mockResolvedValue({ items: [] }),
    createTrainingTask: vi.fn(),
    stopTrainingTask: vi.fn(),
    resumeTrainingTask: vi.fn(),
    getTrainingLogs: vi.fn().mockResolvedValue({ task_id: "", logs: "", line_count: 0 }),
    getTrainingSummary: vi.fn().mockResolvedValue({
      task_id: "",
      status: "pending",
      training_config: {},
      dataset: {},
      progress: { epoch: 0, total_epochs: 0, percent: 0 },
      metrics: {},
      checkpoints: {},
      log_summary: { line_count: 0, tail: [] },
      risks: [],
      next_steps: [],
    }),
    downloadCheckpointUrl: vi.fn().mockReturnValue("#"),
    autoSplitImages: vi.fn(),
  },
}));

const detect: Dataset = { id: "ds-detect", name: "detect-set", description: null, task_type: "detect", image_count: 4, annotated_image_count: 4, class_count: 1, created_at: "", updated_at: "" };
const segment: Dataset = { id: "ds-segment", name: "segment-set", description: null, task_type: "segment", image_count: 2, annotated_image_count: 2, class_count: 1, created_at: "", updated_at: "" };

let root: Root | undefined;
let container: HTMLDivElement | undefined;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => { container = document.createElement("div"); document.body.append(container); root = createRoot(container); });
afterEach(() => { act(() => root?.unmount()); container?.remove(); });

describe("TrainingView dataset selection", () => {
  it("allows selecting the dataset used for a new training task", async () => {
    const onDatasetChange = vi.fn();
    await act(async () => { root?.render(<TrainingView datasets={[detect, segment]} dataset={detect} onDatasetChange={onDatasetChange} onBack={vi.fn()} />); });
    const select = Array.from(container?.querySelectorAll<HTMLSelectElement>("select") ?? []).find((item) => item.value === detect.id);
    expect(select).toBeDefined();
    await act(async () => { if (select) { select.value = segment.id; select.dispatchEvent(new Event("change", { bubbles: true })); } });
    expect(onDatasetChange).toHaveBeenCalledWith(segment);
  });

  it("renders upstream-style workspace sections", async () => {
    await act(async () => { root?.render(<TrainingView datasets={[detect]} dataset={detect} onDatasetChange={vi.fn()} onBack={vi.fn()} />); });
    expect(container?.querySelector("#training-config-anchor")).toBeTruthy();
    expect(container?.querySelector("#experiment-history")).toBeTruthy();
    expect(container?.querySelector(".training-dataset-rail")).toBeTruthy();
    expect(container?.textContent).toContain("开始训练");
  });
});
