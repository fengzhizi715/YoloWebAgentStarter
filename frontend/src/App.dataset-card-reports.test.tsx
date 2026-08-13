// @vitest-environment jsdom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DatasetHome } from "./App";
import type { Dataset, DatasetQualityReport, ValidationReport } from "./types";

const dataset: Dataset = { id: "ds-1", name: "road-signs", description: null, task_type: "detect", image_count: 12, annotated_image_count: 6, class_count: 3, created_at: "", updated_at: "" };
const validation: ValidationReport = { dataset_id: dataset.id, valid: true, error_count: 0, warning_count: 1, issues: [] };
const quality: DatasetQualityReport = { dataset_id: dataset.id, task_type: "detect", summary: { image_count: 12, annotated_image_count: 6, unannotated_image_count: 6, coverage: 0.5, annotation_count: 8, bbox_count: 8, polygon_count: 0, obb_count: 0, classify_count: 0, small_object_count: 2, small_object_ratio: 0.25 }, class_distribution: [], issues: [] };

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

describe("DatasetHome card reports", () => {
  it("runs validation and quality checks from the dataset card", async () => {
    const onValidate = vi.fn().mockResolvedValue(validation);
    const onQuality = vi.fn().mockResolvedValue(quality);
    await act(async () => {
      root?.render(<DatasetHome datasets={[dataset]} busy={false} onSelect={vi.fn()} onCreate={vi.fn()} onImport={vi.fn()} onValidate={onValidate} onQuality={onQuality} onContinueAnnotation={vi.fn()} onDelete={vi.fn()} />);
    });
    const button = (label: string) => Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent === label);

    await act(async () => button("运行校验")?.click());
    expect(onValidate).toHaveBeenCalledWith(dataset);
    expect(container?.textContent).toContain("校验通过");

    await act(async () => button("质量报告")?.click());
    expect(onQuality).toHaveBeenCalledWith(dataset);
    expect(container?.textContent).toContain("标注覆盖率");
  });

  it("shows imported annotation progress and exposes continuation and export actions", async () => {
    const onContinueAnnotation = vi.fn();
    await act(async () => {
      root?.render(<DatasetHome datasets={[dataset]} busy={false} onSelect={vi.fn()} onCreate={vi.fn()} onImport={vi.fn()} onValidate={vi.fn()} onQuality={vi.fn()} onContinueAnnotation={onContinueAnnotation} onDelete={vi.fn()} />);
    });
    expect(container?.textContent).toContain("50%");
    expect(container?.querySelector('a[href$="/api/datasets/ds-1/export/yolo"]')?.textContent).toContain("导出 YOLO");
    act(() => Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent === "继续标注")?.click());
    expect(onContinueAnnotation).toHaveBeenCalledWith(dataset);
  });
});
