// @vitest-environment jsdom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DatasetHome } from "./App";
import type { Dataset, DatasetQualityReport, DuplicateReport, ValidationReport } from "./types";

const dataset: Dataset = { id: "ds-1", name: "road-signs", description: null, task_type: "detect", image_count: 12, annotated_image_count: 6, class_count: 3, created_at: "", updated_at: "" };
const validation: ValidationReport = { dataset_id: dataset.id, valid: true, error_count: 0, warning_count: 1, issues: [] };
const quality: DatasetQualityReport = { dataset_id: dataset.id, task_type: "detect", summary: { image_count: 12, annotated_image_count: 6, unannotated_image_count: 6, coverage: 0.5, annotation_count: 8, bbox_count: 8, polygon_count: 0, obb_count: 0, classify_count: 0, small_object_count: 2, small_object_ratio: 0.25 }, class_distribution: [], issues: [] };
const duplicates: DuplicateReport = { images: 12, duplicate: 2, similar: 1, invalid_images: 0, invalid_image_ids: [], phash_distance: 4, groups: [] };

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
      root?.render(<DatasetHome datasets={[dataset]} busy={false} onCreate={vi.fn()} onImport={vi.fn()} onValidate={onValidate} onQuality={onQuality} onContinueAnnotation={vi.fn()} onUpload={vi.fn()} onTrain={vi.fn()} onDuplicates={vi.fn().mockResolvedValue(duplicates)} onDelete={vi.fn()} />);
    });
    const button = (label: string) => Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent === label);

    await act(async () => button("运行校验")?.click());
    expect(onValidate).toHaveBeenCalledWith(dataset);
    expect(container?.textContent).toContain("校验通过");
    act(() => button("收起校验")?.click());
    expect(container?.textContent).not.toContain("校验通过");

    await act(async () => button("质量报告")?.click());
    expect(onQuality).toHaveBeenCalledWith(dataset);
    expect(container?.textContent).toContain("标注覆盖率");
  });

  it("shows imported annotation progress and opens an export-format dialog", async () => {
    const onContinueAnnotation = vi.fn();
    const onTrain = vi.fn();
    const onDuplicates = vi.fn().mockResolvedValue(duplicates);
    await act(async () => {
      root?.render(<DatasetHome datasets={[dataset]} busy={false} onCreate={vi.fn()} onImport={vi.fn()} onValidate={vi.fn()} onQuality={vi.fn()} onContinueAnnotation={onContinueAnnotation} onUpload={vi.fn()} onTrain={onTrain} onDuplicates={onDuplicates} onDelete={vi.fn()} />);
    });
    expect(container?.textContent).toContain("50%");
    act(() => Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent === "继续标注")?.click());
    expect(onContinueAnnotation).toHaveBeenCalledWith(dataset);
    act(() => Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent === "训练")?.click());
    expect(onTrain).toHaveBeenCalledWith(dataset);
    await act(async () => Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent === "重复/相似图")?.click());
    expect(onDuplicates).toHaveBeenCalledWith(dataset);
    expect(container?.textContent).toContain("精确重复");
    act(() => Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent === "收起结果")?.click());
    expect(container?.textContent).not.toContain("精确重复");
    act(() => Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent === "导出数据集")?.click());
    const dialog = container?.querySelector('[role="dialog"]');
    expect(dialog?.textContent).toContain("导出“road-signs”");
    expect(dialog?.querySelector('a[href$="/api/datasets/ds-1/export/yolo"]')?.textContent).toContain("下载 YOLO ZIP");
  });
});
