// @vitest-environment jsdom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DatasetHome } from "./App";
import type { Dataset, SplitName } from "./types";

const dataset: Dataset = { id: "ds-1", name: "road-signs", description: null, task_type: "detect", image_count: 3, annotated_image_count: 1, class_count: 2, created_at: "", updated_at: "" };

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

describe("DatasetHome image upload", () => {
  it("opens the add-images dialog and submits selected files with split", () => {
    const onUpload = vi.fn();
    act(() => {
      root?.render(<DatasetHome datasets={[dataset]} busy={false} onCreate={vi.fn()} onImport={vi.fn()} onValidate={vi.fn()} onQuality={vi.fn()} onContinueAnnotation={vi.fn()} onUpload={onUpload} onTrain={vi.fn()} onDuplicates={vi.fn()} onDelete={vi.fn()} />);
    });

    act(() => Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent === "+ 添加图片")?.click());
    expect(container?.querySelector('[role="dialog"]')?.textContent).toContain("添加图片到“road-signs”");

    const input = container?.querySelector<HTMLInputElement>('input[type="file"][multiple]');
    const file = new File(["image"], "road-sign.png", { type: "image/png" });
    act(() => {
      Object.defineProperty(input, "files", { value: [file], configurable: true });
      input?.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(container?.textContent).toContain("已选择 1 张图片");

    const split = container?.querySelector<HTMLSelectElement>(".upload-options select");
    act(() => {
      if (split) {
        Object.defineProperty(split, "value", { value: "val", configurable: true });
        split.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });
    act(() => Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent === "添加 1 张图片")?.click());
    expect(onUpload).toHaveBeenCalledWith(dataset, [file], "val" satisfies SplitName);
  });
});
