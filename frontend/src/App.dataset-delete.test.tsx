// @vitest-environment jsdom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DatasetHome } from "./App";
import type { Dataset } from "./types";

const dataset: Dataset = {
  id: "ds-1",
  name: "road-signs",
  description: null,
  task_type: "detect",
  image_count: 12,
  annotated_image_count: 0,
  class_count: 3,
  created_at: "",
  updated_at: "",
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

describe("DatasetHome delete flow", () => {
  it("requires confirmation before deleting a dataset", () => {
    const onDelete = vi.fn();
    act(() => {
      root?.render(<DatasetHome datasets={[dataset]} busy={false} onSelect={vi.fn()} onCreate={vi.fn()} onImport={vi.fn()} onValidate={vi.fn()} onQuality={vi.fn()} onContinueAnnotation={vi.fn()} onDelete={onDelete} />);
    });

    const button = (label: string) => Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent === label);
    act(() => button("删除")?.click());

    expect(container?.querySelector('[role="alertdialog"]')?.textContent).toContain("road-signs");
    expect(container?.querySelector('[role="alertdialog"]')?.textContent).toContain("12 张图片");
    expect(onDelete).not.toHaveBeenCalled();

    act(() => button("确认删除")?.click());

    expect(onDelete).toHaveBeenCalledOnce();
    expect(onDelete).toHaveBeenCalledWith(dataset);
    expect(container?.querySelector('[role="alertdialog"]')).toBeNull();
  });
});
