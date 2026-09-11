// @vitest-environment jsdom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ImportCenterModal } from "./ImportCenterModal";

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

describe("ImportCenterModal", () => {
  it("offers video import in the dataset import center", async () => {
    await act(async () => {
      root?.render(<ImportCenterModal busy={false} onClose={vi.fn()} onArchiveImport={vi.fn()} onVideoCompleted={vi.fn()} />);
    });
    expect(container?.textContent).toContain("YOLO 数据集");
    expect(container?.textContent).toContain("COCO 数据集");

    await act(async () => Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent?.includes("视频文件"))?.click());
    expect(container?.textContent).toContain("拖放或选择 MP4、MOV 或 AVI 视频");
    expect(container?.textContent).toContain("预检视频");

    const dropzone = container?.querySelector<HTMLButtonElement>(".video-import-form .exchange-dropzone");
    const file = new File(["video"], "dropped.mp4", { type: "video/mp4" });
    await act(async () => {
      const event = new Event("drop", { bubbles: true, cancelable: true });
      Object.defineProperty(event, "dataTransfer", { value: { files: [file] } });
      dropzone?.dispatchEvent(event);
    });
    expect(container?.textContent).toContain("dropped.mp4");
  });
});
