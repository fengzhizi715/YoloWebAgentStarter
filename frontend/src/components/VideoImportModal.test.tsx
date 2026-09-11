// @vitest-environment jsdom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { VideoImportModal } from "./VideoImportModal";
import { api } from "../api/client";
import type { VideoImportTask } from "../types";

vi.mock("../api/client", () => ({
  api: {
    createVideoImport: vi.fn(),
    getVideoImport: vi.fn(),
    retryVideoImport: vi.fn(),
    startVideoImport: vi.fn(),
  },
}));

const failedTask: VideoImportTask = {
  id: "vid_1", dataset_id: null, name: "traffic", task_type: "detect", split: "train", status: "failed", start_requested: true,
  checkpoint_next_frame_index: 4, output_bytes: 1200,
  config_json: { sampling_mode: "fps", sampling_value: 1, start_seconds: 0, end_seconds: null, split_strategy: "single", time_block_seconds: null, train_ratio: null, val_ratio: null, test_ratio: null },
  source_file_name: "traffic.mp4", source_checksum: "checksum", video_info_json: { width: 32, height: 24, fps: 4, frame_count: 8, duration_seconds: 2, size_bytes: 1000, estimated_output_bytes: 2000, estimate_sampled_frames: 4, estimated_split_counts: { train: 8, val: 0, test: 0 } },
  total_images: 8, generated_images: 1, progress_percent: 10, error_message: "Interrupted", created_at: "", updated_at: "",
};

let root: Root | undefined;
let container: HTMLDivElement | undefined;

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  vi.useFakeTimers();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
  vi.useRealTimers();
  vi.clearAllMocks();
});

describe("VideoImportModal", () => {
  it.each(["1", "2", "0.5"])("accepts %s as an FPS sampling value", async (samplingValue) => {
    await act(async () => {
      root?.render(<VideoImportModal onClose={vi.fn()} onCompleted={vi.fn()} />);
    });

    const input = container?.querySelectorAll<HTMLInputElement>('input[type="number"]')[0];
    expect(input).toBeDefined();
    await act(async () => {
      const valueSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
      valueSetter?.call(input, samplingValue);
      input?.dispatchEvent(new Event("input", { bubbles: true }));
    });
    expect(input?.validity.valid).toBe(true);
  });

  it("polls automatically after retry queues a pending task", async () => {
    vi.mocked(api.createVideoImport).mockResolvedValue(failedTask);
    vi.mocked(api.retryVideoImport).mockResolvedValue({ ...failedTask, status: "pending", start_requested: true, error_message: null });
    vi.mocked(api.getVideoImport).mockResolvedValue({ ...failedTask, status: "running", error_message: null });
    await act(async () => {
      root?.render(<VideoImportModal onClose={vi.fn()} onCompleted={vi.fn()} />);
    });

    const input = container?.querySelector<HTMLInputElement>('input[type="file"]');
    const file = new File(["video"], "traffic.mp4", { type: "video/mp4" });
    await act(async () => {
      Object.defineProperty(input, "files", { value: [file], configurable: true });
      input?.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await act(async () => {
      container?.querySelector("form")?.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    });
    expect(api.createVideoImport).toHaveBeenCalled();
    expect(api.createVideoImport).toHaveBeenCalledWith(expect.objectContaining({ samplingMode: "fps", samplingValue: 1 }));
    expect(container?.textContent).toContain("导入失败");

    await act(async () => Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent === "重试导入")?.click());
    expect(container?.textContent).toContain("正在排队");
    expect(Array.from(container?.querySelectorAll("button") ?? []).some((item) => item.textContent === "开始导入")).toBe(false);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });
    expect(api.getVideoImport).toHaveBeenCalledWith("vid_1");
  });

  it("starts polling when a preflight task changes from waiting for confirmation to queued", async () => {
    const preflightTask = { ...failedTask, status: "pending" as const, start_requested: false, error_message: null };
    const queuedTask = { ...preflightTask, start_requested: true };
    const completedTask = { ...queuedTask, status: "completed" as const, dataset_id: "ds_1", generated_images: 2, progress_percent: 100 };
    const onCompleted = vi.fn();
    vi.mocked(api.createVideoImport).mockResolvedValue(preflightTask);
    vi.mocked(api.startVideoImport).mockResolvedValue(queuedTask);
    vi.mocked(api.getVideoImport).mockResolvedValue(completedTask);
    await act(async () => {
      root?.render(<VideoImportModal onClose={vi.fn()} onCompleted={onCompleted} />);
    });

    const fileInput = container?.querySelector<HTMLInputElement>('input[type="file"]');
    const file = new File(["video"], "traffic.mp4", { type: "video/mp4" });
    await act(async () => {
      Object.defineProperty(fileInput, "files", { value: [file], configurable: true });
      fileInput?.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await act(async () => {
      container?.querySelector("form")?.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    });
    await act(async () => Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent === "开始导入")?.click());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });

    expect(api.getVideoImport).toHaveBeenCalledWith("vid_1");
    expect(onCompleted).toHaveBeenCalledWith(completedTask);
    expect(container?.textContent).toContain("导入完成");
  });
});
