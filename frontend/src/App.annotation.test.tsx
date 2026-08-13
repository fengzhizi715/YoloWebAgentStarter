// @vitest-environment jsdom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AnnotationView } from "./App";
import type { Annotation, ClassLabel, Dataset, ImageItem } from "./types";

vi.mock("./annotation/AnnotationCanvas", () => ({
  AnnotationCanvas: ({ onChange }: { onChange: (drafts: unknown[]) => void }) => (
    <button onClick={() => onChange([{ id: "ann-1", class_id: "cls-1", type: "obb", obb: { cx: 50, cy: 40, width: 60, height: 40, angle: 45 }, source: "manual" }])}>
      模拟旋转 OBB
    </button>
  ),
}));

const dataset: Dataset = { id: "ds-1", name: "vehicles", description: null, task_type: "obb", image_count: 1, annotated_image_count: 1, class_count: 1, created_at: "", updated_at: "" };
const image: ImageItem = { id: "img-1", dataset_id: dataset.id, file_name: "vehicle.png", width: 100, height: 80, split: "train", status: "annotated", file_url: "/api/images/img-1/file", created_at: "", updated_at: "" };
const secondImage: ImageItem = { ...image, id: "img-2", file_name: "vehicle-02.png", status: "unannotated" };
const classes: ClassLabel[] = [{ id: "cls-1", dataset_id: dataset.id, class_index: 0, name: "vehicle", color: "#22c55e", created_at: "", updated_at: "" }];
const annotation: Annotation = { id: "ann-1", image_id: image.id, dataset_id: dataset.id, class_id: "cls-1", class_index: 0, label: "vehicle", color: "#22c55e", type: "obb", bbox: null, polygon: null, obb: { cx: 45, cy: 35, width: 30, height: 20, angle: 20 }, source: "manual", created_at: "", updated_at: "" };

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

describe("AnnotationView persistence", () => {
  it("shows the complete image list and switches the active image", () => {
    const onImageSelect = vi.fn();
    act(() => {
      root?.render(<AnnotationView dataset={dataset} image={image} images={[image, secondImage]} onImageSelect={onImageSelect} classes={classes} annotations={[annotation]} activeClassId="cls-1" onClassChange={vi.fn()} onBack={vi.fn()} onSave={vi.fn()} onSam={vi.fn()} onSamPoints={vi.fn()} busy={false} />);
    });

    expect(container?.textContent).toContain("2 张图片");
    expect(container?.textContent).toContain("vehicle-02.png");
    expect(container?.textContent).toContain("未标注");
    const item = Array.from(container?.querySelectorAll<HTMLButtonElement>(".annotation-image-item") ?? []).find((button) => button.textContent?.includes("vehicle-02.png"));
    act(() => item?.click());
    expect(onImageSelect).toHaveBeenCalledWith(secondImage);
  });

  it("paginates the image list at twenty items", () => {
    const images = [image, ...Array.from({ length: 20 }, (_, index) => ({ ...image, id: `img-${index + 2}`, file_name: `vehicle-${index + 2}.png` }))];
    act(() => {
      root?.render(<AnnotationView dataset={{ ...dataset, image_count: images.length }} image={image} images={images} onImageSelect={vi.fn()} classes={classes} annotations={[annotation]} activeClassId="cls-1" onClassChange={vi.fn()} onBack={vi.fn()} onSave={vi.fn()} onSam={vi.fn()} onSamPoints={vi.fn()} busy={false} />);
    });

    expect(container?.querySelectorAll(".annotation-image-item")).toHaveLength(20);
    expect(container?.textContent).toContain("第 1 / 2 页");
    const next = Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((button) => button.textContent === "下一页");
    act(() => next?.click());
    expect(container?.querySelectorAll(".annotation-image-item")).toHaveLength(1);
    expect(container?.textContent).toContain("vehicle-21.png");
  });

  it("saves the edited OBB draft without its client-only id", () => {
    const onSave = vi.fn();
    act(() => {
      root?.render(<AnnotationView dataset={dataset} image={image} classes={classes} annotations={[annotation]} activeClassId="cls-1" onClassChange={vi.fn()} onBack={vi.fn()} onSave={onSave} onSam={vi.fn()} onSamPoints={vi.fn()} busy={false} />);
    });

    const rotate = Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent === "模拟旋转 OBB");
    act(() => rotate?.click());
    const save = Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent === "保存标注");
    act(() => save?.click());

    expect(onSave).toHaveBeenCalledWith([
      { id: "ann-1", class_id: "cls-1", type: "obb", obb: { cx: 50, cy: 40, width: 60, height: 40, angle: 45 }, source: "manual" },
    ]);
  });
});
