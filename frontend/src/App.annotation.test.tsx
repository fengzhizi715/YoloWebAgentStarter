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
const personClass: ClassLabel = { id: "cls-2", dataset_id: dataset.id, class_index: 1, name: "person", color: "#3157d5", created_at: "", updated_at: "" };
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

  it("keeps previous and next navigation available by button and keyboard", () => {
    const onPrevious = vi.fn();
    const onNext = vi.fn();
    act(() => {
      root?.render(<AnnotationView dataset={{ ...dataset, image_count: 2 }} image={image} images={[image, secondImage]} onImageSelect={vi.fn()} classes={classes} annotations={[annotation]} activeClassId="cls-1" onClassChange={vi.fn()} onBack={vi.fn()} onPrevious={onPrevious} onNext={onNext} hasPrevious hasNext onSave={vi.fn()} onSam={vi.fn()} onSamPoints={vi.fn()} busy={false} />);
    });

    const previous = container?.querySelector<HTMLButtonElement>(".annotation-navigation button:first-child");
    const next = container?.querySelector<HTMLButtonElement>(".annotation-navigation button:last-child");
    expect(previous?.textContent).toContain("上一张");
    expect(next?.textContent).toContain("下一张");
    act(() => next?.click());
    act(() => window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft" })));
    expect(onNext).toHaveBeenCalledTimes(1);
    expect(onPrevious).toHaveBeenCalledTimes(1);
  });

  it("adds a custom class from the annotation sidebar", async () => {
    const onCreateClass = vi.fn(async () => personClass);
    act(() => {
      root?.render(<AnnotationView dataset={dataset} image={image} classes={classes} annotations={[annotation]} activeClassId="cls-1" onClassChange={vi.fn()} onBack={vi.fn()} onCreateClass={onCreateClass} onSave={vi.fn()} onSam={vi.fn()} onSamPoints={vi.fn()} busy={false} />);
    });

    const open = Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((button) => button.textContent?.includes("新增类别"));
    act(() => open?.click());
    const input = container?.querySelector<HTMLInputElement>("input[placeholder='例如：person']");
    expect(input).toBeTruthy();
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
    act(() => {
      setValue?.call(input, "person");
      input?.dispatchEvent(new Event("input", { bubbles: true }));
    });
    const form = container?.querySelector<HTMLFormElement>(".annotation-class-form");
    await act(async () => { form?.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); });

    expect(onCreateClass).toHaveBeenCalledWith("person", "#ef4444");
  });

  it("allows changing an existing shape to another class", () => {
    const onSave = vi.fn();
    act(() => {
      root?.render(<AnnotationView dataset={{ ...dataset, class_count: 2 }} image={image} classes={[...classes, personClass]} annotations={[annotation]} activeClassId="cls-1" onClassChange={vi.fn()} onBack={vi.fn()} onSave={onSave} onSam={vi.fn()} onSamPoints={vi.fn()} busy={false} />);
    });

    const classSelect = container?.querySelector<HTMLSelectElement>(".shape-class-select");
    expect(classSelect).toBeTruthy();
    act(() => {
      if (classSelect) classSelect.value = personClass.id;
      classSelect?.dispatchEvent(new Event("change", { bubbles: true }));
    });
    const save = Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((button) => button.textContent === "保存标注");
    act(() => save?.click());

    expect(onSave).toHaveBeenCalledWith([{ id: "ann-1", class_id: personClass.id, type: "obb", obb: { cx: 45, cy: 35, width: 30, height: 20, angle: 20 }, source: "manual" }]);
  });
});
