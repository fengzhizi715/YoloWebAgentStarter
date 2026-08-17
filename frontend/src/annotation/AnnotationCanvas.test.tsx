// @vitest-environment jsdom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AnnotationCanvas } from "./AnnotationCanvas";
import type { Annotation, ClassLabel, ImageItem } from "../types";

vi.mock("react-konva", async () => {
  const { forwardRef, useImperativeHandle } = await import("react");
  return {
    Stage: ({ children, ...props }: any) => <div data-testid="stage" {...props}>{children}</div>,
    Layer: ({ children }: any) => <div>{children}</div>,
    Image: () => <div data-testid="image" />,
    Line: () => <div data-testid="line" />,
    Rect: forwardRef((_props: any, ref) => {
      const { draggable, onClick, onTransformEnd } = _props;
      useImperativeHandle(ref, () => null);
      return <div>
        {draggable && <button aria-label="选择 OBB" onClick={onClick}>OBB</button>}
        {onTransformEnd && <button aria-label="旋转 OBB" onClick={() => onTransformEnd({ target: transformTarget() })}>变换</button>}
      </div>;
    }),
    Transformer: forwardRef(({ rotateEnabled }: any, ref) => {
      useImperativeHandle(ref, () => ({ nodes: () => undefined, getLayer: () => ({ batchDraw: () => undefined }) }));
      return <div data-testid="obb-transformer" data-rotate-enabled={String(rotateEnabled)} />;
    }),
  };
});

function transformTarget() {
  return {
    x: () => 50,
    y: () => 40,
    width: () => 40,
    height: () => 20,
    rotation: () => 45,
    scaleX: (value?: number) => value === undefined ? 1.5 : undefined,
    scaleY: (value?: number) => value === undefined ? 2 : undefined,
  };
}

const image: ImageItem = {
  id: "img-1",
  dataset_id: "ds-1",
  file_name: "vehicle.png",
  width: 100,
  height: 80,
  split: "train",
  status: "annotated",
  file_url: "/api/images/img-1/file",
  created_at: "",
  updated_at: "",
};

const classes: ClassLabel[] = [{ id: "cls-1", dataset_id: "ds-1", class_index: 0, name: "vehicle", color: "#22c55e", created_at: "", updated_at: "" }];

const annotation: Annotation = {
  id: "ann-1",
  image_id: image.id,
  dataset_id: image.dataset_id,
  class_id: classes[0].id,
  class_index: 0,
  label: "vehicle",
  color: classes[0].color,
  type: "obb",
  bbox: null,
  polygon: null,
  obb: { cx: 45, cy: 35, width: 30, height: 20, angle: 20 },
  source: "manual",
  created_at: "",
  updated_at: "",
};

let root: Root | undefined;
let container: HTMLDivElement | undefined;
const originalImage = window.Image;

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  class ReadyImage {
    onload: (() => void) | null = null;
    set src(_value: string) { this.onload?.(); }
  }
  Object.defineProperty(window, "Image", { configurable: true, writable: true, value: ReadyImage });
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
  Object.defineProperty(window, "Image", { configurable: true, writable: true, value: originalImage });
});

describe("AnnotationCanvas OBB interactions", () => {
  it("selects an existing OBB, applies a transform and angle edit, then removes it", async () => {
    const onChange = vi.fn();
    await act(async () => {
      root?.render(<AnnotationCanvas image={image} classes={classes} annotations={[annotation]} activeClassId="cls-1" taskType="obb" onChange={onChange} />);
    });

    const select = container?.querySelector<HTMLButtonElement>("[aria-label='选择 OBB']");
    expect(select).toBeTruthy();
    act(() => select?.click());
    expect(container?.textContent).toContain("已选中 OBB");
    expect(container?.querySelector("[data-testid='obb-transformer']")?.getAttribute("data-rotate-enabled")).toBe("true");

    const transform = container?.querySelector<HTMLButtonElement>("[aria-label='旋转 OBB']");
    act(() => transform?.click());
    expect(onChange).toHaveBeenLastCalledWith([
      expect.objectContaining({ obb: { cx: 50, cy: 40, width: 60, height: 40, angle: 45 } }),
    ]);

    const angle = container?.querySelector<HTMLInputElement>("[aria-label='OBB angle']");
    const valueSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
    act(() => {
      valueSetter?.call(angle, "30");
      angle?.dispatchEvent(new Event("input", { bubbles: true }));
    });
    expect(onChange).toHaveBeenLastCalledWith([
      expect.objectContaining({ obb: expect.objectContaining({ angle: 30 }) }),
    ]);

    const remove = Array.from(container?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent === "删除所选 OBB");
    act(() => remove?.click());
    expect(onChange).toHaveBeenLastCalledWith([]);
  });

  it("does not cancel a normal drag when pointerup precedes mouseup", async () => {
    const onChange = vi.fn();
    await act(async () => {
      root?.render(<AnnotationCanvas image={image} classes={classes} annotations={[]} activeClassId="cls-1" taskType="detect" onChange={onChange} />);
    });

    const stage = container?.querySelector<HTMLDivElement>("[data-testid='stage']") as (HTMLDivElement & { getStage?: () => unknown; getClassName?: () => string; getPointerPosition?: () => { x: number; y: number } }) | null;
    expect(stage).toBeTruthy();
    let position = { x: 10, y: 10 };
    stage!.getStage = () => stage;
    stage!.getClassName = () => "Stage";
    stage!.getPointerPosition = () => position;

    act(() => stage?.dispatchEvent(new MouseEvent("mousedown", { bubbles: true })));
    position = { x: 60, y: 50 };
    act(() => stage?.dispatchEvent(new MouseEvent("mousemove", { bubbles: true })));
    act(() => window.dispatchEvent(new Event("pointerup", { bubbles: true })));
    act(() => stage?.dispatchEvent(new MouseEvent("mouseup", { bubbles: true })));

    expect(onChange).toHaveBeenCalledWith([{ class_id: "cls-1", type: "bbox", bbox: { x: 10, y: 10, width: 50, height: 40 }, source: "manual" }]);
  });

  it("blocks drawing until a class is selected", async () => {
    const onChange = vi.fn();
    await act(async () => {
      root?.render(<AnnotationCanvas image={image} classes={[]} annotations={[]} activeClassId="" taskType="detect" onChange={onChange} />);
    });

    const stage = container?.querySelector<HTMLDivElement>("[data-testid='stage']") as (HTMLDivElement & { getStage?: () => unknown; getClassName?: () => string; getPointerPosition?: () => { x: number; y: number } }) | null;
    expect(container?.textContent).toContain("请先在右侧新增一个类别");
    stage!.getStage = () => stage;
    stage!.getClassName = () => "Stage";
    stage!.getPointerPosition = () => ({ x: 10, y: 10 });
    act(() => stage?.dispatchEvent(new MouseEvent("mousedown", { bubbles: true })));
    act(() => stage?.dispatchEvent(new MouseEvent("mouseup", { bubbles: true })));

    expect(onChange).not.toHaveBeenCalled();
  });
});
