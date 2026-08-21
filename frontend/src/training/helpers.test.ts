import { describe, expect, it } from "vitest";
import { modelChoicesFor } from "./helpers";

describe("training model choices", () => {
  it("offers the supported YOLO26, YOLO11 and YOLOv8 families for detection", () => {
    const values = modelChoicesFor("detect").map((item) => item.value);

    expect(values).toEqual(expect.arrayContaining([
      "yolo26n.pt",
      "yolo11n.pt",
      "yolo11m.pt",
      "yolov8n.pt",
    ]));
    expect(values).not.toEqual(expect.arrayContaining(["yolo13n.pt"]));
  });

  it("keeps task-specific suffixes for segmentation, OBB and classification", () => {
    expect(modelChoicesFor("segment").map((item) => item.value)).toContain("yolo26n-seg.pt");
    expect(modelChoicesFor("classify").map((item) => item.value)).toContain("yolov8n-cls.pt");
  });
});
