import { describe, expect, it } from "vitest";
import { clampOBBToImage, normalizeAngle, normalizeBBox, obbCorners, toOriginalPoint } from "./geometry";

describe("annotation geometry helpers", () => {
  it("normalizes a drag in any direction", () => {
    expect(normalizeBBox([80, 60], [20, 10])).toEqual({ x: 20, y: 10, width: 60, height: 50 });
  });

  it("maps display coordinates back to absolute image pixels", () => {
    expect(toOriginalPoint([50, 25], 100, 50, 1000, 500)).toEqual([500, 250]);
  });

  it("normalizes OBB rotation while keeping edited corners inside the image", () => {
    const obb = clampOBBToImage({ cx: 5, cy: 5, width: 80, height: 60, angle: 270 }, 100, 80);
    expect(obb.angle).toBe(-90);
    expect(obb.width).toBe(80);
    expect(obb.height).toBe(60);
    for (const [x, y] of obbCorners(obb)) {
      expect(x).toBeGreaterThanOrEqual(-1e-9);
      expect(x).toBeLessThanOrEqual(100 + 1e-9);
      expect(y).toBeGreaterThanOrEqual(-1e-9);
      expect(y).toBeLessThanOrEqual(80 + 1e-9);
    }
    expect(normalizeAngle(-270)).toBe(90);
  });
});
