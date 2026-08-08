import { describe, expect, it } from "vitest";
import { normalizeBBox, toOriginalPoint } from "./geometry";

describe("annotation geometry helpers", () => {
  it("normalizes a drag in any direction", () => {
    expect(normalizeBBox([80, 60], [20, 10])).toEqual({ x: 20, y: 10, width: 60, height: 50 });
  });

  it("maps display coordinates back to absolute image pixels", () => {
    expect(toOriginalPoint([50, 25], 100, 50, 1000, 500)).toEqual([500, 250]);
  });
});
