import type { BBox } from "../types";

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

export function normalizeBBox(start: [number, number], end: [number, number]): BBox {
  return {
    x: Math.min(start[0], end[0]),
    y: Math.min(start[1], end[1]),
    width: Math.abs(end[0] - start[0]),
    height: Math.abs(end[1] - start[1]),
  };
}

export function toOriginalPoint(
  point: [number, number],
  displayWidth: number,
  displayHeight: number,
  imageWidth: number,
  imageHeight: number,
): [number, number] {
  return [
    clamp((point[0] / displayWidth) * imageWidth, 0, imageWidth),
    clamp((point[1] / displayHeight) * imageHeight, 0, imageHeight),
  ];
}
