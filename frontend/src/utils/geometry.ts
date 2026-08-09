import type { BBox, OBB } from "../types";

export const MIN_SHAPE_SIZE = 3;

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

export function normalizeAngle(angle: number): number {
  const normalized = ((angle % 360) + 360) % 360;
  return normalized > 180 ? normalized - 360 : normalized;
}

export function obbCorners(obb: OBB): [number, number][] {
  const radians = (obb.angle * Math.PI) / 180;
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  const halfWidth = obb.width / 2;
  const halfHeight = obb.height / 2;
  return ([-1, 1, 1, -1] as const).map((horizontal, index) => {
    const vertical = index < 2 ? -1 : 1;
    return [
      obb.cx + horizontal * halfWidth * cosine - vertical * halfHeight * sine,
      obb.cy + horizontal * halfWidth * sine + vertical * halfHeight * cosine,
    ];
  });
}

/** Keep a persisted OBB valid while preserving its centre/size/angle contract. */
export function clampOBBToImage(obb: OBB, imageWidth: number, imageHeight: number): OBB {
  let resolved = {
    ...obb,
    width: Math.min(Math.max(obb.width, MIN_SHAPE_SIZE), imageWidth),
    height: Math.min(Math.max(obb.height, MIN_SHAPE_SIZE), imageHeight),
    angle: normalizeAngle(obb.angle),
  };
  const radians = (resolved.angle * Math.PI) / 180;
  const cosine = Math.abs(Math.cos(radians));
  const sine = Math.abs(Math.sin(radians));
  let halfX = (resolved.width * cosine + resolved.height * sine) / 2;
  let halfY = (resolved.width * sine + resolved.height * cosine) / 2;
  const fitScale = Math.min(1, imageWidth / Math.max(halfX * 2, MIN_SHAPE_SIZE), imageHeight / Math.max(halfY * 2, MIN_SHAPE_SIZE));
  if (fitScale < 1) {
    resolved = { ...resolved, width: Math.max(MIN_SHAPE_SIZE, resolved.width * fitScale), height: Math.max(MIN_SHAPE_SIZE, resolved.height * fitScale) };
    halfX = (resolved.width * cosine + resolved.height * sine) / 2;
    halfY = (resolved.width * sine + resolved.height * cosine) / 2;
  }
  return {
    ...resolved,
    cx: clamp(resolved.cx, Math.min(halfX, imageWidth / 2), Math.max(imageWidth - halfX, imageWidth / 2)),
    cy: clamp(resolved.cy, Math.min(halfY, imageHeight / 2), Math.max(imageHeight - halfY, imageHeight / 2)),
  };
}
