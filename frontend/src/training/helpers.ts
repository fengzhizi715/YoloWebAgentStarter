import type { TaskType, TrainingDevice, TrainingStatus, TrainingTask } from "../types";

export type PresetId = "fast" | "balanced" | "high";
export type DeviceType = "auto" | "cpu" | "mps" | "cuda";
export type ComputeMode = "single" | "multi";
export type HistoryFilter = "all" | "completed" | "failed" | "stopped";

export const PRESETS: Record<PresetId, { epochs: number; batch_size: number; img_size: number; patience: number }> = {
  fast: { epochs: 30, batch_size: 16, img_size: 640, patience: 12 },
  balanced: { epochs: 50, batch_size: 16, img_size: 640, patience: 20 },
  high: { epochs: 100, batch_size: 8, img_size: 640, patience: 30 },
};

export const BATCH_OPTIONS = [1, 2, 4, 8, 16, 32, 64];
export const IMG_SIZE_OPTIONS = [320, 416, 512, 640, 768, 896, 1024];

export interface TrainingFormState {
  name: string;
  model: string;
  epochs: number;
  batch_size: number;
  img_size: number;
  workers: number;
  seed: number;
  val_ratio: number;
  patience: string;
  optimizer: string;
  lr0: string;
  device_type: DeviceType;
  compute_mode: ComputeMode;
  gpu_ids: string[];
}

export function defaultModelFor(taskType: TaskType): string {
  return ({ detect: "yolo11n.pt", segment: "yolo11n-seg.pt", obb: "yolo11n-obb.pt", classify: "yolo11n-cls.pt" })[taskType];
}

const MODEL_FAMILIES: Array<{ stem: string; label: string; sizes: string[]; tasks: TaskType[] }> = [
  { stem: "yolo26", label: "YOLO26", sizes: ["n", "s"], tasks: ["detect", "segment", "obb", "classify"] },
  { stem: "yolo11", label: "YOLO11", sizes: ["n", "s", "m", "l", "x"], tasks: ["detect", "segment", "obb", "classify"] },
  { stem: "yolov8", label: "YOLOv8", sizes: ["n", "s"], tasks: ["detect", "segment", "obb", "classify"] },
];

export function modelChoicesFor(taskType: TaskType): Array<{ value: string; label: string }> {
  const suffix = ({ detect: "", segment: "-seg", obb: "-obb", classify: "-cls" })[taskType];
  const taskLabel = ({ detect: "检测", segment: "分割", obb: "OBB", classify: "分类" })[taskType];
  return MODEL_FAMILIES.filter((family) => family.tasks.includes(taskType)).flatMap((family) => family.sizes.map((size) => {
    const value = `${family.stem}${size}${suffix}.pt`;
    return { value, label: `${family.label}${size.toUpperCase()}（${taskLabel}） · ${value}` };
  }));
}

export function defaultForm(taskType: TaskType): TrainingFormState {
  const preset = PRESETS.balanced;
  return {
    name: "local-training",
    model: defaultModelFor(taskType),
    epochs: preset.epochs,
    batch_size: preset.batch_size,
    img_size: preset.img_size,
    workers: 2,
    seed: 42,
    val_ratio: 0.2,
    patience: String(preset.patience),
    optimizer: "",
    lr0: "",
    device_type: "auto",
    compute_mode: "single",
    gpu_ids: [],
  };
}

export function statusLabel(status: TrainingStatus): string {
  return ({ pending: "排队中", running: "运行中", completed: "已完成", failed: "失败", stopped: "已停止" })[status];
}

export function taskTypeLabel(taskType: TaskType): string {
  return ({ detect: "检测", segment: "分割", obb: "旋转框", classify: "分类" })[taskType];
}

export function formatShortId(id: string): string {
  return id.length > 12 ? `${id.slice(0, 8)}…` : id;
}

export function formatDateTime(value: string | null): string {
  if (!value) return "—";
  return value.slice(0, 19).replace("T", " ");
}

export function formatRelativeTime(value: string | null): string {
  if (!value) return "—";
  const ts = Date.parse(value);
  if (!Number.isFinite(ts)) return formatDateTime(value);
  const deltaSec = Math.round((Date.now() - ts) / 1000);
  if (deltaSec < 60) return "刚刚";
  if (deltaSec < 3600) return `${Math.floor(deltaSec / 60)} 分钟前`;
  if (deltaSec < 86400) return `${Math.floor(deltaSec / 3600)} 小时前`;
  if (deltaSec < 86400 * 7) return `${Math.floor(deltaSec / 86400)} 天前`;
  return formatDateTime(value);
}

export function formatDuration(startedAt: string | null, finishedAt: string | null): string | null {
  if (!startedAt) return null;
  const start = Date.parse(startedAt);
  const end = Date.parse(finishedAt || new Date().toISOString());
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return null;
  const sec = Math.round((end - start) / 1000);
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export function resolveDeviceString(form: TrainingFormState): string {
  if (form.device_type === "cuda") return form.gpu_ids.join(",") || "0";
  return form.device_type;
}

export function deviceStatusLabel(device: TrainingDevice | undefined, fallback: string): string {
  if (!device) return fallback;
  return ({ available: "可用", idle: "空闲", busy: "忙碌", unavailable: "不可用", unknown: "未知" })[device.status] ?? fallback;
}

export function formatGpuMemory(device: TrainingDevice): string {
  if (!device.memory_total_mb) return "显存未知";
  const total = Math.round(device.memory_total_mb / 1024);
  if (device.memory_free_mb == null) return `${total} GB`;
  return `${Math.round(device.memory_free_mb / 1024)} / ${total} GB 可用`;
}

export function estimateMinutes(epochs: number, imageCount: number, batchSize: number, gpuCount: number): number {
  const steps = Math.max(1, Math.ceil(Math.max(imageCount, 1) / Math.max(batchSize, 1)) * epochs);
  const perStepSec = gpuCount > 1 ? 0.35 : 0.55;
  return Math.max(1, Math.round((steps * perStepSec) / 60));
}

export function estimateOptimizerSteps(epochs: number, imageCount: number, batchSize: number): number {
  return Math.max(1, Math.ceil(Math.max(imageCount, 1) / Math.max(batchSize, 1)) * epochs);
}

export function applyTaskToForm(task: TrainingTask): TrainingFormState {
  const device = task.device || "auto";
  const isCudaList = /^\d+(,\d+)*$/.test(device);
  const gpuIds = isCudaList ? device.split(",") : [];
  return {
    name: `${task.name}-retry`,
    model: task.model_name,
    epochs: task.epochs,
    batch_size: task.batch_size,
    img_size: task.img_size,
    workers: task.workers,
    seed: task.seed,
    val_ratio: task.val_ratio,
    patience: task.patience != null ? String(task.patience) : "",
    optimizer: task.optimizer ?? "",
    lr0: task.lr0 != null ? String(task.lr0) : "",
    device_type: isCudaList ? "cuda" : (device as DeviceType),
    compute_mode: gpuIds.length > 1 ? "multi" : "single",
    gpu_ids: gpuIds,
  };
}

export function pickPreset(form: TrainingFormState): PresetId | null {
  for (const id of ["fast", "balanced", "high"] as const) {
    const preset = PRESETS[id];
    if (form.epochs === preset.epochs && form.batch_size === preset.batch_size && form.img_size === preset.img_size) return id;
  }
  return null;
}

export function metricNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}
