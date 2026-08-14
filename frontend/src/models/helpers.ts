import type { ModelVersion } from "../types";
import { taskTypeLabel } from "../training/helpers";

export type ModelListFilter = "all" | "pt" | "onnx" | "best" | "archived";
export type ModelListSort = "newest" | "map50";

export function newerFirst(a: ModelVersion, b: ModelVersion) {
  return Date.parse(b.created_at) - Date.parse(a.created_at);
}

export function formatMetric(value: number | null | undefined, digits = 2): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

export function formatDelta(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(3)}`;
}

export function metricBarWidth(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "0%";
  return `${Math.max(0, Math.min(100, value * 100))}%`;
}

export function pickBestModel(models: ModelVersion[]): ModelVersion | undefined {
  const scored = models.filter((item) => item.status === "active" && typeof item.map50 === "number");
  if (!scored.length) return models.find((item) => item.status === "active") ?? models[0];
  return [...scored].sort((a, b) => (b.map50 ?? 0) - (a.map50 ?? 0) || newerFirst(a, b))[0];
}

export function artifactTagsFor(model: ModelVersion, all: ModelVersion[]): Array<{ label: string; tone: "pt" | "onnx" | "best" | "last" | "archived" }> {
  const tags: Array<{ label: string; tone: "pt" | "onnx" | "best" | "last" | "archived" }> = [
    { label: model.format.toUpperCase(), tone: model.format === "onnx" ? "onnx" : "pt" },
  ];
  if (model.artifact_type === "best") tags.push({ label: "BEST", tone: "best" });
  if (model.artifact_type === "last") tags.push({ label: "LAST", tone: "last" });
  if (model.artifact_type === "onnx") tags.push({ label: "EXPORT", tone: "onnx" });
  if (model.status === "archived") tags.push({ label: "ARCHIVED", tone: "archived" });
  if (model.format === "pt" && all.some((item) => item.source_model_id === model.id && item.format === "onnx")) {
    tags.push({ label: "HAS ONNX", tone: "onnx" });
  }
  return tags;
}

export function modelTaskLabel(model: ModelVersion): string {
  return taskTypeLabel(model.task_type);
}

export function fileName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

export function sortModels(models: ModelVersion[], sort: ModelListSort): ModelVersion[] {
  const items = [...models];
  if (sort === "map50") {
    return items.sort((a, b) => (b.map50 ?? -1) - (a.map50 ?? -1) || newerFirst(a, b));
  }
  return items.sort(newerFirst);
}

export function filterModels(models: ModelVersion[], filter: ModelListFilter, search: string): ModelVersion[] {
  const q = search.trim().toLowerCase();
  return models.filter((item) => {
    if (filter === "pt" && item.format !== "pt") return false;
    if (filter === "onnx" && item.format !== "onnx") return false;
    if (filter === "best" && item.artifact_type !== "best") return false;
    if (filter === "archived" && item.status !== "archived") return false;
    if (!q) return true;
    return item.name.toLowerCase().includes(q)
      || item.version.toLowerCase().includes(q)
      || item.format.toLowerCase().includes(q)
      || item.artifact_type.toLowerCase().includes(q)
      || (item.base_model ?? "").toLowerCase().includes(q);
  });
}
