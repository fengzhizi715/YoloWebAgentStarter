import type {
  Annotation,
  ClassLabel,
  Dataset,
  DatasetQualityReport,
  DuplicateReport,
  ImageItem,
  ImagePage,
  SplitName,
  TaskType,
  TrainingLog,
  TrainingTask,
  ModelVersion,
  InferenceResult,
  ModelComparison,
  ModelEvaluationRecord,
  ModelTestRecord,
  SamPrediction,
  SamSettings,
  SystemInfo,
  RuntimeLogResponse,
  TrainingDevice,
  ValidationReport,
} from "../types";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { error?: { message?: string } };
      message = body.error?.message ?? message;
    } catch {
      // Keep the HTTP status when the server does not return JSON.
    }
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

const json = (body: unknown, method = "POST"): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  getSystemInfo: () => request<SystemInfo>("/api/system/info"),
  getSamSettings: () => request<SamSettings>("/api/settings/sam"),
  updateSamSettings: (payload: Omit<SamSettings, "model_configured">) => request<SamSettings>("/api/settings/sam", json(payload, "PUT")),
  listDatasets: () => request<Dataset[]>("/api/datasets"),
  createDataset: (name: string, taskType: TaskType, description?: string) =>
    request<Dataset>("/api/datasets", json({ name, task_type: taskType, description: description || null })),
  deleteDataset: (datasetId: string) => request<void>(`/api/datasets/${datasetId}`, { method: "DELETE" }),
  listClasses: (datasetId: string) => request<ClassLabel[]>(`/api/datasets/${datasetId}/classes`),
  createClass: (datasetId: string, name: string, color = "#22c55e") =>
    request<ClassLabel>(`/api/datasets/${datasetId}/classes`, json({ name, color })),
  listImages: async (datasetId: string) => {
    const pageSize = 500;
    const first = await request<ImagePage>(`/api/datasets/${datasetId}/images?offset=0&limit=${pageSize}`);
    if (first.total <= first.items.length) return first;
    const offsets = Array.from({ length: Math.ceil((first.total - first.items.length) / pageSize) }, (_, index) => first.items.length + index * pageSize);
    const pages = await Promise.all(offsets.map((offset) => request<ImagePage>(`/api/datasets/${datasetId}/images?offset=${offset}&limit=${pageSize}`)));
    return { total: first.total, items: [first, ...pages].flatMap((page) => page.items) };
  },
  uploadImages: async (datasetId: string, files: File[], split: SplitName) => {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    form.append("split", split);
    return request<{ imported: number; items: ImageItem[] }>(`/api/datasets/${datasetId}/images/upload`, {
      method: "POST",
      body: form,
    });
  },
  updateImageSplit: (datasetId: string, imageId: string, split: SplitName) =>
    request<ImageItem>(`/api/datasets/${datasetId}/images/${imageId}`, json({ split }, "PATCH")),
  updateImageSplits: (datasetId: string, imageIds: string[], split: SplitName) => request<{ updated: number; split_counts: Record<SplitName, number> }>(`/api/datasets/${datasetId}/images/bulk-split`, json({ image_ids: imageIds, split })),
  autoSplitImages: (datasetId: string, data: { train_ratio: number; val_ratio: number; test_ratio: number; seed: number }) => request<{ updated: number; split_counts: Record<SplitName, number> }>(`/api/datasets/${datasetId}/images/auto-split`, json(data)),
  scanImages: (datasetId: string, path: string, split: SplitName) =>
    request<{ total_found: number; imported: number; skipped: number; invalid: number }>(`/api/datasets/${datasetId}/images/scan`, json({ path, recursive: true, split })),
  getAnnotations: (datasetId: string, imageId: string) =>
    request<Annotation[]>(`/api/datasets/${datasetId}/images/${imageId}/annotations`),
  replaceAnnotations: (datasetId: string, imageId: string, annotations: unknown[]) =>
    request<Annotation[]>(`/api/datasets/${datasetId}/images/${imageId}/annotations`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ annotations }),
    }),
  samPredict: (payload: { image_id: string; class_id: string; prompt_type: "box" | "point"; box?: { x: number; y: number; width: number; height: number }; points?: { x: number; y: number; label?: 0 | 1 }[] }) =>
    request<SamPrediction>("/api/sam/predict", json(payload)),
  validateDataset: (datasetId: string) => request<ValidationReport>(`/api/datasets/${datasetId}/validate`, { method: "POST" }),
  qualityReport: (datasetId: string) => request<DatasetQualityReport>(`/api/datasets/${datasetId}/quality/report`),
  duplicateReport: (datasetId: string) => request<DuplicateReport>(`/api/datasets/${datasetId}/duplicates`),
  importVideo: async (datasetId: string, file: File, split: SplitName, frameInterval: number) => { const form = new FormData(); form.append("file", file); form.append("split", split); form.append("frame_interval", String(frameInterval)); return request<{ imported: number; source_fps: number; frame_count: number }>(`/api/datasets/${datasetId}/video/import`, { method: "POST", body: form }); },
  importYolo: async (file: File, name: string, taskType: TaskType) => {
    const form = new FormData();
    form.append("file", file);
    form.append("name", name);
    form.append("task_type", taskType);
    return request<{ dataset: Dataset; imported_images: number; imported_annotations: number }>("/api/datasets/import/yolo", {
      method: "POST",
      body: form,
    });
  },
  exportYoloUrl: (datasetId: string) => apiUrl(`/api/datasets/${datasetId}/export/yolo`),
  exportCocoUrl: (datasetId: string) => apiUrl(`/api/datasets/${datasetId}/export/coco`),
  importCoco: async (file: File, name: string, taskType: TaskType) => { const form = new FormData(); form.append("file", file); form.append("name", name); form.append("task_type", taskType); return request<{ dataset: Dataset; imported_images: number; imported_annotations: number }>("/api/datasets/import/coco", { method: "POST", body: form }); },
  tileDataset: (datasetId: string, data: { name: string; description?: string; tile_size: number; overlap: number; keep_empty_tiles: boolean }) => request<{ dataset_id: string; source_dataset_id: string; generated_images: number; generated_annotations: number; skipped_empty_tiles: number }>(`/api/datasets/${datasetId}/tile`, json(data)),
  listTrainingTasks: (datasetId: string) => request<{ items: TrainingTask[] }>(`/api/training/tasks?dataset_id=${datasetId}`),
  listTrainingDevices: () => request<{ items: TrainingDevice[] }>("/api/training/devices"),
  createTrainingTask: (payload: {
    dataset_id: string;
    name: string;
    model: string;
    task_type: TaskType;
    epochs: number;
    img_size: number;
    batch_size: number;
    device: string;
    workers: number;
    seed: number;
    val_ratio?: number;
    optimizer?: string;
    lr0?: number;
    patience?: number;
  }) => request<TrainingTask>("/api/training/tasks", json(payload)),
  stopTrainingTask: (taskId: string) => request<TrainingTask>(`/api/training/tasks/${taskId}/stop`, { method: "POST" }),
  resumeTrainingTask: (taskId: string, data: { name?: string; epochs?: number; resume_epoch?: boolean } = {}) => request<TrainingTask>(`/api/training/tasks/${taskId}/resume`, json(data)),
  getTrainingLogs: (taskId: string, tail = 200) => request<TrainingLog>(`/api/training/tasks/${taskId}/logs?tail=${tail}`),
  getTrainingSummary: (taskId: string) => request<import("../types").TrainingSummary>(`/api/training/tasks/${taskId}/summary`),
  downloadCheckpointUrl: (taskId: string, checkpoint: "best" | "last") => apiUrl(`/api/training/tasks/${taskId}/checkpoints/${checkpoint}`),
  listModels: (datasetId: string, includeArchived = true) => request<{ items: ModelVersion[]; total: number }>(`/api/models?dataset_id=${datasetId}&include_archived=${includeArchived}`),
  getModel: (modelId: string) => request<ModelVersion>(`/api/models/${modelId}`),
  updateModel: (modelId: string, payload: { name?: string; version?: string; notes?: string }) => request<ModelVersion>(`/api/models/${modelId}`, json(payload, "PATCH")),
  archiveModel: (modelId: string) => request<ModelVersion>(`/api/models/${modelId}/archive`, { method: "POST" }),
  restoreModel: (modelId: string) => request<ModelVersion>(`/api/models/${modelId}/restore`, { method: "POST" }),
  deleteModel: (modelId: string) => request<void>(`/api/models/${modelId}`, { method: "DELETE" }),
  exportModelOnnx: (modelId: string) => request<ModelVersion>(`/api/models/${modelId}/export-onnx`, { method: "POST" }),
  testModel: async (modelId: string, file: File, confidence: number, iou: number) => { const form = new FormData(); form.append("file", file); return request<InferenceResult>(`/api/models/${modelId}/test?confidence=${confidence}&iou=${iou}`, { method: "POST", body: form }); },
  listModelTests: (modelId: string) => request<ModelTestRecord[]>(`/api/models/${modelId}/tests`),
  evaluateModel: (modelId: string, split: SplitName = "val") => request<ModelEvaluationRecord>(`/api/models/${modelId}/evaluate`, json({ split })),
  listModelEvaluations: (modelId: string) => request<ModelEvaluationRecord[]>(`/api/models/${modelId}/evaluations`),
  modelEvaluationLogs: (modelId: string, evaluationId: string, tail = 500) => request<{ evaluation_id: string; logs: string; line_count: number }>(`/api/models/${modelId}/evaluations/${evaluationId}/logs?tail=${tail}`),
  modelEvaluationArtifactUrl: (modelId: string, evaluationId: string, artifact: "confusion_matrix" | "pr_curve" | "box_pr_curve" | "mask_pr_curve" | "predictions") => apiUrl(`/api/models/${modelId}/evaluations/${evaluationId}/artifacts/${artifact}`),
  compareModels: (baselineModelId: string, candidateModelId: string) => request<ModelComparison>("/api/models/compare", json({ baseline_model_id: baselineModelId, candidate_model_id: candidateModelId })),
  downloadModelUrl: (modelId: string) => apiUrl(`/api/models/${modelId}/download`),
  runtimeLogs: (options: { lines?: number; level?: string } = {}) => {
    const query = new URLSearchParams();
    if (options.lines) query.set("lines", String(options.lines));
    if (options.level) query.set("level", options.level);
    return request<RuntimeLogResponse>(`/api/logs/runtime${query.toString() ? `?${query.toString()}` : ""}`);
  },
};
