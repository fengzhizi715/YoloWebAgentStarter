import type {
  Annotation,
  ClassLabel,
  Dataset,
  ImageItem,
  ImagePage,
  SplitName,
  TaskType,
  TrainingLog,
  TrainingTask,
  ModelVersion,
  SamPrediction,
  SystemInfo,
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
  listDatasets: () => request<Dataset[]>("/api/datasets"),
  createDataset: (name: string, taskType: TaskType, description?: string) =>
    request<Dataset>("/api/datasets", json({ name, task_type: taskType, description: description || null })),
  listClasses: (datasetId: string) => request<ClassLabel[]>(`/api/datasets/${datasetId}/classes`),
  createClass: (datasetId: string, name: string, color = "#22c55e") =>
    request<ClassLabel>(`/api/datasets/${datasetId}/classes`, json({ name, color })),
  listImages: (datasetId: string) => request<ImagePage>(`/api/datasets/${datasetId}/images?limit=500`),
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
  listTrainingTasks: (datasetId: string) => request<{ items: TrainingTask[] }>(`/api/training/tasks?dataset_id=${datasetId}`),
  createTrainingTask: (payload: { dataset_id: string; name: string; model: string; task_type: TaskType; epochs: number; img_size: number; batch_size: number; device: string; workers: number; seed: number }) =>
    request<TrainingTask>("/api/training/tasks", json(payload)),
  stopTrainingTask: (taskId: string) => request<TrainingTask>(`/api/training/tasks/${taskId}/stop`, { method: "POST" }),
  getTrainingLogs: (taskId: string, tail = 200) => request<TrainingLog>(`/api/training/tasks/${taskId}/logs?tail=${tail}`),
  downloadCheckpointUrl: (taskId: string, checkpoint: "best" | "last") => apiUrl(`/api/training/tasks/${taskId}/checkpoints/${checkpoint}`),
  listModels: (datasetId: string, includeArchived = true) => request<{ items: ModelVersion[]; total: number }>(`/api/models?dataset_id=${datasetId}&include_archived=${includeArchived}`),
  getModel: (modelId: string) => request<ModelVersion>(`/api/models/${modelId}`),
  updateModel: (modelId: string, payload: { name?: string; version?: string; notes?: string }) => request<ModelVersion>(`/api/models/${modelId}`, json(payload, "PATCH")),
  archiveModel: (modelId: string) => request<ModelVersion>(`/api/models/${modelId}/archive`, { method: "POST" }),
  restoreModel: (modelId: string) => request<ModelVersion>(`/api/models/${modelId}/restore`, { method: "POST" }),
  deleteModel: (modelId: string) => request<void>(`/api/models/${modelId}`, { method: "DELETE" }),
  exportModelOnnx: (modelId: string) => request<ModelVersion>(`/api/models/${modelId}/export-onnx`, { method: "POST" }),
  downloadModelUrl: (modelId: string) => apiUrl(`/api/models/${modelId}/download`),
};
