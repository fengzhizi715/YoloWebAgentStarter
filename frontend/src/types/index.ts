export type TaskType = "detect" | "segment" | "obb" | "classify";
export type SplitName = "train" | "val" | "test";

export interface Dataset {
  id: string;
  name: string;
  description: string | null;
  task_type: TaskType;
  image_count: number;
  annotated_image_count: number;
  class_count: number;
  created_at: string;
  updated_at: string;
}

export interface ClassLabel {
  id: string;
  dataset_id: string;
  class_index: number;
  name: string;
  color: string;
  created_at: string;
  updated_at: string;
}

export interface ImageItem {
  id: string;
  dataset_id: string;
  file_name: string;
  width: number;
  height: number;
  split: SplitName;
  status: string;
  file_url: string;
  created_at: string;
  updated_at: string;
}

export interface BBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface OBB {
  cx: number;
  cy: number;
  width: number;
  height: number;
  angle: number;
}

export interface Annotation {
  id: string;
  image_id: string;
  dataset_id: string;
  class_id: string;
  class_index: number;
  label: string;
  color: string;
  type: "bbox" | "polygon" | "obb" | "classify";
  bbox: BBox | null;
  polygon: [number, number][] | null;
  obb: OBB | null;
  source: "manual" | "imported" | "sam";
  created_at: string;
  updated_at: string;
}

export interface SamPrediction {
  image_id: string;
  class_id: string;
  mask_id: string;
  polygon: [number, number][];
  score: number;
  backend_used: string;
  device: string | null;
}

export interface SamCapabilities {
  model_configured: boolean;
  box_prompt_available: boolean;
  point_prompt_available: boolean;
  box_backend: "ultralytics_sam" | "box_stub";
}

export interface SystemInfo {
  name: string;
  edition: string;
  version: string;
  task_types: TaskType[];
  data_dir: string;
  import_root: string;
  auth_enabled: boolean;
  sam: SamCapabilities;
}

export interface ValidationIssue {
  level: "error" | "warning";
  code: string;
  message: string;
  image_id: string | null;
  annotation_id: string | null;
}

export interface ValidationReport {
  dataset_id: string;
  valid: boolean;
  error_count: number;
  warning_count: number;
  issues: ValidationIssue[];
}

export interface QualityIssue {
  level: "error" | "warning" | "info";
  type: string;
  message: string;
  image_id: string | null;
  annotation_ids: string[];
  class_id: string | null;
  iou: number | null;
  value: number | null;
}

export interface DatasetQualityReport {
  dataset_id: string;
  task_type: TaskType;
  summary: { image_count: number; annotated_image_count: number; unannotated_image_count: number; coverage: number; annotation_count: number; bbox_count: number; polygon_count: number; obb_count: number; classify_count: number; small_object_count: number; small_object_ratio: number };
  class_distribution: Array<{ class_id: string; class_index: number; name: string; count: number; ratio: number }>;
  issues: QualityIssue[];
}
export interface DuplicateReport { images: number; duplicate: number; similar: number; invalid_images: number; invalid_image_ids: string[]; phash_distance: number; groups: Array<{ canonical_image_id: string; image_ids: string[]; kind: "exact" | "similar"; score: number; hamming_distance?: number }>; }

export interface ImagePage {
  items: ImageItem[];
  total: number;
}

export type TrainingStatus = "pending" | "running" | "completed" | "failed" | "stopped";

export interface TrainingTask {
  id: string;
  dataset_id: string;
  profile_id: string | null;
  name: string;
  status: TrainingStatus;
  task_type: TaskType;
  model_name: string;
  model_path: string;
  epochs: number;
  img_size: number;
  batch_size: number;
  device: string;
  workers: number;
  val_ratio: number;
  seed: number;
  optimizer: string | null;
  lr0: number | null;
  patience: number | null;
  command_preview: string | null;
  export_path: string | null;
  data_yaml_path: string | null;
  run_dir: string | null;
  logs_path: string | null;
  summary_path: string | null;
  best_model_path: string | null;
  last_model_path: string | null;
  progress_epoch: number;
  progress_total_epochs: number;
  progress_percent: number;
  metrics_json: Record<string, number>;
  error_message: string | null;
  stop_requested: boolean;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface TrainingLog {
  task_id: string;
  logs: string;
  line_count: number;
}

export interface TrainingSummary {
  task_id: string;
  status: TrainingStatus;
  training_config: Record<string, unknown>;
  dataset: Record<string, unknown>;
  progress: { epoch: number; total_epochs: number; percent: number };
  metrics: Record<string, unknown> & { history?: Array<Record<string, number>> };
  checkpoints: Record<string, string | null>;
  log_summary: { line_count: number; tail: string[] };
  risks: string[];
  next_steps: string[];
}

export type ModelStatus = "active" | "archived";
export type ModelFormat = "pt" | "onnx";
export type ModelArtifactType = "best" | "last" | "onnx";

export interface ModelVersion {
  id: string;
  name: string;
  version: string;
  dataset_id: string | null;
  training_task_id: string | null;
  source_model_id: string | null;
  source: "training_task" | "exported";
  artifact_type: ModelArtifactType;
  format: ModelFormat;
  task_type: TaskType;
  engine_type: string;
  model_path: string;
  base_model: string | null;
  status: ModelStatus;
  precision: number | null;
  recall: number | null;
  map50: number | null;
  map50_95: number | null;
  metrics_json: Record<string, number>;
  notes: string;
  archived_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface InferenceDetection { class_index: number; class_name: string; confidence: number; x: number; y: number; width: number; height: number; polygon: number[][] | null; obb_points: number[][] | null; }
export interface InferenceResult { model_id: string; task_type: TaskType; detections: InferenceDetection[]; inference_time_ms: number; }
export interface ModelComparison { dataset_id: string; baseline: { id: string; name: string; metrics: Record<string, number> }; candidate: { id: string; name: string; metrics: Record<string, number> }; deltas: Record<string, number | null>; suggestions: string[]; }
export interface ModelTestRecord { id: string; model_id: string; file_name: string; result_json: InferenceResult; created_at: string; }
export interface ModelEvaluationRecord { id: string; model_id: string; dataset_id: string; split: SplitName; status: "pending" | "running" | "completed" | "failed"; confidence: number; iou: number; result_json: { split?: SplitName; task_type?: TaskType; metrics?: Record<string, number>; artifacts?: Record<string, string | null>; error_samples?: Array<{ image_file?: string; class_index?: number; confidence?: number; type: string; message: string }> }; error_message: string | null; export_path: string | null; data_path: string | null; run_dir: string | null; logs_path: string | null; created_at: string; started_at: string | null; finished_at: string | null; }
