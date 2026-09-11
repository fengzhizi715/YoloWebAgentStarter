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
  source_type?: string;
  source_file?: string | null;
  source_group_id?: string | null;
  source_video_task_id?: string | null;
  source_checksum?: string | null;
  frame_index?: number | null;
  timestamp?: number | null;
  file_url: string;
  created_at: string;
  updated_at: string;
}

export interface VideoImportTask {
  id: string;
  dataset_id: string | null;
  name: string;
  task_type: TaskType;
  split: SplitName;
  status: "pending" | "running" | "completed" | "failed";
  start_requested: boolean;
  checkpoint_next_frame_index: number;
  output_bytes: number;
  config_json: {
    sampling_mode: "fps" | "interval_seconds" | "frame_interval";
    sampling_value: number;
    start_seconds: number;
    end_seconds: number | null;
    split_strategy: "single" | "time_blocks";
    time_block_seconds: number | null;
    train_ratio: number | null;
    val_ratio: number | null;
    test_ratio: number | null;
  };
  source_file_name: string;
  source_checksum: string;
  video_info_json: {
    width: number;
    height: number;
    fps: number;
    frame_count: number;
    duration_seconds: number;
    size_bytes: number;
    estimated_output_bytes: number;
    estimate_sampled_frames: number;
    estimated_split_counts: Record<SplitName, number>;
  };
  total_images: number;
  generated_images: number;
  progress_percent: number;
  error_message: string | null;
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
  source: "manual" | "imported" | "sam" | "auto";
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
  enabled?: boolean;
  model_configured: boolean;
  box_prompt_available: boolean;
  point_prompt_available: boolean;
  box_backend: "ultralytics_sam" | "box_stub";
  device?: string;
}

export interface SamSettings {
  enabled: boolean;
  model: string;
  device: string;
  img_size: number;
  fallback_mode: "box" | "disabled";
  model_configured: boolean;
}

/** Upstream-compatible LLM settings (GET never returns api_key). */
export type LLMAuthScheme = "bearer" | "header" | "raw_authorization";

export interface LLMSettings {
  enabled: boolean;
  provider: string;
  api_base: string;
  model: string;
  temperature: number;
  timeout_seconds: number;
  api_key_configured: boolean;
  auth_scheme: LLMAuthScheme;
  auth_header_name: string;
}

export interface LLMConnectionTestResult {
  ok: boolean;
  message: string;
  remote_test_performed: boolean;
}

export interface TrainingDevice {
  id: string;
  type: "cpu" | "mps" | "cuda";
  name: string;
  index: number | null;
  memory_total_mb: number | null;
  memory_free_mb: number | null;
  status: "available" | "idle" | "busy" | "unavailable" | "unknown";
}

export interface RuntimeLogResponse {
  path: string;
  level: string | null;
  lines: string[];
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
  training_devices?: TrainingDevice[];
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
  timing: Record<string, number>;
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
export type AutoAnnotationStatus = "pending" | "running" | "completed" | "failed" | "stopped";
export interface AutoAnnotationTask {
  id: string;
  dataset_id: string;
  model_id: string;
  task_type: TaskType;
  status: AutoAnnotationStatus;
  clean_old_annotations: boolean;
  skip_annotated_images: boolean;
  confidence: number;
  iou: number;
  class_mapping: Record<string, string>;
  total_images: number;
  processed_images: number;
  created_annotations: number;
  skipped_images: number;
  progress_percent: number;
  logs_path: string | null;
  error_message: string | null;
  stop_requested: boolean;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  finished_at: string | null;
}
export interface AutoAnnotationLog { task_id: string; logs: string; line_count: number; }
export interface ModelEvaluationRecord { id: string; model_id: string; dataset_id: string; split: SplitName; status: "pending" | "running" | "completed" | "failed"; confidence: number; iou: number; result_json: { split?: SplitName; task_type?: TaskType; metrics?: Record<string, number>; artifacts?: Record<string, string | null>; error_samples?: Array<{ image_file?: string; class_index?: number; confidence?: number; type: string; message: string }> }; error_message: string | null; export_path: string | null; data_path: string | null; run_dir: string | null; logs_path: string | null; created_at: string; started_at: string | null; finished_at: string | null; }

export type AgentRunStatus = "pending" | "running" | "awaiting_approval" | "completed" | "failed" | "cancelled";
export type AgentMessageRole = "user" | "assistant" | "system" | "tool";
export type AgentToolCallStatus = "pending" | "running" | "completed" | "failed" | "awaiting_approval";
export type AgentApprovalStatus = "pending" | "approved" | "executing" | "rejected" | "expired" | "executed";

export interface AgentMessage {
  id: string;
  session_id: string;
  run_id: string | null;
  role: AgentMessageRole;
  content: string;
  sequence: number;
  created_at: string;
  updated_at: string;
}

export interface AgentToolCall {
  id: string;
  run_id: string;
  name: string;
  arguments_json: Record<string, unknown>;
  result_json: Record<string, unknown> | null;
  status: AgentToolCallStatus;
  error_message: string | null;
  sequence: number;
  created_at: string;
  updated_at: string;
}

export interface AgentApproval {
  id: string;
  run_id: string;
  tool_call_id: string | null;
  tool_name: string;
  payload_json: Record<string, unknown>;
  status: AgentApprovalStatus;
  idempotency_key: string;
  expires_at: string;
  decided_at: string | null;
  result_task_id: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface AgentRun {
  id: string;
  session_id: string;
  status: AgentRunStatus;
  provider: string;
  model: string;
  error_message: string | null;
  stop_requested: boolean;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  finished_at: string | null;
  messages: AgentMessage[];
  tool_calls: AgentToolCall[];
  approvals: AgentApproval[];
}

export interface AgentSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  latest_run_status: AgentRunStatus | null;
}

export interface AgentSessionDetail extends AgentSession {
  messages: AgentMessage[];
  runs: AgentRun[];
}

export interface AgentProviderStatus {
  provider: string;
  model: string;
  configured: boolean;
  api_key_configured: boolean;
  max_tool_rounds: number;
  approval_ttl_seconds: number;
}
