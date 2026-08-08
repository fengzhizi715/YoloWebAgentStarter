export type TaskType = "detect" | "segment";
export type SplitName = "train" | "val" | "test";

export interface Dataset {
  id: string;
  name: string;
  description: string | null;
  task_type: TaskType;
  image_count: number;
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

export interface Annotation {
  id: string;
  image_id: string;
  dataset_id: string;
  class_id: string;
  class_index: number;
  label: string;
  color: string;
  type: "bbox" | "polygon";
  bbox: BBox | null;
  polygon: [number, number][] | null;
  source: "manual" | "imported";
  created_at: string;
  updated_at: string;
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

export interface ImagePage {
  items: ImageItem[];
  total: number;
}
