import type { ModelEvaluationRecord } from "../../types";

export type ErrorSample = NonNullable<ModelEvaluationRecord["result_json"]["error_samples"]>[number];
export type ErrorFilter = "all" | "missed_detection" | "false_positive" | "low_confidence";

export type SampleGroup = {
  imageKey: string;
  imageFile: string;
  items: ErrorSample[];
  types: ErrorSample["type"][];
  classIndexes: number[];
  bestConfidence: number | null;
  primaryType: ErrorSample["type"];
  preview: string;
};

// Community port of upstream evaluationDetailHelpers.ts with pose-only fields removed.
export function summarizeSamples(samples: ErrorSample[]) {
  const imageFiles = new Set<string>();
  const summary = { missed_detection: 0, false_positive: 0, low_confidence: 0 };
  for (const sample of samples) {
    if (sample.type in summary) summary[sample.type as keyof typeof summary] += 1;
    imageFiles.add(sample.image_file ?? "unknown");
  }
  return { ...summary, imageCount: imageFiles.size };
}

export function groupSamples(samples: ErrorSample[]): SampleGroup[] {
  const groups = new Map<string, ErrorSample[]>();
  for (const sample of samples) {
    const key = sample.image_file ?? "unknown";
    groups.set(key, [...(groups.get(key) ?? []), sample]);
  }
  return [...groups.entries()]
    .map(([imageKey, items]) => {
      const types = [...new Set(items.map((item) => item.type))];
      const classIndexes = [...new Set(items.map((item) => item.class_index).filter((value): value is number => value != null))].sort((a, b) => a - b);
      const confidences = items.map((item) => item.confidence).filter((value): value is number => value != null);
      const primaryType = types.includes("missed_detection") ? "missed_detection" : types.includes("false_positive") ? "false_positive" : "low_confidence";
      return { imageKey, imageFile: items[0]?.image_file ?? imageKey, items, types, classIndexes, bestConfidence: confidences.length ? Math.max(...confidences) : null, primaryType, preview: items[0]?.message ?? "" };
    })
    .sort((left, right) => severityScore(right.primaryType) - severityScore(left.primaryType) || right.items.length - left.items.length);
}

export function formatEvalMetric(value?: number) {
  return value == null || Number.isNaN(value) ? "-" : value.toFixed(3);
}

function severityScore(type: string) {
  if (type === "missed_detection") return 3;
  if (type === "false_positive") return 2;
  return 1;
}
