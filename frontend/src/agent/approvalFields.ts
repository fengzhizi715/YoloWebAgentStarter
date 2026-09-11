/** PlanPreview-style editable fields for Community write-tool approvals. */

export type ApprovalFieldType = "text" | "number" | "checkbox" | "select";

export interface ApprovalEditableField {
  key: string;
  labelZh: string;
  labelEn: string;
  type: ApprovalFieldType;
  min?: number;
  max?: number;
  step?: number;
  options?: Array<{ value: string; labelZh: string; labelEn: string }>;
}

export function editableApprovalFields(toolName: string): ApprovalEditableField[] {
  if (toolName === "create_training_task") {
    return [
      { key: "name", labelZh: "任务名", labelEn: "Name", type: "text" },
      { key: "model", labelZh: "模型", labelEn: "Model", type: "text" },
      { key: "epochs", labelZh: "Epochs", labelEn: "Epochs", type: "number", min: 1, step: 1 },
      { key: "img_size", labelZh: "图像尺寸", labelEn: "Image size", type: "number", min: 32, step: 32 },
      { key: "batch_size", labelZh: "Batch", labelEn: "Batch", type: "number", min: 1, step: 1 },
      { key: "val_ratio", labelZh: "验证比例", labelEn: "Val ratio", type: "number", min: 0, max: 0.9, step: 0.01 },
      { key: "device", labelZh: "设备", labelEn: "Device", type: "text" },
    ];
  }
  if (toolName === "create_model_evaluation") {
    return [
      {
        key: "split",
        labelZh: "Split",
        labelEn: "Split",
        type: "select",
        options: [
          { value: "train", labelZh: "train", labelEn: "train" },
          { value: "val", labelZh: "val", labelEn: "val" },
          { value: "test", labelZh: "test", labelEn: "test" },
        ],
      },
      { key: "confidence", labelZh: "置信度", labelEn: "Confidence", type: "number", min: 0, max: 1, step: 0.01 },
      { key: "iou", labelZh: "IoU", labelEn: "IoU", type: "number", min: 0, max: 1, step: 0.01 },
    ];
  }
  if (toolName === "create_auto_annotation_task") {
    return [
      { key: "confidence", labelZh: "置信度", labelEn: "Confidence", type: "number", min: 0, max: 1, step: 0.01 },
      { key: "iou", labelZh: "IoU", labelEn: "IoU", type: "number", min: 0, max: 1, step: 0.01 },
      { key: "skip_annotated_images", labelZh: "跳过已标注图片", labelEn: "Skip annotated images", type: "checkbox" },
      { key: "clean_old_annotations", labelZh: "清理旧自动标注", labelEn: "Clean old auto labels", type: "checkbox" },
    ];
  }
  return [];
}

export function summarizeApprovalParams(
  payload: Record<string, unknown>,
  locale: "zh" | "en",
): Array<{ label: string; value: string }> {
  const labels: Record<string, { zh: string; en: string }> = {
    dataset_id: { zh: "数据集", en: "Dataset" },
    model_id: { zh: "模型 ID", en: "Model ID" },
    name: { zh: "任务名", en: "Name" },
    model: { zh: "模型", en: "Model" },
    epochs: { zh: "Epochs", en: "Epochs" },
    img_size: { zh: "图像尺寸", en: "Image size" },
    batch_size: { zh: "Batch", en: "Batch" },
    device: { zh: "设备", en: "Device" },
    val_ratio: { zh: "验证比例", en: "Val ratio" },
    split: { zh: "Split", en: "Split" },
    confidence: { zh: "置信度", en: "Confidence" },
    iou: { zh: "IoU", en: "IoU" },
    skip_annotated_images: { zh: "跳过已标注", en: "Skip annotated" },
    clean_old_annotations: { zh: "清理旧标注", en: "Clean old" },
  };
  const items: Array<{ label: string; value: string }> = [];
  for (const [key, meta] of Object.entries(labels)) {
    const value = payload[key];
    if (value == null || value === "") continue;
    if (typeof value === "boolean") {
      items.push({ label: meta[locale], value: value ? (locale === "zh" ? "是" : "yes") : locale === "zh" ? "否" : "no" });
      continue;
    }
    items.push({ label: meta[locale], value: String(value) });
  }
  return items;
}

export function expectedApprovalOutputs(toolName: string, locale: "zh" | "en"): string[] {
  if (toolName === "create_training_task") {
    return locale === "zh"
      ? ["训练任务", "训练日志", "受管 PT 权重"]
      : ["Training task", "Training logs", "Managed PT weights"];
  }
  if (toolName === "create_model_evaluation") {
    return locale === "zh"
      ? ["评估任务", "指标摘要", "错误样本"]
      : ["Evaluation task", "Metrics", "Error samples"];
  }
  if (toolName === "create_auto_annotation_task") {
    return locale === "zh"
      ? ["自动标注任务", "待审核标注"]
      : ["Auto-annotation task", "Labels pending review"];
  }
  return [];
}

/** Drop empty numeric fields before approve API calls. */
export function sanitizeApprovalPayload(
  payload: Record<string, unknown>,
  toolName: string,
): Record<string, unknown> {
  const fields = editableApprovalFields(toolName);
  const numericKeys = new Set(fields.filter((field) => field.type === "number").map((field) => field.key));
  const cleaned: Record<string, unknown> = { ...payload };
  for (const key of numericKeys) {
    const value = cleaned[key];
    if (value === "" || value === null || value === undefined) {
      delete cleaned[key];
    }
  }
  return cleaned;
}
