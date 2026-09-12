import type { AppLocale } from "../locale";

const RUN_STATUS: Record<string, { zh: string; en: string }> = {
  pending: { zh: "等待中", en: "Pending" },
  running: { zh: "运行中", en: "Running" },
  awaiting_approval: { zh: "待确认", en: "Awaiting approval" },
  completed: { zh: "已完成", en: "Completed" },
  failed: { zh: "失败", en: "Failed" },
  cancelled: { zh: "已取消", en: "Cancelled" },
};

const TOOL_STATUS: Record<string, { zh: string; en: string }> = {
  pending: { zh: "等待中", en: "Pending" },
  running: { zh: "运行中", en: "Running" },
  completed: { zh: "已完成", en: "Completed" },
  failed: { zh: "失败", en: "Failed" },
  awaiting_approval: { zh: "待确认", en: "Awaiting approval" },
};

const APPROVAL_STATUS: Record<string, { zh: string; en: string }> = {
  pending: { zh: "待确认", en: "Pending" },
  approved: { zh: "已批准", en: "Approved" },
  executing: { zh: "执行中", en: "Executing" },
  rejected: { zh: "已拒绝", en: "Rejected" },
  expired: { zh: "已过期", en: "Expired" },
  executed: { zh: "已执行", en: "Executed" },
};

const TOOL_NAMES: Record<string, { zh: string; en: string }> = {
  global_summary: { zh: "全局概览", en: "Global summary" },
  list_datasets: { zh: "数据集列表", en: "List datasets" },
  get_dataset: { zh: "数据集详情", en: "Get dataset" },
  dataset_quality_report: { zh: "质量报告", en: "Quality report" },
  validate_dataset: { zh: "校验数据集", en: "Validate dataset" },
  list_training_tasks: { zh: "训练任务列表", en: "List training tasks" },
  get_training_task: { zh: "训练任务详情", en: "Get training task" },
  training_summary: { zh: "训练摘要", en: "Training summary" },
  training_logs: { zh: "训练日志", en: "Training logs" },
  list_models: { zh: "模型列表", en: "List models" },
  get_model: { zh: "模型详情", en: "Get model" },
  model_latest_for_dataset: { zh: "最新模型", en: "Latest model" },
  compare_models: { zh: "模型对比", en: "Compare models" },
  list_model_evaluations: { zh: "评估列表", en: "List evaluations" },
  get_model_evaluation: { zh: "评估详情", en: "Get evaluation" },
  create_training_task: { zh: "创建训练任务", en: "Create training task" },
  create_model_evaluation: { zh: "创建模型评估", en: "Create model evaluation" },
  create_auto_annotation_task: { zh: "创建自动标注", en: "Create auto-annotation" },
};

const FACT_KEYS: Record<string, { zh: string; en: string }> = {
  valid: { zh: "校验通过", en: "Valid" },
  image_count: { zh: "图片数", en: "Images" },
  annotated_image_count: { zh: "已标注图片", en: "Annotated images" },
  class_count: { zh: "类别数", en: "Classes" },
  status: { zh: "状态", en: "Status" },
  coverage: { zh: "覆盖率", en: "Coverage" },
};

function pick(map: Record<string, { zh: string; en: string }>, key: string, locale: AppLocale): string {
  const entry = map[key];
  if (!entry) return key;
  return entry[locale];
}

export function runStatusLabel(status: string | null | undefined, locale: AppLocale): string {
  if (!status) return "—";
  return pick(RUN_STATUS, status, locale);
}

export function toolStatusLabel(status: string, locale: AppLocale): string {
  return pick(TOOL_STATUS, status, locale);
}

export function approvalStatusLabel(status: string, locale: AppLocale): string {
  return pick(APPROVAL_STATUS, status, locale);
}

export function toolNameLabel(name: string, locale: AppLocale): string {
  return pick(TOOL_NAMES, name, locale);
}

export function factKeyLabel(key: string, locale: AppLocale): string {
  return pick(FACT_KEYS, key, locale);
}

export function reportTitle(name: string, locale: AppLocale): string {
  const titles: Record<string, { zh: string; en: string }> = {
    global_summary: { zh: "数据集概览", en: "Dataset overview" },
    list_datasets: { zh: "数据集列表", en: "Dataset list" },
    training_list: { zh: "训练概览", en: "Training overview" },
    list_training_tasks: { zh: "训练概览", en: "Training overview" },
    model_list: { zh: "模型概览", en: "Model overview" },
    list_models: { zh: "模型概览", en: "Model overview" },
    evaluation_list: { zh: "评估概览", en: "Evaluation overview" },
    list_model_evaluations: { zh: "评估概览", en: "Evaluation overview" },
    dataset_quality_report: { zh: "质量报告", en: "Quality report" },
    dataset_validate: { zh: "校验结果", en: "Validation result" },
    validate_dataset: { zh: "校验结果", en: "Validation result" },
  };
  const entry = titles[name];
  if (entry) return entry[locale];
  return locale === "zh" ? "工具结果摘要" : "Tool result summary";
}
