import type { AgentToolCall } from "../types";

export type AgentEntityKind = "dataset" | "training" | "model";

export interface AgentEntityLink {
  kind: AgentEntityKind;
  id: string;
  datasetId?: string;
  label: string;
}

const ID_RE = /\b((?:ds|train|eval)_[a-f0-9]+|model_[a-zA-Z0-9_]+)\b/gi;

function pushUnique(target: AgentEntityLink[], link: AgentEntityLink) {
  if (target.some((item) => item.kind === link.kind && item.id === link.id)) return;
  target.push(link);
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

/** Collect dataset / training / model jump targets from a tool call result. */
export function linksFromToolCall(tool: AgentToolCall): AgentEntityLink[] {
  const links: AgentEntityLink[] = [];
  const result = asRecord(tool.result_json);
  if (!result || result.error) return links;
  const args = asRecord(tool.arguments_json) || {};

  const addDataset = (id?: string, name?: string) => {
    if (!id?.startsWith("ds_")) return;
    pushUnique(links, { kind: "dataset", id, label: name || id });
  };
  const addTraining = (id?: string, datasetId?: string, name?: string) => {
    if (!id?.startsWith("train_")) return;
    pushUnique(links, { kind: "training", id, datasetId, label: name || id });
  };
  const addModel = (id?: string, datasetId?: string, name?: string) => {
    if (!id?.startsWith("model_")) return;
    pushUnique(links, { kind: "model", id, datasetId, label: name || id });
  };

  if (tool.name === "list_datasets" || tool.name === "global_summary") {
    for (const item of (result.items as unknown[]) || []) {
      const row = asRecord(item);
      addDataset(asString(row?.id), asString(row?.name));
    }
  }
  if (
    tool.name === "get_dataset" ||
    tool.name === "dataset_summary" ||
    tool.name === "dataset_quality_report" ||
    tool.name === "validate_dataset" ||
    tool.name === "dataset_validate"
  ) {
    addDataset(asString(result.id) || asString(result.dataset_id) || asString(args.dataset_id), asString(result.name));
  }
  if (tool.name === "list_training_tasks" || tool.name === "training_list") {
    for (const item of (result.items as unknown[]) || []) {
      const row = asRecord(item);
      addTraining(asString(row?.id), asString(row?.dataset_id), asString(row?.name));
      addDataset(asString(row?.dataset_id));
    }
  }
  if (
    tool.name === "get_training_task" ||
    tool.name === "training_status" ||
    tool.name === "training_summary" ||
    tool.name === "training_latest_result" ||
    tool.name === "training_logs"
  ) {
    addTraining(
      asString(result.id) || asString(result.task_id) || asString(args.task_id) || asString(args.training_task_id),
      asString(result.dataset_id),
      asString(result.name),
    );
    addDataset(asString(result.dataset_id));
  }
  if (tool.name === "list_models" || tool.name === "model_list") {
    for (const item of (result.items as unknown[]) || []) {
      const row = asRecord(item);
      addModel(asString(row?.id), asString(row?.dataset_id), asString(row?.name));
      addDataset(asString(row?.dataset_id));
    }
  }
  if (tool.name === "get_model" || tool.name === "model_get") {
    addModel(
      asString(result.id) || asString(args.model_id) || asString(args.model_version_id),
      asString(result.dataset_id),
      asString(result.name),
    );
    addDataset(asString(result.dataset_id));
  }
  if (tool.name === "model_latest_for_dataset") {
    const model = asRecord(result.model);
    addModel(asString(model?.id), asString(model?.dataset_id) || asString(args.dataset_id), asString(model?.name));
    addDataset(asString(model?.dataset_id) || asString(args.dataset_id));
  }
  if (tool.name === "compare_models" || tool.name === "model_compare") {
    const baseline = asRecord(result.baseline);
    const candidate = asRecord(result.candidate);
    addModel(asString(baseline?.id), asString(result.dataset_id), asString(baseline?.name));
    addModel(asString(candidate?.id), asString(result.dataset_id), asString(candidate?.name));
    addDataset(asString(result.dataset_id));
  }
  if (
    tool.name === "list_model_evaluations" ||
    tool.name === "evaluation_list" ||
    tool.name === "get_model_evaluation" ||
    tool.name === "evaluation_summary" ||
    tool.name === "evaluation_error_summary"
  ) {
    addModel(
      asString(result.model_id) || asString(args.model_id) || asString(args.model_version_id),
      asString(result.dataset_id),
    );
    addDataset(asString(result.dataset_id));
    if (tool.name === "list_model_evaluations" || tool.name === "evaluation_list") {
      for (const item of (result.items as unknown[]) || []) {
        const row = asRecord(item);
        addModel(asString(row?.model_id), asString(row?.dataset_id));
        addDataset(asString(row?.dataset_id));
      }
    }
  }
  return links;
}

export function linksFromText(text: string): AgentEntityLink[] {
  const links: AgentEntityLink[] = [];
  for (const match of text.matchAll(ID_RE)) {
    const id = match[1];
    if (id.startsWith("ds_")) pushUnique(links, { kind: "dataset", id, label: id });
    else if (id.startsWith("train_")) pushUnique(links, { kind: "training", id, label: id });
    else if (id.startsWith("model_")) pushUnique(links, { kind: "model", id, label: id });
  }
  return links;
}

export function isActiveRunStatus(status: string | null | undefined): boolean {
  return status === "pending" || status === "running" || status === "awaiting_approval";
}
