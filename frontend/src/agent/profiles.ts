import type { AgentContext, AgentProfileId, AgentSession } from "../types";
import type { AppLocale } from "../locale";

export const PROFILE_IDS: AgentProfileId[] = ["global", "dataset", "training", "model"];
const titles = {
  global: ["综合助手", "General assistant"], dataset: ["数据集助手", "Dataset assistant"],
  training: ["训练助手", "Training assistant"], model: ["模型助手", "Model assistant"],
};
export const profileTitle = (id: AgentProfileId, locale: AppLocale) => titles[id][locale === "zh" ? 0 : 1];
export const inferProfile = (context: AgentContext): AgentProfileId => context.model_id ? "model" : context.training_task_id ? "training" : context.dataset_id ? "dataset" : "global";

export function bindingKey(profile: AgentProfileId, context: AgentContext = {}): string {
  // A task/model ID has a server-validated dataset parent; tolerate the UI not
  // having loaded that parent yet, never use display names as identity.
  const object = context.model_id || context.training_task_id
    ? `model:${context.model_id ?? ""}:training:${context.training_task_id ?? ""}`
    : `dataset:${context.dataset_id ?? ""}`;
  return `${profile}:${object}`;
}

export function sameBinding(leftProfile: AgentProfileId, left: AgentContext = {}, rightProfile: AgentProfileId, right: AgentContext = {}): boolean {
  return bindingKey(leftProfile, left) === bindingKey(rightProfile, right) &&
    (!left.dataset_id || !right.dataset_id || left.dataset_id === right.dataset_id);
}

export function matchingSession(items: AgentSession[], profile: AgentProfileId, context: AgentContext): AgentSession | undefined {
  return items.find((item) => sameBinding(item.profile_id ?? "global", item.context, profile, context));
}
