import { describe, expect, it } from "vitest";
import { bindingKey, inferProfile, matchingSession, sameBinding } from "./profiles";
import type { AgentSession } from "../types";

describe("assistant bindings", () => {
  it("infers the most specific mode", () => {
    expect(inferProfile({})).toBe("global");
    expect(inferProfile({ dataset_id: "ds_a" })).toBe("dataset");
    expect(inferProfile({ dataset_id: "ds_a", training_task_id: "train_a" })).toBe("training");
    expect(inferProfile({ dataset_id: "ds_a", model_id: "model_a" })).toBe("model");
  });
  it("does not reopen an unrelated recent chat", () => {
    const base = { created_at: "", updated_at: "", title: "chat", message_count: 1, latest_run_status: "completed" as const };
    const sessions: AgentSession[] = [
      { ...base, id: "recent", profile_id: "dataset", context: { dataset_id: "ds_b" } },
      { ...base, id: "matched", profile_id: "dataset", context: { dataset_id: "ds_a" } },
    ];
    expect(matchingSession(sessions, "dataset", { dataset_id: "ds_a" })?.id).toBe("matched");
    expect(matchingSession(sessions, "model", { dataset_id: "ds_a" })).toBeUndefined();
  });
  it("matches task context before its derived parent has loaded", () => {
    expect(bindingKey("training", { training_task_id: "train_a" })).toBe(bindingKey("training", { training_task_id: "train_a", dataset_id: "ds_a" }));
  });
  it("compares every bound object and rejects conflicting known parents", () => {
    const context = { model_id: "model_a", training_task_id: "train_a", dataset_id: "ds_a" };
    expect(sameBinding("model", context, "model", { ...context, training_task_id: "train_b" })).toBe(false);
    expect(sameBinding("model", context, "model", { model_id: "model_a" })).toBe(false);
    expect(sameBinding("model", context, "model", { ...context, dataset_id: "ds_b" })).toBe(false);
    expect(sameBinding("model", context, "model", { model_id: "model_a", training_task_id: "train_a" })).toBe(true);
  });
});
