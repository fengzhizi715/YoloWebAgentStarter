// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { agentApi } from "../api/agent";
import { api } from "../api/client";
import type { AgentMessage, AgentRun, AgentSessionDetail } from "../types";
import { mergeMessages, mergeRun } from "../agent/history";
import { AgentView } from "./AgentView";

vi.mock("../api/client", () => ({ api: {
  listDatasets: vi.fn().mockResolvedValue([]), listTrainingTasks: vi.fn().mockResolvedValue({ items: [] }),
  listModels: vi.fn().mockResolvedValue({ items: [] }),
} }));
vi.mock("../api/agent", () => ({ agentApi: {
  status: vi.fn(), listSessionPage: vi.fn(), getSession: vi.fn(), getRun: vi.fn(),
  createSession: vi.fn(), postMessage: vi.fn(), approveApproval: vi.fn(), rejectApproval: vi.fn(),
} }));

const stamp = "2026-01-01T00:00:00Z";
const message = (sequence: number, runId = "arun_current"): AgentMessage => ({
  id: `amsg_${sequence}`, session_id: "asess_1", run_id: runId, sequence,
  role: sequence % 2 ? "user" : "assistant", content: `History message ${sequence}`, created_at: stamp, updated_at: stamp,
});
const run: AgentRun = {
  id: "arun_current", session_id: "asess_1", status: "completed", provider: "mock", model: "mock-model",
  profile_id: "global", profile_version: 1, context: {}, read_only: true, stop_requested: false,
  error_message: null, created_at: stamp, updated_at: stamp, started_at: stamp, finished_at: stamp,
  messages: [message(51), message(52)], tool_calls: [], approvals: [],
};
const detail: AgentSessionDetail = {
  id: "asess_1", title: "Current conversation", context: {}, profile_id: "global", profile_version: 1,
  created_at: stamp, updated_at: stamp, message_count: 52, latest_run_status: "completed",
  messages: run.messages, runs: [run], next_before_sequence: 51,
};
let root: Root;
let container: HTMLDivElement;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const props = { locale: "en" as const, onOpenDataset: vi.fn(), onOpenTrainingTask: vi.fn(), onOpenModel: vi.fn() };
const render = async () => { await act(async () => { root.render(<AgentView {...props} />); }); };
const click = async (selector: string) => { await act(async () => { container.querySelector<HTMLButtonElement>(selector)!.click(); }); };
const search = async (value: string) => {
  await act(async () => {
    const input = container.querySelector<HTMLInputElement>('input[type="search"]')!;
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await act(async () => { await vi.advanceTimersByTimeAsync(250); });
};
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.listDatasets).mockResolvedValue([]);
  vi.mocked(api.listTrainingTasks).mockResolvedValue({ items: [] });
  vi.mocked(api.listModels).mockResolvedValue({ items: [], total: 0 });
  vi.mocked(agentApi.status).mockResolvedValue({ provider: "mock", model: "mock-model", configured: true,
    api_key_configured: false, max_tool_rounds: 8, approval_ttl_seconds: 3600 });
  vi.mocked(agentApi.listSessionPage).mockResolvedValue({ items: [detail], next_cursor: null });
  vi.mocked(agentApi.getSession).mockResolvedValue(detail);
  vi.mocked(agentApi.getRun).mockResolvedValue(run);
  container = document.createElement("div"); document.body.append(container); root = createRoot(container);
});
afterEach(() => { act(() => root.unmount()); container.remove(); vi.useRealTimers(); });

describe("paginated Agent history", () => {
  it("prepends older messages, preserves the reading position, and retains pages during run-only polling", async () => {
    vi.useFakeTimers();
    const active = { ...run, status: "running" as const, messages: [message(51)] };
    vi.mocked(agentApi.getSession).mockResolvedValueOnce({ ...detail, messages: active.messages, runs: [active] });
    await render();
    const pane = container.querySelector<HTMLDivElement>(".agent-conversation-body")!;
    Object.defineProperty(pane, "scrollHeight", { configurable: true, get: () => 1000 + container.querySelectorAll(".agent-message").length * 100 });
    Object.defineProperty(pane, "clientHeight", { value: 200 });
    pane.scrollTop = 150;
    await act(async () => { pane.dispatchEvent(new Event("scroll")); });
    vi.mocked(agentApi.getSession).mockResolvedValueOnce({ ...detail, messages: [message(49, "arun_old"), message(50, "arun_old")], runs: [], next_before_sequence: 49 });
    await click(".agent-load-history");
    expect(agentApi.getSession).toHaveBeenLastCalledWith("asess_1", { beforeSequence: 51 });
    expect(pane.scrollTop).toBe(350);
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(agentApi.getRun).toHaveBeenLastCalledWith("arun_current");
    expect(agentApi.getSession).toHaveBeenCalledTimes(2);
    expect(agentApi.listSessionPage).toHaveBeenCalledTimes(1);
    expect([...container.querySelectorAll(".agent-message")].map((node) => node.textContent)).toEqual([
      "YouHistory message 49", "AgentHistory message 50", "YouHistory message 51", "AgentHistory message 52",
    ]);
    expect(pane.scrollTop).toBe(350);
    expect(container.querySelector(".agent-jump-latest")?.textContent).toContain("New reply");
    await click(".agent-jump-latest");
    expect(pane.scrollTop).toBe(pane.scrollHeight);
    expect(container.querySelector(".agent-jump-latest")).toBeNull();
  });

  it("appends and deduplicates conversation pages without loading or selecting them", async () => {
    const second = { ...detail, id: "asess_2", title: "Older conversation" };
    vi.mocked(agentApi.listSessionPage).mockResolvedValueOnce({ items: [detail], next_cursor: "older-cursor" })
      .mockResolvedValueOnce({ items: [detail, second], next_cursor: null });
    await render();
    await click(".agent-load-more");
    expect(agentApi.listSessionPage).toHaveBeenLastCalledWith({ cursor: "older-cursor", query: "" });
    expect(container.querySelectorAll(".agent-session-item")).toHaveLength(2);
    expect(agentApi.getSession).toHaveBeenCalledTimes(1);
    expect(container.querySelector(".agent-load-more")).toBeNull();
    expect(container.querySelector(".agent-title-line h2")?.textContent).toBe(detail.title);
  });

  it("discards outdated search results and keeps the current conversation", async () => {
    vi.useFakeTimers();
    await render();
    const first = deferred<Awaited<ReturnType<typeof agentApi.listSessionPage>>>();
    vi.mocked(agentApi.listSessionPage).mockReturnValueOnce(first.promise)
      .mockResolvedValueOnce({ items: [{ ...detail, id: "asess_new", title: "Latest search result" }], next_cursor: null });
    await search("first"); await search("second");
    await act(async () => { first.resolve({ items: [{ ...detail, title: "Outdated result" }], next_cursor: null }); });
    expect(container.querySelector(".agent-session-list")?.textContent).toContain("Latest search result");
    expect(container.querySelector(".agent-session-list")?.textContent).not.toContain("Outdated result");
    expect(container.querySelector(".agent-title-line h2")?.textContent).toBe(detail.title);
  });

  it("finds a matching bound conversation beyond the first page", async () => {
    vi.mocked(agentApi.listSessionPage).mockResolvedValueOnce({ items: [{ ...detail, id: "other", profile_id: "dataset", context: { dataset_id: "ds_other" } }], next_cursor: "more" })
      .mockResolvedValueOnce({ items: [detail], next_cursor: null });
    await render();
    expect(agentApi.listSessionPage).toHaveBeenLastCalledWith({ profileId: "global", context: {}, limit: 1 });
    expect(agentApi.getSession).toHaveBeenCalledWith("asess_1");
  });

  it("ignores an older-message response after switching conversations", async () => {
    const second = { ...detail, id: "asess_2", title: "Second", messages: [], runs: [], next_before_sequence: null };
    vi.mocked(agentApi.listSessionPage).mockResolvedValue({ items: [detail, second], next_cursor: null });
    await render();
    const older = deferred<AgentSessionDetail>();
    vi.mocked(agentApi.getSession).mockReturnValueOnce(older.promise).mockResolvedValueOnce(second);
    await click(".agent-load-history");
    await act(async () => { container.querySelectorAll<HTMLButtonElement>(".agent-session-item")[1].click(); });
    await act(async () => { older.resolve({ ...detail, messages: [message(1)], runs: [], next_before_sequence: null }); });
    expect(container.querySelector(".agent-title-line h2")?.textContent).toBe("Second");
    expect(container.querySelectorAll(".agent-message")).toHaveLength(0);
  });

  it("does not let a sidebar search choose an older chat during object navigation", async () => {
    vi.useFakeTimers();
    await render();
    const older = { ...detail, id: "asess_old", profile_id: "dataset" as const, context: { dataset_id: "ds_1" }, title: "Search match" };
    const newest = { ...older, id: "asess_new", title: "Newest bound conversation" };
    vi.mocked(agentApi.listSessionPage).mockImplementation(async (options) => ({
      items: options?.profileId === "dataset" ? [newest] : [older], next_cursor: null,
    }));
    await search("Search match");
    vi.mocked(agentApi.getSession).mockResolvedValue(newest);
    await act(async () => { root.render(<AgentView {...props} context={{ dataset: { id: "ds_1", name: "Dataset" } }} />); });
    expect(agentApi.listSessionPage).toHaveBeenLastCalledWith({ profileId: "dataset", context: { dataset_id: "ds_1" }, limit: 1 });
    expect(agentApi.getSession).toHaveBeenLastCalledWith("asess_new");
  });

  it("loads historical failed-run evidence once and never exposes execution controls in that record", async () => {
    vi.mocked(agentApi.getSession).mockResolvedValue({ ...detail, messages: [message(49, "arun_old"), ...detail.messages] });
    vi.mocked(agentApi.getRun).mockResolvedValue({ ...run, id: "arun_old", status: "failed", model: "historical-model",
      approvals: [{ id: "aappr_old", run_id: "arun_old", tool_call_id: null, tool_name: "create_training_task",
        payload_json: {}, status: "executed", idempotency_key: "history-test", expires_at: stamp, decided_at: stamp, result_task_id: "train_old",
        error_message: null, created_at: stamp, updated_at: stamp }] });
    await render();
    expect(agentApi.getRun).not.toHaveBeenCalled();
    await click(".agent-evidence-toggle");
    const evidence = container.querySelector(".agent-turn-evidence")!;
    expect(evidence.textContent).toContain("historical-model");
    expect(evidence.textContent).toContain("train_old");
    expect(evidence.textContent).not.toContain("Confirm");
    await click(".agent-evidence-toggle"); await click(".agent-evidence-toggle");
    expect(agentApi.getRun).toHaveBeenCalledTimes(1);
    expect(agentApi.approveApproval).not.toHaveBeenCalled();
  });

  it("merges snapshots without duplicating messages or regressing a newer run", () => {
    expect(mergeMessages(detail.messages, [...detail.messages.map((row) => ({ ...row })), { ...message(53), role: "tool" }]))
      .toEqual(detail.messages);
    expect(mergeMessages(detail.messages, [{ ...detail.messages[0] }])[0]).toBe(detail.messages[0]);
    expect(mergeRun(detail, { ...run, session_id: "other" })).toBe(detail);
    expect(mergeRun(detail, { ...run, updated_at: "2025-01-01T00:00:00Z" })).toBe(detail);
    expect(mergeRun(detail, { ...run, id: "older", created_at: "2025-01-01T00:00:00Z" })).toBe(detail);
    const merged = mergeRun(detail, { ...run, messages: [message(53)] });
    expect(merged.messages).toHaveLength(3);
    expect(merged.next_before_sequence).toBe(51);
  });
});
