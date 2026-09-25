// @vitest-environment jsdom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AgentView } from "./AgentView";
import { linksFromToolCall } from "../agent/entityLinks";
import type { AgentToolCall } from "../types";

vi.mock("../api/client", () => ({ api: {
  listDatasets: vi.fn().mockResolvedValue([]),
  listTrainingTasks: vi.fn().mockResolvedValue({ items: [] }),
  listModels: vi.fn().mockResolvedValue({ items: [] }),
} }));

vi.mock("../api/agent", () => {
  const session = {
    id: "asess_1",
    title: "质量问答",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    message_count: 2,
    latest_run_status: "completed" as const,
  };
  const run = {
    id: "arun_1",
    session_id: "asess_1",
    status: "completed" as const,
    provider: "mock",
    model: "mock-model",
    error_message: null,
    stop_requested: false,
    read_only: false,
    context: {},
    actual_source: "mock" as const,
    inference_steps: [{ round: 1, source: "mock" as const, provider: "mock", model: "mock-model", duration_ms: 12, outcome: "completed" as const, reason: null }],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:01Z",
    started_at: "2026-01-01T00:00:00Z",
    finished_at: "2026-01-01T00:00:01Z",
    messages: [],
    tool_calls: [
      {
        id: "atool_1",
        run_id: "arun_1",
        name: "global_summary",
        arguments_json: {},
        result_json: {
          items: [{ id: "ds_abc", name: "agent-demo", task_type: "detect", image_count: 1, annotated_image_count: 0, class_count: 1 }],
          total: 1,
          truncated: 0,
        },
        status: "completed" as const,
        error_message: null,
        sequence: 1,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      },
    ],
    approvals: [],
  };
  const detail = {
    ...session,
    messages: [
      {
        id: "amsg_1",
        session_id: "asess_1",
        run_id: "arun_1",
        role: "user" as const,
        content: "列出有哪些数据集",
        sequence: 1,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      },
      {
        id: "amsg_2",
        session_id: "asess_1",
        run_id: "arun_1",
        role: "assistant" as const,
        content: "数据集列表：agent-demo (ds_abc)",
        sequence: 2,
        created_at: "2026-01-01T00:00:01Z",
        updated_at: "2026-01-01T00:00:01Z",
      },
    ],
    runs: [run],
  };
  return {
    agentApi: {
      status: vi.fn().mockResolvedValue({
        provider: "mock",
        model: "mock-model",
        configured: true,
        api_key_configured: false,
        max_tool_rounds: 8,
        approval_ttl_seconds: 3600,
      }),
      listSessions: vi.fn().mockResolvedValue([session]),
      createSession: vi.fn().mockResolvedValue(session),
      getSession: vi.fn().mockResolvedValue(detail),
      updateSession: vi.fn(),
      deleteSession: vi.fn().mockResolvedValue(undefined),
      postMessage: vi.fn().mockResolvedValue(run),
      retryRun: vi.fn().mockResolvedValue(run),
      getRun: vi.fn().mockResolvedValue(run),
      cancelRun: vi.fn().mockResolvedValue({ ...run, status: "cancelled" }),
      approveApproval: vi.fn(),
      rejectApproval: vi.fn(),
    },
  };
});

let root: Root | undefined;
let container: HTMLDivElement | undefined;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
});

describe("AgentView", () => {
  it("disables actions until initial conversation selection has finished", async () => {
    const { agentApi } = await import("../api/agent");
    const original = await agentApi.getSession("asess_1");
    let resolve!: (detail: typeof original) => void;
    vi.mocked(agentApi.getSession).mockReturnValueOnce(new Promise((done) => { resolve = done; }));
    await act(async () => {
      root?.render(<AgentView locale="en" onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />);
    });
    expect(container?.querySelector<HTMLTextAreaElement>("textarea")?.disabled).toBe(true);
    expect([...container!.querySelectorAll<HTMLButtonElement>(".agent-profile-options button")].every((button) => button.disabled)).toBe(true);
    await act(async () => { resolve(original); });
    expect(container?.querySelector<HTMLTextAreaElement>("textarea")?.disabled).toBe(false);
  });

  it("cannot send into the previous chat while another conversation is loading", async () => {
    const { agentApi } = await import("../api/agent");
    const original = await agentApi.getSession("asess_1");
    const second = { ...original, id: "asess_2", title: "Second chat", profile_id: "training" as const, context: {} };
    vi.mocked(agentApi.listSessions).mockResolvedValueOnce([original, second]);
    await act(async () => {
      root?.render(<AgentView locale="en" onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />);
    });
    await act(async () => {
      const textarea = container!.querySelector<HTMLTextAreaElement>("textarea")!;
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set?.call(textarea, "Question for the old chat");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    });
    let resolve!: (detail: typeof second) => void;
    vi.mocked(agentApi.getSession).mockReturnValueOnce(new Promise((done) => { resolve = done; }));
    vi.mocked(agentApi.postMessage).mockClear();
    await act(async () => { container!.querySelectorAll<HTMLButtonElement>(".agent-session-item")[1].click(); });
    expect(container?.querySelector<HTMLTextAreaElement>("textarea")?.disabled).toBe(true);
    expect([...container!.querySelectorAll<HTMLButtonElement>(".agent-profile-options button")].every((button) => button.disabled)).toBe(true);
    await act(async () => { container?.querySelector(".agent-composer")?.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); });
    expect(agentApi.postMessage).not.toHaveBeenCalled();
    await act(async () => { resolve(second); });
    expect(container?.querySelector(".agent-title-line h2")?.textContent).toBe("Second chat");
    expect(container?.querySelector<HTMLTextAreaElement>("textarea")?.disabled).toBe(false);
    expect(container?.querySelector<HTMLTextAreaElement>("textarea")?.value).toBe("");
  });

  it("shows a session-load error and restores interaction with the previous chat", async () => {
    const { agentApi } = await import("../api/agent");
    await act(async () => {
      root?.render(<AgentView locale="en" onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />);
    });
    vi.mocked(agentApi.getSession).mockRejectedValueOnce(new Error("Session unavailable"));
    await act(async () => { container?.querySelector<HTMLButtonElement>(".agent-session-item")?.click(); });
    expect(container?.textContent).toContain("Session unavailable");
    expect(container?.querySelector(".agent-title-line h2")?.textContent).toBe("质量问答");
    expect(container?.querySelector<HTMLTextAreaElement>("textarea")?.disabled).toBe(false);
  });

  it("opens the matching dataset conversation instead of the most recent chat", async () => {
    const { agentApi } = await import("../api/agent");
    const original = await agentApi.getSession("asess_1");
    const bound = { ...original, id: "asess_bound", title: "Bound conversation", profile_id: "dataset" as const, context: { dataset_id: "ds_bound" } };
    vi.mocked(agentApi.listSessions).mockResolvedValueOnce([original, bound]);
    vi.mocked(agentApi.getSession).mockResolvedValueOnce(bound);
    await act(async () => {
      root?.render(<AgentView locale="en" context={{ dataset: { id: "ds_bound", name: "Bound" } }} onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />);
    });
    expect(agentApi.getSession).toHaveBeenLastCalledWith("asess_bound");
    expect(container?.querySelector(".agent-title-line h2")?.textContent).toBe("Bound conversation");
  });

  it("requires a binding choice before sending under a changed profile", async () => {
    const { agentApi } = await import("../api/agent");
    vi.mocked(agentApi.postMessage).mockClear();
    await act(async () => {
      root?.render(<AgentView locale="zh" onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />);
    });
    const modelMode = [...container!.querySelectorAll<HTMLButtonElement>(".agent-profile-options button")].find((button) => button.textContent === "模型助手")!;
    await act(async () => { modelMode.click(); });
    expect(container?.querySelector(".agent-binding-warning")?.textContent).toContain("对象或助手模式已改变");
    expect(container?.querySelector<HTMLTextAreaElement>("textarea")?.disabled).toBe(true);
    const proceed = [...container!.querySelectorAll<HTMLButtonElement>("button")].find((button) => button.textContent === "继续当前会话")!;
    await act(async () => { proceed.click(); });
    expect(container?.querySelector<HTMLTextAreaElement>("textarea")?.disabled).toBe(false);
    const quick = [...container!.querySelectorAll<HTMLButtonElement>("button")].find((button) => button.textContent === "选择模型")!;
    await act(async () => { quick.click(); });
    expect(agentApi.postMessage).toHaveBeenCalledWith("asess_1", expect.any(String), expect.objectContaining({ profileId: "model", readOnly: true, allowContextChange: true }));
  });

  it("can create a separate bound chat without continuing or sending to the old one", async () => {
    const { agentApi } = await import("../api/agent");
    vi.mocked(agentApi.postMessage).mockClear();
    await act(async () => {
      root?.render(<AgentView locale="zh" onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />);
    });
    const trainingMode = [...container!.querySelectorAll<HTMLButtonElement>(".agent-profile-options button")].find((button) => button.textContent === "训练助手")!;
    await act(async () => { trainingMode.click(); });
    await act(async () => { container?.querySelector<HTMLButtonElement>(".agent-binding-warning .primary")?.click(); });
    expect(agentApi.createSession).toHaveBeenLastCalledWith("新会话", expect.objectContaining({ profileId: "training" }));
    expect(agentApi.postMessage).not.toHaveBeenCalled();
  });

  it.each(["zh", "en"] as const)("shows mixed inference sources and the fallback reason in %s", async (locale) => {
    const { agentApi } = await import("../api/agent");
    const original = await agentApi.getSession("asess_1");
    vi.mocked(agentApi.getSession).mockResolvedValueOnce({
      ...original,
      runs: [{ ...original.runs[0], actual_source: "mixed", inference_steps: [
        { round: 1, source: "fallback", provider: "openai-compatible", model: "test-model", duration_ms: 120, outcome: "completed", reason: "no_tool_calls" },
        { round: 2, source: "llm", provider: "openai-compatible", model: "test-model", duration_ms: 230, outcome: "completed", reason: null },
      ] }],
    });
    await act(async () => {
      root?.render(<AgentView locale={locale} onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />);
    });
    expect(container?.textContent).toContain(locale === "zh" ? "混合来源" : "Mixed sources");
    expect(container?.textContent).toContain("350 ms");
    expect(container?.textContent).toContain(locale === "zh" ? "模型未选择工具，改用规则规划" : "Model selected no tools");
    expect(container?.querySelector('[role="note"]')?.textContent).toContain(locale === "zh" ? "并非全部由 LLM 完成" : "not completed entirely by the LLM");
  });

  it("does not infer LLM provenance for old runs without inference records", async () => {
    const { agentApi } = await import("../api/agent");
    const original = await agentApi.getSession("asess_1");
    vi.mocked(agentApi.getSession).mockResolvedValueOnce({
      ...original, runs: [{ ...original.runs[0], provider: "openai-compatible", actual_source: "unknown", inference_steps: [] }],
    });
    await act(async () => {
      root?.render(<AgentView locale="zh" onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />);
    });
    expect(container?.textContent).toContain("实际来源: 未记录");
    expect(container?.textContent).not.toContain("推理总耗时");
  });

  it("renders sessions, messages, tool calls and entity jump links", async () => {
    const onOpenDataset = vi.fn();
    await act(async () => {
      root?.render(
        <AgentView
          locale="zh"
          onOpenDataset={onOpenDataset}
          onOpenTrainingTask={vi.fn()}
          onOpenModel={vi.fn()}
        />,
      );
    });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });

    expect(container?.textContent).toContain("智能体");
    expect(container?.textContent).toContain("质量问答");
    expect(container?.textContent).toContain("列出有哪些数据集");
    expect(container?.textContent).toContain("全局概览");
    expect(container?.textContent).toContain("数据集: agent-demo");

    const jump = Array.from(container?.querySelectorAll("button") ?? []).find((button) => button.textContent?.includes("数据集: agent-demo"));
    expect(jump).toBeTruthy();
    await act(async () => { jump?.click(); });
    expect(onOpenDataset).toHaveBeenCalledWith("ds_abc");
  });

  it("uses a contextual quick prompt for the active dataset", async () => {
    const { agentApi } = await import("../api/agent");
    await act(async () => {
      root?.render(
        <AgentView
          locale="zh"
          context={{ dataset: { id: "ds_abc", name: "agent-demo" } }}
          onOpenDataset={vi.fn()}
          onOpenTrainingTask={vi.fn()}
          onOpenModel={vi.fn()}
        />,
      );
    });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });

    expect(container?.textContent).toContain("当前数据集上下文");
    const quickPrompt = Array.from(container?.querySelectorAll("button") ?? []).find((button) => button.textContent === "数据集质量报告");
    expect(quickPrompt).toBeTruthy();
    await act(async () => { quickPrompt?.click(); });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(agentApi.postMessage).toHaveBeenCalledWith("asess_1", expect.stringContaining("ds_abc"), { readOnly: true, context: { dataset_id: "ds_abc" }, profileId: "dataset", allowContextChange: false });
  });

  it("offers read-only prompts for the current training task", async () => {
    const { agentApi } = await import("../api/agent");
    await act(async () => {
      root?.render(<AgentView locale="zh" context={{ trainingTask: { id: "train_abc", name: "试验训练" } }}
        onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />);
    });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(container?.textContent).toContain("当前训练任务");
    const prompt = Array.from(container?.querySelectorAll("button") ?? []).find((button) => button.textContent === "训练状态");
    await act(async () => { prompt?.click(); });
    expect(agentApi.postMessage).toHaveBeenCalledWith("asess_1", expect.stringContaining("train_abc"), { readOnly: true, context: { training_task_id: "train_abc" }, profileId: "training", allowContextChange: false });
  });

  it("shows a quality summary without a raw tool JSON chat bubble", async () => {
    const { agentApi } = await import("../api/agent");
    const baseline = await agentApi.getSession("asess_1");
    const qualityTool = {
      ...baseline.runs[0].tool_calls[0],
      id: "atool_quality",
      name: "dataset_quality_report",
      result_json: { summary: { coverage: 0.5, annotation_count: 4 }, issues: [] },
    };
    vi.mocked(agentApi.getSession).mockResolvedValueOnce({
      ...baseline,
      messages: [
        ...baseline.messages,
        { ...baseline.messages[0], id: "amsg_tool", role: "tool", content: "RAW_PRIVATE_TOOL_JSON", sequence: 3 },
      ],
      runs: [{ ...baseline.runs[0], tool_calls: [qualityTool] }],
    });
    await act(async () => {
      root?.render(<AgentView locale="zh" onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />);
    });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(container?.textContent).toContain("50.0%");
    expect(container?.textContent).toContain("标注数");
    expect(container?.textContent).not.toContain("RAW_PRIVATE_TOOL_JSON");
  });

  it("sends structured context with a manually typed question", async () => {
    const { agentApi } = await import("../api/agent");
    await act(async () => {
      root?.render(<AgentView locale="zh"
        context={{ dataset: { id: "ds_abc", name: "Data" }, trainingTask: { id: "train_abc", name: "Task" } }}
        onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />);
    });
    const input = container?.querySelector("textarea") as HTMLTextAreaElement;
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set?.call(input, "这个训练为什么失败？");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => {
      container?.querySelector("form")?.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    });
    expect(agentApi.postMessage).toHaveBeenLastCalledWith("asess_1", "这个训练为什么失败？", {
      readOnly: false, context: { dataset_id: "ds_abc", training_task_id: "train_abc" },
      profileId: "training", allowContextChange: false,
    });
  });

  it("shows progress and allows cancellation after asynchronous submission", async () => {
    const { agentApi } = await import("../api/agent");
    const baseline = await agentApi.getSession("asess_1");
    const user = { ...baseline.messages[0], id: "amsg_async", run_id: "arun_async", content: "查看数据集", sequence: 3 };
    const pending = { ...baseline.runs[0], id: "arun_async", status: "pending" as const, messages: [user], tool_calls: [] };
    const pendingDetail = { ...baseline, messages: [...baseline.messages, user], runs: [pending, ...baseline.runs] };
    vi.mocked(agentApi.getSession).mockResolvedValueOnce(baseline).mockResolvedValueOnce(pendingDetail);
    vi.mocked(agentApi.postMessage).mockResolvedValueOnce(pending);
    vi.useFakeTimers();
    try {
      await act(async () => {
        root?.render(<AgentView locale="zh" onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />);
      });
      const prompt = Array.from(container?.querySelectorAll("button") ?? []).find((button) => button.textContent === "列出数据集");
      await act(async () => { prompt?.click(); });
      expect(container?.querySelector('[role="status"]')?.textContent).toContain("等待运行");
      let cancel = Array.from(container?.querySelectorAll("button") ?? []).find((button) => button.textContent === "取消运行");
      expect(cancel?.disabled).toBe(false);
      const running = { ...pending, status: "running" as const,
        tool_calls: [{ ...baseline.runs[0].tool_calls[0], name: "dataset_summary", status: "running" as const }] };
      vi.mocked(agentApi.getSession).mockResolvedValueOnce({ ...pendingDetail, runs: [running] });
      await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
      expect(container?.querySelector('[role="status"]')?.textContent).toContain("正在调用：数据集概览");
      const cancelled = { ...running, status: "cancelled" as const, stop_requested: true };
      vi.mocked(agentApi.cancelRun).mockResolvedValueOnce(cancelled);
      vi.mocked(agentApi.getSession).mockResolvedValueOnce({ ...pendingDetail, runs: [cancelled] });
      cancel = Array.from(container?.querySelectorAll("button") ?? []).find((button) => button.textContent === "取消运行");
      await act(async () => { cancel?.click(); });
      expect(agentApi.cancelRun).toHaveBeenLastCalledWith("arun_async");
      expect(container?.querySelector('[role="status"]')).toBeNull();
      expect(container?.textContent).toContain("已取消");
    } finally {
      vi.useRealTimers();
    }
  });

  it("renames the current session through the Agent API", async () => {
    const { agentApi } = await import("../api/agent");
    await act(async () => {
      root?.render(
        <AgentView locale="zh" onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />,
      );
    });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    const rename = Array.from(container?.querySelectorAll("button") ?? []).find((button) => button.textContent === "重命名");
    expect(rename).toBeTruthy();
    await act(async () => { rename?.click(); });
    const input = container?.querySelector(".agent-title-editor input") as HTMLInputElement | null;
    expect(input).toBeTruthy();
    if (input) {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
      setter?.call(input, "重命名后的会话");
      await act(async () => { input.dispatchEvent(new Event("input", { bubbles: true })); });
    }
    const save = Array.from(container?.querySelectorAll("button") ?? []).find((button) => button.textContent === "保存名称");
    await act(async () => { save?.click(); });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(agentApi.updateSession).toHaveBeenCalledWith("asess_1", "重命名后的会话");
  });

  it("confirms pending approvals via approve API", async () => {
    const { agentApi } = await import("../api/agent");
    const awaitingRun = {
      id: "arun_1",
      session_id: "asess_1",
      status: "awaiting_approval" as const,
      provider: "mock",
      model: "mock-model",
      error_message: null,
      stop_requested: false,
      read_only: false,
      context: {},
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:01Z",
      started_at: "2026-01-01T00:00:00Z",
      finished_at: null,
      messages: [],
      tool_calls: [],
      approvals: [
        {
          id: "aappr_1",
          run_id: "arun_1",
          tool_call_id: null,
          tool_name: "create_training_task",
          payload_json: { dataset_id: "ds_abc", preview: { dataset_id: "ds_abc", name: "agent-training" } },
          status: "pending" as const,
          idempotency_key: "k1",
          expires_at: "2026-01-02T00:00:00Z",
          decided_at: null,
          result_task_id: null,
          error_message: null,
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
        },
      ],
    };
    vi.mocked(agentApi.getSession).mockResolvedValue({
      id: "asess_1",
      title: "写操作",
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
      message_count: 1,
      latest_run_status: "awaiting_approval",
      messages: [
        {
          id: "amsg_1",
          session_id: "asess_1",
          run_id: "arun_1",
          role: "assistant",
          content: "待确认创建训练",
          sequence: 1,
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
        },
      ],
      runs: [awaitingRun],
    });
    vi.mocked(agentApi.approveApproval).mockResolvedValue({
      approval: { ...awaitingRun.approvals[0], status: "executed", result_task_id: "train_1" },
      run: { ...awaitingRun, status: "completed", finished_at: "2026-01-01T00:00:02Z" },
    });
    vi.mocked(agentApi.getSession).mockResolvedValueOnce({
      id: "asess_1",
      title: "写操作",
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
      message_count: 1,
      latest_run_status: "awaiting_approval",
      messages: [
        {
          id: "amsg_1",
          session_id: "asess_1",
          run_id: "arun_1",
          role: "assistant",
          content: "待确认创建训练",
          sequence: 1,
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
        },
      ],
      runs: [awaitingRun],
    });

    await act(async () => {
      root?.render(
        <AgentView
          locale="zh"
          onOpenDataset={vi.fn()}
          onOpenTrainingTask={vi.fn()}
          onOpenModel={vi.fn()}
        />,
      );
    });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(container?.querySelector('[data-testid="agent-approval-slot"]')).toBeTruthy();
    expect(container?.textContent).toContain("只读校验会自动执行");

    const approve = Array.from(container?.querySelectorAll("button") ?? []).find((button) => button.textContent === "确认执行");
    expect(approve).toBeTruthy();
    vi.mocked(agentApi.getSession).mockResolvedValue({
      id: "asess_1",
      title: "写操作",
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:02Z",
      message_count: 2,
      latest_run_status: "completed",
      messages: [],
      runs: [{ ...awaitingRun, status: "completed", approvals: [{ ...awaitingRun.approvals[0], status: "executed", result_task_id: "train_1" }] }],
    });
    await act(async () => { approve?.click(); });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(agentApi.approveApproval).toHaveBeenCalledWith(
      "aappr_1",
      expect.objectContaining({ dataset_id: "ds_abc" }),
    );
  });

  it("rejects a pending approval", async () => {
    const { agentApi } = await import("../api/agent");
    const awaitingRun = {
      id: "arun_wait",
      session_id: "asess_1",
      status: "awaiting_approval" as const,
      provider: "mock",
      model: "mock-model",
      error_message: null,
      stop_requested: false,
      read_only: false,
      context: {},
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:01Z",
      started_at: "2026-01-01T00:00:00Z",
      finished_at: null,
      messages: [],
      tool_calls: [],
      approvals: [
        {
          id: "aappr_reject",
          run_id: "arun_wait",
          tool_call_id: null,
          tool_name: "create_training_task",
          payload_json: { dataset_id: "ds_abc" },
          status: "pending" as const,
          idempotency_key: "k2",
          expires_at: "2026-01-02T00:00:00Z",
          decided_at: null,
          result_task_id: null,
          error_message: null,
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
        },
      ],
    };
    vi.mocked(agentApi.getSession).mockResolvedValue({
      id: "asess_1",
      title: "交互",
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
      message_count: 1,
      latest_run_status: "awaiting_approval",
      messages: [
        {
          id: "amsg_w",
          session_id: "asess_1",
          run_id: "arun_wait",
          role: "assistant",
          content: "待确认",
          sequence: 1,
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
        },
      ],
      runs: [awaitingRun],
    });
    vi.mocked(agentApi.rejectApproval).mockResolvedValue({
      approval: { ...awaitingRun.approvals[0], status: "rejected" },
      run: { ...awaitingRun, status: "cancelled" },
    });

    await act(async () => {
      root?.render(
        <AgentView locale="zh" onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />,
      );
    });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    const reject = Array.from(container?.querySelectorAll("button") ?? []).find((button) => button.textContent === "拒绝");
    expect(reject).toBeTruthy();
    await act(async () => { reject?.click(); });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(agentApi.rejectApproval).toHaveBeenCalledWith("aappr_reject");
  });

  it("cancels an active run", async () => {
    const { agentApi } = await import("../api/agent");
    const running = {
      id: "arun_run",
      session_id: "asess_1",
      status: "running" as const,
      provider: "mock",
      model: "mock-model",
      error_message: null,
      stop_requested: false,
      read_only: false,
      context: {},
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:01Z",
      started_at: "2026-01-01T00:00:00Z",
      finished_at: null,
      messages: [],
      tool_calls: [],
      approvals: [],
    };
    vi.mocked(agentApi.listSessions).mockResolvedValue([
      {
        id: "asess_1",
        title: "交互",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
        message_count: 1,
        latest_run_status: "running",
      },
    ]);
    vi.mocked(agentApi.getSession).mockResolvedValue({
      id: "asess_1",
      title: "交互",
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
      message_count: 1,
      latest_run_status: "running",
      messages: [],
      runs: [running],
    });
    vi.mocked(agentApi.cancelRun).mockResolvedValue({ ...running, status: "cancelled" });

    await act(async () => {
      root?.render(
        <AgentView locale="zh" onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />,
      );
    });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve(); });
    const cancel = Array.from(container?.querySelectorAll("button") ?? []).find((button) => button.textContent === "取消运行");
    expect(cancel).toBeTruthy();
    await act(async () => { cancel?.click(); });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(agentApi.cancelRun).toHaveBeenCalledWith("arun_run");
  });

  it("shows send failures and allows retrying another message", async () => {
    const { agentApi } = await import("../api/agent");
    vi.mocked(agentApi.getSession).mockResolvedValue({
      id: "asess_1",
      title: "交互",
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
      message_count: 0,
      latest_run_status: "completed",
      messages: [],
      runs: [],
    });
    vi.mocked(agentApi.postMessage).mockRejectedValueOnce(new Error("provider unavailable"));

    await act(async () => {
      root?.render(
        <AgentView locale="zh" onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />,
      );
    });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });

    const textarea = container?.querySelector("textarea");
    const send = Array.from(container?.querySelectorAll("button") ?? []).find((button) => button.textContent === "发送");
    expect(textarea).toBeTruthy();
    expect(send).toBeTruthy();
    await act(async () => {
      if (!textarea) return;
      const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value")?.set;
      setter?.call(textarea, "第一次");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => { send?.click(); });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(container?.textContent).toContain("provider unavailable");

    vi.mocked(agentApi.postMessage).mockResolvedValueOnce({
      id: "arun_retry",
      session_id: "asess_1",
      status: "completed",
      provider: "mock",
      model: "mock-model",
      error_message: null,
      stop_requested: false,
      read_only: false,
      context: {},
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:01Z",
      started_at: "2026-01-01T00:00:00Z",
      finished_at: "2026-01-01T00:00:01Z",
      messages: [],
      tool_calls: [],
      approvals: [],
    });
    await act(async () => {
      if (!textarea) return;
      const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value")?.set;
      setter?.call(textarea, "再次发送");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => { send?.click(); });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(agentApi.postMessage).toHaveBeenLastCalledWith("asess_1", "再次发送", { readOnly: false, context: {}, profileId: "global", allowContextChange: false });
  });

  it("retries the original run after the page context and language change", async () => {
    const { agentApi } = await import("../api/agent");
    vi.mocked(agentApi.getSession).mockResolvedValue({
      id: "asess_1",
      title: "失败重试",
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
      message_count: 2,
      latest_run_status: "failed",
      messages: [
        {
          id: "amsg_u",
          session_id: "asess_1",
          run_id: "arun_fail",
          role: "user",
          content: "列出数据集",
          sequence: 1,
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
        },
        {
          id: "amsg_a",
          session_id: "asess_1",
          run_id: "arun_fail",
          role: "assistant",
          content: "失败了",
          sequence: 2,
          created_at: "2026-01-01T00:00:01Z",
          updated_at: "2026-01-01T00:00:01Z",
        },
      ],
      runs: [
        {
          id: "arun_fail",
          session_id: "asess_1",
          status: "failed",
          provider: "mock",
          model: "mock-model",
          error_message: "boom",
          stop_requested: false,
          read_only: true,
          context: { dataset_id: "ds_original" },
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:01Z",
          started_at: "2026-01-01T00:00:00Z",
          finished_at: "2026-01-01T00:00:01Z",
          messages: [],
          tool_calls: [],
          approvals: [],
        },
      ],
    });
    vi.mocked(agentApi.retryRun).mockResolvedValue({
      id: "arun_retry2",
      session_id: "asess_1",
      status: "completed",
      provider: "mock",
      model: "mock-model",
      error_message: null,
      stop_requested: false,
      read_only: true,
      context: { dataset_id: "ds_original" },
      created_at: "2026-01-01T00:00:02Z",
      updated_at: "2026-01-01T00:00:03Z",
      started_at: "2026-01-01T00:00:02Z",
      finished_at: "2026-01-01T00:00:03Z",
      messages: [],
      tool_calls: [],
      approvals: [],
    });

    await act(async () => {
      root?.render(
        <AgentView locale="en" context={{ dataset: { id: "ds_other", name: "Other dataset" } }} onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />,
      );
    });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    // A different page context must not auto-open this historical conversation.
    await act(async () => { (container?.querySelector(".agent-session-item") as HTMLButtonElement)?.click(); });
    const retry = Array.from(container?.querySelectorAll("button") ?? []).find((button) => button.textContent === "Retry last question");
    expect(retry).toBeTruthy();
    await act(async () => { retry?.click(); });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(agentApi.retryRun).toHaveBeenCalledWith("arun_fail");
  });
});

describe("linksFromToolCall", () => {
  it("extracts dataset links from global_summary and legacy list_datasets", () => {
    const base = {
      id: "atool_1",
      run_id: "arun_1",
      arguments_json: {},
      result_json: { items: [{ id: "ds_abc", name: "demo" }] },
      status: "completed" as const,
      error_message: null,
      sequence: 1,
      created_at: "",
      updated_at: "",
    };
    expect(linksFromToolCall({ ...base, name: "global_summary" })).toEqual([
      { kind: "dataset", id: "ds_abc", label: "demo" },
    ]);
    expect(linksFromToolCall({ ...base, name: "list_datasets" })).toEqual([
      { kind: "dataset", id: "ds_abc", label: "demo" },
    ]);
  });
});
