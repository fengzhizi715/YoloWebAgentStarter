// @vitest-environment jsdom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AgentView } from "./AgentView";
import { linksFromToolCall } from "../agent/entityLinks";
import type { AgentToolCall } from "../types";

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

    expect(container?.textContent).toContain("智能助手");
    expect(container?.textContent).toContain("质量问答");
    expect(container?.textContent).toContain("列出有哪些数据集");
    expect(container?.textContent).toContain("global_summary");
    expect(container?.textContent).toContain("数据集: agent-demo");

    const jump = Array.from(container?.querySelectorAll("button") ?? []).find((button) => button.textContent?.includes("数据集: agent-demo"));
    expect(jump).toBeTruthy();
    await act(async () => { jump?.click(); });
    expect(onOpenDataset).toHaveBeenCalledWith("ds_abc");
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
    expect(agentApi.postMessage).toHaveBeenLastCalledWith("asess_1", "再次发送");
  });

  it("retries the last user question after a failed run", async () => {
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
    vi.mocked(agentApi.postMessage).mockResolvedValue({
      id: "arun_retry2",
      session_id: "asess_1",
      status: "completed",
      provider: "mock",
      model: "mock-model",
      error_message: null,
      stop_requested: false,
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
        <AgentView locale="zh" onOpenDataset={vi.fn()} onOpenTrainingTask={vi.fn()} onOpenModel={vi.fn()} />,
      );
    });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    const retry = Array.from(container?.querySelectorAll("button") ?? []).find((button) => button.textContent === "重试上一问");
    expect(retry).toBeTruthy();
    await act(async () => { retry?.click(); });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(agentApi.postMessage).toHaveBeenCalledWith("asess_1", "列出数据集");
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
