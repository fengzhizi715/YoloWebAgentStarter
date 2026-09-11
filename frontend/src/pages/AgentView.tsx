import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { agentApi } from "../api/agent";
import {
  editableApprovalFields,
  expectedApprovalOutputs,
  sanitizeApprovalPayload,
  summarizeApprovalParams,
} from "../agent/approvalFields";
import { isActiveRunStatus, linksFromText, linksFromToolCall, type AgentEntityLink } from "../agent/entityLinks";
import type { AppLocale } from "../locale";
import type { AgentApproval, AgentMessage, AgentProviderStatus, AgentRun, AgentSession, AgentSessionDetail, AgentToolCall } from "../types";

const copy = {
  zh: {
    eyebrow: "AGENT",
    title: "智能助手",
    subtitle: "可查询数据集、训练与模型；写操作需人工确认后才会创建任务。",
    newChat: "新会话",
    sessions: "会话",
    emptySessions: "还没有会话。",
    delete: "删除",
    placeholder: "询问质量报告，或说明要创建训练/评估/自动标注…",
    send: "发送",
    cancel: "取消运行",
    refreshing: "同步中…",
    provider: "Provider",
    tools: "工具调用",
    sources: "可跳转",
    noTools: "本轮未调用工具。",
    approvals: "待确认操作",
    approvalHint: "可先调整参数再确认。只读校验会自动执行；写操作确认后才会创建任务，本轮结束，不会自动串联下一步。",
    approve: "确认执行",
    reject: "拒绝",
    adjustBeforeRun: "执行前可调整",
    keyParams: "关键参数",
    expectedOutputs: "预期产物",
    viewDetails: "查看原始 JSON",
    retryLast: "重试上一问",
    providerUnsupported: "当前 Provider 未就绪，请到「设置 → Agent LLM」配置或关闭 LLM 使用本地规则。",
    status: "状态",
    emptyChat: "选择或新建会话后开始提问。",
    openDataset: "数据集",
    openTraining: "训练任务",
    openModel: "模型",
    user: "你",
    assistant: "助手",
    tool: "工具",
    system: "系统",
  },
  en: {
    eyebrow: "AGENT",
    title: "Assistant",
    subtitle: "Query datasets, training, and models. Write actions run only after you confirm.",
    newChat: "New chat",
    sessions: "Sessions",
    emptySessions: "No sessions yet.",
    delete: "Delete",
    placeholder: "Ask about quality, or request training / evaluation / auto-annotation…",
    send: "Send",
    cancel: "Cancel run",
    refreshing: "Syncing…",
    provider: "Provider",
    tools: "Tool calls",
    sources: "Open",
    noTools: "No tools in this turn.",
    approvals: "Pending approvals",
    approvalHint: "Adjust parameters if needed. Read-only validation runs automatically; confirming a write creates the task and ends this turn.",
    approve: "Confirm",
    reject: "Reject",
    adjustBeforeRun: "Adjust before run",
    keyParams: "Key params",
    expectedOutputs: "Expected outputs",
    viewDetails: "View raw JSON",
    retryLast: "Retry last question",
    providerUnsupported: "Provider is not ready. Configure it under Settings → Agent LLM, or disable LLM for local rules.",
    status: "Status",
    emptyChat: "Select or create a session to start.",
    openDataset: "Dataset",
    openTraining: "Training task",
    openModel: "Model",
    user: "You",
    assistant: "Assistant",
    tool: "Tool",
    system: "System",
  },
} as const;

type AgentCopy = (typeof copy)[AppLocale];

export interface AgentNavigationHandlers {
  onOpenDataset: (datasetId: string) => void;
  onOpenTrainingTask: (datasetId: string | undefined, taskId: string) => void;
  onOpenModel: (datasetId: string | undefined, modelId: string) => void;
}

interface Props extends AgentNavigationHandlers {
  locale: AppLocale;
}

export function AgentView({ locale, onOpenDataset, onOpenTrainingTask, onOpenModel }: Props) {
  const text = copy[locale];
  const [sessions, setSessions] = useState<AgentSession[]>([]);
  const [sessionId, setSessionId] = useState<string>();
  const [detail, setDetail] = useState<AgentSessionDetail>();
  const [status, setStatus] = useState<AgentProviderStatus>();
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [expandedTools, setExpandedTools] = useState<Record<string, boolean>>({});
  const bottomRef = useRef<HTMLDivElement>(null);

  const activeRun = useMemo(() => {
    const runs = detail?.runs || [];
    return runs.find((run) => isActiveRunStatus(run.status)) || runs[0];
  }, [detail]);

  const refreshSessions = async () => {
    const items = await agentApi.listSessions();
    setSessions(items);
    return items;
  };

  const loadSession = async (id: string) => {
    const next = await agentApi.getSession(id);
    setDetail(next);
    setSessionId(next.id);
    return next;
  };

  useEffect(() => {
    void (async () => {
      try {
        const [provider, items] = await Promise.all([agentApi.status(), refreshSessions()]);
        setStatus(provider);
        if (items[0]) await loadSession(items[0].id);
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Request failed");
      }
    })();
  }, []);

  useEffect(() => {
    if (!activeRun || !isActiveRunStatus(activeRun.status)) return;
    const timer = window.setInterval(() => {
      void agentApi.getRun(activeRun.id).then(async () => {
        if (sessionId) await loadSession(sessionId);
        await refreshSessions();
      }).catch(() => undefined);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [activeRun?.id, activeRun?.status, sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [detail?.messages.length, activeRun?.status]);

  const createSession = async () => {
    setBusy(true); setError("");
    try {
      const created = await agentApi.createSession();
      await refreshSessions();
      await loadSession(created.id);
      setDraft("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Request failed");
    } finally {
      setBusy(false);
    }
  };

  const removeSession = async (id: string) => {
    setBusy(true); setError("");
    try {
      await agentApi.deleteSession(id);
      const items = await refreshSessions();
      if (sessionId === id) {
        setDetail(undefined);
        setSessionId(undefined);
        if (items[0]) await loadSession(items[0].id);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Request failed");
    } finally {
      setBusy(false);
    }
  };

  const sendMessage = async (event: FormEvent) => {
    event.preventDefault();
    const content = draft.trim();
    if (!content || busy) return;
    setBusy(true); setError("");
    try {
      let targetId = sessionId;
      if (!targetId) {
        const created = await agentApi.createSession(content.slice(0, 80));
        targetId = created.id;
        await refreshSessions();
      }
      const run = await agentApi.postMessage(targetId, content);
      setDraft("");
      await loadSession(targetId);
      await refreshSessions();
      if (isActiveRunStatus(run.status) && sessionId) await loadSession(targetId);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Request failed");
    } finally {
      setBusy(false);
    }
  };

  const cancelRun = async () => {
    if (!activeRun || !isActiveRunStatus(activeRun.status)) return;
    setBusy(true); setError("");
    try {
      await agentApi.cancelRun(activeRun.id);
      if (sessionId) await loadSession(sessionId);
      await refreshSessions();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Request failed");
    } finally {
      setBusy(false);
    }
  };

  const decideApproval = async (
    approvalId: string,
    decision: "approve" | "reject",
    payload?: Record<string, unknown>,
  ) => {
    if (!sessionId) return;
    setBusy(true);
    setError("");
    try {
      if (decision === "approve") await agentApi.approveApproval(approvalId, payload);
      else await agentApi.rejectApproval(approvalId);
      await loadSession(sessionId);
      await refreshSessions();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Request failed");
    } finally {
      setBusy(false);
    }
  };

  const openLink = (link: AgentEntityLink) => {
    if (link.kind === "dataset") onOpenDataset(link.id);
    else if (link.kind === "training") onOpenTrainingTask(link.datasetId, link.id);
    else onOpenModel(link.datasetId, link.id);
  };

  const messages = detail?.messages || [];
  const focusRun =
    (activeRun?.status === "awaiting_approval" ? activeRun : undefined) ||
    detail?.runs.find((run) => run.id === messages[messages.length - 1]?.run_id) ||
    activeRun;
  const pendingApprovals = (focusRun?.approvals || []).filter(
    (item) => item.status === "pending" || item.status === "approved",
  );
  const lastUserMessage = [...messages].reverse().find((item) => item.role === "user");
  const canRetryLast =
    Boolean(lastUserMessage?.content) &&
    !busy &&
    Boolean(activeRun) &&
    (activeRun?.status === "failed" || activeRun?.status === "cancelled") &&
    pendingApprovals.length === 0;

  const retryLast = async () => {
    if (!lastUserMessage?.content || !sessionId || busy) return;
    setDraft(lastUserMessage.content);
    setBusy(true);
    setError("");
    try {
      await agentApi.postMessage(sessionId, lastUserMessage.content);
      await loadSession(sessionId);
      await refreshSessions();
      setDraft("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Request failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="agent-layout">
      <header className="agent-head">
        <div>
          <span className="eyebrow">{text.eyebrow}</span>
          <h1>{text.title}</h1>
          <p>{text.subtitle}</p>
        </div>
        <div className="header-actions">
          {status ? (
            <span className="agent-provider-badge">
              {text.provider}: {status.provider}/{status.model}
              {status.configured ? (status.api_key_configured ? "" : " · local-mock") : ""}
              {!status.configured ? ` · ${text.providerUnsupported}` : ""}
            </span>
          ) : null}
          <button className="button primary" onClick={() => void createSession()} disabled={busy}>{text.newChat}</button>
        </div>
      </header>
      {error ? <div className="validation invalid"><span>{error}</span></div> : null}

      <div className="agent-section-layout">
        <aside className="agent-session-rail panel" aria-label={text.sessions}>
          <header className="agent-rail-head"><strong>{text.sessions}</strong><span>{sessions.length}</span></header>
          {sessions.length === 0 ? <p className="muted agent-empty">{text.emptySessions}</p> : (
            <ul className="agent-session-list">
              {sessions.map((session) => (
                <li key={session.id} className={session.id === sessionId ? "active" : undefined}>
                  <button className="agent-session-item" onClick={() => void loadSession(session.id)} disabled={busy}>
                    <strong>{session.title || session.id}</strong>
                    <small>
                      {session.message_count} · {session.latest_run_status || "—"}
                    </small>
                  </button>
                  <button className="agent-session-delete" onClick={() => void removeSession(session.id)} disabled={busy} aria-label={text.delete}>×</button>
                </li>
              ))}
            </ul>
          )}
        </aside>

        <section className="agent-chat panel">
          {!detail ? (
            <p className="muted agent-empty">{text.emptyChat}</p>
          ) : (
            <>
              <header className="agent-chat-head">
                <div>
                  <h2>{detail.title}</h2>
                  <p>{text.status}: {activeRun?.status || "—"}{busy ? ` · ${text.refreshing}` : ""}</p>
                </div>
                {activeRun && isActiveRunStatus(activeRun.status) ? (
                  <button className="button" onClick={() => void cancelRun()} disabled={busy}>{text.cancel}</button>
                ) : null}
                {canRetryLast ? (
                  <button className="button" onClick={() => void retryLast()} disabled={busy}>{text.retryLast}</button>
                ) : null}
              </header>

              <div className="agent-message-list">
                {messages.map((message) => (
                  <MessageBubble
                    key={message.id}
                    message={message}
                    text={text}
                    onOpenLink={openLink}
                  />
                ))}
                <div ref={bottomRef} />
              </div>

              {focusRun ? (
                <ToolPanel
                  run={focusRun}
                  text={text}
                  expanded={expandedTools}
                  onToggle={(id) => setExpandedTools((current) => ({ ...current, [id]: !current[id] }))}
                  onOpenLink={openLink}
                />
              ) : null}

              {pendingApprovals.length > 0 ? (
                <ApprovalCards
                  approvals={pendingApprovals}
                  text={text}
                  locale={locale}
                  busy={busy}
                  onApprove={(id, payload) => void decideApproval(id, "approve", payload)}
                  onReject={(id) => void decideApproval(id, "reject")}
                />
              ) : null}

              <form className="agent-composer" onSubmit={(event) => void sendMessage(event)}>
                <textarea
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  placeholder={text.placeholder}
                  rows={3}
                  disabled={busy || (activeRun ? isActiveRunStatus(activeRun.status) : false)}
                />
                <button className="button primary" type="submit" disabled={busy || !draft.trim() || (activeRun ? isActiveRunStatus(activeRun.status) : false)}>
                  {text.send}
                </button>
              </form>
            </>
          )}
        </section>
      </div>
    </main>
  );
}

function MessageBubble({
  message,
  text,
  onOpenLink,
}: {
  message: AgentMessage;
  text: AgentCopy;
  onOpenLink: (link: AgentEntityLink) => void;
}) {
  const roleLabel = message.role === "user" ? text.user : message.role === "assistant" ? text.assistant : message.role === "tool" ? text.tool : text.system;
  const links = message.role === "assistant" || message.role === "user" ? linksFromText(message.content) : [];
  let body = message.content;
  if (message.role === "tool") {
    try {
      const parsed = JSON.parse(message.content) as { tool_name?: string; result?: unknown };
      body = `${parsed.tool_name || "tool"}\n${JSON.stringify(parsed.result ?? parsed, null, 2)}`;
    } catch {
      // keep raw
    }
  }
  return (
    <article className={`agent-message role-${message.role}`}>
      <header><span>{roleLabel}</span></header>
      <pre>{body}</pre>
      {links.length > 0 ? (
        <div className="agent-entity-links">
          {links.map((link) => (
            <button key={`${link.kind}-${link.id}`} type="button" className="agent-entity-link" onClick={() => onOpenLink(link)}>
              {link.kind === "dataset" ? text.openDataset : link.kind === "training" ? text.openTraining : text.openModel}: {link.label}
            </button>
          ))}
        </div>
      ) : null}
    </article>
  );
}

function ToolPanel({
  run,
  text,
  expanded,
  onToggle,
  onOpenLink,
}: {
  run: AgentRun;
  text: AgentCopy;
  expanded: Record<string, boolean>;
  onToggle: (id: string) => void;
  onOpenLink: (link: AgentEntityLink) => void;
}) {
  if (!run.tool_calls.length) {
    return <section className="agent-tools"><h3>{text.tools}</h3><p className="muted">{text.noTools}</p></section>;
  }
  return (
    <section className="agent-tools">
      <h3>{text.tools}</h3>
      <ul>
        {run.tool_calls.map((tool) => {
          const links = linksFromToolCall(tool);
          const open = expanded[tool.id];
          return (
            <li key={tool.id} className="agent-tool-card">
              <button type="button" className="agent-tool-toggle" onClick={() => onToggle(tool.id)}>
                <strong>{tool.name}</strong>
                <span className={`agent-tool-status status-${tool.status}`}>{tool.status}</span>
              </button>
              {links.length > 0 ? (
                <div className="agent-entity-links">
                  <span className="muted">{text.sources}</span>
                  {links.map((link) => (
                    <button key={`${tool.id}-${link.kind}-${link.id}`} type="button" className="agent-entity-link" onClick={() => onOpenLink(link)}>
                      {link.kind === "dataset" ? text.openDataset : link.kind === "training" ? text.openTraining : text.openModel}: {link.label}
                    </button>
                  ))}
                </div>
              ) : null}
              {open ? <ToolResultPreview tool={tool} /> : null}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function ToolResultPreview({ tool }: { tool: AgentToolCall }) {
  return (
    <pre className="agent-tool-result">
      {JSON.stringify({ arguments: tool.arguments_json, result: tool.result_json, error: tool.error_message }, null, 2)}
    </pre>
  );
}

function ApprovalCards({
  approvals,
  text,
  locale,
  busy,
  onApprove,
  onReject,
}: {
  approvals: AgentApproval[];
  text: AgentCopy;
  locale: AppLocale;
  busy: boolean;
  onApprove: (approvalId: string, payload: Record<string, unknown>) => void;
  onReject: (approvalId: string) => void;
}) {
  return (
    <section className="agent-approvals" data-testid="agent-approval-slot">
      <h3>{text.approvals}</h3>
      <p className="muted">{text.approvalHint}</p>
      {approvals.map((approval) => (
        <ApprovalCard
          key={approval.id}
          approval={approval}
          text={text}
          locale={locale}
          busy={busy}
          onApprove={onApprove}
          onReject={onReject}
        />
      ))}
    </section>
  );
}

function ApprovalCard({
  approval,
  text,
  locale,
  busy,
  onApprove,
  onReject,
}: {
  approval: AgentApproval;
  text: AgentCopy;
  locale: AppLocale;
  busy: boolean;
  onApprove: (approvalId: string, payload: Record<string, unknown>) => void;
  onReject: (approvalId: string) => void;
}) {
  const basePayload = useMemo(() => {
    const raw = approval.payload_json || {};
    const { action: _a, preview: _p, reserved_task_id: _r, requires_approval: _req, approval_id: _id, ...rest } = raw;
    return rest as Record<string, unknown>;
  }, [approval.payload_json]);
  const [draft, setDraft] = useState<Record<string, unknown>>(basePayload);
  useEffect(() => {
    setDraft(basePayload);
  }, [basePayload]);

  const fields = editableApprovalFields(approval.tool_name);
  const params = summarizeApprovalParams(draft, locale);
  const outputs = expectedApprovalOutputs(approval.tool_name, locale);
  const editable = approval.status === "pending";

  return (
    <article className="agent-approval-card" data-testid={`agent-approval-${approval.id}`}>
      <header>
        <strong>{approval.tool_name}</strong>
        <span>{approval.status}</span>
      </header>
      {params.length > 0 ? (
        <div className="agent-plan-step-card__section">
          <span className="agent-plan-step-card__label">{text.keyParams}</span>
          <div className="agent-plan-kv-list">
            {params.map((item) => (
              <span key={`${approval.id}-${item.label}`} className="agent-plan-kv-pill">
                <strong>{item.label}</strong>
                <em>{item.value}</em>
              </span>
            ))}
          </div>
        </div>
      ) : null}
      {fields.length > 0 && editable ? (
        <div className="agent-plan-step-card__section">
          <span className="agent-plan-step-card__label">{text.adjustBeforeRun}</span>
          <div className="agent-plan-edit-grid">
            {fields.map((field) => (
              <label key={`${approval.id}-${field.key}`} className="agent-plan-edit-field">
                <span>{locale === "zh" ? field.labelZh : field.labelEn}</span>
                {field.type === "checkbox" ? (
                  <input
                    type="checkbox"
                    disabled={busy}
                    checked={Boolean(draft[field.key])}
                    onChange={(event) => {
                      setDraft((current) => ({ ...current, [field.key]: event.target.checked }));
                    }}
                  />
                ) : field.type === "select" ? (
                  <select
                    disabled={busy}
                    value={String(draft[field.key] ?? field.options?.[0]?.value ?? "")}
                    onChange={(event) => {
                      setDraft((current) => ({ ...current, [field.key]: event.target.value }));
                    }}
                  >
                    {(field.options || []).map((option) => (
                      <option key={option.value} value={option.value}>
                        {locale === "zh" ? option.labelZh : option.labelEn}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type={field.type}
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    disabled={busy}
                    value={String(draft[field.key] ?? "")}
                    onChange={(event) => {
                      const raw = event.target.value;
                      const nextValue = field.type === "number" ? (raw === "" ? "" : Number(raw)) : raw;
                      setDraft((current) => ({ ...current, [field.key]: nextValue }));
                    }}
                  />
                )}
              </label>
            ))}
          </div>
        </div>
      ) : null}
      {outputs.length > 0 ? (
        <div className="agent-plan-step-card__section">
          <span className="agent-plan-step-card__label">{text.expectedOutputs}</span>
          <div className="agent-plan-output-list">
            {outputs.map((item) => (
              <span key={`${approval.id}-${item}`} className="agent-plan-output-pill">
                {item}
              </span>
            ))}
          </div>
        </div>
      ) : null}
      <details className="agent-plan-details">
        <summary>{text.viewDetails}</summary>
        <pre>{JSON.stringify(draft, null, 2)}</pre>
      </details>
      <div className="agent-approval-actions">
        <button
          type="button"
          className="button primary"
          disabled={busy || !editable}
          onClick={() => onApprove(approval.id, sanitizeApprovalPayload(draft, approval.tool_name))}
        >
          {text.approve}
        </button>
        <button type="button" className="button" disabled={busy || !editable} onClick={() => onReject(approval.id)}>
          {text.reject}
        </button>
      </div>
    </article>
  );
}
