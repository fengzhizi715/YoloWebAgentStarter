import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { agentApi } from "../api/agent";
import {
  editableApprovalFields,
  expectedApprovalOutputs,
  sanitizeApprovalPayload,
  summarizeApprovalParams,
} from "../agent/approvalFields";
import { isActiveRunStatus, linksFromText, linksFromToolCall, type AgentEntityLink } from "../agent/entityLinks";
import { inferenceReasonLabel, inferenceSourceLabel } from "../agent/inferenceLabels";
import { AgentContextPicker } from "../agent/AgentContextPicker";
import { sameBinding, inferProfile, matchingSession, profileTitle } from "../agent/profiles";
import {
  approvalStatusLabel,
  factKeyLabel,
  reportTitle,
  runStatusLabel,
  toolNameLabel,
  toolStatusLabel,
} from "../agent/labels";
import type { AppLocale } from "../locale";
import type { AgentApproval, AgentContext, AgentProfileId, AgentMessage, AgentProviderStatus, AgentRun, AgentSession, AgentSessionDetail, AgentToolCall, Dataset, ModelVersion, TrainingTask } from "../types";

const copy = {
  zh: {
    eyebrow: "AGENT",
    title: "智能体",
    subtitle: "可查询数据集、训练与模型；写操作需人工确认后才会创建任务。",
    newChat: "新会话",
    rename: "重命名",
    saveName: "保存名称",
    cancelRename: "取消",
    sessions: "会话",
    emptySessions: "还没有会话。",
    delete: "删除",
    placeholder: "询问质量报告，或说明要创建训练/评估/自动标注…",
    send: "发送",
    cancel: "取消运行",
    refreshing: "同步中…",
    thinking: "正在分析并生成回答…",
    querying: "正在调用：",
    queued: "等待运行…",
    runContext: "本轮对象",
    readOnly: "只读",
    confirmWrites: "写操作需确认",
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
    quickPrompts: "快捷提问",
    contextDataset: "当前数据集上下文",
    contextTraining: "当前训练任务",
    contextModel: "当前模型",
    openDataset: "数据集",
    openTraining: "训练任务",
    openModel: "模型",
    user: "你",
    assistant: "智能体",
    tool: "工具",
    system: "系统",
    itemCount: "共 {count} 项",
    imagesCount: "{count} 张图片",
  },
  en: {
    eyebrow: "AGENT",
    title: "Agent",
    subtitle: "Query datasets, training, and models. Write actions run only after you confirm.",
    newChat: "New chat",
    rename: "Rename",
    saveName: "Save name",
    cancelRename: "Cancel",
    sessions: "Sessions",
    emptySessions: "No sessions yet.",
    delete: "Delete",
    placeholder: "Ask about quality, or request training / evaluation / auto-annotation…",
    send: "Send",
    cancel: "Cancel run",
    refreshing: "Syncing…",
    thinking: "Analyzing and preparing an answer…",
    querying: "Running: ",
    queued: "Waiting to run…",
    runContext: "This turn's context",
    readOnly: "Read-only",
    confirmWrites: "Writes require confirmation",
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
    quickPrompts: "Quick prompts",
    contextDataset: "Current dataset context",
    contextTraining: "Current training task",
    contextModel: "Current model",
    openDataset: "Dataset",
    openTraining: "Training task",
    openModel: "Model",
    user: "You",
    assistant: "Agent",
    tool: "Tool",
    system: "System",
    itemCount: "{count} items",
    imagesCount: "{count} images",
  },
} as const;

type AgentCopy = (typeof copy)[AppLocale];

export interface AgentNavigationHandlers {
  onOpenDataset: (datasetId: string) => void;
  onOpenTrainingTask: (datasetId: string | undefined, taskId: string) => void;
  onOpenModel: (datasetId: string | undefined, modelId: string) => void;
}

export interface AgentPageContext {
  dataset?: Pick<Dataset, "id" | "name">;
  trainingTask?: Pick<TrainingTask, "id" | "name">;
  model?: Pick<ModelVersion, "id" | "name">;
}

interface Props extends AgentNavigationHandlers {
  locale: AppLocale;
  context?: AgentPageContext;
}

function contextIds(context?: AgentPageContext): AgentContext {
  return { ...(context?.dataset ? { dataset_id: context.dataset.id } : {}),
    ...(context?.trainingTask ? { training_task_id: context.trainingTask.id } : {}),
    ...(context?.model ? { model_id: context.model.id } : {}) };
}

export function AgentView({ locale, context: pageContext, onOpenDataset, onOpenTrainingTask, onOpenModel }: Props) {
  const text = copy[locale];
  const [bindingContext, setBindingContext] = useState<AgentContext>(() => contextIds(pageContext));
  const [profileId, setProfileId] = useState<AgentProfileId>(() => inferProfile(contextIds(pageContext)));
  const [bindingConfirmed, setBindingConfirmed] = useState(false);
  const selectionEpoch = useRef(0);
  const selectedSessionId = useRef<string>();
  const context: AgentPageContext = {
    ...(bindingContext.dataset_id ? { dataset: { id: bindingContext.dataset_id, name: pageContext?.dataset?.id === bindingContext.dataset_id ? pageContext.dataset.name : bindingContext.dataset_id } } : {}),
    ...(bindingContext.training_task_id ? { trainingTask: { id: bindingContext.training_task_id, name: pageContext?.trainingTask?.id === bindingContext.training_task_id ? pageContext.trainingTask.name : bindingContext.training_task_id } } : {}),
    ...(bindingContext.model_id ? { model: { id: bindingContext.model_id, name: pageContext?.model?.id === bindingContext.model_id ? pageContext.model.name : bindingContext.model_id } } : {}),
  };
  const [sessions, setSessions] = useState<AgentSession[]>([]);
  const [sessionId, setSessionId] = useState<string>();
  const [detail, setDetail] = useState<AgentSessionDetail>();
  const [status, setStatus] = useState<AgentProviderStatus>();
  const [draft, setDraft] = useState("");
  const [actionBusy, setBusy] = useState(false);
  const [loadingSession, setLoadingSession] = useState(true);
  const busy = actionBusy || loadingSession;
  const [error, setError] = useState("");
  const [expandedTools, setExpandedTools] = useState<Record<string, boolean>>({});
  const [renaming, setRenaming] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const bindingChanged = Boolean(detail && detail.message_count > 0 &&
    !sameBinding(detail.profile_id ?? "global", detail.context, profileId, bindingContext));
  const needsBindingChoice = bindingChanged && !bindingConfirmed;

  const activeRun = useMemo(() => {
    const runs = detail?.runs || [];
    return runs.find((run) => isActiveRunStatus(run.status)) || runs[0];
  }, [detail]);

  const refreshSessions = async () => {
    const items = await agentApi.listSessions();
    setSessions(items);
    return items;
  };

  const loadSession = async (id: string, restoreBinding = false) => {
    const epoch = ++selectionEpoch.current;
    selectedSessionId.current = id;
    setLoadingSession(true);
    try {
      const next = await agentApi.getSession(id);
      if (epoch !== selectionEpoch.current) return next;
      setDetail(next);
      setSessionId(next.id);
      setTitleDraft(next.title);
      setRenaming(false);
      if (restoreBinding) {
        setBindingContext(next.context ?? {});
        setProfileId(next.profile_id ?? "global");
        setBindingConfirmed(false);
        setDraft("");
      }
      return next;
    } catch (reason) {
      if (epoch === selectionEpoch.current) {
        selectedSessionId.current = sessionId;
        setError(reason instanceof Error ? reason.message : "Request failed");
      }
      throw reason;
    } finally {
      if (epoch === selectionEpoch.current) setLoadingSession(false);
    }
  };

  useEffect(() => {
    let disposed = false;
    const incoming = contextIds(pageContext);
    const incomingProfile = inferProfile(incoming);
    setBindingContext(incoming); setProfileId(incomingProfile); setBindingConfirmed(false);
    setDetail(undefined); setSessionId(undefined);
    setLoadingSession(true); setDraft(""); setRenaming(false); setError("");
    selectedSessionId.current = undefined;
    ++selectionEpoch.current;
    void (async () => {
      try {
        const [provider, items] = await Promise.all([agentApi.status(), refreshSessions()]);
        if (disposed) return;
        setStatus(provider);
        const match = matchingSession(items, incomingProfile, incoming);
        if (match) await loadSession(match.id);
      } catch (reason) {
        if (!disposed) setError(reason instanceof Error ? reason.message : "Request failed");
      } finally {
        if (!disposed) setLoadingSession(false);
      }
    })();
    return () => { disposed = true; ++selectionEpoch.current; };
  }, [pageContext?.dataset?.id, pageContext?.trainingTask?.id, pageContext?.model?.id]);

  useEffect(() => {
    if (!activeRun || !sessionId || !isActiveRunStatus(activeRun.status)) return;
    let disposed = false;
    let inFlight = false;
    const timer = window.setInterval(() => {
      if (inFlight) return;
      inFlight = true;
      void agentApi.getSession(sessionId).then((next) => {
        if (disposed || sessionId !== selectedSessionId.current) return;
        setDetail(next);
        void refreshSessions().catch(() => undefined);
      }).catch((reason) => {
        if (!disposed) setError(reason instanceof Error ? reason.message : "Request failed");
      }).finally(() => { inFlight = false; });
    }, 1000);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [activeRun?.id, activeRun?.status, sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [detail?.messages.length, activeRun?.status]);

  const createSession = async () => {
    setBusy(true); setError("");
    try {
      const created = await agentApi.createSession(text.newChat, { profileId, context: bindingContext });
      await refreshSessions();
      await loadSession(created.id);
      setBindingConfirmed(false);
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
        selectedSessionId.current = undefined;
        ++selectionEpoch.current;
        const match = matchingSession(items, profileId, bindingContext);
        if (match) await loadSession(match.id);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Request failed");
    } finally {
      setBusy(false);
    }
  };

  const acceptRun = (run: AgentRun) => {
    setSessionId(run.session_id);
    setDetail((current) => current?.id === run.session_id ? {
      ...current,
      latest_run_status: run.status,
      runs: [run, ...current.runs.filter((item) => item.id !== run.id)],
      messages: [...current.messages.filter((item) => item.run_id !== run.id), ...run.messages]
        .sort((left, right) => left.sequence - right.sequence),
    } : current);
  };

  const submitMessage = async (raw: string, readOnly = false) => {
    const content = raw.trim();
    if (!content || busy || needsBindingChoice) return;
    setBusy(true); setError("");
    try {
      let targetId = sessionId;
      if (!targetId) {
        const created = await agentApi.createSession(content.slice(0, 80), { profileId, context: bindingContext });
        targetId = created.id;
        await refreshSessions();
      }
      const run = await agentApi.postMessage(targetId, content, {
        readOnly,
        context: bindingContext,
        profileId,
        allowContextChange: bindingChanged && bindingConfirmed,
      });
      acceptRun(run);
      setDraft("");
      await loadSession(targetId);
      setBindingConfirmed(false);
      await refreshSessions();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Request failed");
    } finally {
      setBusy(false);
    }
  };

  const sendMessage = (event: FormEvent) => {
    event.preventDefault();
    void submitMessage(draft);
  };

  const renameSession = async () => {
    if (!sessionId || !titleDraft.trim() || busy) return;
    setBusy(true); setError("");
    try {
      await agentApi.updateSession(sessionId, titleDraft.trim());
      await loadSession(sessionId);
      await refreshSessions();
      setRenaming(false);
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
      const run = await agentApi.cancelRun(activeRun.id);
      acceptRun(run);
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
  const quickPrompts = buildQuickPrompts(locale, context, profileId);
  const canRetryLast =
    Boolean(lastUserMessage?.content) &&
    !busy &&
    Boolean(activeRun) &&
    (activeRun?.status === "failed" || activeRun?.status === "cancelled") &&
    pendingApprovals.length === 0;

  const retryLast = async () => {
    if (!lastUserMessage?.content || !activeRun || !sessionId || busy) return;
    setDraft(lastUserMessage.content);
    setBusy(true);
    setError("");
    try {
      const run = await agentApi.retryRun(activeRun.id);
      acceptRun(run);
      await loadSession(sessionId, true);
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
              {status.provider === "mock" ? " · local rules" : ""}
              {!status.configured ? ` · ${text.providerUnsupported}` : ""}
            </span>
          ) : null}
          <button className="button primary" onClick={() => void createSession()} disabled={busy}>{text.newChat}</button>
        </div>
      </header>
      {error ? <div className="validation invalid"><span>{error}</span></div> : null}

      <AgentContextPicker locale={locale} profile={profileId} context={bindingContext} disabled={busy}
        onChange={(profile, next) => { setProfileId(profile); setBindingContext(next); setBindingConfirmed(false); }} />
      {needsBindingChoice ? <div className="agent-binding-warning" role="alert">
        <p>{locale === "zh" ? "对象或助手模式已改变。请选择新建会话，或明确继续当前会话；历史运行保持原绑定。" : "Context or mode changed. Start a new chat or explicitly continue this one; historical runs keep their original binding."}</p>
        <button className="button primary" disabled={busy} onClick={() => void createSession()}>{text.newChat}</button>
        <button className="button" disabled={busy || Boolean(activeRun && isActiveRunStatus(activeRun.status))} onClick={() => setBindingConfirmed(true)}>{locale === "zh" ? "继续当前会话" : "Continue this chat"}</button>
      </div> : null}

      <div className="agent-section-layout">
        <aside className="agent-session-rail panel" aria-label={text.sessions}>
          <header className="agent-rail-head"><strong>{text.sessions}</strong><span>{sessions.length}</span></header>
          {sessions.length === 0 ? <p className="muted agent-empty">{text.emptySessions}</p> : (
            <ul className="agent-session-list">
              {sessions.map((session) => (
                <li key={session.id} className={session.id === sessionId ? "active" : undefined}>
                  <button className="agent-session-item" onClick={() => void loadSession(session.id, true).catch(() => undefined)} disabled={busy}>
                    <strong>{session.title || session.id}</strong>
                    <small>
                      {profileTitle(session.profile_id ?? "global", locale)} · {session.message_count} · {runStatusLabel(session.latest_run_status, locale)}
                      {Object.values(session.context ?? {}).filter(Boolean).length ? ` · ${Object.values(session.context ?? {}).filter(Boolean).join(" / ")}` : ""}
                    </small>
                  </button>
                  <button className="agent-session-delete" onClick={() => void removeSession(session.id)} disabled={busy} aria-label={text.delete}>×</button>
                </li>
              ))}
            </ul>
          )}
        </aside>

        <section className="agent-chat panel">
            <>
              {!detail ? <p className="muted agent-empty">{loadingSession ? text.refreshing : locale === "zh" ? "没有此模式与对象的会话，发送问题将新建会话。" : "No chat matches this mode and context. Sending a question starts a new chat."}</p> : null}
              <header className="agent-chat-head">
                <div>
                  {renaming ? (
                    <div className="agent-title-editor">
                      <input value={titleDraft} onChange={(event) => setTitleDraft(event.target.value)} maxLength={255} />
                      <button className="button primary" type="button" onClick={() => void renameSession()} disabled={busy || !titleDraft.trim()}>{text.saveName}</button>
                      <button className="button" type="button" onClick={() => { setTitleDraft(detail?.title ?? ""); setRenaming(false); }} disabled={busy}>{text.cancelRename}</button>
                    </div>
                  ) : (
                    <div className="agent-title-line"><h2>{detail?.title ?? text.newChat}</h2><button className="button" type="button" onClick={() => setRenaming(true)} disabled={busy || !detail}>{text.rename}</button></div>
                  )}
                  <p>{text.status}: {runStatusLabel(activeRun?.status, locale)}{busy ? ` · ${text.refreshing}` : ""}{activeRun ? ` · ${activeRun.read_only ? text.readOnly : text.confirmWrites}` : ""}</p>
                  {activeRun ? <p>{locale === "zh" ? "本轮模式" : "Run mode"}: {profileTitle(activeRun.profile_id ?? "global", locale)} v{activeRun.profile_version ?? 1}</p> : null}
                  {activeRun && Object.values(activeRun.context ?? {}).some(Boolean) ? (
                    <p>{text.runContext}: {Object.values(activeRun.context).filter(Boolean).join(" · ")}</p>
                  ) : null}
                </div>
                {activeRun && isActiveRunStatus(activeRun.status) ? (
                  <button className="button" onClick={() => void cancelRun()} disabled={busy}>{text.cancel}</button>
                ) : null}
                {canRetryLast ? (
                  <button className="button" onClick={() => void retryLast()} disabled={busy}>{text.retryLast}</button>
                ) : null}
              </header>

              {activeRun ? (
                <div className="agent-inference-summary">
                  <p>{locale === "zh" ? "实际来源" : "Actual source"}: {inferenceSourceLabel(activeRun.actual_source, locale)}
                    {activeRun.inference_steps?.length ? ` · ${locale === "zh" ? "推理总耗时" : "Inference time"} ${activeRun.inference_steps.reduce((total, step) => total + step.duration_ms, 0)} ms` : ""}
                  </p>
                  {activeRun.inference_steps?.some((step) => step.source === "fallback") ? (
                    <p role="note">{locale === "zh" ? "本轮发生过规则降级，并非全部由 LLM 完成。" : "This run used rule fallback; it was not completed entirely by the LLM."}</p>
                  ) : null}
                  {activeRun.inference_steps?.length ? (
                    <details>
                      <summary>{locale === "zh" ? "推理记录" : "Inference history"}</summary>
                      <ul>{activeRun.inference_steps.map((step) => (
                        <li key={step.round}>
                          #{step.round} · {inferenceSourceLabel(step.source, locale)} · {step.provider} / {step.model} · {step.duration_ms} ms
                          {` · ${step.outcome === "discarded" ? (locale === "zh" ? "取消后丢弃" : "Discarded after cancellation") : runStatusLabel(step.outcome, locale)}`}
                          {step.reason ? ` · ${inferenceReasonLabel(step.reason, locale)}` : ""}
                        </li>
                      ))}</ul>
                    </details>
                  ) : null}
                </div>
              ) : null}

              {activeRun?.status === "pending" || activeRun?.status === "running" ? (
                <p role="status" className="agent-empty">
                  {activeRun.status === "pending" ? text.queued : activeRun.tool_calls.some((tool) => tool.status === "running")
                    ? `${text.querying}${toolNameLabel(activeRun.tool_calls.find((tool) => tool.status === "running")!.name, locale)}`
                    : text.thinking}
                </p>
              ) : null}
              {activeRun?.status === "failed" && activeRun.error_message ? <p className="validation invalid">{activeRun.error_message}</p> : null}

              <div className="agent-message-list">
                {messages.filter((message) => message.role !== "tool").map((message) => (
                  <MessageBubble
                    key={message.id}
                    message={message}
                    text={text}
                    onOpenLink={openLink}
                  />
                ))}
                <div ref={bottomRef} />
              </div>

              <QuickPromptBar
                prompts={quickPrompts}
                title={text.quickPrompts}
                disabled={busy || needsBindingChoice || Boolean(activeRun && isActiveRunStatus(activeRun.status))}
                onSelect={(prompt) => void submitMessage(prompt, true)}
              />

              {context?.dataset || context?.trainingTask || context?.model ? (
                <div className="agent-context-card">
                  {context?.dataset ? <><strong>{text.contextDataset}</strong><span>{context.dataset.name} · <code>{context.dataset.id}</code></span></> : null}
                  {context?.trainingTask ? <><strong>{text.contextTraining}</strong><span>{context.trainingTask.name} · <code>{context.trainingTask.id}</code></span></> : null}
                  {context?.model ? <><strong>{text.contextModel}</strong><span>{context.model.name} · <code>{context.model.id}</code></span></> : null}
                </div>
              ) : null}

              {focusRun ? (
                <ToolPanel
                  run={focusRun}
                  text={text}
                  locale={locale}
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

              <form className="agent-composer" onSubmit={sendMessage}>
                <textarea
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  placeholder={text.placeholder}
                  rows={3}
                  disabled={busy || needsBindingChoice || (activeRun ? isActiveRunStatus(activeRun.status) : false)}
                />
                <button className="button primary" type="submit" disabled={busy || needsBindingChoice || !draft.trim() || (activeRun ? isActiveRunStatus(activeRun.status) : false)}>
                  {text.send}
                </button>
              </form>
            </>
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
  locale,
  expanded,
  onToggle,
  onOpenLink,
}: {
  run: AgentRun;
  text: AgentCopy;
  locale: AppLocale;
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
                <strong>{toolNameLabel(tool.name, locale)}</strong>
                <span className={`agent-tool-status status-${tool.status}`}>{toolStatusLabel(tool.status, locale)}</span>
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
              <ToolReportCard tool={tool} locale={locale} text={text} />
              {open ? <ToolResultPreview tool={tool} /> : null}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function QuickPromptBar({
  prompts,
  title,
  disabled,
  onSelect,
}: {
  prompts: Array<{ label: string; prompt: string }>;
  title: string;
  disabled: boolean;
  onSelect: (prompt: string) => void;
}) {
  return (
    <section className="agent-quick-prompts" aria-label={title}>
      <strong>{title}</strong>
      <div>{prompts.map((item) => <button key={item.label} type="button" className="button" disabled={disabled} onClick={() => onSelect(item.prompt)}>{item.label}</button>)}</div>
    </section>
  );
}

function buildQuickPrompts(locale: AppLocale, context?: AgentPageContext, profile: AgentProfileId = "global"): Array<{ label: string; prompt: string }> {
  if (profile === "training" && !context?.trainingTask) return [{ label: locale === "zh" ? "选择训练任务" : "Find a training task", prompt: locale === "zh" ? `列出训练任务 ${context?.dataset?.id ?? ""}，让我选择要诊断的对象。` : `List training tasks ${context?.dataset?.id ?? ""} so I can choose one to diagnose.` }];
  if (profile === "model" && !context?.model) return [{ label: locale === "zh" ? "选择模型" : "Find a model", prompt: locale === "zh" ? `列出模型 ${context?.dataset?.id ?? ""}，让我选择评估或比较的对象。` : `List models ${context?.dataset?.id ?? ""} so I can choose ones to evaluate or compare.` }];
  if (context?.trainingTask) {
    const id = context.trainingTask.id;
    return locale === "zh"
      ? [
          { label: "训练状态", prompt: `请总结训练任务 ${id} 的进度、指标和风险。` },
          { label: "训练日志", prompt: `请查看训练任务 ${id} 的日志并总结异常。` },
          { label: "失败诊断", prompt: `请诊断训练任务 ${id} 的失败原因，给出证据和人工排查步骤，不要创建任务。` },
        ]
      : [
          { label: "Training status", prompt: `Summarize progress, metrics and risks for training task ${id}.` },
          { label: "Training logs", prompt: `Review logs and anomalies for training task ${id}.` },
          { label: "Failure diagnosis", prompt: `Diagnose failure for training task ${id}, show evidence and manual next steps; do not create tasks.` },
        ];
  }
  if (context?.model) {
    const id = context.model.id;
    return locale === "zh"
      ? [
          { label: "模型与评估", prompt: `请总结模型 ${id} 的指标和最近评估结果。` },
          { label: "模型详情", prompt: `请查看模型 ${id} 的详情和使用边界。` },
          { label: "模型比较准备", prompt: `我想比较模型 ${id}，请列出同数据集候选让我选择，再检查评估可比性。` },
        ]
      : [
          { label: "Model evaluation", prompt: `Summarize metrics and recent evaluations for model ${id}.` },
          { label: "Model details", prompt: `Review model ${id} and its limitations.` },
          { label: "Prepare comparison", prompt: `I want to compare model ${id}. List candidates from the same dataset for me to choose, then check evaluation comparability.` },
        ];
  }
  const dataset = context?.dataset;
  if (dataset) {
    return locale === "zh"
      ? [
          { label: "数据集质量报告", prompt: `请给出数据集 ${dataset.name}（${dataset.id}）的质量报告。` },
          { label: "校验数据集", prompt: `请校验数据集 ${dataset.id}，并总结阻断训练的问题。` },
          { label: "训练建议", prompt: `请基于数据集 ${dataset.id} 给出训练建议，不要创建任务。` },
          { label: "查看最新模型", prompt: `请列出数据集 ${dataset.id} 的最新受管模型和评估结果。` },
        ]
      : [
          { label: "Dataset quality", prompt: `Give a quality report for dataset ${dataset.name} (${dataset.id}).` },
          { label: "Validate dataset", prompt: `Validate dataset ${dataset.id} and summarize blockers for training.` },
          { label: "Training advice", prompt: `Give training advice for dataset ${dataset.id}; do not create a task.` },
          { label: "Latest model", prompt: `List the latest managed models and evaluations for dataset ${dataset.id}.` },
        ];
  }
  return locale === "zh"
    ? [
        { label: "列出数据集", prompt: "列出当前数据集并给出概览。" },
        { label: "查看训练", prompt: "总结最近训练任务的状态和失败原因。" },
        { label: "查看模型", prompt: "列出当前受管模型和最近评估结果。" },
      ]
    : [
        { label: "List datasets", prompt: "List the current datasets with a brief overview." },
        { label: "Review training", prompt: "Summarize recent training status and failures." },
        { label: "Review models", prompt: "List managed models and recent evaluation results." },
      ];
}

function ToolReportCard({ tool, locale, text }: { tool: AgentToolCall; locale: AppLocale; text: AgentCopy }) {
  const result = tool.result_json;
  if (!result || typeof result !== "object" || Array.isArray(result)) return null;
  const payload = result as Record<string, unknown>;
  if (typeof payload.summary === "string" && Array.isArray(payload.next_steps)) {
    const evidence = Array.isArray(payload.evidence) ? payload.evidence : [];
    return <section className="agent-report-card">
      <strong>{toolNameLabel(tool.name, locale)}</strong><p>{payload.summary}</p>
      <strong>{locale === "zh" ? "证据" : "Evidence"}</strong>
      <ul>{evidence.map((item, index) => {
        const row = item as Record<string, unknown>;
        const facts = (row.facts ?? row) as Record<string, unknown>;
        return <li key={index}>{String(row.tool ?? row.model_id ?? "")}: {Object.entries(facts).filter(([key, value]) => ["id", "task_id", "evaluation_id", "status", "split", "image_count", "class_count", "line_count", "error", "error_message"].includes(key) && value != null).map(([key, value]) => `${key}=${String(value)}`).join(" · ") || (locale === "zh" ? "详见工具记录" : "See tool record")}</li>;
      })}</ul>
      <strong>{locale === "zh" ? "问题" : "Issues"}</strong><ul>{(Array.isArray(payload.issues) ? payload.issues : []).map((item, i) => <li key={i}>{String(item)}</li>)}</ul>
      <strong>{locale === "zh" ? "下一步" : "Next steps"}</strong><ul>{payload.next_steps.map((item, i) => <li key={i}>{String(item)}</li>)}</ul>
    </section>;
  }
  const items = Array.isArray(payload.items) ? payload.items.filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object")) : [];
  const isList = ["global_summary", "list_datasets", "training_list", "list_training_tasks", "model_list", "list_models", "evaluation_list", "list_model_evaluations"].includes(tool.name);
  const title = reportTitle(tool.name, locale);
  const summary = payload.summary && typeof payload.summary === "object" && !Array.isArray(payload.summary)
    ? payload.summary as Record<string, unknown> : {};
  const model = payload.model && typeof payload.model === "object" && !Array.isArray(payload.model)
    ? payload.model as Record<string, unknown> : {};
  const source = tool.name === "dataset_quality_report" || tool.name === "dataset_validate" || tool.name === "validate_dataset"
    ? { ...payload, ...summary }
    : tool.name === "model_latest_for_dataset" ? model : payload;
  if (isList && items.length) {
    return (
      <section className="agent-report-card">
        <strong>{title}</strong>
        <span>{text.itemCount.replace("{count}", String(payload.total ?? items.length))}</span>
        <ul>{items.slice(0, 4).map((item, index) => <li key={`${String(item.id ?? index)}`}>{reportItem(item, text)}</li>)}</ul>
      </section>
    );
  }
  const facts = ["valid", "image_count", "annotated_image_count", "class_count", "status", "coverage", "annotation_count", "error_count", "warning_count", "map50", "split"].flatMap((key) => source[key] === undefined ? [] : [[key, source[key]] as const]);
  if (Array.isArray(payload.issues)) facts.push(["issue_count", payload.issues.length]);
  if (!facts.length) return null;
  return (
    <section className="agent-report-card">
      <strong>{title}</strong>
      <dl>{facts.map(([key, value]) => <div key={key}><dt>{factKeyLabel(key, locale)}</dt><dd>{key === "coverage" && typeof value === "number" && value <= 1 ? `${(value * 100).toFixed(1)}%` : String(value)}</dd></div>)}</dl>
    </section>
  );
}

function reportItem(item: Record<string, unknown>, text: AgentCopy): string {
  const name = String(item.name ?? item.id ?? "item");
  const metrics = item.metrics && typeof item.metrics === "object" && !Array.isArray(item.metrics)
    ? item.metrics as Record<string, unknown> : {};
  const details = [
    item.task_type,
    item.status,
    item.split === undefined ? undefined : `split=${String(item.split)}`,
    item.map50 === undefined ? undefined : `mAP50=${String(item.map50)}`,
    metrics.map50 === undefined ? undefined : `mAP50=${String(metrics.map50)}`,
    item.error_message === undefined || item.error_message === null ? undefined : String(item.error_message),
    item.image_count === undefined ? undefined : text.imagesCount.replace("{count}", String(item.image_count)),
  ].filter(Boolean);
  return details.length ? `${name} · ${details.join(" · ")}` : name;
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
        <strong>{toolNameLabel(approval.tool_name, locale)}</strong>
        <span>{approvalStatusLabel(approval.status, locale)}</span>
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
