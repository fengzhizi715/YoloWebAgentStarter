import type {
  AgentApproval,
  AgentContext,
  AgentProfileId,
  AgentProviderStatus,
  AgentRun,
  AgentSession,
  AgentSessionDetail,
  AgentSessionPage,
} from "../types";

export interface AgentApprovalDecision {
  approval: AgentApproval;
  run: AgentRun;
}

const API_BASE = (import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { error?: { message?: string } };
      message = body.error?.message ?? message;
    } catch {
      // Keep the HTTP status when the server does not return JSON.
    }
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

const json = (body: unknown, method = "POST"): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export interface SessionPageOptions {
  cursor?: string;
  query?: string;
  limit?: number;
  profileId?: AgentProfileId;
  context?: AgentContext;
}

/** Agent MVP HTTP adapters. Pages import from here, not hard-coded URLs. */
export const agentApi = {
  status: () => request<AgentProviderStatus>("/api/agent/status"),
  listSessions: () => request<AgentSession[]>("/api/agent/sessions"),
  listSessionPage: (options: SessionPageOptions = {}) => {
    const params = new URLSearchParams({ limit: String(options.limit ?? 30) });
    if (options.cursor) params.set("cursor", options.cursor);
    if (options.query) params.set("q", options.query);
    if (options.profileId) params.set("profile_id", options.profileId);
    if (options.context) {
      params.set("match_context", "true");
      for (const [key, value] of Object.entries(options.context)) if (value) params.set(key, value);
    }
    return request<AgentSessionPage>(`/api/agent/sessions/page?${params}`);
  },
  createSession: (title?: string, binding?: { profileId: AgentProfileId; context: AgentContext }) =>
    request<AgentSession>("/api/agent/sessions", json({ title, profile_id: binding?.profileId, context: binding?.context })),
  getSession: (sessionId: string, options: { beforeSequence?: number } = {}) => {
    const params = new URLSearchParams({ limit: "50" });
    if (options.beforeSequence !== undefined) params.set("before_sequence", String(options.beforeSequence));
    return request<AgentSessionDetail>(`/api/agent/sessions/${sessionId}/timeline?${params}`);
  },
  updateSession: (sessionId: string, title: string) =>
    request<AgentSession>(`/api/agent/sessions/${sessionId}`, json({ title }, "PATCH")),
  deleteSession: (sessionId: string) => request<void>(`/api/agent/sessions/${sessionId}`, { method: "DELETE" }),
  postMessage: (sessionId: string, content: string, options?: { readOnly?: boolean; context?: AgentContext; profileId?: AgentProfileId; allowContextChange?: boolean }) =>
    request<AgentRun>(`/api/agent/sessions/${sessionId}/messages`, json({
      content, read_only: options?.readOnly ?? false, context: options?.context,
      profile_id: options?.profileId, allow_context_change: options?.allowContextChange ?? false,
    })),
  retryRun: (runId: string) => request<AgentRun>(`/api/agent/runs/${runId}/retry`, { method: "POST" }),
  getRun: (runId: string) => request<AgentRun>(`/api/agent/runs/${runId}`),
  cancelRun: (runId: string) => request<AgentRun>(`/api/agent/runs/${runId}/cancel`, { method: "POST" }),
  approveApproval: (approvalId: string, payload?: Record<string, unknown>) =>
    request<AgentApprovalDecision>(`/api/agent/approvals/${approvalId}/approve`, json({ payload: payload ?? null })),
  rejectApproval: (approvalId: string) =>
    request<AgentApprovalDecision>(`/api/agent/approvals/${approvalId}/reject`, { method: "POST" }),
};
