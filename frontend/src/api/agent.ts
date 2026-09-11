import type {
  AgentApproval,
  AgentProviderStatus,
  AgentRun,
  AgentSession,
  AgentSessionDetail,
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

/** Agent MVP HTTP adapters. Pages import from here, not hard-coded URLs. */
export const agentApi = {
  status: () => request<AgentProviderStatus>("/api/agent/status"),
  listSessions: () => request<AgentSession[]>("/api/agent/sessions"),
  createSession: (title?: string) => request<AgentSession>("/api/agent/sessions", json(title ? { title } : {})),
  getSession: (sessionId: string) => request<AgentSessionDetail>(`/api/agent/sessions/${sessionId}`),
  updateSession: (sessionId: string, title: string) =>
    request<AgentSession>(`/api/agent/sessions/${sessionId}`, json({ title }, "PATCH")),
  deleteSession: (sessionId: string) => request<void>(`/api/agent/sessions/${sessionId}`, { method: "DELETE" }),
  postMessage: (sessionId: string, content: string) =>
    request<AgentRun>(`/api/agent/sessions/${sessionId}/messages`, json({ content })),
  getRun: (runId: string) => request<AgentRun>(`/api/agent/runs/${runId}`),
  cancelRun: (runId: string) => request<AgentRun>(`/api/agent/runs/${runId}/cancel`, { method: "POST" }),
  approveApproval: (approvalId: string, payload?: Record<string, unknown>) =>
    request<AgentApprovalDecision>(`/api/agent/approvals/${approvalId}/approve`, json({ payload: payload ?? null })),
  rejectApproval: (approvalId: string) =>
    request<AgentApprovalDecision>(`/api/agent/approvals/${approvalId}/reject`, { method: "POST" }),
};
