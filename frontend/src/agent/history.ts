import type { AgentMessage, AgentRun, AgentSessionDetail } from "../types";

export function mergeMessages(current: AgentMessage[], incoming: AgentMessage[]): AgentMessage[] {
  const rows = new Map(current.map((message) => [message.id, message]));
  for (const message of incoming) {
    if (message.role === "tool") continue;
    const saved = rows.get(message.id);
    // Unchanged snapshots should not re-render every Markdown bubble each poll.
    if (saved && saved.updated_at === message.updated_at && saved.content === message.content) continue;
    rows.set(message.id, message);
  }
  return [...rows.values()].filter((message) => message.role !== "tool").sort((a, b) => a.sequence - b.sequence);
}

// Polling must never replace previously loaded historical pages. Run snapshots
// provide the authoritative mode/context after submissions, retries or approval.
export function mergeRun(current: AgentSessionDetail, run: AgentRun): AgentSessionDetail {
  if (current.id !== run.session_id) return current;
  if (current.runs.some((item) => Date.parse(item.created_at) > Date.parse(run.created_at))) return current;
  const saved = current.runs.find((item) => item.id === run.id);
  if (saved && Date.parse(saved.updated_at) > Date.parse(run.updated_at)) return current;
  const messages = mergeMessages(current.messages, run.messages);
  return {
    ...current, profile_id: run.profile_id ?? current.profile_id,
    profile_version: run.profile_version ?? current.profile_version,
    context: run.context, updated_at: run.updated_at, latest_run_status: run.status,
    message_count: Math.max(current.message_count, messages.length),
    runs: [run], messages,
  };
}
