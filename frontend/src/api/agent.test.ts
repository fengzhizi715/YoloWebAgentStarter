import { afterEach, describe, expect, it, vi } from "vitest";
import { agentApi } from "./agent";

afterEach(() => vi.unstubAllGlobals());

describe("Agent history HTTP adapters", () => {
  it("encodes server search, cursor and exact context matching", async () => {
    const fetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({ items: [], next_cursor: null }) });
    vi.stubGlobal("fetch", fetch);
    await agentApi.listSessionPage({ query: "质量 %_ &", cursor: "opaque+=", profileId: "training", context: { training_task_id: "train_1" } });
    const url = new URL(fetch.mock.calls[0][0]);
    expect(url.pathname).toBe("/api/agent/sessions/page");
    expect(Object.fromEntries(url.searchParams)).toEqual({ limit: "30", q: "质量 %_ &", cursor: "opaque+=",
      profile_id: "training", match_context: "true", training_task_id: "train_1" });
  });

  it("requests only the timeline page, including its exclusive sequence boundary", async () => {
    const fetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({}) });
    vi.stubGlobal("fetch", fetch);
    await agentApi.getSession("asess_1");
    await agentApi.getSession("asess_1", { beforeSequence: 51 });
    const first = new URL(fetch.mock.calls[0][0]);
    const second = new URL(fetch.mock.calls[1][0]);
    expect(first.pathname).toBe("/api/agent/sessions/asess_1/timeline");
    expect(first.search).toBe("?limit=50");
    expect(second.search).toBe("?limit=50&before_sequence=51");
  });
});
