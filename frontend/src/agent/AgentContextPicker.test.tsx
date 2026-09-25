// @vitest-environment jsdom
import { act, useState } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import { AgentContextPicker } from "./AgentContextPicker";
import type { AgentContext, AgentProfileId } from "../types";

vi.mock("../api/client", () => ({ api: {
  listDatasets: vi.fn().mockResolvedValue([{ id: "ds_a", name: "Dataset A" }]),
  listTrainingTasks: vi.fn().mockResolvedValue({ items: [] }),
  listModels: vi.fn().mockResolvedValue({ items: [] }),
} }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function Harness() {
  const [profile, setProfile] = useState<AgentProfileId>("model");
  const [context, setContext] = useState<AgentContext>({});
  return <AgentContextPicker locale="en" profile={profile} context={context} disabled={false}
    onChange={(nextProfile, nextContext) => { setProfile(nextProfile); setContext(nextContext); }} />;
}

describe("Agent context controls", () => {
  it("shows relevant selectors and keeps the chosen assistant when changing its dataset", async () => {
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    try {
      await act(async () => { root.render(<Harness />); });
      const labels = () => [...container.querySelectorAll("label")].map((label) => label.childNodes[0].textContent);
      expect(labels()).toEqual(["Dataset", "Model"]);
      const dataset = container.querySelector("select")!;
      await act(async () => {
        dataset.value = "ds_a";
        dataset.dispatchEvent(new Event("change", { bubbles: true }));
      });
      expect(container.querySelector('[aria-label="Model assistant"]')?.getAttribute("aria-pressed")).toBe("true");
      expect(labels()).toEqual(["Dataset", "Model"]);
      await act(async () => { container.querySelector<HTMLButtonElement>('[aria-label="Training assistant"]')!.click(); });
      expect(labels()).toEqual(["Dataset", "Training task"]);
      expect(container.querySelector("select")?.value).toBe("ds_a");
    } finally {
      act(() => root.unmount());
      container.remove();
    }
  });
});
