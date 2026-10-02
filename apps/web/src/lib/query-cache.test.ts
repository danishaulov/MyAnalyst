import { afterEach, describe, expect, it, vi } from "vitest";
import { answerQuestionAI } from "./query";
import { profileTable } from "./profile";
import type { Table } from "./types";

function dataset(value: number): Table {
  return { name: "same.csv", rowCount: 6, columns: ["Revenue"], rows: Array.from({ length: 6 }, () => ({ Revenue: value })) };
}

describe("AI answer cache isolation", () => {
  afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); });

  it("never shares answers across files with identical names and row counts", async () => {
    vi.stubEnv("NEXT_PUBLIC_LLM_ENABLED", "1");
    const fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ answer: "Revenue is 600." }) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ answer: "Revenue is 1200." }) });
    vi.stubGlobal("fetch", fetch);
    const first = dataset(100), second = dataset(200);
    expect((await answerQuestionAI("total Revenue", first, profileTable(first))).answer).toContain("600");
    expect((await answerQuestionAI("total Revenue", first, profileTable(first))).answer).toContain("600");
    expect((await answerQuestionAI("total Revenue", second, profileTable(second))).answer).toContain("1200");
    expect(fetch).toHaveBeenCalledTimes(2);
  });
});
