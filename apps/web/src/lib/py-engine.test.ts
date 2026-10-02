import { describe, it, expect, vi, afterEach } from "vitest";
import { sampleForPayload, runPythonConclusions, runPythonAnalysis, runPythonAsk, type PyAnalysisSpec } from "./py-engine";

const MAX = 3_800_000;

function rows(n: number, cols: string[], wide = false): Record<string, unknown>[] {
  return Array.from({ length: n }, (_, i) => {
    const r: Record<string, unknown> = {};
    for (const c of cols) r[c] = wide ? `${c}-value-${i}-padding-padding-padding` : i % 1000;
    return r;
  });
}

describe("sampleForPayload (adaptive byte-budget sampling)", () => {
  it("sends all rows when the dataset is small", () => {
    const out = sampleForPayload(["a", "b"], rows(5_000, ["a", "b"]));
    expect(out.length).toBe(5_000);
  });

  it("sends far more than the old 40k cap for a narrow table that fits", () => {
    const out = sampleForPayload(["a", "b"], rows(500_000, ["a", "b"]));
    expect(out.length).toBeGreaterThan(40_000); // adaptive — narrow rows fit many
    expect(out.length).toBeLessThanOrEqual(100_000); // bounded by the hard cap
    expect(JSON.stringify(out).length).toBeLessThan(MAX); // stays under the Vercel budget
  });

  it("keeps a wide table's payload under the Vercel limit (no 413)", () => {
    const cols = Array.from({ length: 30 }, (_, i) => `col${i}`);
    const out = sampleForPayload(cols, rows(200_000, cols, true));
    expect(JSON.stringify({ columns: cols, rows: out }).length).toBeLessThan(4_500_000);
    expect(out.length).toBeGreaterThan(0);
  });

  it("preserves column order in the emitted arrays", () => {
    const out = sampleForPayload(["x", "y", "z"], [{ z: 3, x: 1, y: 2 }]);
    expect(out[0]).toEqual([1, 2, 3]);
  });

  it("measures actual UTF-8 bytes for Hebrew and emoji, including metadata", () => {
    const cols = ["שם", "תיאור"];
    const data = Array.from({ length: 15_000 }, (_, i) => ({ "שם": `לקוח ${i}`, "תיאור": "שלום 📊".repeat(40) }));
    const out = sampleForPayload(cols, data);
    expect(new TextEncoder().encode(JSON.stringify({ columns: cols, rows: out, sourceRowCount: data.length })).byteLength).toBeLessThanOrEqual(MAX);
    expect(out.length).toBeLessThan(data.length);
  });

  it("checks bytes after estimation even when probes miss unusually large cells", () => {
    const data = Array.from({ length: 10_000 }, (_, i) => ({ a: i % 100 === 1 ? "x".repeat(60_000) : "x" }));
    const out = sampleForPayload(["a"], data);
    expect(new TextEncoder().encode(JSON.stringify(out)).byteLength).toBeLessThan(MAX);
  });

  it("rejects a single oversized row and an unsupported number of columns", () => {
    expect(() => sampleForPayload(["a"], [{ a: "x".repeat(4_500_000) }])).toThrow("row exceeds");
    expect(() => sampleForPayload(Array(101).fill("a"), [{ a: 1 }])).toThrow("100 columns");
  });
});

// Transient backend failures used to silently blank the conclusions/KPIs (no retry). These pin the retry
// policy: retry the transient classes (5xx / network) a couple times; fail fast on a 4xx the retry can't fix.
const CONCLUDE_SPEC = {
  facts: [],
  kpis: [],
  chartReadings: [],
  domain: { domain: "generic", confidence: 1, reason: "" },
  narrative: "",
} as unknown as PyAnalysisSpec;
const okRes = (data: unknown) => ({ ok: true, status: 200, json: async () => data });
const failRes = (status: number, error = "boom") => ({ ok: false, status, json: async () => ({ error }) });

describe("postJson retry (via runPythonConclusions)", () => {
  afterEach(() => vi.restoreAllMocks());

  it("retries a transient 5xx, then resolves with the eventual success", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(failRes(503))
      .mockResolvedValueOnce(okRes({ provider: "none", conclusions: [] }));
    vi.stubGlobal("fetch", fetchMock);
    await expect(runPythonConclusions(CONCLUDE_SPEC)).resolves.toMatchObject({ provider: "none" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("retries a network error (rejected fetch), then resolves", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(okRes({ provider: "groq", conclusions: [] }));
    vi.stubGlobal("fetch", fetchMock);
    await expect(runPythonConclusions(CONCLUDE_SPEC)).resolves.toMatchObject({ provider: "groq" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("does NOT retry a 4xx client error — fails fast", async () => {
    const fetchMock = vi.fn().mockResolvedValue(failRes(400, "bad payload"));
    vi.stubGlobal("fetch", fetchMock);
    await expect(runPythonConclusions(CONCLUDE_SPEC)).rejects.toThrow("bad payload");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("bounds each request and does not repeat a timed out computation", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new DOMException("timeout", "TimeoutError"));
    vi.stubGlobal("fetch", fetchMock);
    await expect(runPythonConclusions(CONCLUDE_SPEC)).rejects.toThrow("timed out");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][1].signal).toBeInstanceOf(AbortSignal);
  });

  it("sends source row count and excludes untrusted facts from the ask computation", async () => {
    const fetchMock = vi.fn().mockResolvedValue(okRes({ provider: "none" }));
    vi.stubGlobal("fetch", fetchMock);
    await runPythonAnalysis(["Revenue"], [{ Revenue: 1 }]);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body).sourceRowCount).toBe(1);
    await runPythonAsk("total Revenue?", ["Revenue"], [{ Revenue: 1 }], [{ id: "x", kind: "kpi", text: "fabricated" }]);
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).not.toHaveProperty("facts");
  });
});
