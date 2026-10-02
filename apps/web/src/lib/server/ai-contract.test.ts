import { describe, expect, it } from "vitest";
import { readAiRequest, validateAiRequest, hasSupportedNumbers } from "./ai-contract";

describe("AI request contract", () => {
  it("rejects invalid roots, tasks and malformed BYOK before calling a provider", () => {
    for (const body of [null, [], { task: "exec" }, { task: "answer", question: 123 },
      { task: "answer", question: "hi", byok: { provider: "groq", apiKey: 123 } },
      { task: "humanize", conclusions: [null] }, { kpis: [], correlations: "invalid" }]) {
      expect(() => validateAiRequest(body)).toThrow();
    }
  });

  it("accepts valid metadata and bounded client keys", () => {
    const body = { task: "answer", question: "Revenue?", grounded: "Revenue 100", byok: { provider: "groq", apiKey: "device-key" } };
    expect(validateAiRequest(body)).toBe(body);
  });

  it("bounds actual streamed bytes when Content-Length is missing", async () => {
    const req = new Request("https://example.com", { method: "POST", body: JSON.stringify({ task: "answer", question: "📊".repeat(40_000) }) });
    await expect(readAiRequest(req)).rejects.toMatchObject({ status: 413 });
  });

  it("rejects oversized arrays and invalid JSON", async () => {
    expect(() => validateAiRequest({ kpis: Array(301).fill({}) })).toThrow();
    await expect(readAiRequest(new Request("https://example.com", { method: "POST", body: "invalid" }))).rejects.toThrow("Invalid JSON");
  });

  it("does not forward raw rows or CSV embedded in metadata", () => {
    expect(() => validateAiRequest({ task: "answer", question: "hi", dataset: { rows: [{ email: "private" }] } })).toThrow("raw records");
    expect(() => validateAiRequest({ kpis: [], csv: "email\nprivate" })).toThrow("raw records");
  });
});

describe("numeric evidence gate", () => {
  const evidence = { fact: "Revenue $51.4M, margin 0.2%, 3 orders in 2024." };
  it("accepts evidence figures and equivalent scaled values", () => {
    expect(hasSupportedNumbers("Revenue $51,400,000, margin 0.2%, 3 orders in 2024.", evidence)).toBe(true);
  });
  it("rejects invented scales, currencies, tiny figures, counts and years", () => {
    for (const text of ["Revenue $51.4B.", "Revenue €51.4M.", "Margin 0.8%.", "9 orders.", "In 2028.", "Margin 51.4%."]) {
      expect(hasSupportedNumbers(text, evidence), text).toBe(false);
    }
  });
  it("does not derive arbitrary ratios or exempt small invented numbers", () => {
    expect(hasSupportedNumbers("Share 50%.", { text: "Revenue 100 and cost 200." })).toBe(false);
    expect(hasSupportedNumbers("2 customers", { count: 100 })).toBe(false);
  });
});
