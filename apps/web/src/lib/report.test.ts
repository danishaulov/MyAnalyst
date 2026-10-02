import { describe, expect, it } from "vitest";
import { analyze } from "./analyze";
import { buildExecutiveSummary } from "./report";
import type { Table } from "./types";

function salesTable(): Table {
  const rows: Record<string, unknown>[] = [];
  const start = Date.UTC(2023, 0, 1);
  for (let i = 0; i < 40; i++) {
    rows.push({
      Date: new Date(start + i * 7 * 86400000).toISOString().slice(0, 10),
      Region: ["North", "South", "East"][i % 3],
      Spend: 500 + (i % 7) * 30,
      Units: 20 + i + (i % 5),
      Revenue: 1000 + i * 25,
    });
  }
  return { name: "sales.csv", columns: ["Date", "Region", "Spend", "Units", "Revenue"], rows, rowCount: rows.length };
}

describe("buildExecutiveSummary", () => {
  it("produces a grounded multi-paragraph summary from a real analysis", async () => {
    const spec = await analyze(salesTable(), { skipCharts: true });
    const paras = buildExecutiveSummary(spec);
    expect(paras.length).toBeGreaterThanOrEqual(2);
    // Opening paragraph states the size and quality grade.
    expect(paras[0]).toContain("40 rows");
    expect(paras[0]).toMatch(/grades [A-F]/);
    // The whole summary should be plain text (no leftover template artifacts).
    for (const p of paras) expect(p.length).toBeGreaterThan(0);
  });

  it("carries sampled scope into exported report prose", async () => {
    const spec = await analyze(salesTable(), { skipCharts: true });
    spec.computeScope = { sourceRows: 1000, analyzedRows: 40, sampled: true };
    expect(buildExecutiveSummary(spec).join(" ")).toContain("40 out of 1,000 rows");
    expect(buildExecutiveSummary(spec).join(" ")).toContain("sample only");
  });
});
