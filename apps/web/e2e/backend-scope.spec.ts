import { test, expect } from "@playwright/test";

test("main dashboard makes the scope of server totals visible", async ({ page }) => {
  await page.route("**/api/analyze", async (route) => {
    const request = route.request().postDataJSON();
    expect(request.sourceRowCount).toBeGreaterThan(0);
    await route.fulfill({ json: {
      engine: "python", rowCount: 20, domain: { domain: "generic", confidence: 1, reason: "test" },
      columns: [], kpis: [{ id: "server-total", name: "Server revenue", value: "$100", howComputed: "sum", relevance: 1 }],
      charts: [], facts: [{ id: "f", kind: "kpi", text: "Revenue $100." }], narrative: "Revenue $100.",
      scope: { sourceRows: 9000, analyzedRows: 20, sampled: true },
    } });
  });
  await page.route("**/api/conclude", async (route) => {
    expect(route.request().postDataJSON().facts[0].text).toContain("sample");
    await route.fulfill({ json: {
      provider: "none", bottomLine: "Revenue $100.", summary: "Sample totals only.", conclusions: [],
      actions: [], chartInsights: [], grounding: { grounded: true, unverified: [] }, disclaimer: "Automated analysis.",
    } });
  });
  await page.goto("/analyze?demo=1");
  await expect(page.getByText("Server revenue", { exact: true })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("status").filter({ hasText: "Server KPIs and charts use 20 of 9,000 rows" })).toBeVisible();
});
