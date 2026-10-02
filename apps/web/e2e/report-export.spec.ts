import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";

test("report and deck export valid PDFs after the library upgrade", async ({ page }, testInfo) => {
  await page.goto("/analyze?demo=1");
  await expect(page.getByRole("heading", { name: "Key metrics" })).toBeVisible({ timeout: 30_000 });
  for (const name of ["Report", "Deck"]) {
    const pending = page.waitForEvent("download");
    await page.getByRole("button", { name: new RegExp(name) }).click();
    const download = await pending;
    expect(download.suggestedFilename()).toMatch(/\.pdf$/);
    const path = testInfo.outputPath(`${name}.pdf`);
    await download.saveAs(path);
    const data = await readFile(path);
    expect(data.subarray(0, 5).toString()).toBe("%PDF-");
    expect(data.length).toBeGreaterThan(1000);
  }
});
