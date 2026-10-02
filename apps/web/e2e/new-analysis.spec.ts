import { test, expect } from "@playwright/test";

// The old column-controls workflow was removed from the app. Exercise the current reanalysis path,
// including clearing the previous dashboard rather than carrying its metrics into the next upload.
test("a new analysis replaces the previous dataset", async ({ page }) => {
  await page.goto("/analyze?demo=1");
  await expect(page.getByRole("heading", { name: "Key metrics" })).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: "New analysis", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Key metrics" })).toHaveCount(0);
  await page.setInputFiles('input[type="file"]', {
    name: "replacement.csv", mimeType: "text/csv",
    buffer: Buffer.from("Region,Revenue\nEast,10\nWest,20\nEast,30\nWest,40\nEast,50\nWest,60"),
  });
  await expect(page.getByRole("heading", { name: /Ready to analyze/ })).toBeVisible();
  await page.getByRole("button", { name: /Start analyzing/ }).click();
  await expect(page.getByRole("heading", { name: "Key metrics" })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("replacement.csv", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("sample-orders.csv", { exact: true })).toHaveCount(0);
});
