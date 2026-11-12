import { readFileSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";

// Expectations come from the data files the page is served (dist/data, exported from the committed
// results) plus a few literal anchors from the published official test evaluation.
const data = (file: string) => JSON.parse(readFileSync(new URL(`../dist/data/${file}`, import.meta.url), "utf8"));
const official = data("official_test.json");
const cv = data("cv.json");
const ablation = data("ablation.json");
const conformal = data("conformal.json");
const bench = data("benchmarks.json");
const validation = data("phase1_validation.json");
const LABELS: Record<string, string> = Object.fromEntries(data("meta.json").families.map((f: { id: string; label: string }) => [f.id, f.label]));
const FILES = ["meta.json", "official_test.json", "phase1_validation.json", "cv.json", "ablation.json", "conformal.json", "benchmarks.json"];

/** Open the dashboard, fail on any uncaught exception or console error, wait until every section has rendered. */
async function open(page: Page, { allowErrors = false } = {}) {
  const problems: string[] = [];
  page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
  if (!allowErrors) page.on("console", (m) => m.type() === "error" && problems.push(`console: ${m.text()}`));
  await page.goto("./");
  await expect(page.locator("#foot")).toContainText("Generated from committed experiment artifacts");
  return problems;
}

test("loads every data file without uncaught errors @mobile", async ({ page }) => {
  const loaded = new Map<string, number>();
  page.on("response", (r) => {
    const m = r.url().match(/\/engine-rul\/data\/([\w.]+\.json)$/);
    if (m) loaded.set(m[1], r.status());
  });
  const problems = await open(page);
  expect([...loaded.keys()].sort()).toEqual([...FILES].sort());
  expect([...loaded.values()].every((s) => s === 200)).toBe(true);
  expect(problems).toEqual([]);
  await expect(page.locator("figure.chart svg").first()).toBeVisible();
  await expect(page.locator(".error")).toHaveCount(0);
  await expect(page.locator(".status")).toHaveCount(0);
});

test("headline metrics, model names and protocol labels @mobile", async ({ page }) => {
  await open(page);
  const cards = page.locator("#cards .card");
  await expect(cards).toHaveCount(4);
  await expect(cards.nth(0)).toContainText("13.26 cycles");
  await expect(cards.nth(0)).toContainText("Gradient boosting");
  await expect(cards.nth(1)).toContainText("−63.9%");
  await expect(cards.nth(1)).toContainText("36.76 → 13.26 cycles");
  await expect(cards.nth(2)).toContainText("100 engines");
  await expect(cards.nth(3)).toContainText("13 / 15");
  // three official-test cards, one clearly marked validation-only
  await expect(page.locator("#cards .badge.official")).toHaveCount(3);
  await expect(cards.nth(3).locator(".badge.validation")).toHaveText("Validation only");
  await expect(page.locator("#takeaway")).toContainText("63.9% lower");

  const rows = page.locator("#models-table tbody tr");
  await expect(rows.locator("td:first-child")).toHaveText(["constant", "life_prior", "ridge", "hgb", "lstm"].map((f) => LABELS[f]));
  await expect(page.locator("#models-table tr.best")).toContainText("Gradient boosting");
  for (const id of ["models", "engines"]) await expect(page.locator(`#${id} .section-head:not(.sub) .badge`)).toHaveText("Official test");
  for (const id of ["cv", "ablations", "uncertainty"]) await expect(page.locator(`#${id} .section-head .badge`)).toHaveText("Validation only");
  await expect(page.locator("#engines .section-head.sub .badge")).toHaveText("Validation only"); // trajectories are validation data
});

test("switches between RMSE, MAE and NASA score", async ({ page }) => {
  await open(page);
  const chart = page.locator("#models-chart svg");
  const hgb = official.families.find((f: { id: string }) => f.id === "hgb");
  await expect(chart).toContainText("Test RMSE (cycles");
  await expect(chart).toContainText(hgb.rmse.mean.toFixed(2));
  for (const [name, axis, value] of [
    ["MAE", "Test MAE (cycles", hgb.mae.mean.toFixed(2)],
    ["NASA score", "NASA asymmetric score", Math.round(hgb.nasa_score.mean).toLocaleString("en-US")],
    ["RMSE", "Test RMSE (cycles", hgb.rmse.mean.toFixed(2)],
  ]) {
    const radio = page.getByRole("radiogroup", { name: "Metric" }).getByRole("radio", { name, exact: true });
    await radio.click();
    await expect(radio).toHaveAttribute("aria-checked", "true");
    await expect(chart).toContainText(axis);
    await expect(chart).toContainText(value);
  }
  await page.getByRole("radio", { name: "NASA score" }).click();
  await expect(page.locator("#models-cap")).toContainText("penalises late predictions");
});

test("shows and hides individual models, keeping at least one", async ({ page }) => {
  await open(page);
  const picks = page.locator("#models-pick");
  const ticks = page.locator("#models-chart svg [aria-label='y-axis tick label'] text");
  await expect(ticks).toHaveCount(5);
  await picks.getByLabel("LSTM").uncheck();
  await expect(ticks).toHaveCount(4);
  await expect(ticks).not.toContainText(["LSTM"]);
  await picks.getByLabel("LSTM").check();
  await expect(ticks).toHaveCount(5);
  for (const f of ["constant", "life_prior", "ridge", "lstm"]) await picks.getByLabel(LABELS[f], { exact: true }).uncheck();
  await expect(ticks).toHaveText(["Gradient boosting"]);
  await picks.getByLabel("Gradient boosting").click(); // the last series cannot be hidden
  await expect(picks.getByLabel("Gradient boosting")).toBeChecked();
  await expect(ticks).toHaveCount(1);
});

test("selecting an engine updates its prediction details @mobile", async ({ page }) => {
  await open(page);
  const e = official.engines.find((x: { unit: number }) => x.unit === 34);
  await page.locator("#test-engine").selectOption("34");
  const detail = page.locator("#test-detail");
  await expect(detail.getByRole("heading")).toHaveText("Test engine 34");
  await expect(detail).toContainText(`${e.y_true} cycles`);
  await expect(detail).toContainText(String(e.last_cycle));
  const rows = detail.locator("tbody tr");
  await expect(rows).toHaveCount(5);
  await expect(rows.filter({ hasText: "Gradient boosting" })).toContainText(e.pred.hgb.toFixed(1));
  // the model picker recolours the scatter and relabels its axis, without changing the engine
  await page.locator("#test-model").selectOption("lstm");
  await expect(page.locator("#test-scatter svg")).toContainText("Predicted RUL, LSTM");
  await expect(detail.getByRole("heading")).toHaveText("Test engine 34");
});

test("clicking a scatter point selects that engine", async ({ page }) => {
  await open(page);
  // the engine farthest from any other in (true, predicted) space, so the nearest-point pick is unambiguous
  const pts = official.engines.map((e: { unit: number; y_true: number; pred: Record<string, number> }) => ({ unit: e.unit, t: e.y_true, p: e.pred.hgb }));
  const gap = (a: (typeof pts)[number]) => Math.min(...pts.filter((b: typeof a) => b !== a).map((b: typeof a) => Math.hypot(a.t - b.t, a.p - b.p)));
  const index = pts.reduce((best: number, a: typeof pts[number], i: number) => (gap(a) > gap(pts[best]) ? i : best), 0);
  const target = pts[index];
  expect(target.unit).not.toBe(official.engines[0].unit); // the default selection
  const dot = page.locator("#test-scatter svg g[aria-label='dot']").first().locator("circle").nth(index);
  await dot.scrollIntoViewIfNeeded();
  const box = (await dot.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await expect(page.locator("#test-scatter")).toContainText(`Engine ${target.unit}`); // tooltip
  await page.mouse.down();
  await page.mouse.up();
  await expect(page.locator("#test-detail h3")).toHaveText(`Test engine ${target.unit}`);
  await expect(page.locator("#test-engine")).toHaveValue(String(target.unit));
});

test("switches validation-engine trajectories and their models @mobile", async ({ page }) => {
  await open(page);
  const chart = page.locator("#traj-chart");
  await expect(chart).toContainText("training cap 125");
  const lines = chart.locator("g[aria-label='line'] path");
  const before = await chart.innerHTML();
  const second = validation.trajectories[1];
  await page.locator("#val-engine").selectOption(String(second.unit));
  await expect(page.locator("#val-engine")).toHaveValue(String(second.unit));
  await expect.poll(() => chart.innerHTML()).not.toBe(before);
  const n = await lines.count();
  await page.locator("#traj-pick").getByLabel("Constant mean").check();
  await expect(lines).toHaveCount(n + 1);
  await expect(page.locator("#traj-note")).toContainText("not test results");
});

test("cross-validation, ablation, uncertainty and benchmark controls", async ({ page }) => {
  await open(page);
  // CV: the table follows the selected cut-point strategy
  const late = cv.strategies.find((s: { id: string }) => s.id === "late");
  await page.locator("#cv-strategy").selectOption("late");
  for (const f of late.families) {
    await expect(page.locator("#cv-table tbody tr").filter({ hasText: LABELS[f.id] })).toContainText(f.pooled_rmse.toFixed(2));
  }
  await expect(page.locator("#cv-folds-cap")).toContainText(late.label);

  // Ablations: one model at a time, full-model RMSE in the caption
  for (const fam of ["ridge", "lstm", "hgb"]) {
    await page.getByRole("radiogroup", { name: "Model" }).getByRole("radio", { name: LABELS[fam] }).click();
    const base = ablation.rows.find((r: { family: string; variant: string }) => r.family === fam && r.variant === "base");
    await expect(page.locator("#ab-cap")).toContainText(`${LABELS[fam]}: full-model pooled RMSE ${base.pooled_rmse.toFixed(2)}`);
  }

  // Uncertainty: method, shift check and nominal level
  const note = page.locator("#unc-method-note");
  await expect(note).toContainText("Finite-sample guarantee");
  await page.locator("#unc-method").selectOption("global_pooled");
  await expect(note).toContainText("Empirical coverage only");
  await page.locator("#unc-strategy").selectOption("late");
  await expect(note).toContainText("Shift check");
  for (const level of conformal.levels) {
    const pct = `${Math.round(level * 100)}%`;
    await page.getByRole("radiogroup", { name: "Nominal level" }).getByRole("radio", { name: pct }).click();
    await expect(page.locator("#unc-stage-cap")).toContainText(`at the ${pct} level`);
  }

  // Benchmarks: scope switches the table, statistic switches the axis
  const modelRows = bench.rows.filter((r: { scope: string }) => r.scope === "model");
  const maxN = Math.max(...modelRows.map((r: { n: number }) => r.n));
  await page.getByRole("radio", { name: "Model only" }).click();
  await expect(page.locator("#bench-table thead")).toContainText(`Batch (${modelRows[0].unit})`);
  await expect(page.locator("#bench-table tbody tr")).toHaveCount(modelRows.filter((r: { n: number }) => r.n === maxN).length);
  await page.getByRole("radio", { name: "p95" }).click();
  await expect(page.locator("#bench-chart")).toContainText("p95 latency per call");
});

for (const [width, height] of [
  [375, 812],
  [768, 1024],
  [1440, 900],
]) {
  test(`no horizontal overflow or overlapping controls at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    await open(page);
    const layout = await page.evaluate(() => {
      const doc = document.documentElement;
      const vw = doc.clientWidth;
      // anything wider than the viewport must sit inside a horizontally scrollable table wrapper
      const escaped = [...document.querySelectorAll("main *")]
        .filter((e) => !e.closest(".table-wrap, .top nav, pre"))
        .filter((e) => e.getBoundingClientRect().right > vw + 1)
        .map((e) => `${e.tagName.toLowerCase()}#${e.id}.${e.getAttribute("class")}`);
      const overlaps: string[] = [];
      document.querySelectorAll(".controls").forEach((c) => {
        const kids = [...c.children].map((k) => k.getBoundingClientRect());
        kids.forEach((a, i) =>
          kids.slice(i + 1).forEach((b) => {
            if (a.left < b.right - 1 && b.left < a.right - 1 && a.top < b.bottom - 1 && b.top < a.bottom - 1) overlaps.push(c.parentElement!.id);
          }),
        );
      });
      // plot SVGs only (colour legends add small swatch SVGs)
      const charts = [...document.querySelectorAll("figure.chart svg")].map((s) => s.getBoundingClientRect()).filter((r) => r.height > 100).map((r) => r.width);
      return { scroll: doc.scrollWidth - vw, escaped, overlaps, charts };
    });
    expect(layout.scroll).toBeLessThanOrEqual(0);
    expect(layout.escaped).toEqual([]);
    expect(layout.overlaps).toEqual([]);
    expect(Math.min(...layout.charts)).toBeGreaterThan(200); // charts resized, not collapsed
    expect(Math.max(...layout.charts)).toBeLessThanOrEqual(width);
    await expect(page.getByRole("navigation", { name: "Dashboard sections" })).toBeVisible();
  });
}

test("navigation links and keyboard-accessible controls @mobile", async ({ page }) => {
  await open(page);
  // skip link is the first tab stop and moves focus past the header
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Skip to content" })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/#main$/);

  for (const [name, id] of [
    ["Models", "models"],
    ["Engines", "engines"],
    ["Cross-validation", "cv"],
    ["Ablations", "ablations"],
    ["Uncertainty", "uncertainty"],
    ["Benchmarks", "benchmarks"],
    ["Limitations", "limitations"],
  ]) {
    await page.getByRole("navigation", { name: "Dashboard sections" }).getByRole("link", { name, exact: true }).click();
    await expect(page).toHaveURL(new RegExp(`#${id}$`));
    await expect(page.locator(`#${id} h2`)).toBeInViewport();
  }
  await expect(page.getByRole("link", { name: "GitHub", exact: true })).toHaveAttribute("href", "https://github.com/Eliot-2006/engine-rul");

  // radio groups: one tab stop, arrow keys move and select
  const group = page.getByRole("radiogroup", { name: "Metric" });
  await expect(group.locator("[tabindex='0']")).toHaveCount(1);
  await group.getByRole("radio", { name: "RMSE" }).focus();
  await page.keyboard.press("ArrowRight");
  await expect(group.getByRole("radio", { name: "MAE" })).toBeFocused();
  await expect(group.getByRole("radio", { name: "MAE" })).toHaveAttribute("aria-checked", "true");
  await expect(page.locator("#models-chart svg")).toContainText("Test MAE");
  await page.keyboard.press("ArrowLeft");
  await page.keyboard.press("ArrowLeft");
  await expect(group.getByRole("radio", { name: "NASA score" })).toHaveAttribute("aria-checked", "true");

  // checkboxes toggle with Space
  const lstm = page.locator("#models-pick").getByLabel("LSTM");
  await lstm.focus();
  await page.keyboard.press("Space");
  await expect(lstm).not.toBeChecked();

  // every chart and form control has an accessible name
  for (const fig of await page.locator("figure.chart").all()) await expect(fig).toHaveAttribute("aria-label", /\w/);
  for (const sel of await page.locator("select").all()) await expect(sel).toHaveAccessibleName(/\w/);
});

test("respects reduced-motion preferences", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await open(page);
  expect(await page.evaluate(() => getComputedStyle(document.documentElement).scrollBehavior)).toBe("auto");
});

test("a missing data file gives a section error, not a blank or broken page", async ({ page }) => {
  await page.route("**/data/cv.json", (r) => r.fulfill({ status: 404, body: "not found" }));
  const problems = await open(page, { allowErrors: true });
  expect(problems).toEqual([]); // no uncaught exception
  await expect(page.locator("#cv-folds .error")).toContainText("could not load cv.json (HTTP 404)");
  await expect(page.locator("#cv .error").first()).toBeVisible();
  await expect(page.locator("#cv .status")).toHaveCount(0); // no chart left on "Loading…"
  await expect(page.locator("#cards .card")).toHaveCount(3); // official-test cards still render
  await expect(page.locator("#models-chart svg")).toBeVisible();
});

test("data with the wrong protocol is refused", async ({ page }) => {
  // official test results served in place of validation data must not be drawn as validation results
  await page.route("**/data/conformal.json", (r) => r.fulfill({ json: official }));
  await open(page, { allowErrors: true });
  await expect(page.locator("#unc-coverage .error")).toContainText('expected protocol "nested_cv_conformal", got "official_test"');
});

test("malformed core data shows a clear error everywhere @mobile", async ({ page }) => {
  await page.route("**/data/official_test.json", (r) => r.fulfill({ status: 200, contentType: "application/json", body: "{not json" }));
  const problems: string[] = [];
  page.on("pageerror", (e) => problems.push(e.message));
  await page.goto("./");
  await expect(page.locator("#cards .error")).toContainText("Could not load results");
  await expect(page.locator("figure.chart > .status")).toHaveCount(0);
  await expect(page.locator("#models-chart .error")).toContainText("core results could not be loaded");
  expect(problems).toEqual([]);
});
