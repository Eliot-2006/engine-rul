import * as Plot from "@observablehq/plot";
import "./style.css";
import {
  load,
  type Ablation,
  type Benchmarks,
  type Conformal,
  type CV,
  type FamilyId,
  type Interval,
  type Meta,
  type OfficialTest,
  type Paired,
  type Phase1Validation,
} from "./data";

// ---------------------------------------------------------------- helpers

const ORDER: FamilyId[] = ["constant", "life_prior", "ridge", "hgb", "lstm"];
const LEARNED: FamilyId[] = ["ridge", "hgb", "lstm"];
// Narrow screens: forest-plot row labels use these.
const SHORT: Record<FamilyId, string> = {
  constant: "Constant",
  life_prior: "Life prior",
  ridge: "Ridge",
  hgb: "Grad. boost.",
  lstm: "LSTM",
};
const LABEL: Record<FamilyId, string> = {
  constant: "Constant mean",
  life_prior: "Life prior (cycle only)",
  ridge: "Ridge",
  hgb: "Gradient boosting",
  lstm: "LSTM",
};

const $ = <T extends HTMLElement = HTMLElement>(id: string) => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
};
const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const color = (f: FamilyId) => css(`--c-${f}`);
const f2 = (x: number) => x.toFixed(2);
const f1 = (x: number) => x.toFixed(1);
const ci = ([a, b]: Interval, d = 2) => `[${a.toFixed(d)}, ${b.toFixed(d)}]`;
const pct = (x: number, d = 0) => `${(100 * x).toFixed(d)}%`;
const signed = (x: number, d = 2) => `${x > 0 ? "+" : x < 0 ? "−" : "±"}${Math.abs(x).toFixed(d)}`;
const int = (x: number) => Math.round(x).toLocaleString("en-US");
const short = (sha: string | null) => (sha ? sha.slice(0, 7) : "unknown");
function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Record<string, unknown> = {},
  ...kids: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  Object.assign(e, props);
  e.append(...kids);
  return e;
}
const swatch = (f: FamilyId) => {
  const s = el("span", { className: "swatch" });
  s.style.background = color(f);
  return s;
};
const famScale = (fams: FamilyId[]) => ({ domain: fams.map((f) => LABEL[f]), range: fams.map(color) });

function baseStyle(width: number, height: number): Plot.PlotOptions {
  return {
    width,
    height,
    marginBottom: 46,
    style: { background: "transparent", color: css("--ink-2"), fontSize: "12px", fontFamily: css("--sans") },
  };
}

// Charts redraw on resize, colour-scheme change and control changes.
const redraws: (() => void)[] = [];
function chart(target: HTMLElement, make: (width: number) => Element): () => void {
  const draw = () => target.replaceChildren(make(Math.max(280, target.clientWidth)));
  redraws.push(draw);
  draw();
  return draw;
}
let lastWidth = window.innerWidth;
let frame = 0;
window.addEventListener("resize", () => {
  if (window.innerWidth === lastWidth) return;
  lastWidth = window.innerWidth;
  cancelAnimationFrame(frame);
  frame = requestAnimationFrame(() => redraws.forEach((d) => d()));
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => redraws.forEach((d) => d()));

function seg<T extends string | number>(host: HTMLElement, options: [T, string][], value: T, onChange: (v: T) => void) {
  host.replaceChildren(
    ...options.map(([v, label]) => {
      const b = el("button", { type: "button", textContent: label });
      b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", String(v === value));
      b.addEventListener("click", () => {
        host.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
        onChange(v);
      });
      return b;
    }),
  );
}

function fillSelect(sel: HTMLSelectElement, options: [string, string][], value: string, onChange: (v: string) => void) {
  sel.replaceChildren(...options.map(([v, label]) => el("option", { value: v, textContent: label, selected: v === value })));
  sel.onchange = () => onChange(sel.value);
}

function familyChecks(host: HTMLElement, fams: FamilyId[], selected: Set<FamilyId>, onChange: () => void) {
  host.replaceChildren(
    ...fams.map((f) => {
      const box = el("input", { type: "checkbox", checked: selected.has(f) });
      box.addEventListener("change", () => {
        if (box.checked) selected.add(f);
        else if (selected.size > 1) selected.delete(f);
        else box.checked = true; // keep at least one series
        onChange();
      });
      return el("label", {}, box, swatch(f), LABEL[f]);
    }),
  );
}

function table(host: HTMLElement, head: string[], rows: (string | Node)[][], best?: number) {
  const t = el(
    "table",
    {},
    el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { scope: "col", textContent: h })))),
    el(
      "tbody",
      {},
      ...rows.map((r, i) => el("tr", { className: i === best ? "best" : "" }, ...r.map((c) => el("td", {}, c)))),
    ),
  );
  host.replaceChildren(t);
}
const famCell = (f: FamilyId) => el("span", {}, swatch(f), LABEL[f]);

async function guard(host: HTMLElement, fn: () => Promise<void> | void) {
  try {
    await fn();
  } catch (e) {
    host.replaceChildren(el("p", { className: "error", textContent: `Could not render this section: ${(e as Error).message}` }));
    console.error(e);
  }
}

/** Forest plot: one row per comparison, point estimate with a 95% CI, reference line at 0. */
function forest(
  width: number,
  input: { label: string; short: string; delta: number; ci: Interval; clear: boolean; tip: string }[],
  xLabel: string,
) {
  const rows = input.map((r) => ({ ...r, label: width < 600 ? r.short : r.label }));
  const lo = Math.min(0, ...rows.map((r) => r.ci[0]));
  const hi = Math.max(0, ...rows.map((r) => r.ci[1]));
  const pad = (hi - lo) * 0.12 || 1;
  return Plot.plot({
    ...baseStyle(width, 46 + rows.length * 34),
    marginLeft: Math.min(230, 12 + 7 * Math.max(...rows.map((r) => r.label.length))),
    marginRight: 48,
    x: { labelOffset: 40, label: xLabel, domain: [lo - pad, hi + pad], grid: true },
    y: { label: null, domain: rows.map((r) => r.label) },
    marks: [
      Plot.ruleX([0], { stroke: css("--ink-2"), strokeDasharray: "4,3" }),
      Plot.ruleY(rows, { y: "label", x1: (r) => r.ci[0], x2: (r) => r.ci[1], stroke: css("--ink"), strokeWidth: 2 }),
      Plot.dot(rows, {
        y: "label",
        x: "delta",
        r: 5,
        fill: (r) => (r.clear ? css("--ink") : css("--surface")),
        stroke: css("--ink"),
        strokeWidth: 2,
        title: "tip",
        tip: true,
      }),
      Plot.text(rows, { y: "label", x: (r) => Math.max(r.ci[1], 0), text: (r) => signed(r.delta), dx: 8, textAnchor: "start", fill: css("--ink") }),
    ],
  });
}

// ---------------------------------------------------------------- A. overview

function overview(meta: Meta, test: OfficialTest, cv: CV | null) {
  const h = test.headline;
  const best = test.families.find((f) => f.id === h.best_family)!;
  const card = (k: string, badge: "official" | "validation", v: string, d: string) =>
    el(
      "div",
      { className: "card" },
      el("span", { className: `badge ${badge}`, textContent: badge === "official" ? "Official test" : "Validation only" }),
      el("div", { className: "k", textContent: k }),
      el("div", { className: "v", innerHTML: v }),
      el("div", { className: "d", textContent: d }),
    );
  const cards = [
    card("Best test RMSE", "official", `${f2(h.best_rmse)} <small>cycles</small>`, `${LABEL[h.best_family]}, 95% CI ${ci(best.rmse_ci95, 1)}`),
    card(
      "RMSE reduction vs cycle-only baseline",
      "official",
      `−${h.rmse_reduction_pct.toFixed(1)}%`,
      `${f2(h.baseline_rmse)} → ${f2(h.best_rmse)} cycles; the baseline predicts from the current cycle alone`,
    ),
    card("Evaluation", "official", `${test.n_engines} <small>engines</small>`, `${test.dataset}, official test split, one locked evaluation`),
  ];
  if (cv) {
    const prim = cv.strategies.find((s) => s.id === cv.primary_strategy)!;
    const top = [...prim.families].sort((a, b) => b.rank1_frac - a.rank1_frac)[0];
    cards.push(
      card(
        "Best model across nested-CV folds",
        "validation",
        `${Math.round(top.rank1_frac * cv.n_outer_folds)} / ${cv.n_outer_folds} <small>folds</small>`,
        `${LABEL[top.id]} had the lowest RMSE in ${Math.round(top.rank1_frac * cv.n_outer_folds)} of ${cv.n_outer_folds} outer folds`,
      ),
    );
  }
  $("cards").replaceChildren(...cards);
  $("disclaimer").textContent = meta.disclaimer;
}

// ---------------------------------------------------------------- B. model comparison

function models(test: OfficialTest) {
  type Metric = "rmse" | "mae" | "nasa_score";
  const metricLabel: Record<Metric, string> = { rmse: "RMSE", mae: "MAE", nasa_score: "NASA score" };
  let metric: Metric = "rmse";
  const selected = new Set<FamilyId>(ORDER);
  const fams = ORDER.filter((f) => test.families.some((r) => r.id === f));
  const byId = Object.fromEntries(test.families.map((r) => [r.id, r])) as Record<FamilyId, OfficialTest["families"][number]>;
  const seeds = (f: FamilyId) => (byId[f].seeds_identical ? `${byId[f].seeds} (identical)` : String(byId[f].seeds));
  $("models-note").textContent =
    `${test.split}. Target: ${test.target}. Bars show the mean over seeds; whiskers are the ${test.ci_method}. Evaluated ${test.evaluated_at.slice(0, 10)}.`;

  const draw = chart($("models-chart"), (w) => {
    const rows = fams
      .filter((f) => selected.has(f))
      .map((f) => {
        const r = byId[f];
        const m = r[metric];
        const interval = metric === "rmse" ? r.rmse_ci95 : metric === "mae" ? r.mae_ci95 : null;
        return {
          f,
          label: LABEL[f],
          v: m.mean,
          lo: interval?.[0],
          hi: interval?.[1],
          tip: [
            LABEL[f],
            `${metricLabel[metric]}: ${metric === "nasa_score" ? int(m.mean) : f2(m.mean)}${m.std ? ` ± ${metric === "nasa_score" ? int(m.std) : f2(m.std)} (seeds)` : ""}`,
            interval ? `95% CI (median seed): ${ci(interval)}` : "No CI computed for the NASA score",
            `Seeds: ${seeds(f)}`,
          ].join("\n"),
        };
      })
      .sort((a, b) => a.v - b.v);
    const max = Math.max(...rows.map((r) => r.hi ?? r.v));
    return Plot.plot({
      ...baseStyle(w, 40 + rows.length * 44),
      marginLeft: 160,
      marginRight: 64,
      x: {
        label: metric === "nasa_score" ? "NASA asymmetric score (sum over 100 engines; lower is better) →" : `Test ${metricLabel[metric]} (cycles; lower is better) →`,
        grid: true,
        domain: [0, max * 1.08],
      },
      y: { label: null, domain: rows.map((r) => r.label), padding: 0.35 },
      color: famScale(fams),
      marks: [
        Plot.barX(rows, { y: "label", x: "v", fill: "label", rx: 3, title: "tip", tip: true }),
        Plot.ruleY(rows.filter((r) => r.lo !== undefined), { y: "label", x1: "lo", x2: "hi", stroke: css("--ink"), strokeWidth: 1.5 }),
        Plot.text(rows, {
          y: "label",
          x: (r) => r.hi ?? r.v,
          text: (r) => (metric === "nasa_score" ? int(r.v) : f2(r.v)),
          dx: 6,
          textAnchor: "start",
          fill: css("--ink"),
        }),
        Plot.ruleX([0], { stroke: css("--axis") }),
      ],
    });
  });
  seg($("metric"), [["rmse", "RMSE"], ["mae", "MAE"], ["nasa_score", "NASA score"]] as [Metric, string][], metric, (v) => {
    metric = v;
    $("models-cap").textContent = capFor(v);
    draw();
  });
  familyChecks($("models-pick"), fams, selected, draw);
  const capFor = (m: Metric) =>
    m === "nasa_score"
      ? "The NASA score penalises late predictions (more life predicted than remains) exponentially harder than early ones; it is a sum, so one bad engine can dominate."
      : "Root-mean-square and mean absolute error between predicted and true remaining cycles, one prediction per test engine.";
  $("models-cap").textContent = capFor(metric);

  const bestIdx = [...fams].sort((a, b) => byId[a].rmse.mean - byId[b].rmse.mean)[0];
  table(
    $("models-table"),
    ["Model", "RMSE", "RMSE 95% CI", "MAE", "NASA score", "Bias", "Late predictions", "Seeds"],
    fams.map((f) => {
      const r = byId[f];
      const ms = (m: { mean: number; std: number }, fmt: (x: number) => string) => (m.std ? `${fmt(m.mean)} ± ${fmt(m.std)}` : fmt(m.mean));
      return [famCell(f), ms(r.rmse, f2), ci(r.rmse_ci95, 1), ms(r.mae, f2), ms(r.nasa_score, int), signed(r.bias.mean), pct(r.late_fraction), seeds(f)];
    }),
    fams.indexOf(bestIdx),
  );

  chart($("paired-chart"), (w) =>
    forest(
      w,
      test.paired.map((p: Paired) => ({
        label: `${LABEL[p.a]} − ${LABEL[p.b]}`,
        short: `${SHORT[p.a]} − ${SHORT[p.b]}`,
        delta: p.delta_rmse,
        ci: p.ci95,
        clear: p.ci_excludes_0,
        tip: `${LABEL[p.a]} − ${LABEL[p.b]}\nΔ RMSE ${signed(p.delta_rmse)} cycles\n95% CI ${ci(p.ci95)}\n${p.ci_excludes_0 ? "CI excludes 0: clear difference" : "CI includes 0: no clear difference"}`,
      })),
      "Δ test RMSE (cycles)",
    ),
  );
  const pair = test.paired.find((p) => new Set([p.a, p.b]).size === 2 && [p.a, p.b].includes("hgb") && [p.a, p.b].includes("lstm"));
  if (pair) {
    const d = pair.a === "lstm" ? pair.delta_rmse : -pair.delta_rmse;
    const c: Interval = pair.a === "lstm" ? pair.ci95 : [-pair.ci95[1], -pair.ci95[0]];
    $("why").textContent =
      `On the official test set the LSTM's RMSE was ${f2(d)} cycles ${d >= 0 ? "higher" : "lower"} than gradient boosting's (paired 95% CI ${ci(c)}` +
      `${pair.ci_excludes_0 ? ", which excludes 0" : ", which includes 0"}). Likely reasons:`;
  }
}

// ---------------------------------------------------------------- C. engines

function engines(test: OfficialTest, val: Phase1Validation | null) {
  const fams = ORDER.filter((f) => f in test.engines[0].pred);
  let unit = test.engines[0].unit;
  let model: FamilyId = test.headline.best_family;
  const sel = $<HTMLSelectElement>("test-engine");
  const byUnit = new Map(test.engines.map((e) => [e.unit, e]));

  const detail = () => {
    const e = byUnit.get(unit)!;
    const rows = fams.map((f) => {
      const err = e.pred[f] - e.y_true;
      return [famCell(f), f1(e.pred[f]), `${signed(err, 1)} ${err > 0 ? "late" : err < 0 ? "early" : ""}`.trim()];
    });
    const host = $("test-detail");
    const tableHost = el("div", { className: "table-wrap" });
    host.replaceChildren(
      el("h3", { textContent: `Test engine ${e.unit}` }),
      el(
        "dl",
        {},
        el("dt", { textContent: "Last observed cycle" }),
        el("dd", { textContent: String(e.last_cycle) }),
        el("dt", { textContent: "True RUL (official label)" }),
        el("dd", { textContent: `${e.y_true} cycles` }),
      ),
      tableHost,
      el("p", {
        className: "cap",
        textContent: "Error = predicted − true. Late (positive) means more life was predicted than remained: the dangerous direction.",
      }),
    );
    table(tableHost, ["Model", "Predicted", "Error"], rows);
  };

  const draw = chart($("test-scatter"), (w) => {
    const pts = test.engines.map((e) => ({ unit: e.unit, t: e.y_true, p: e.pred[model] }));
    const cur = pts.filter((d) => d.unit === unit);
    const lim = Math.ceil(Math.max(...pts.flatMap((d) => [d.t, d.p])) / 25) * 25;
    const size = Math.min(w, 520);
    const plot = Plot.plot({
      ...baseStyle(size, size),
      marginLeft: 48,
      x: { labelOffset: 40, label: "True RUL (cycles) →", domain: [0, lim], grid: true },
      y: { label: `↑ Predicted RUL, ${LABEL[model]}`, domain: [0, lim], grid: true },
      marks: [
        Plot.line([[0, 0], [lim, lim]], { stroke: css("--ink-2"), strokeDasharray: "4,3" }),
        Plot.dot(pts, {
          x: "t",
          y: "p",
          r: 4.5,
          fill: color(model),
          fillOpacity: 0.85,
          stroke: css("--surface"),
          strokeWidth: 1,
          title: (d) => `Engine ${d.unit}\nTrue ${d.t} · predicted ${f1(d.p)}\nError ${signed(d.p - d.t, 1)} cycles\nClick to inspect`,
          tip: true,
        }),
        Plot.dot(cur, { x: "t", y: "p", r: 9, stroke: css("--ink"), strokeWidth: 2.5 }),
      ],
    }) as SVGSVGElement & { value?: { unit: number } };
    plot.style.cursor = "pointer";
    plot.addEventListener("click", () => {
      if (plot.value) {
        unit = plot.value.unit;
        sel.value = String(unit);
        update();
      }
    });
    return plot;
  });
  const update = () => {
    detail();
    draw();
  };
  fillSelect(
    sel,
    test.engines.map((e) => [String(e.unit), `Engine ${e.unit} (true RUL ${e.y_true})`]),
    String(unit),
    (v) => {
      unit = Number(v);
      update();
    },
  );
  fillSelect(
    $<HTMLSelectElement>("test-model"),
    fams.map((f) => [f, LABEL[f]]),
    model,
    (v) => {
      model = v as FamilyId;
      draw();
    },
  );
  detail();

  if (!val) return;
  const tfams = ORDER.filter((f) => f in val.trajectories[0].pred);
  const show = new Set<FamilyId>(LEARNED.filter((f) => tfams.includes(f)));
  let vunit = val.trajectories[0].unit;
  $("traj-note").textContent =
    `${val.split}. ${val.target}. The curves show the frozen median-seed models predicting at every cycle of an engine's life; ` +
    `they are validation data used for Phase 1 model selection, not test results.`;
  const drawTraj = chart($("traj-chart"), (w) => {
    const t = val.trajectories.find((x) => x.unit === vunit)!;
    const fams = tfams.filter((f) => show.has(f));
    const rows = fams.flatMap((f) => t.cycle.map((c, i) => ({ f: LABEL[f], c, p: t.pred[f][i] })));
    const truth = t.cycle.map((c, i) => ({ c, y: t.y_true[i], i }));
    const tip = (d: { c: number; y: number; i: number }) =>
      [`Cycle ${d.c} · true RUL ${d.y}`, ...fams.map((f) => `${LABEL[f]}: ${f1(t.pred[f][d.i])} (error ${signed(t.pred[f][d.i] - d.y, 1)})`)].join("\n");
    return Plot.plot({
      ...baseStyle(w, Math.min(420, Math.max(300, w * 0.45))),
      marginLeft: 48,
      x: { labelOffset: 40, label: "Engine cycle →", grid: true },
      y: { label: "↑ Remaining useful life (cycles)", grid: true, domain: [0, Math.max(...t.y_true, 130) + 5] },
      color: famScale(fams), // the checkboxes above carry the legend
      marks: [
        Plot.ruleY([125], { stroke: css("--ink-2"), strokeDasharray: "2,3" }),
        Plot.text([125], { y: (d) => d, x: t.cycle[t.cycle.length - 1], text: () => "training cap 125", dy: -7, textAnchor: "end", fill: css("--ink-2") }),
        Plot.line(truth, { x: "c", y: "y", stroke: css("--truth"), strokeWidth: 2.5 }),
        Plot.line(rows, { x: "c", y: "p", stroke: "f", strokeWidth: 2 }),
        Plot.ruleX(truth, Plot.pointerX({ x: "c", stroke: css("--axis") })),
        Plot.tip(truth, Plot.pointerX({ x: "c", y: "y", title: tip })),
      ],
    });
  });
  fillSelect(
    $<HTMLSelectElement>("val-engine"),
    val.trajectories.map((t) => [String(t.unit), `Engine ${t.unit} (failed at cycle ${t.failure_cycle})`]),
    String(vunit),
    (v) => {
      vunit = Number(v);
      drawTraj();
    },
  );
  familyChecks($("traj-pick"), tfams, show, drawTraj);
}

// ---------------------------------------------------------------- D. cross-validation

function crossValidation(cv: CV) {
  let sid = cv.primary_strategy;
  $("cv-intro").textContent = `${cv.split}. ${cv.endpoints}. Target: ${cv.target}.`;
  const strategy = () => cv.strategies.find((s) => s.id === sid)!;

  const drawFolds = chart($("cv-folds"), (w) => {
    const st = strategy();
    const fams = LEARNED.filter((f) => st.families.some((x) => x.id === f));
    const rows = st.folds
      .filter((r) => fams.includes(r.family))
      .map((r) => ({ ...r, label: LABEL[r.family], key: `R${r.repeat + 1}·F${r.fold + 1}` }));
    const keys = [...new Set(rows.map((r) => r.key))];
    return Plot.plot({
      ...baseStyle(w, 320),
      marginLeft: 44,
      marginBottom: 64,
      x: { labelOffset: w < 640 ? 58 : 40, label: "Outer fold (repeat · fold) →", domain: keys, tickRotate: w < 640 ? -50 : 0 },
      y: { label: "↑ RMSE (cycles, capped truth)", grid: true, zero: false },
      color: { ...famScale(fams), legend: true },
      marks: [
        Plot.line(rows, { x: "key", y: "rmse", stroke: "label", z: (r) => `${r.label}${r.repeat}`, strokeOpacity: 0.5, strokeWidth: 1.5 }),
        Plot.dot(rows, {
          x: "key",
          y: "rmse",
          fill: "label",
          r: 4.5,
          stroke: css("--surface"),
          title: (r) => `${r.label}\nRepeat ${r.repeat + 1}, fold ${r.fold + 1}\nRMSE ${f2(r.rmse)} · MAE ${f2(r.mae)}`,
          tip: true,
        }),
      ],
    });
  });

  const drawPaired = chart($("cv-paired"), (w) =>
    forest(
      w,
      strategy()
        .paired.filter((p) => LEARNED.includes(p.a) && LEARNED.includes(p.b))
        .map((p) => ({
          label: `${LABEL[p.a]} − ${LABEL[p.b]}`,
          short: `${SHORT[p.a]} − ${SHORT[p.b]}`,
          delta: p.delta_rmse,
          ci: p.ci95,
          clear: p.ci_excludes_0,
          tip: `${LABEL[p.a]} − ${LABEL[p.b]}\nΔ pooled RMSE ${signed(p.delta_rmse)}\n95% CI ${ci(p.ci95)}`,
        })),
      "Δ pooled CV RMSE (cycles)",
    ),
  );

  const rest = () => {
    const st = strategy();
    const fams = ORDER.filter((f) => st.families.some((x) => x.id === f));
    const by = Object.fromEntries(st.families.map((x) => [x.id, x])) as Record<FamilyId, CV["strategies"][number]["families"][number]>;
    const best = fams.reduce((a, b) => (by[a].pooled_rmse < by[b].pooled_rmse ? a : b));
    table(
      $("cv-table"),
      ["Model", "Pooled RMSE", "Approx. 95% CI", `Fold mean ± sd (${cv.n_outer_folds})`, "Pooled MAE", "Mean rank", "Best in folds", "Bootstrap P(best)"],
      fams.map((f) => {
        const r = by[f];
        return [famCell(f), f2(r.pooled_rmse), ci(r.pooled_rmse_ci95), `${f2(r.fold_rmse_mean)} ± ${f2(r.fold_rmse_std)}`, f2(r.pooled_mae), r.mean_rank.toFixed(2), `${Math.round(r.rank1_frac * cv.n_outer_folds)} / ${cv.n_outer_folds}`, pct(r.bootstrap_rank1_frac, 1)];
      }),
      fams.indexOf(best),
    );
    $("cv-folds-cap").textContent =
      `${st.label}: ${st.n_points.toLocaleString("en-US")} held-out cut points per model. ` +
      `${LABEL[best]} has the lowest pooled RMSE and was best in ${Math.round(by[best].rank1_frac * cv.n_outer_folds)} of ${cv.n_outer_folds} outer folds. ` +
      `Baselines (RMSE > 30) are in the table. Pooled RMSE is computed over all held-out predictions; the fold mean averages the ${cv.n_outer_folds} per-fold RMSEs.`;
    $("cv-paired-cap").textContent = `Paired engine-bootstrap 95% CIs on identical cut points; ${cv.ci_method}.`;
    const taus = cv.rank_agreement.map((x) => x.kendall_tau);
    const agree = Object.entries(cv.selection_agreement_with_primary).filter(([s]) => s !== cv.primary_strategy).flatMap(([, v]) => Object.values(v));
    $("cv-stability").textContent =
      `The model ranking is identical whichever cut points are used for scoring (Kendall τ ${Math.min(...taus) === Math.max(...taus) ? `= ${taus[0].toFixed(2)} for every pair of strategies` : `${Math.min(...taus).toFixed(2)}–${Math.max(...taus).toFixed(2)}`}). ` +
      `The selected hyperparameters are much less stable: re-selecting on late-life or stage-stratified cut points picks the same configuration as uniform cut points in only ${pct(Math.min(...agree))}–${pct(Math.max(...agree))} of folds, i.e. validation error is nearly flat across the grids. ` +
      `Fold-to-fold spread (sd ≈ ${f2(by[best].fold_rmse_std)} cycles for ${LABEL[best]}) is comparable to the gap between the two best models, which is why the paired comparison on identical engines is the more informative test.`;
  };
  rest();
  fillSelect(
    $<HTMLSelectElement>("cv-strategy"),
    cv.strategies.map((s) => [s.id, s.id === cv.primary_strategy ? `${s.label} (selection)` : s.label]),
    sid,
    (v) => {
      sid = v;
      drawFolds();
      drawPaired();
      rest();
    },
  );
}

// ---------------------------------------------------------------- E. ablations

function ablations(ab: Ablation) {
  const fams = LEARNED.filter((f) => ab.rows.some((r) => r.family === f));
  let fam: FamilyId = fams.includes("hgb") ? "hgb" : fams[0];
  const det = Object.entries(ab.determinism_check_max_abs_diff);
  $("ab-intro").textContent =
    `Each variant changes one thing relative to the same model's full configuration and is scored on identical engines and cut points. ${ab.split}. ` +
    `Positive Δ means the variant is worse. Filled dots: the 95% CI excludes 0. ` +
    (det.every(([, v]) => v === 0) ? "Re-running the unchanged full models reproduced the CV predictions exactly." : "");
  const draw = chart($("ab-chart"), (w) => {
    const rows = ab.rows
      .filter((r) => r.family === fam && r.variant !== "base")
      .map((r) => ({
        label: `${r.group}: ${r.variant}`,
        short: r.variant,
        delta: r.delta_rmse,
        ci: r.ci95,
        clear: r.ci_excludes_0,
        tip: `${r.variant} (${r.group})\n${r.description}\nPooled RMSE ${f2(r.pooled_rmse)} · Δ ${signed(r.delta_rmse)}\n95% CI ${ci(r.ci95)}\n${r.repeats} repeat${r.repeats > 1 ? "s" : ""}, ${r.n_folds} folds`,
      }));
    return forest(w, rows, `Δ pooled RMSE vs full ${LABEL[fam]} (cycles; positive = worse)`);
  });
  const cap = () => {
    const base = ab.rows.find((r) => r.family === fam && r.variant === "base")!;
    $("ab-cap").textContent =
      `${LABEL[fam]}: full-model pooled RMSE ${f2(base.pooled_rmse)} (capped truth). ` +
      `${base.repeats === 3 ? `All ${base.repeats} CV repeats (${base.n_folds} outer folds)` : `Only ${base.repeats} CV repeat (${base.n_folds} outer folds) because LSTM training is slow, so treat its smaller effects with extra caution`}. ` +
      `Validation-only: no variant was adopted, and the frozen Phase 1 models are unchanged.`;
  };
  cap();
  seg($("ab-family"), fams.map((f) => [f, LABEL[f]] as [FamilyId, string]), fam, (v) => {
    fam = v;
    draw();
    cap();
  });
}

// ---------------------------------------------------------------- F. uncertainty

function uncertainty(c: Conformal) {
  let method = c.methods.find((m) => m.guarantee === "finite_sample")?.id ?? c.methods[0].id;
  let strat = c.calibration_strategy;
  let level = c.levels[1] ?? c.levels[0];
  const fams = LEARNED.filter((f) => c.summary.some((r) => r.family === f));
  const fr = c.inner_fractions;
  $("unc-intro").textContent =
    `${c.split}${fr.cal ? ` (inner split ${pct(fr.train)} / ${pct(fr.val)} / ${pct(fr.cal)} of development engines)` : ""}. ` +
    `Each interval is [prediction − q, prediction + q], with q the conformal quantile of absolute calibration errors. ${c.caveat}`;
  const note = () => {
    const m = c.methods.find((x) => x.id === method)!;
    const shift = strat !== c.calibration_strategy;
    $("unc-method-note").innerHTML = "";
    $("unc-method-note").append(
      el("strong", { textContent: m.guarantee === "finite_sample" ? "Finite-sample guarantee (under assumptions). " : "Empirical coverage only. " }),
      m.note,
      shift ? ` Shift check: these held-out cut points are drawn differently from the calibration cut points (${c.calibration_strategy}), so no guarantee applies.` : "",
    );
  };
  const rows = () => c.summary.filter((r) => r.method === method && r.strategy === strat && fams.includes(r.family));
  const off = (f: FamilyId) => (fams.indexOf(f) - (fams.length - 1) / 2) * 0.012;

  const drawCov = chart($("unc-coverage"), (w) => {
    const rs = rows().map((r) => ({ ...r, label: LABEL[r.family], x: r.level + off(r.family) }));
    return Plot.plot({
      ...baseStyle(w, 300),
      marginLeft: 48,
      x: { labelOffset: 40, label: "Nominal coverage →", domain: [0.76, 0.99], ticks: c.levels, tickFormat: (d: number) => pct(d) },
      y: { label: "↑ Observed coverage", domain: [0.5, 1], grid: true, tickFormat: (d: number) => pct(d) },
      color: { ...famScale(fams), legend: true },
      marks: [
        Plot.line([[0.76, 0.76], [0.99, 0.99]], { stroke: css("--ink-2"), strokeDasharray: "4,3" }),
        Plot.ruleX(rs, { x: "x", y1: "coverage_fold_min", y2: "coverage_fold_max", stroke: "label", strokeWidth: 2, strokeOpacity: 0.6 }),
        Plot.dot(rs, {
          x: "x",
          y: "coverage",
          fill: "label",
          r: 5,
          stroke: css("--surface"),
          title: (r) => `${r.label} at ${pct(r.level)}\nCoverage ${pct(r.coverage, 1)} (folds ${pct(r.coverage_fold_min ?? 0, 0)}–${pct(r.coverage_fold_max ?? 0, 0)})\nMean width ${f1(r.mean_width)} cycles\n${r.n_points.toLocaleString("en-US")} held-out points`,
          tip: true,
        }),
      ],
    });
  });
  const drawWidth = chart($("unc-width"), (w) => {
    const rs = rows().map((r) => ({ ...r, label: LABEL[r.family], lv: pct(r.level) }));
    return Plot.plot({
      ...baseStyle(w, 300),
      marginLeft: 48,
      fx: { label: null, padding: 0.15, tickFormat: (d: string) => `${d} nominal` },
      x: { axis: null, domain: fams.map((f) => LABEL[f]) },
      y: { label: "↑ Mean width (cycles)", grid: true },
      color: { ...famScale(fams), legend: true },
      marks: [
        Plot.barY(rs, { fx: "lv", x: "label", y: "mean_width", fill: "label", rx: 3, title: (r) => `${r.label} at ${r.lv}\nMean width ${f1(r.mean_width)} cycles`, tip: true }),
        Plot.ruleY([0], { stroke: css("--axis") }),
      ],
    });
  });
  const drawStage = chart($("unc-stage"), (w) => {
    const stages = [...new Set(c.per_stage.map((r) => r.stage!))];
    const rs = c.per_stage
      .filter((r) => r.method === method && r.level === level && fams.includes(r.family))
      .map((r) => ({ ...r, label: LABEL[r.family] }));
    return Plot.plot({
      ...baseStyle(w, 300),
      marginBottom: w < 560 ? 72 : 46,
      marginLeft: 48,
      x: { labelOffset: w < 560 ? 58 : 40, label: "True RUL stage (cycles) →", domain: stages, padding: 0.4, tickRotate: w < 560 ? -40 : 0 },
      y: { label: "↑ Observed coverage", domain: [0.4, 1], grid: true, tickFormat: (d: number) => pct(d) },
      color: { ...famScale(fams), legend: true },
      marks: [
        Plot.ruleY([level], { stroke: css("--ink-2"), strokeDasharray: "4,3" }),
        Plot.line(rs, { x: "stage", y: "coverage", stroke: "label", strokeWidth: 2 }),
        Plot.dot(rs, {
          x: "stage",
          y: "coverage",
          fill: "label",
          r: 4.5,
          stroke: css("--surface"),
          title: (r) => `${r.label}, true RUL ${r.stage}\nCoverage ${pct(r.coverage, 1)}\nMean width ${f1(r.mean_width)} cycles\n${r.n_points} points`,
          tip: true,
        }),
      ],
    });
  });
  const stageCap = () => {
    $("unc-stage-cap").textContent =
      `Coverage within each true-RUL band at the ${pct(level)} level (dashed), ${c.calibration_strategy} held-out cut points. ` +
      `The guarantee is marginal: it does not promise this level inside any particular stage.`;
  };
  const all = () => {
    note();
    drawCov();
    drawWidth();
    drawStage();
    stageCap();
  };
  note();
  stageCap();
  fillSelect($<HTMLSelectElement>("unc-method"), c.methods.map((m) => [m.id, m.label]), method, (v) => {
    method = v;
    all();
  });
  const strategies = [...new Set(c.summary.map((r) => r.strategy))];
  fillSelect(
    $<HTMLSelectElement>("unc-strategy"),
    strategies.map((s) => [s, s === c.calibration_strategy ? `${s} (same as calibration)` : `${s} (shift check)`]),
    strat,
    (v) => {
      strat = v;
      note();
      drawCov();
      drawWidth();
    },
  );
  seg($("unc-level"), c.levels.map((l) => [l, pct(l)] as [number, string]), level, (v) => {
    level = v;
    drawStage();
    stageCap();
  });
}

// ---------------------------------------------------------------- G. benchmarks

function benchmarks(b: Benchmarks) {
  let scope: "model" | "end_to_end" = "end_to_end";
  let stat: "latency_ms_median" | "latency_ms_p95" = "latency_ms_median";
  const env = b.environment;
  $("bench-intro").textContent =
    `Measured on ${env.cpu} (${env.cpu_count} cores, ${env.ram_gb} GB RAM, ${env.os}), Python ${env.python}, PyTorch ${env.torch}, scikit-learn ${env.scikit_learn}. ${b.method}. ` +
    `Model scope: ${b.scope_definitions.model}. End-to-end scope: ${b.scope_definitions.end_to_end}.`;
  const series = (r: Benchmarks["rows"][number]) => `${LABEL[r.family]}${r.device === "cpu" ? "" : ` (${r.device.toUpperCase()})`}`;
  const fams = ORDER.filter((f) => b.rows.some((r) => r.family === f));
  const draw = chart($("bench-chart"), (w) => {
    type Row = Benchmarks["rows"][number] & { s: string; y: number };
    const rs: Row[] = b.rows.filter((r) => r.scope === scope).map((r) => ({ ...r, s: series(r), y: r[stat] }));
    const unit = rs[0]?.unit ?? "items";
    return Plot.plot({
      ...baseStyle(w, 340),
      marginLeft: 56,
      x: { type: "log", labelOffset: 40, label: `Batch size (${unit}) →`, ticks: [...new Set(rs.map((r: Row) => r.n))], tickFormat: (d: number) => String(d) },
      y: { type: "log", label: `↑ ${stat === "latency_ms_median" ? "Median" : "p95"} latency per call (ms, log scale)`, grid: true },
      color: { ...famScale(fams), legend: true },
      marks: [
        Plot.line(rs.filter((r: Row) => r.device === "cpu"), { x: "n", y: "y", stroke: (r: Row) => LABEL[r.family], z: "s", strokeWidth: 2 }),
        Plot.line(rs.filter((r: Row) => r.device !== "cpu"), { x: "n", y: "y", stroke: (r: Row) => LABEL[r.family], z: "s", strokeWidth: 2, strokeDasharray: "5,3" }),
        Plot.dot(rs, {
          x: "n",
          y: "y",
          fill: (r: Row) => LABEL[r.family],
          r: 4,
          stroke: css("--surface"),
          symbol: (r: Row) => (r.device === "cpu" ? "circle" : "square"),
          title: (r: Row) =>
            `${r.s}, ${r.n} ${r.unit}\nMedian ${r.latency_ms_median.toPrecision(3)} ms · p95 ${r.latency_ms_p95.toPrecision(3)} ms\nThroughput ${int(r.throughput_per_s)} ${r.unit}/s\nPython peak allocation ${r.python_peak_alloc_mb.toPrecision(3)} MB`,
          tip: true,
        }),
      ],
    });
  });
  const tbl = () => {
    const rs = b.rows.filter((r) => r.scope === scope);
    const maxN = Math.max(...rs.map((r) => r.n));
    table(
      $("bench-table"),
      ["Model", "Device", `Batch (${rs[0]?.unit})`, "Median (ms)", "p95 (ms)", "Throughput (/s)", "Python peak (MB)"],
      rs
        .filter((r) => r.n === maxN)
        .map((r) => [famCell(r.family), r.device.toUpperCase(), String(r.n), r.latency_ms_median.toPrecision(3), r.latency_ms_p95.toPrecision(3), int(r.throughput_per_s), r.python_peak_alloc_mb.toPrecision(3)]),
    );
    $("bench-cap").textContent =
      `Dashed line / squares: LSTM on Apple-GPU (MPS); solid: CPU. Table: largest batch. "Python peak" is tracemalloc's peak of Python allocations (including NumPy buffers, excluding PyTorch's native allocator); ` +
      `the whole benchmark process peaked at ${f1(b.process_max_rss_mb)} MB resident memory.`;
  };
  tbl();
  seg($("bench-scope"), [["end_to_end", "End-to-end"], ["model", "Model only"]] as ["model" | "end_to_end", string][], scope, (v) => {
    scope = v;
    draw();
    tbl();
  });
  seg($("bench-stat"), [["latency_ms_median", "Median"], ["latency_ms_p95", "p95"]] as [typeof stat, string][], stat, (v) => {
    stat = v;
    draw();
  });
  table(
    $("train-table"),
    ["Model", "Device", "Train time (s)", "Epochs", "Parameters / tree nodes", "Artifact (KB)"],
    ORDER.filter((f) => b.training.some((r) => r.family === f)).map((f) => {
      const r = b.training.find((x) => x.family === f)!;
      return [famCell(f), r.device.toUpperCase(), r.train_time_s < 0.01 ? "< 0.01" : f2(r.train_time_s), r.epochs == null ? "—" : String(r.epochs), r.n_params == null ? "—" : int(r.n_params), f1(r.artifact_bytes / 1024)];
    }),
  );
}

// ---------------------------------------------------------------- boot

async function main() {
  const [meta, test] = await Promise.all([load("meta.json"), load("official_test.json")]).catch((e) => {
    $("cards").replaceChildren(el("p", { className: "error", textContent: `Could not load results: ${(e as Error).message}` }));
    throw e;
  });
  for (const f of meta.families) LABEL[f.id] = f.label;
  const opt = <T,>(p: Promise<T>) => p.catch((e) => (console.error(e), null));
  const [val, cv, ab, conf, bench] = await Promise.all([
    opt(load("phase1_validation.json")),
    opt(load("cv.json")),
    opt(load("ablation.json")),
    opt(load("conformal.json")),
    opt(load("benchmarks.json")),
  ]);
  const missing = (id: string, file: string) => $(id).replaceChildren(el("p", { className: "error", textContent: `${file} could not be loaded.` }));
  await guard($("cards"), () => overview(meta, test, cv));
  await guard($("models-chart"), () => models(test));
  await guard($("test-scatter"), () => engines(test, val));
  if (!val) missing("traj-chart", "phase1_validation.json");
  if (cv) await guard($("cv-folds"), () => crossValidation(cv));
  else missing("cv-folds", "cv.json");
  if (ab) await guard($("ab-chart"), () => ablations(ab));
  else missing("ab-chart", "ablation.json");
  if (conf) await guard($("unc-coverage"), () => uncertainty(conf));
  else missing("unc-coverage", "conformal.json");
  if (bench) await guard($("bench-chart"), () => benchmarks(bench));
  else missing("bench-chart", "benchmarks.json");
  $("foot").innerHTML = "";
  $("foot").append(
    `Generated from committed experiment artifacts. Official test evaluated ${test.evaluated_at.slice(0, 10)} (commit ${short(test.git_commit)}). `,
    cv ? `Phase 2 experiments from commit ${short(cv.provenance.git_commit)}${cv.provenance.source_snapshot ? " plus the published source snapshot" : ""}. ` : "",
    el("a", { href: meta.repository, textContent: "Source on GitHub" }),
    ".",
  );
}

main().catch((e) => console.error(e));
