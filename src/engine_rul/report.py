"""Generate reports/report.md, result figures, and README result tables (PRD 9, M12).

Reads only committed machine-readable results in results/final/ and reports/eda_summary.json,
so the report can be regenerated without re-training. Never computes metrics from scratch
(apart from formatting); every number comes from those files (PRES-01)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from engine_rul import plots as P
from engine_rul.plots import FAMILY_LABELS, FAMILY_ORDER
from engine_rul.utils import REPO_ROOT

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _f(x, nd=2):
    return "–" if x is None else f"{x:,.{nd}f}"


def _ms(d, nd=2):
    return f"{d['mean']:,.{nd}f} ± {d['std']:,.{nd}f}" if d["std"] > 0 else f"{d['mean']:,.{nd}f}"


def _table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load_results(results_dir: Path) -> dict:
    fd = results_dir / "final"
    r = {"summary": json.loads((fd / "summary.json").read_text())}
    for name in ("benchmarks", "sensitivity", "training_details"):
        p = fd / f"{name}.json"
        r[name] = json.loads(p.read_text()) if p.exists() else None
    r["tuning"] = {
        p.stem.removeprefix("tuning_"): json.loads(p.read_text())
        for p in sorted(fd.glob("tuning_*.json"))
    }
    r["test_preds"] = pd.read_csv(fd / "test_predictions_median_seed.csv")
    r["val_dense"] = pd.read_csv(fd / "val_dense_predictions_median_seed.csv")
    log = results_dir / "test_evaluations.jsonl"
    r["test_log"] = (
        [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        if log.exists()
        else []
    )
    eda = REPO_ROOT / "reports" / "eda_summary.json"
    r["eda"] = json.loads(eda.read_text()) if eda.exists() else None
    return r


def main_table(s: dict, fams: list[str], key="test_primary") -> str:
    rows = []
    for f in fams:
        fam = s["families"][f]
        m = fam[key]
        ci = fam["test_ci95_median_seed"]["rmse"] if key == "test_primary" else None
        rows.append(
            [
                f"**{FAMILY_LABELS[f]}**",
                len(fam["seeds"]),
                _ms(m["rmse"]),
                _ms(m["mae"]),
                _ms(m["nasa_sum"], 0),
                _ms(m["nasa_mean"]),
                _ms(m["bias"]),
                _f(m["late_fraction"]["mean"]),
                *([f"[{ci[0]:.2f}, {ci[1]:.2f}]"] if ci else []),
            ]
        )
    head = [
        "Model",
        "Seeds",
        "RMSE",
        "MAE",
        "NASA score (sum)",
        "NASA score / engine",
        "Bias",
        "Late frac.",
    ]
    return _table(head + (["RMSE 95% CI†"] if key == "test_primary" else []), rows)


def paired_table(s: dict) -> str:
    rows = [
        [
            f"{FAMILY_LABELS[p['a']]} − {FAMILY_LABELS[p['b']]}",
            _f(p["delta_rmse"]),
            f"[{p['ci95'][0]:.2f}, {p['ci95'][1]:.2f}]",
            "yes" if p["ci_excludes_0"] else "no",
        ]
        for p in s["paired_comparisons"]
    ]
    return _table(
        ["Comparison (RMSE a − b)", "Δ RMSE", "95% paired bootstrap CI", "CI excludes 0"], rows
    )


def findings(s: dict, fams: list[str]) -> list[str]:
    """Auto-generated factual statements; wording is gated on the paired CIs (EVAL-28/29)."""
    out = []
    best = min(fams, key=lambda f: s["families"][f]["test_primary"]["rmse"]["mean"])
    b = s["families"][best]["test_primary"]
    out.append(
        f"Lowest mean test RMSE: **{FAMILY_LABELS[best]}** ({_ms(b['rmse'])} cycles over "
        f"{len(s['families'][best]['seeds'])} seed(s); NASA score {_ms(b['nasa_sum'], 0)})."
    )
    vbest = min(fams, key=lambda f: s["families"][f]["val_points_capped"]["rmse"]["mean"])
    if vbest != best:
        v = {f: s["families"][f]["val_points_capped"]["rmse"]["mean"] for f in (vbest, best)}
        out.append(
            f"On validation (the selection data), **{FAMILY_LABELS[vbest]}** had the lowest RMSE "
            f"({v[vbest]:.2f} vs {v[best]:.2f} for {FAMILY_LABELS[best]}); the ranking flipped on the test "
            "set, a reminder of how noisy 20 validation engines are."
        )
    for p in s["paired_comparisons"]:
        a, bb = FAMILY_LABELS[p["a"]], FAMILY_LABELS[p["b"]]
        lo, hi = p["ci95"]
        if p["ci_excludes_0"]:
            verdict = "lower" if p["delta_rmse"] < 0 else "higher"
            out.append(
                f"{a} has {verdict} test RMSE than {bb} (median seeds, Δ = {p['delta_rmse']:+.2f}, "
                f"paired 95% CI [{lo:.2f}, {hi:.2f}])."
            )
        else:
            out.append(
                f"{a} vs {bb}: no clear difference on 100 test engines (Δ = {p['delta_rmse']:+.2f}, "
                f"95% CI [{lo:.2f}, {hi:.2f}] includes 0)."
            )
    return out


def build_figures(r: dict, fig_root: Path) -> dict[str, str]:
    P.style()
    s = r["summary"]
    fams = [f for f in FAMILY_ORDER if f in s["families"]]
    learned = [f for f in ("ridge", "hgb", "lstm") if f in fams]
    figs = {}
    res = fig_root / "results"
    figs["comparison"] = P.model_comparison(s, res / "model_comparison.png", fams)
    figs["pred_vs_true"] = P.pred_vs_true(
        r["test_preds"], s["r_max"], res / "pred_vs_true_test.png", fams
    )
    figs["sorted"] = P.test_engines_sorted(
        r["test_preds"], res / "test_engines_sorted.png", ["life_prior", *learned]
    )
    figs["errors"] = P.error_distribution(
        r["test_preds"], res / "error_distribution.png", ["life_prior", *learned]
    )
    figs["stage"] = P.error_by_stage(s, res / "error_by_stage.png", ["life_prior", *learned])
    vd_fams = [f for f in ("hgb", "lstm") if f in fams]
    if vd_fams:
        figs["val_traj"] = P.val_engine_trajectories(
            r["val_dense"], res / "val_engine_trajectories.png", vd_fams
        )
    td = r["training_details"] or {}
    if td.get("lstm_histories"):
        figs["loss"] = P.loss_curves(
            td["lstm_histories"], fig_root / "training" / "lstm_loss_curves.png"
        )
    if td.get("permutation_importance"):
        figs["importance"] = P.feature_importance(
            td["permutation_importance"], res / "feature_importance.png"
        )
    if r["sensitivity"]:
        figs["sensitivity"] = P.sensitivity(r["sensitivity"]["rows"], res / "sensitivity.png")
    if r["tuning"]:
        figs["hparams"] = P.hparam_sensitivity(r["tuning"], res / "hparam_sensitivity.png")
    if r["benchmarks"]:
        figs["bench"] = P.benchmarks(r["benchmarks"]["rows"], res / "benchmarks.png")
    return {k: str(Path(v).relative_to(fig_root.parent)) for k, v in figs.items() if v is not None}


def build_report(results_dir="results", out="reports", update_readme=True) -> Path:
    results_dir, out = Path(results_dir), Path(out)
    r = load_results(results_dir)
    s = r["summary"]
    fams = [f for f in FAMILY_ORDER if f in s["families"]]
    figs = build_figures(r, out / "figures")
    env = s["env"]
    L = []
    add = L.append
    add("# EngineRUL — Experiment Report (FD001)\n")
    add("> Generated by `engine-rul report` from `results/final/*.json`. Do not edit by hand.\n>")
    add(
        "> All numbers are measured on NASA's **simulated** C-MAPSS data and are not validated for real-world "
        "maintenance or flight-safety decisions.\n"
    )
    dirty = any(e.get("git_dirty") for e in r["test_log"])
    add(
        f"- Results generated: {s['generated_at']} · git `{s['git_commit'][:10]}`"
        f"{' (uncommitted working-tree changes at evaluation time)' if dirty else ''} · training data SHA-256 "
        f"`{s['data_sha256'][:12]}…` · split SHA-256 `{s['split_sha256'][:12]}…`"
    )
    add(
        f"- Hardware: {env['cpu']} · {env['ram_gb']} GB RAM · {env['os']} · Python {env['python']} · "
        f"torch {env['torch']} · scikit-learn {env['scikit_learn']}"
    )
    add(
        f"- Training device per family: {', '.join(f'{FAMILY_LABELS[f]}={d}' for f, d in s['device'].items())}\n"
    )

    add("## 1. Data\n")
    e = r["eda"]
    if e:
        add(
            _table(
                ["Item", "Value"],
                [
                    ["Training engines / rows", f"{e['train']['units']} / {e['train']['rows']:,}"],
                    [
                        "Training lifetimes (cycles)",
                        f"min {e['train']['lifetime']['min']}, median {e['train']['lifetime']['median']:.0f}, max {e['train']['lifetime']['max']}",
                    ],
                    [
                        "Test engines / rows",
                        f"{e['test_sensor_data']['units']} / {e['test_sensor_data']['rows']:,}",
                    ],
                    [
                        "Test observed lengths (cycles)",
                        f"min {e['test_sensor_data']['observed_length']['min']}, max {e['test_sensor_data']['observed_length']['max']}",
                    ],
                    [
                        "Dropped (constant / two-valued)",
                        ", ".join(e["constant_columns"] + e["two_valued_columns"]),
                    ],
                    ["Dropped by config (op settings)", "op1, op2, op3"],
                    ["Model inputs", f"{len(e['informative_sensors'])} sensors + normalized cycle"],
                    [
                        "Test engines with true RUL > R_max",
                        f"{s['n_test_rul_above_r_max']} (R_max = {s['r_max']})",
                    ],
                ],
            )
        )
    add(
        "\nSplit: 80 train / 20 validation engines (engine-level, lifetime-stratified, committed in "
        "`splits/fd001_split.json`). Validation model selection uses 10 seeded truncation points per validation "
        "engine (200 points); early stopping uses every validation window.\n"
    )

    add("## 2. Validation results (model selection — test set not used)\n")
    rows = []
    for f in fams:
        t = r["tuning"].get(f)
        v = s["families"][f]["val_points_capped"]
        rng = (
            f"{min(x['score'] for x in t['rows']):.2f} – {max(x['score'] for x in t['rows']):.2f}"
            if t
            else "–"
        )
        rows.append(
            [
                FAMILY_LABELS[f],
                f"`{json.dumps(s['families'][f]['params'])}`",
                t["n_configs"] if t else 1,
                rng,
                _ms(v["rmse"]),
                _ms(v["mae"]),
                _ms(v["nasa_mean"]),
            ]
        )
    add(
        _table(
            [
                "Model",
                "Selected params",
                "Configs tried",
                "Val RMSE range (tuning)",
                "Val RMSE (final seeds)",
                "Val MAE",
                "Val NASA / point",
            ],
            rows,
        )
    )
    add(
        "\nSelection metric: RMSE on the 200 truncated validation points against truth capped at R_max "
        "(EVAL-12). Ties within 0.01 go to the first (simplest) grid entry.\n"
    )
    if "hparams" in figs:
        add(f"![Hyperparameter sensitivity]({figs['hparams']})\n")

    add("## 3. Official test results\n")
    add(
        "**Primary — official uncapped RUL labels** (one prediction per engine at its last observed cycle, N = 100):\n"
    )
    add(main_table(s, fams))
    add(
        "\n† Percentile bootstrap over test engines (2000 resamples) for the median-RMSE seed. "
        "± is the sample std over seeds; deterministic models were trained once.\n"
    )
    add("**Secondary — labels capped at R_max (literature-style):**\n")
    add(main_table(s, fams, "test_capped"))
    add(f"\n![Model comparison]({figs['comparison']})\n")
    add(f"![Predicted vs true]({figs['pred_vs_true']})\n")
    add(f"![Per-engine predictions]({figs['sorted']})\n")
    add(f"![Error distribution]({figs['errors']})\n")

    add("## 4. Paired comparisons (EVAL-28)\n")
    add(paired_table(s))
    add(
        "\nNegative Δ means model *a* has lower RMSE. Engines are resampled jointly for both models.\n"
    )
    add("**Auto-generated findings:**\n")
    L.extend(f"- {x}" for x in findings(s, fams))
    add("")

    add("## 5. Performance by degradation stage\n")
    for key, title in (
        ("stage_metrics_test", "Official test set (bins with n < 5 not scored)"),
        ("stage_metrics_val_dense", "Validation engines, every window"),
    ):
        add(f"**{title}** — RMSE (cycles) vs uncapped truth:\n")
        bins = [b["bin"] for b in s[key][fams[0]]]
        rows = [["n", *[b["n"] for b in s[key][fams[0]]]]]
        rows += [[FAMILY_LABELS[f], *[_f(b["rmse"]) for b in s[key][f]]] for f in fams]
        add(_table(["True RUL bin", *bins], rows) + "\n")
        add("Bias (mean predicted − true):\n")
        add(
            _table(
                ["True RUL bin", *bins],
                [[FAMILY_LABELS[f], *[_f(b["bias"]) for b in s[key][f]]] for f in fams],
            )
            + "\n"
        )
    add(f"![Error by stage]({figs['stage']})\n")
    add(
        "Errors in the `[125, inf)` bin are partly irreducible for models trained on targets capped at 125 (ADR-03).\n"
    )

    add("## 6. Error analysis — largest test errors (median seed)\n")
    for f in [f for f in ("hgb", "lstm") if f in fams]:
        add(f"**{FAMILY_LABELS[f]}**\n")
        add(
            _table(
                ["Unit", "Observed length", "True RUL", "Predicted", "Error"],
                [
                    [w["unit"], w["length"], _f(w["true"], 0), _f(w["pred"], 1), _f(w["error"], 1)]
                    for w in s["worst_engines_test"][f]
                ],
            )
            + "\n"
        )
    if "val_traj" in figs:
        add(f"![Validation trajectories]({figs['val_traj']})\n")

    if "importance" in figs:
        add("## 7. Feature importance (validation set only)\n")
        add(f"![Permutation importance]({figs['importance']})\n")
        td = r["training_details"]
        if td.get("ridge_coefficients"):
            top = sorted(td["ridge_coefficients"].items(), key=lambda kv: -abs(kv[1]))[:10]
            add("Ridge — 10 largest standardized coefficients:\n")
            add(
                _table(["Feature", "Coefficient (cycles per std)"], [[k, _f(v)] for k, v in top])
                + "\n"
            )

    if "loss" in figs:
        add("## 8. LSTM training curves\n")
        fi = s["families"]["lstm"]["fit_info"]
        add(
            f"Median seed: {fi.get('epochs_run')} epochs run, best epoch {fi.get('best_epoch')}, "
            f"{_f(fi.get('seconds_per_epoch'))} s/epoch on {fi.get('device')}.\n"
        )
        add(f"![LSTM loss curves]({figs['loss']})\n")

    if r["benchmarks"]:
        add("## 9. Computational benchmarks (measured)\n")
        rows = []
        for b in r["benchmarks"]["rows"]:
            rows.append(
                [
                    FAMILY_LABELS[b["family"]],
                    b["device"],
                    _f(b["train_time_s"], 2),
                    b["epochs_run"] or "–",
                    _f(b["latency_ms_median"], 3),
                    _f(b["latency_ms_p95"], 3),
                    f"{b['throughput_windows_per_s']:,.0f}",
                    f"{b['n_params']:,}",
                    f"{b['artifact_bytes']:,}",
                ]
            )
        add(
            _table(
                [
                    "Model",
                    "Inference device",
                    "Train time (s, CPU)",
                    "Epochs",
                    "Latency median (ms)",
                    "Latency p95 (ms)",
                    "Throughput (windows/s, batch 100)",
                    "Params / tree nodes",
                    "Artifact bytes",
                ],
                rows,
            )
        )
        add(
            "\nLatency: batch of 1 window, 20 warm-up + 200 timed calls, including feature extraction for tabular "
            "models. Training time is wall-clock for one run. Comparable only within this machine.\n"
        )
        if "bench" in figs:
            add(f"![Benchmarks]({figs['bench']})\n")

    if r["sensitivity"]:
        add("## 10. Sensitivity study (validation only, seed 0)\n")
        rows = [
            [
                FAMILY_LABELS[x["family"]],
                "none" if x["r_max"] is None else x["r_max"],
                x["window"],
                _f(x["val_rmse_cap125"]),
                _f(x["val_rmse_uncapped"]),
            ]
            for x in r["sensitivity"]["rows"]
        ]
        add(
            _table(
                [
                    "Model",
                    "R_max",
                    "Window W",
                    "Val RMSE (truth capped at 125)",
                    "Val RMSE (uncapped truth)",
                ],
                rows,
            )
        )
        add(f"\n![Sensitivity]({figs['sensitivity']})\n")

    add("## 11. Test-set usage log\n")
    rounds = sorted({e["timestamp_utc"][:16] for e in r["test_log"]})
    add(
        f"`results/test_evaluations.jsonl` holds {len(r['test_log'])} entries across {len(rounds)} evaluation "
        "round(s). Notes recorded per round:\n"
    )
    notes = {}
    for e in r["test_log"]:
        notes.setdefault(e["timestamp_utc"][:16], set()).add(e.get("note") or "(none)")
    L.extend(f"- {k} UTC: {'; '.join(sorted(v))}" for k, v in sorted(notes.items()))
    add(
        "\nSee `docs/decisions.md` → *Post-test changes* for any change made after a test evaluation.\n"
    )

    add("## 12. Limitations\n")
    L.extend(
        [
            "- Simulated data (C-MAPSS); not validated for real engines, maintenance planning or flight safety.",
            "- Single subset (FD001: one operating condition, one fault mode).",
            "- One fixed 80/20 engine split; final models are not retrained on all 100 engines, and split variance is not captured.",
            "- Only 100 test engines: confidence intervals are wide; small differences are not distinguishable.",
            "- Models trained on targets capped at 125 cannot predict long RULs exactly (see stage table).",
            "- Mirror provenance: files match pinned checksums, but byte-identity with NASA's original ZIP is unverified.",
            "- Published results use different protocols; see `docs/external_results.md` — no ranking against them is claimed.",
        ]
    )
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.md"
    path.write_text("\n".join(L) + "\n")
    if update_readme:
        update_readme_tables(REPO_ROOT / "README.md", r, figs)
    return path


def readme_block(r: dict, figs: dict) -> str:
    s = r["summary"]
    fams = [f for f in FAMILY_ORDER if f in s["families"]]
    rows = []
    for f in fams:
        m = s["families"][f]["test_primary"]
        ci = s["families"][f]["test_ci95_median_seed"]["rmse"]
        rows.append(
            [
                f"**{FAMILY_LABELS[f]}**",
                _ms(m["rmse"]),
                f"[{ci[0]:.1f}, {ci[1]:.1f}]",
                _ms(m["mae"]),
                _ms(m["nasa_sum"], 0),
                len(s["families"][f]["seeds"]),
            ]
        )
    lines = [
        _table(
            ["Model", "Test RMSE ↓", "RMSE 95% CI", "Test MAE ↓", "NASA score ↓", "Seeds"], rows
        ),
        "",
        f"<sub>Official FD001 test set, 100 engines, uncapped labels; mean ± std over seeds; CI = engine bootstrap "
        f"of the median seed. Source: `results/final/summary.json` (generated {s['generated_at'][:10]}).</sub>",
        "",
        *[f"- {x}" for x in findings(s, fams)],
    ]
    return "\n".join(lines)


def update_readme_tables(readme: Path, r: dict, figs: dict) -> None:
    text = readme.read_text()
    if START not in text or END not in text:
        return
    block = f"{START}\n{readme_block(r, figs)}\n{END}"
    readme.write_text(
        re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    )
