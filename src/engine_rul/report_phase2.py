"""reports/phase2_report.md + reports/figures/phase2/*, generated only from results/phase2/ artifacts.

Every number in the report is read from an experiment's summary/manifest files; nothing here reads
the official test files. The Phase-1 report (reports/report.md) is a separate document."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine_rul import plots as P
from engine_rul.report import _f, _table

plt = P.plt
# CVD-validated categorical slots (dataviz validator: deutan dE >= 9.2); baselines stay neutral.
COLORS = {
    "ridge": "#2a78d6",
    "hgb": "#eb6834",
    "lstm": "#1baf7a",
    "constant": "#9a9a9a",
    "life_prior": "#5f5f5f",
}
MARKERS = {"ridge": "o", "hgb": "s", "lstm": "D", "constant": "v", "life_prior": "^"}
STRAT_LABELS = {
    "uniform": "Uniform",
    "late": "Late-life (RUL ≤ 125)",
    "stratified": "Stage-stratified",
}
METHOD_LABELS = {
    "global_one_per_engine": "Global, one endpoint per engine",
    "global_pooled": "Global, pooled endpoints",
    "mondrian_pooled": "Mondrian by predicted RUL",
}


def _L(f: str) -> str:
    return P.FAMILY_LABELS.get(f, f)


def _ci(x) -> str:
    return f"[{x[0]:.2f}, {x[1]:.2f}]"


def _pct(x: float) -> str:
    return ">99%" if 0.995 <= x < 1 else "<1%" if 0 < x < 0.005 else f"{x:.0%}"


def _fracs(fr: dict) -> str:
    return " / ".join(f"{k} {v:.0%}" for k, v in fr.items())


def _load(d: Path, *names) -> list:
    return [json.loads((d / n).read_text()) if (d / n).exists() else None for n in names]


def _prov(man: dict) -> str:
    p = man.get("provenance", {})
    snap = p.get("source_snapshot") or {}
    snap = f", source snapshot `{snap['path']}`" if snap.get("path") else ""
    return (
        f"commit `{str(p.get('git_commit'))[:10]}` (working tree "
        f"{'dirty' if p.get('git_dirty') else 'clean'}{snap}); "
        f"spec `{man.get('spec_file', '–')}` sha256 `{str(man.get('spec_sha256', ''))[:12]}`"
    )


# --------------------------------------------------------------------------- figures


def fig_cv_folds(fm: pd.DataFrame, fams, path) -> Path:
    d = fm[(fm.strategy == "uniform") & (fm.truth == "capped")]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for i, f in enumerate(fams):
        v = d[d.family == f]["rmse"].to_numpy()
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(v))
        ax.scatter(
            v,
            i + jitter,
            s=22,
            color=COLORS[f],
            marker=MARKERS[f],
            alpha=0.8,
            edgecolor="white",
            lw=0.5,
        )
        ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=P.INK, lw=2)
    ax.set_yticks(range(len(fams)), [_L(f) for f in fams])
    ax.set(
        xlabel="Outer-fold RMSE (cycles, capped truth)",
        title="Nested CV: RMSE of each outer fold (bar = mean)",
    )
    ax.invert_yaxis()
    return P._save(fig, path)


def fig_strategies(cv: dict, fams, path) -> Path:
    strats = list(cv["strategies"])
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(strats))
    for j, f in enumerate(fams):
        y = [cv["strategies"][s]["families"][f]["pooled"]["capped"]["rmse"] for s in strats]
        lo = [
            cv["strategies"][s]["families"][f]["rmse_ci95_engine_bootstrap"]["capped"]
            for s in strats
        ]
        off = (j - (len(fams) - 1) / 2) * 0.12
        ax.errorbar(
            x + off,
            y,
            yerr=np.array([[a - c[0], c[1] - a] for a, c in zip(y, lo, strict=True)]).T,
            fmt=MARKERS[f],
            color=COLORS[f],
            ms=6,
            capsize=2,
            lw=1.2,
            label=_L(f),
        )
    ax.set_xticks(x, [STRAT_LABELS[s] for s in strats])
    ax.set(ylabel="Pooled RMSE (capped truth)", title="Same models, different validation endpoints")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return P._save(fig, path)


def fig_ablation(ab: dict, path) -> Path:
    fams = [f for f in P.FAMILY_ORDER if f in ab["families"]]
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 3.6), sharex=False)
    for ax, f in zip(np.atleast_1d(axes), fams, strict=True):
        rows = [r for r in ab["families"][f] if r["variant"] != "base"]
        y = np.arange(len(rows))
        d = np.array([r["delta_rmse"] for r in rows])
        ci = np.array([r["ci95"] for r in rows])
        ax.errorbar(d, y, xerr=np.abs(ci.T - d), fmt=MARKERS[f], color=COLORS[f], capsize=2, lw=1.2)
        ax.axvline(0, color=P.INK, lw=0.8)
        ax.set_yticks(y, [r["variant"] for r in rows])
        ax.invert_yaxis()
        ax.set(title=_L(f), xlabel="Δ RMSE vs base (cycles; + = worse)")
        if np.nanmax(ci) > 8:  # keep small deltas readable next to cycle_only's large one
            ax.set_xscale("symlog", linthresh=2)
            ax.set_xticks(
                [t for t in (-5, -2, 0, 2, 5, 10, 20, 40) if ci.min() - 1 <= t <= ci.max() + 5]
            )
            ax.xaxis.set_major_formatter(P.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    fig.suptitle("Ablations: paired engine-bootstrap 95% CI of the RMSE change", y=1.02)
    fig.tight_layout()
    return P._save(fig, path)


def fig_coverage(conf: dict, path) -> Path:
    s = pd.DataFrame(conf["summary"])
    s = s[s.method == "global_one_per_engine"]
    fams = [f for f in P.FAMILY_ORDER if f in set(s.family)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, strat in zip(axes, ("uniform", "late"), strict=True):
        for f in fams:
            g = s[(s.family == f) & (s.strategy == strat)].sort_values("level")
            ax.plot(
                g["level"], g["coverage"], marker=MARKERS[f], color=COLORS[f], lw=2, label=_L(f)
            )
        ax.plot([0.78, 0.97], [0.78, 0.97], color=P.INK, lw=0.8, ls="--", label="nominal")
        ax.set(
            xlabel="Nominal level",
            ylabel="Empirical coverage (engine-weighted)",
            title=f"{STRAT_LABELS[strat]} endpoints",
            xlim=(0.78, 0.97),
        )
    axes[0].legend(fontsize=8)
    fig.suptitle("Split-conformal coverage (global, one calibration endpoint per engine)", y=1.02)
    fig.tight_layout()
    return P._save(fig, path)


def fig_stage_coverage(conf: dict, path, level=0.9) -> Path:
    s = pd.DataFrame(conf["per_stage"])
    s = s[(s.method == "global_one_per_engine") & (s.level == level) & (s.strategy == "uniform")]
    fams = [f for f in P.FAMILY_ORDER if f in set(s.family)]
    stages = list(dict.fromkeys(s["stage"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(len(stages))
    for j, f in enumerate(fams):
        g = s[s.family == f].set_index("stage").reindex(stages)
        off = (j - (len(fams) - 1) / 2) * 0.25
        axes[0].bar(
            x + off,
            g["coverage"],
            width=0.23,
            color=COLORS[f],
            label=_L(f),
            edgecolor="white",
            lw=1,
        )
        axes[1].bar(x + off, g["mean_width"], width=0.23, color=COLORS[f], edgecolor="white", lw=1)
    axes[0].axhline(level, color=P.INK, lw=0.8, ls="--")
    axes[0].set(
        ylabel="Coverage",
        ylim=(0, 1.02),
        title=f"Coverage by true-RUL stage ({level:.0%} intervals)",
    )
    axes[1].set(ylabel="Mean width (cycles)", title="Interval width by stage")
    for ax in axes:
        ax.set_xticks(x, stages, rotation=30, fontsize=8)
        ax.set_xlabel("True RUL stage (cycles)")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        ncol=3,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout()
    return P._save(fig, path)


def fig_intervals(cv_dir: Path, path, families=("hgb", "lstm"), n_engines=2) -> Path | None:
    from engine_rul import data as D
    from engine_rul.inference import predict_engines

    dirs = [cv_dir / "models" / f"r0_f0_{f}" for f in families]
    if not all((d / "conformal.json").exists() for d in dirs):
        return None
    rj = json.loads((dirs[0] / "run.json").read_text())
    df = D.load_train(rj["config"]["data"]["data_dir"], rj["config"]["data"]["subset"])
    life = df.groupby("unit")["cycle"].max()
    outer = sorted(rj["engines"]["outer"], key=lambda u: life[u])
    units = [outer[len(outer) // 4], outer[3 * len(outer) // 4]][:n_engines]
    cap = rj["config"]["data"]["r_max"]
    fig, axes = plt.subplots(
        len(families),
        len(units),
        figsize=(5 * len(units), 3.1 * len(families)),
        squeeze=False,
        sharey=True,
    )
    for i, (f, d) in enumerate(zip(families, dirs, strict=True)):
        for j, u in enumerate(units):
            ax = axes[i, j]
            raw = df[df.unit == u].drop(columns=["rul"])
            pr = predict_engines(d, raw, levels=(0.8, 0.95), all_cycles=True)
            t = life[u] - pr["cycle"]
            ax.fill_between(
                pr["cycle"],
                pr["lower_0.95"],
                pr["upper_0.95"],
                color=COLORS[f],
                alpha=0.15,
                lw=0,
                label="95% interval",
            )
            ax.fill_between(
                pr["cycle"],
                pr["lower_0.8"],
                pr["upper_0.8"],
                color=COLORS[f],
                alpha=0.3,
                lw=0,
                label="80% interval",
            )
            ax.plot(pr["cycle"], pr["y_pred"], color=COLORS[f], lw=2, label="prediction")
            ax.plot(
                pr["cycle"],
                np.minimum(t, cap),
                color=P.INK,
                lw=1.5,
                label=f"true RUL capped at {cap}",
            )
            ax.plot(pr["cycle"], t, color=P.INK, lw=1, ls=":", label="true RUL (uncapped)")
            ax.set(
                title=f"{_L(f)} — held-out engine {u} (life {life[u]})",
                xlabel="Cycle",
                ylabel="RUL (cycles)",
            )
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Per-cycle predictions with split-conformal intervals (outer fold, repeat 0, fold 0)",
        y=1.01,
    )
    fig.tight_layout()
    return P._save(fig, path)


def fig_bench(b: dict, path) -> Path:
    r = pd.DataFrame(b["rows"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, scope in zip(axes, ("model", "end_to_end"), strict=True):
        g = r[r.scope == scope]
        for (f, dev), h in g.groupby(["family", "device"], sort=False):
            ax.plot(
                h["n"],
                h["latency_ms_median"],
                marker=MARKERS[f],
                color=COLORS[f],
                lw=2,
                ls="-" if dev == "cpu" else "--",
                label=f"{_L(f)} ({dev})",
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel=f"Batch size ({g['unit'].iloc[0]})",
            ylabel="Median latency (ms)",
            title="Model only" if scope == "model" else "End to end (raw rows)",
        )
    fmt = P.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    for ax in axes:
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(P.matplotlib.ticker.NullFormatter())
    axes[1].yaxis.set_major_locator(P.matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        ncol=3,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout()
    return P._save(fig, path)


# --------------------------------------------------------------------------- report


def _tau_range(cv: dict) -> str:
    taus = [r["kendall_tau"] for r in cv["strategy_rank_agreement"]]
    return (
        f"= {taus[0]:.2f} for every pair"
        if min(taus) == max(taus)
        else f"{min(taus):.2f}–{max(taus):.2f}"
    )


def key_findings(cv: dict, phase1_summary: Path) -> list[str]:
    """Auto-generated statements; 'better' is only claimed when the paired CI excludes 0."""
    prim = cv["strategies"][cv["primary_strategy"]]
    learned = [f for f in ("ridge", "hgb", "lstm") if f in prim["families"]]
    best = min(learned, key=lambda f: prim["families"][f]["pooled"]["capped"]["rmse"])
    n_folds = prim["families"][best]["fold_rmse_capped"]["n_folds"]
    out = [
        f"Under nested, repeated, engine-grouped CV, **{_L(best)}** has the lowest RMSE "
        f"({prim['families'][best]['pooled']['capped']['rmse']:.2f} cycles, capped truth) and is best in "
        f"{prim['ranking'][best]['rank1_frac'] * n_folds:.0f} of {n_folds} outer folds."
    ]
    for p in prim["paired"]:
        if p["truth"] == "capped" and p["a"] in learned and p["b"] in learned:
            word = "lower" if p["delta_rmse"] < 0 else "higher"
            out.append(
                f"{_L(p['a'])} vs {_L(p['b'])}: Δ RMSE {p['delta_rmse']:+.2f}, paired 95% CI {_ci(p['ci95'])} — "
                + (f"{_L(p['a'])} is {word}." if p["ci_excludes_0"] else "no clear difference.")
            )
    if phase1_summary.exists():
        val = json.loads(phase1_summary.read_text())["families"]
        v = {f: val[f]["val_points_capped"]["rmse"]["mean"] for f in learned if f in val}
        vbest = min(v, key=v.get)
        if vbest != best:
            out.append(
                f"Phase 1's single 20-engine validation split ranked {_L(vbest)} first "
                f"({v[vbest]:.2f} vs {v[best]:.2f} for {_L(best)}); with every engine held out and selection "
                "nested, that ordering does not hold. A single small validation split was not enough to rank "
                "these models. (Phase-1 test results were not used for this or any Phase-2 analysis.)"
            )
    agree = [
        x
        for s, a in cv["selection_agreement_with_primary"].items()
        if s != cv["primary_strategy"]
        for x in a.values()
    ]
    out.append(
        f"Model *rankings* do not depend on the endpoint strategy (Kendall τ {_tau_range(cv)}), "
        f"but the *selected hyperparameters* do: another strategy would have picked the same config in only "
        f"{min(agree):.0%}–{max(agree):.0%} of outer folds, so validation-error landscapes are flat relative to "
        "fold noise."
    )
    out.append(
        "Engine-bootstrap rank-1 shares are much more confident than fold rank-1 shares because the bootstrap "
        "ignores training-set variability; read the fold spread as the more honest stability measure."
    )
    return out


# --------------------------------------------------------------------------- report


def build_phase2_report(
    root: str | Path = "results/phase2",
    out: str | Path = "reports",
    cv_name="fd001_nested_cv",
    ablation_name="fd001_ablation",
    conformal_name="fd001_conformal",
    bench_name="fd001_latency",
    repro_name="frozen_configs",
    readme: str | Path | None = "README.md",
) -> Path:
    root, out = Path(root), Path(out)
    figdir = out / "figures" / "phase2"
    figdir.mkdir(parents=True, exist_ok=True)
    P.style()
    rel = lambda p: str(Path(p).relative_to(out))  # noqa: E731
    lines: list[str] = []
    add = lines.append
    add("# EngineRUL — Phase 2: validation-only evaluation and reliability\n")
    add(
        "> **Everything in this report uses FD001 *training* engines only.** The official FD001 test set was "
        "evaluated once, in Phase 1, with the frozen configurations; those results, the frozen configs and the "
        "test-evaluation log are unchanged and are reported only in [`report.md`](report.md). No Phase-2 "
        "experiment read the official test files, and nothing here was used to change the Phase-1 models. "
        "The numbers below are therefore **not comparable** to published FD001 test scores: they use capped truth "
        "and many truncation points per engine.\n"
    )
    add(
        "Generated by `engine-rul report-phase2` from `results/phase2/`. Method details and assumptions: "
        "[`docs/phase2.md`](../docs/phase2.md).\n"
    )

    # ---------------------------------------------------------------- CV
    cv_dir = root / "cv" / cv_name
    cv, cv_man = _load(cv_dir, "summary.json", "manifest.json")
    if cv:
        fm = pd.read_csv(cv_dir / "fold_metrics.csv")
        fams = [
            f for f in P.FAMILY_ORDER if f in cv["strategies"][cv["primary_strategy"]]["families"]
        ]
        prim = cv["strategies"][cv["primary_strategy"]]
        spec = cv_man["spec"]
        ep = spec["endpoints"]
        add("## Key findings\n")
        for line in key_findings(cv, root.parent / "final" / "summary.json"):
            add(f"* {line}")
        add("")
        add("## 1. Protocol\n")
        add(
            f"* **Outer loop:** {cv['n_repeats']} repeats × {spec.get('n_folds', 5)}-fold engine-grouped CV "
            f"(`{spec['fold_file']}`, sha256 `{cv_man['fold_file_sha256'][:12]}`); folds are lifetime-stratified, "
            f"each of the {cv['n_engines']} training engines is held out exactly once per repeat."
        )
        fr = spec.get("inner_fractions", {"train": 0.8, "val": 0.2})
        add(
            f"* **Inner loop (nested):** each outer fold's development engines are split {_fracs(fr)} by engine. "
            "Preprocessing (sensor selection, z-scoring) is fit on inner-train engines only; hyperparameters are "
            "re-selected per outer fold from the Phase-1 grids on inner-validation endpoints; LSTM early stopping "
            f"uses the same inner-validation endpoints (`early_stopping: {spec.get('early_stopping', 'dense')}`). "
            "The outer engines influence nothing but the score."
        )
        add(
            f"* **Endpoints:** {ep['n_per_engine']} truncation cycles per engine (cycle ≥ {ep['min_cycle']}), drawn "
            f"with a fixed seed per (strategy, engine); every model and variant is scored on identical endpoints. "
            f"Selection uses `{cv['primary_strategy']}` endpoints; other strategies are scored, never selected on."
        )
        add(
            f"* **Truth:** RUL capped at {cv['truth_cap']} (the training target) unless stated as uncapped."
        )
        add(
            "* **Statistics at the engine level:** per-engine mean errors first, then engines are averaged with equal "
            "weight; confidence intervals resample engines (cluster bootstrap, 2000 resamples). Windows and "
            "endpoints are never treated as independent samples. Because CV training sets overlap, these CIs ignore "
            "training-set variability and are **approximate (likely somewhat narrow)**."
        )
        add(f"* Provenance: {_prov(cv_man)}.\n")

        add("## 2. Cross-validation results\n")
        rows = []
        for f in fams:
            v = prim["families"][f]
            fr_ = v["fold_rmse_capped"]
            rows.append(
                [
                    f"**{_L(f)}**",
                    _f(v["pooled"]["capped"]["rmse"]),
                    _ci(v["rmse_ci95_engine_bootstrap"]["capped"]),
                    f"{fr_['mean']:.2f} ± {fr_['std']:.2f} ({fr_['min']:.1f}–{fr_['max']:.1f})",
                    _f(v["pooled"]["capped"]["mae"]),
                    _f(v["pooled"]["capped"]["nasa_mean"]),
                    _f(v["pooled"]["capped"]["bias"]),
                    _f(v["pooled"]["uncapped"]["rmse"]),
                    " / ".join(f"{x:.2f}" for x in v["repeat_mean_rmse_capped"]),
                ]
            )
        add(
            _table(
                [
                    "Model",
                    "RMSE",
                    "95% CI†",
                    "Fold RMSE mean ± sd (range)",
                    "MAE",
                    "NASA / point",
                    "Bias",
                    "RMSE (uncapped truth)",
                    "Mean fold RMSE per repeat",
                ],
                rows,
            )
        )
        add(
            f"\n{prim['n_points']} held-out endpoints per model in total ({cv['n_repeats']} repeats × "
            f"{cv['n_engines']} engines × {ep['n_per_engine']}; {cv['n_outer_folds']} outer folds). † engine cluster bootstrap over pooled predictions (approximate).\n"
        )
        add(f"![CV folds]({rel(fig_cv_folds(fm, fams, figdir / 'cv_fold_rmse.png'))})\n")
        add(
            "**Hyperparameters selected inside the outer folds** (how often each config won the inner validation):\n"
        )
        for f in ("ridge", "hgb", "lstm"):
            if f in cv["selected_params"]:
                ps = sorted(cv["selected_params"][f].items(), key=lambda kv: -kv[1])
                add(
                    f"* {_L(f)}: "
                    + "; ".join(f"`{k}` × {n}" for k, n in ps[:4])
                    + (f"; {len(ps) - 4} other configs" if len(ps) > 4 else "")
                )
        add("")

        add("## 3. Ranking stability and paired comparisons\n")
        rk = prim["ranking"]
        add(
            _table(
                [
                    "Model",
                    "Pooled rank",
                    "Mean rank over outer folds",
                    "Rank-1 share (folds)",
                    "Rank-1 share (engine bootstrap)",
                ],
                [
                    [
                        _L(f),
                        rk[f]["pooled_rank"],
                        _f(rk[f]["mean_rank"]),
                        f"{rk[f]['rank1_frac']:.0%}",
                        _pct(rk[f]["bootstrap_rank1_frac"]),
                    ]
                    for f in fams
                ],
            )
        )
        add(
            "\nPaired comparisons on the same held-out engines and endpoints (Δ = RMSE(a) − RMSE(b), capped truth, "
            "engine cluster bootstrap):\n"
        )
        prs = [
            p
            for p in prim["paired"]
            if p["truth"] == "capped"
            and p["a"] in ("ridge", "hgb", "lstm")
            and p["b"] in ("ridge", "hgb", "lstm", "life_prior")
        ]
        add(
            _table(
                ["a", "b", "Δ RMSE", "95% CI", "CI excludes 0"],
                [
                    [
                        _L(p["a"]),
                        _L(p["b"]),
                        f"{p['delta_rmse']:+.2f}",
                        _ci(p["ci95"]),
                        "yes" if p["ci_excludes_0"] else "no",
                    ]
                    for p in prs
                ],
            )
        )
        add("")

        add("## 4. Validation strategy comparison\n")
        add(
            "| Strategy | What it measures | Trade-off |\n|---|---|---|\n"
            "| Uniform | average accuracy over an engine's observable life (cycle ≥ min) | dominated by early/mid life where truth is capped and errors are small |\n"
            "| Late-life | accuracy when RUL ≤ 125, i.e. where maintenance decisions happen | ignores the plateau; fewer distinct cycles per engine |\n"
            "| Stage-stratified | equal attention to each 25-cycle RUL stage | weights rare very-late stages up; the mix is a design choice |\n"
        )
        strats = list(cv["strategies"])
        rows = []
        for f in fams:
            rows.append(
                [_L(f)]
                + [
                    f"{cv['strategies'][s]['families'][f]['pooled']['capped']['rmse']:.2f} "
                    f"(#{cv['strategies'][s]['ranking'][f]['pooled_rank']})"
                    for s in strats
                ]
            )
        add(_table(["Model"] + [STRAT_LABELS[s] for s in strats], rows))
        add(
            "\nPooled RMSE (capped truth) and rank, same outer-fold models scored on each endpoint set.\n"
        )
        add(
            f"![Strategies]({rel(fig_strategies(cv, fams, figdir / 'validation_strategies.png'))})\n"
        )
        add(
            "Rank agreement between strategies (Kendall τ over the 5 models): "
            + "; ".join(
                f"{STRAT_LABELS[r['a']]} vs {STRAT_LABELS[r['b']]} τ = {r['kendall_tau']:.2f}"
                for r in cv["strategy_rank_agreement"]
            )
            + ".\n"
        )
        agree = cv["selection_agreement_with_primary"]
        add(
            "If the inner-validation endpoints had used another strategy, the selected hyperparameters would have "
            "been the same in this share of outer folds: "
            + "; ".join(
                f"{STRAT_LABELS[s]}: " + ", ".join(f"{_L(f)} {v:.0%}" for f, v in a.items())
                for s, a in agree.items()
                if s != cv["primary_strategy"]
            )
            + ".\n"
        )
        late_pairs = [
            p
            for p in cv["strategies"].get("late", {}).get("paired", [])
            if p["truth"] == "capped" and {p["a"], p["b"]} <= {"ridge", "hgb", "lstm"}
        ]
        if late_pairs:
            add(
                "Paired comparisons on **late-life** endpoints: "
                + "; ".join(
                    f"{_L(p['a'])} − {_L(p['b'])} = {p['delta_rmse']:+.2f} {_ci(p['ci95'])}"
                    for p in late_pairs
                )
                + ".\n"
            )

    # ---------------------------------------------------------------- ablation
    ab_dir = root / "ablation" / ablation_name
    ab, ab_man = _load(ab_dir, "summary.json", "manifest.json")
    if ab:
        add("## 5. Ablations\n")
        add(
            "Each variant reuses the hyperparameters selected on inner validation for that (repeat, fold, model) in "
            "the CV run, the same inner roles and the same held-out endpoints (uniform, capped truth), and changes "
            "one thing. Tabular models: 3 repeats; LSTM: repeat 0 (5 outer folds). Δ is paired on engines.\n"
        )
        chk = ab["base_vs_source_cv_max_abs_diff"]
        add(
            "Determinism check — recomputed `base` vs the CV run's predictions, max |difference|: "
            + ", ".join(f"{_L(f)} {v:.2g}" for f, v in chk.items())
            + ".\n"
        )
        for f in [x for x in P.FAMILY_ORDER if x in ab["families"]]:
            add(f"**{_L(f)}**\n")
            add(
                _table(
                    [
                        "Variant",
                        "RMSE",
                        "Δ vs base",
                        "95% CI",
                        "CI excludes 0",
                        "Fold RMSE mean ± sd",
                    ],
                    [
                        [
                            r["variant"],
                            _f(r["rmse"]),
                            f"{r['delta_rmse']:+.2f}",
                            _ci(r["ci95"]),
                            "–"
                            if r["variant"] == "base"
                            else ("yes" if r["ci_excludes_0"] else "no"),
                            f"{r['fold_rmse_mean']:.2f} ± {r['fold_rmse_std']:.2f}",
                        ]
                        for r in ab["families"][f]
                    ],
                )
            )
            add("")
        add(f"![Ablations]({rel(fig_ablation(ab, figdir / 'ablations.png'))})\n")
        add(
            "Variant definitions: `sensors_only` drops the normalized-cycle channel; `cycle_only` keeps only it "
            "(tabular: its value at the end cycle); `stats_*` restrict the per-sensor window statistics (default "
            "last, mean, std, slope); `window_20/40` change the LSTM sequence length (default 30); `layers_1/2` fix "
            "the LSTM depth; `early_stop_dense` stops on dense inner-validation windows instead of the selection "
            f"endpoints. Provenance: {_prov(ab_man)}.\n"
        )

    # ---------------------------------------------------------------- conformal
    cf_dir = root / "conformal" / conformal_name
    conf, cf_cv, cf_man = _load(cf_dir, "conformal_summary.json", "summary.json", "manifest.json")
    if conf:
        add("## 6. Prediction intervals (split conformal)\n")
        fr = cf_man["spec"]["inner_fractions"]
        add(
            f"Repeat 0 of the same outer folds. Development engines split {_fracs(fr)} by engine: models are fit on "
            "inner-train, selected/early-stopped on inner-validation, and **calibrated on engines used for "
            "nothing else**; coverage is measured on the outer engines. Score = |capped truth − prediction|; "
            "interval = prediction ± q, lower end clipped at 0.\n"
        )
        s = pd.DataFrame(conf["summary"])
        fams_c = [f for f in P.FAMILY_ORDER if f in set(s.family)]
        for strat in ("uniform", "late"):
            g = s[s.strategy == strat]
            rows = []
            for f in fams_c:
                for m in conf["methods"]:
                    h = g[(g.family == f) & (g.method == m)].set_index("level")
                    rows.append(
                        [_L(f), METHOD_LABELS[m]]
                        + [
                            f"{h.loc[lv, 'coverage']:.3f} ({h.loc[lv, 'coverage_fold_min']:.2f}–"
                            f"{h.loc[lv, 'coverage_fold_max']:.2f}) / {h.loc[lv, 'mean_width']:.1f}"
                            for lv in conf["levels"]
                        ]
                    )
            add(
                f"**{STRAT_LABELS[strat]} held-out endpoints** — coverage (fold range) / mean width in cycles:\n"
            )
            add(_table(["Model", "Method"] + [f"{lv:.0%}" for lv in conf["levels"]], rows))
            add("")
        add(f"![Coverage]({rel(fig_coverage(conf, figdir / 'conformal_coverage.png'))})\n")
        add(
            f"![Stage coverage]({rel(fig_stage_coverage(conf, figdir / 'conformal_stage_coverage.png'))})\n"
        )
        iv = fig_intervals(cf_dir, figdir / "conformal_intervals.png")
        if iv is None and (figdir / "conformal_intervals.png").exists():
            iv = (
                figdir / "conformal_intervals.png"
            )  # fold models are not committed; keep the last rendering
        if iv is not None:
            add(f"![Intervals]({rel(iv)})\n")
        ps = pd.DataFrame(conf["per_stage"])
        ps = ps[
            (ps.method == "global_one_per_engine") & (ps.level == 0.9) & (ps.strategy == "uniform")
        ]
        add(
            _table(
                ["Stage (true RUL)"] + [_L(f) for f in fams_c],
                [
                    [st]
                    + [
                        f"{ps[(ps.stage == st) & (ps.family == f)]['coverage'].iloc[0]:.2f}"
                        for f in fams_c
                    ]
                    for st in dict.fromkeys(ps["stage"])
                ],
            )
        )
        add(
            "\nCoverage of 90% intervals by true-RUL stage (global, one endpoint per calibration engine).\n"
        )
        add(
            "**How to read this.** `global_one_per_engine` is the only variant with a finite-sample guarantee: it "
            "uses one randomly drawn endpoint per calibration engine, so if engines are exchangeable and a new "
            "engine is cut at a point drawn the same way, its interval covers the (capped) truth with probability "
            "≥ the nominal level — *marginally*, averaged over engines and cut points. The pooled variants use all "
            "endpoints of an engine, which are dependent, so they have no exact guarantee and are judged "
            "empirically. No variant promises coverage for a *particular* stage or for differently drawn cut points "
            "(the late-life rows), and none applies to the official test set, whose truncation process differs. "
            "Coverage is reported against capped truth; against uncapped truth it is lower early in life by "
            "construction. The guarantee is also averaged over the random calibration set: with only 20 "
            "calibration scores the realised coverage of any one calibrated model varies a lot (see the fold "
            "ranges above), which is the price of the exact-guarantee variant on a 100-engine dataset.\n"
        )
        add(
            "**Performance uncertainty vs prediction uncertainty.** The confidence intervals in §2–5 describe how "
            "well we know a model's *average* error (e.g. RMSE ± a few cycles, shrinking with more engines). A "
            "prediction interval describes where *one engine's* RUL is likely to lie (tens of cycles wide, and it "
            "does not shrink with more data, only with a better model).\n"
        )

    # ---------------------------------------------------------------- benchmarks
    b_dir = root / "benchmark" / bench_name
    (b,) = _load(b_dir, "benchmarks.json")
    if b:
        add("## 7. Computational benchmarks\n")
        for k, v in b["scope_definitions"].items():
            add(f"* **{'Model inference' if k == 'model' else 'End-to-end'}:** {v}.")
        add(f"* Memory: {b['memory_note']}; process peak RSS {b['process_max_rss_mb']:.0f} MB.")
        env = b["environment"]
        add(
            f"* Phase-1 median-seed models; inputs are training-file sensor rows. {env.get('cpu', '')} "
            f"({env.get('os', '')}), torch {env.get('torch', '')}, scikit-learn {env.get('scikit_learn', '')}; 5 warm-up calls, 20–50 timed repetitions, "
            "device synchronisation after every call.\n"
        )
        r = pd.DataFrame(b["rows"])
        r = r[
            ~((r.scope == "model") & r.family.isin(["constant", "life_prior"]))
        ]  # trivial arithmetic
        add(
            "Model-only rows for the two baselines are omitted (they are a few arithmetic operations). Their "
            "end-to-end rows are kept: they measure the **pipeline overhead alone** (preprocessing, windowing, "
            "output frame), which dominates end-to-end latency for every model except gradient boosting.\n"
        )
        r["family"] = r["family"].map(_L)
        add(
            _table(
                [
                    "Model",
                    "Device",
                    "Scope",
                    "Batch",
                    "Median ms",
                    "p95 ms",
                    "Throughput / s",
                    "Py peak MB",
                ],
                [
                    [
                        x.family,
                        x.device,
                        x.scope,
                        f"{x.n} {x.unit}",
                        f"{x.latency_ms_median:.3g}",
                        f"{x.latency_ms_p95:.3g}",
                        f"{x.throughput_per_s:,.0f}",
                        f"{x.python_peak_alloc_mb:.1f}",
                    ]
                    for x in r.itertuples()
                ],
            )
        )
        add(f"\n![Benchmarks]({rel(fig_bench(b, figdir / 'latency.png'))})\n")

    # ---------------------------------------------------------------- reproducibility
    (rp,) = _load(root / "repro" / repro_name, "repro.json")
    add("## 8. Reproducibility\n")
    if rp:
        add(
            f"`engine-rul repro-check` retrained the frozen configurations into an isolated directory and compared "
            f"validation RMSE with `results/final/summary.json` (tolerance {rp['tolerance']:g}):\n"
        )
        add(
            _table(
                ["Model", "Seed", "Published", "Reproduced", "|Δ|", "Config hash match", "OK"],
                [
                    [
                        _L(x["family"]),
                        x["seed"],
                        f"{x['published_val_rmse']:.6f}",
                        f"{x['reproduced_val_rmse']:.6f}",
                        f"{x['abs_diff']:.1e}",
                        x["config_hash_matches"],
                        "✅" if x["ok"] else "❌",
                    ]
                    for x in rp["rows"]
                ],
            )
        )
        add("")
    add(
        "Every Phase-2 experiment directory has a `manifest.json` with the git commit, dirty flag and (when dirty) "
        "a content-addressed source snapshot, the spec and its hash, fold-file and data checksums, library "
        "versions and device. Experiment directories are never overwritten without `--force`.\n"
    )

    add("## 9. Limitations\n")
    add(
        "* FD001 only (one operating condition, one fault mode, 100 training engines); 20 engines per outer fold."
    )
    add(
        "* Engine bootstrap CIs on pooled CV predictions ignore the dependence created by overlapping training sets "
        "and by hyperparameter selection noise; they are approximate. Fold-to-fold spread is shown alongside."
    )
    add(
        "* All CV numbers use capped truth and several endpoints per engine; they are not comparable to the "
        "official last-cycle test metric, and none of them was used to revisit Phase-1 choices."
    )
    add(
        "* LSTM ablations use one repeat; LSTM training is seeded and deterministic on CPU in this environment, but "
        "seed-to-seed variation is not separated from fold variation."
    )
    add(
        "* Conformal guarantees are marginal and rest on engine exchangeability and on how cut points are drawn; "
        "simulated C-MAPSS engines say nothing about real aircraft maintenance."
    )
    p = out / "phase2_report.md"
    p.write_text("\n".join(lines) + "\n")
    if readme is not None and cv:
        _update_readme(Path(readme), readme_block(cv, ab, conf))
    return p


START, END = "<!-- PHASE2:START -->", "<!-- PHASE2:END -->"


def _update_readme(readme: Path, block: str) -> None:
    text = readme.read_text()
    if START in text and END in text:
        i, j = text.index(START) + len(START), text.index(END)
        readme.write_text(text[:i] + "\n" + block + "\n" + text[j:])


def readme_block(cv: dict, ab: dict | None, conf: dict | None) -> str:
    prim = cv["strategies"][cv["primary_strategy"]]
    fams = [f for f in P.FAMILY_ORDER if f in prim["families"]]
    rows = []
    for f in fams:
        v, rk = prim["families"][f], prim["ranking"][f]
        rows.append(
            [
                f"**{_L(f)}**",
                f"{v['fold_rmse_capped']['mean']:.2f} ± {v['fold_rmse_capped']['std']:.2f}",
                _ci(v["rmse_ci95_engine_bootstrap"]["capped"]),
                f"{rk['rank1_frac']:.0%}",
            ]
        )
    out = [
        _table(
            ["Model", "CV RMSE (fold mean ± sd)", "Pooled 95% CI (approx.)", "Best in % of folds"],
            rows,
        ),
        "",
        f"<sub>{cv['n_repeats']} × 5-fold nested, engine-grouped CV on the {cv['n_engines']} training engines; uniform "
        f"truncation endpoints; truth capped at {cv['truth_cap']}. Source: `results/phase2/cv/`.</sub>",
        "",
    ]
    for p_ in prim["paired"]:
        if p_["truth"] == "capped" and {p_["a"], p_["b"]} == {"hgb", "lstm"}:
            verdict = (
                "the CI excludes 0"
                if p_["ci_excludes_0"]
                else "the CI includes 0, so no clear difference"
            )
            out.append(
                f"- {_L(p_['a'])} − {_L(p_['b'])}: Δ RMSE = {p_['delta_rmse']:+.2f}, paired engine-bootstrap "
                f"95% CI {_ci(p_['ci95'])}; {verdict}."
            )
    out.append(
        f"- Model rankings across uniform / late-life / stage-stratified validation endpoints: Kendall τ "
        f"{_tau_range(cv)}."
    )
    if ab:
        for f, rows_ in ab["families"].items():
            sig = [r for r in rows_ if r["variant"] != "base" and r["ci_excludes_0"]]
            worse = ", ".join(
                f"`{r['variant']}` ({r['delta_rmse']:+.1f})" for r in sig if r["delta_rmse"] > 0
            )
            better = ", ".join(
                f"`{r['variant']}` ({r['delta_rmse']:+.1f})" for r in sig if r["delta_rmse"] < 0
            )
            out.append(
                f"- Ablation, {_L(f)}: clearly worse than the full model: {worse or 'none'}; clearly better: "
                f"{better or 'none'} (Δ RMSE, paired CI excludes 0)."
            )
        out.append(
            "- Ablation improvements are validation-only observations; the frozen Phase-1 models were not "
            "changed, and adopting a variant would need a new frozen config and a fresh held-out test."
        )
    if conf:
        s = pd.DataFrame(conf["summary"])
        g = s[(s.method == "global_one_per_engine") & (s.level == 0.9)]
        parts = [
            f"{_L(f)} {g[(g.family == f) & (g.strategy == 'uniform')]['coverage'].iloc[0]:.0%} "
            f"(mean width {g[(g.family == f) & (g.strategy == 'uniform')]['mean_width'].iloc[0]:.0f} cycles)"
            for f in P.FAMILY_ORDER
            if f in set(g.family)
        ]
        out.append(
            "- 90% split-conformal prediction intervals, empirical coverage on held-out engines: "
            + ", ".join(parts)
            + "."
        )
    return "\n".join(out)
