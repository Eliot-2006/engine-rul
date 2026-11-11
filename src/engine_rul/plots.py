"""All figures (PRD 9.1). Matplotlib only; every figure is generated from saved data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine_rul.data import SENSOR_DESCRIPTIONS  # noqa: E402

FAMILY_ORDER = ["constant", "life_prior", "ridge", "hgb", "lstm"]
FAMILY_LABELS = {
    "constant": "Constant mean",
    "life_prior": "Life prior (cycle only)",
    "ridge": "Ridge",
    "hgb": "Gradient boosting",
    "lstm": "LSTM",
}
_tab = plt.get_cmap("tab10").colors
FAMILY_COLORS = {
    "constant": _tab[7],
    "life_prior": _tab[8],
    "ridge": _tab[0],
    "hgb": _tab[2],
    "lstm": _tab[3],
}
INK = "#222222"
GRID = "#d9d9d9"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#888888",
            "axes.labelcolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _short(s: str) -> str:
    return SENSOR_DESCRIPTIONS.get(s, s).split(" – ")[0]


# --------------------------------------------------------------------------- EDA


def lifetime_hist(train_life, test_len, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.arange(0, max(train_life.max(), test_len.max()) + 20, 15)
    ax.hist(
        train_life,
        bins=bins,
        alpha=0.75,
        color=_tab[0],
        label=f"Training engines: lifetime (n={len(train_life)})",
    )
    ax.hist(
        test_len,
        bins=bins,
        alpha=0.6,
        color=_tab[1],
        label=f"Test engines: observed length (n={len(test_len)})",
    )
    ax.set(xlabel="Cycles", ylabel="Engines", title="FD001 trajectory lengths")
    ax.legend()
    return _save(fig, path)


def sensor_trajectories(df, sensors, path, n_engines=10, seed=0):
    """Sensors vs cycles-to-failure for sampled engines plus the cross-engine median."""
    rng = np.random.default_rng(seed)
    units = rng.choice(
        df["unit"].unique(), size=min(n_engines, df["unit"].nunique()), replace=False
    )
    ncol = 4
    nrow = int(np.ceil(len(sensors) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.4 * nrow), sharex=True)
    med = df.assign(rul=df["rul"].astype(int)).groupby("rul")[sensors].median()
    med = med[med.index <= 300]
    for ax, s in zip(axes.flat, sensors, strict=False):
        for u in units:
            g = df[df["unit"] == u]
            ax.plot(-g["rul"], g[s], color=_tab[0], alpha=0.25, lw=0.7)
        ax.plot(-med.index, med[s], color=_tab[3], lw=1.6)
        ax.set_title(f"{s}: {_short(s)}", fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(sensors) :]:
        ax.set_visible(False)
    for col in range(ncol):  # x labels on the last *visible* panel of each column
        last = max(i for i in range(col, len(sensors), ncol))
        axes.flat[last].set_xlabel("Cycles relative to failure")
        axes.flat[last].xaxis.set_tick_params(labelbottom=True)
    fig.suptitle(
        "Sensor degradation trajectories (train-split engines; red = median across engines)", y=1.0
    )
    fig.tight_layout()
    return _save(fig, path)


def sensor_distributions(df, sensors, path):
    ncol = 4
    nrow = int(np.ceil(len(sensors) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.2 * nrow))
    for ax, s in zip(axes.flat, sensors, strict=False):
        ax.hist(df[s], bins=40, color=_tab[0], alpha=0.85)
        ax.set_title(f"{s}: {_short(s)}", fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(sensors) :]:
        ax.set_visible(False)
    fig.suptitle("Sensor value distributions (train-split rows)", y=1.0)
    fig.tight_layout()
    return _save(fig, path)


def correlation_heatmap(df, cols, path):
    c = df[cols].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(
                j,
                i,
                f"{c[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if abs(c[i, j]) > 0.6 else INK,
            )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Correlation of informative sensors and RUL (train split)")
    return _save(fig, path)


def sensor_rul_correlation(rho: pd.Series, path):
    rho = rho.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(
        [f"{s} ({_short(s)})" for s in rho.index],
        rho.values,
        color=[_tab[3] if v < 0 else _tab[0] for v in rho.values],
    )
    ax.axvline(0, color=INK, lw=0.8)
    ax.set(
        xlabel="Spearman ρ with uncapped RUL",
        title="Sensor–RUL monotonic association (train split)",
    )
    return _save(fig, path)


def rul_target_shapes(df, units, r_max, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, u in enumerate(units):
        g = df[df["unit"] == u]
        ax.plot(g["cycle"], g["rul"], color=_tab[i], lw=1, ls="--", alpha=0.7)
        ax.plot(g["cycle"], np.minimum(g["rul"], r_max), color=_tab[i], lw=2, label=f"Engine {u}")
    ax.axhline(r_max, color="#888888", lw=0.8, ls=":")
    ax.text(2, r_max + 4, f"R_max = {r_max}", fontsize=8, color="#555555")
    ax.set(
        xlabel="Cycle",
        ylabel="RUL (cycles)",
        title="Training target: uncapped (dashed) vs piecewise-linear (solid)",
    )
    ax.legend()
    return _save(fig, path)


# --------------------------------------------------------------------------- results


def loss_curves(histories: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, (seed, h) in enumerate(sorted(histories.items())):
        h = pd.DataFrame(h)
        c = plt.get_cmap("Reds")(0.45 + 0.1 * i)
        axes[0].plot(h["epoch"], h["train_loss"], color=c, label=f"seed {seed}")
        axes[1].plot(h["epoch"], h["val_rmse"], color=c, label=f"seed {seed}")
        best = h.loc[h["val_rmse"].idxmin()]
        axes[1].scatter(best["epoch"], best["val_rmse"], color=c, edgecolor=INK, zorder=3, s=30)
    axes[0].set(
        xlabel="Epoch",
        ylabel="MSE on scaled target (RUL / R_max)",
        title="LSTM training loss",
        yscale="log",
    )
    axes[1].set(
        xlabel="Epoch",
        ylabel="RMSE (cycles, capped truth)",
        title="LSTM validation RMSE (dense windows; dot = restored best epoch)",
    )
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def pred_vs_true(preds: pd.DataFrame, r_max, path, families):
    fig, axes = plt.subplots(
        1, len(families), figsize=(3.2 * len(families), 3.4), sharex=True, sharey=True
    )
    lim = max(preds["y_true"].max(), preds["y_pred"].max()) * 1.05
    for ax, f in zip(np.atleast_1d(axes), families, strict=True):
        p = preds[preds["family"] == f]
        ax.plot([0, lim], [0, lim], color="#888888", lw=0.8)
        if r_max:
            ax.axhline(r_max, color="#888888", lw=0.8, ls=":")
        ax.scatter(p["y_true"], p["y_pred"], s=12, alpha=0.8, color=FAMILY_COLORS[f])
        ax.set(title=FAMILY_LABELS[f], xlabel="True RUL (cycles)", xlim=(0, lim), ylim=(0, lim))
    np.atleast_1d(axes)[0].set_ylabel("Predicted RUL (cycles)")
    fig.suptitle(
        "Official FD001 test set: predicted vs true RUL at the last observed cycle (median seed)",
        y=1.02,
    )
    fig.tight_layout()
    return _save(fig, path)


def test_engines_sorted(preds: pd.DataFrame, path, families):
    ref = preds[preds["family"] == families[0]].sort_values("y_true").reset_index(drop=True)
    order = ref["unit"].tolist()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(range(len(order)), ref["y_true"], color=INK, lw=2, label="True RUL")
    for f in families:
        p = preds[preds["family"] == f].set_index("unit").loc[order]
        ax.plot(
            range(len(order)),
            p["y_pred"],
            marker="o",
            ms=2.5,
            lw=0.8,
            alpha=0.85,
            color=FAMILY_COLORS[f],
            label=FAMILY_LABELS[f],
        )
    ax.set(
        xlabel="Test engines sorted by true RUL",
        ylabel="RUL (cycles)",
        title="Per-engine predictions on the official test set (median seed)",
    )
    ax.legend(ncol=3, fontsize=8)
    return _save(fig, path)


def error_distribution(preds: pd.DataFrame, path, families):
    fig, ax = plt.subplots(figsize=(8, 4))
    err = {
        f: (
            preds[preds["family"] == f]["y_pred"] - preds[preds["family"] == f]["y_true"]
        ).to_numpy()
        for f in families
    }
    lo = min(e.min() for e in err.values())
    hi = max(e.max() for e in err.values())
    bins = np.linspace(lo, hi, 31)
    ax.axvspan(0, hi, color=_tab[3], alpha=0.06)
    ax.text(
        hi * 0.97,
        0.5,
        "late (over-estimates RUL)",
        ha="right",
        fontsize=8,
        color=_tab[3],
        transform=ax.get_xaxis_transform(),
    )
    for f in families:
        ax.hist(
            err[f],
            bins=bins,
            histtype="step",
            lw=1.8,
            color=FAMILY_COLORS[f],
            label=FAMILY_LABELS[f],
        )
    ax.axvline(0, color=INK, lw=0.8)
    ax.set(
        xlabel="Error d = predicted − true (cycles)",
        ylabel="Test engines",
        title="Test-set error distribution (median seed)",
    )
    ax.legend(fontsize=8)
    return _save(fig, path)


def error_by_stage(summary: dict, path, families):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    w = 0.8 / len(families)
    for ax, key, title in (
        (axes[0], "stage_metrics_test", "Official test set (100 engines)"),
        (axes[1], "stage_metrics_val_dense", "Validation engines, every window"),
    ):
        bins = [r["bin"] for r in summary[key][families[0]]]
        ns = [r["n"] for r in summary[key][families[0]]]
        for i, f in enumerate(families):
            vals = [r["rmse"] if r["rmse"] is not None else np.nan for r in summary[key][f]]
            ax.bar(
                np.arange(len(bins)) + (i - (len(families) - 1) / 2) * w,
                vals,
                w,
                color=FAMILY_COLORS[f],
                label=FAMILY_LABELS[f],
            )
        ax.set_xticks(
            range(len(bins)), [f"{b}\nn={n}" for b, n in zip(bins, ns, strict=True)], fontsize=8
        )
        ax.set(xlabel="True RUL bin (cycles)", title=title)
    axes[0].set_ylabel("RMSE (cycles, uncapped truth)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Error by degradation stage (bins with n < 5 omitted on test)", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def model_comparison(summary: dict, path, families):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, label in (
        (axes[0], "rmse", "RMSE (cycles)"),
        (axes[1], "nasa_mean", "NASA score per engine"),
    ):
        for i, f in enumerate(families):
            fam = summary["families"][f]
            per = fam["test_primary"][metric]["per_seed"]
            lo, hi = fam["test_ci95_median_seed"][metric]
            mean = fam["test_primary"][metric]["mean"]
            ax.errorbar(
                i,
                mean,
                yerr=[[mean - lo], [hi - mean]],
                fmt="o",
                ms=8,
                color=FAMILY_COLORS[f],
                capsize=5,
                lw=1.5,
                zorder=3,
            )
            jitter = np.linspace(-0.12, 0.12, len(per)) if len(per) > 1 else [0]
            ax.scatter(
                i + np.asarray(jitter) + 0.25,
                per,
                s=12,
                color=FAMILY_COLORS[f],
                alpha=0.6,
                zorder=2,
            )
        ax.set_xticks(
            range(len(families)), [FAMILY_LABELS[f] for f in families], rotation=20, ha="right"
        )
        ax.set(ylabel=label + " — lower is better")
    axes[0].set_title("Test RMSE: mean of seeds, 95% bootstrap CI (dots = seeds)")
    axes[1].set_title("Test NASA score (mean per engine)")
    axes[1].set_yscale("log")
    fig.tight_layout()
    return _save(fig, path)


def feature_importance(perm: dict, path, top=15):
    fams = [f for f in ("ridge", "hgb") if f in perm]
    fig, axes = plt.subplots(1, len(fams), figsize=(6 * len(fams), 5))
    for ax, f in zip(np.atleast_1d(axes), fams, strict=True):
        items = sorted(perm[f].items(), key=lambda kv: kv[1]["mean"], reverse=True)[:top][::-1]
        names = [k for k, _ in items]
        ax.barh(
            names,
            [v["mean"] for _, v in items],
            xerr=[v["std"] for _, v in items],
            color=FAMILY_COLORS[f],
            alpha=0.85,
        )
        ax.set(
            xlabel="Increase in validation RMSE when permuted (cycles)",
            title=f"{FAMILY_LABELS[f]}: top {top} features",
        )
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return _save(fig, path)


def val_engine_trajectories(vd: pd.DataFrame, path, families=("hgb", "lstm"), n=4):
    units = sorted(vd["unit"].unique())
    pick = [units[int(i)] for i in np.linspace(0, len(units) - 1, n)]
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4), sharey=True)
    for ax, u in zip(axes, pick, strict=True):
        ref = vd[(vd["unit"] == u) & (vd["family"] == families[0])]
        ax.plot(ref["cycle"], ref["y_true"], color="#888888", ls="--", lw=1, label="True RUL")
        ax.plot(ref["cycle"], ref["y_capped"], color=INK, lw=1.5, label="Capped target")
        for f in families:
            g = vd[(vd["unit"] == u) & (vd["family"] == f)]
            ax.plot(g["cycle"], g["y_pred"], color=FAMILY_COLORS[f], lw=1.2, label=FAMILY_LABELS[f])
        ax.set(title=f"Validation engine {u}", xlabel="Cycle")
    axes[0].set_ylabel("RUL (cycles)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Predictions at every cycle for held-out validation engines (median seed)", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def sensitivity(rows: list[dict], path):
    df = pd.DataFrame(rows)
    df["r_max_label"] = df["r_max"].apply(
        lambda v: "none" if v is None or (isinstance(v, float) and np.isnan(v)) else str(int(v))
    )
    fams = list(dict.fromkeys(df["family"]))
    fig, axes = plt.subplots(1, len(fams), figsize=(5.5 * len(fams), 3.8), sharey=True)
    for ax, f in zip(np.atleast_1d(axes), fams, strict=True):
        for j, w in enumerate(sorted(df["window"].unique())):
            g = df[(df["family"] == f) & (df["window"] == w)]
            ax.plot(
                g["r_max_label"],
                g["val_rmse_cap125"],
                marker="o",
                color=FAMILY_COLORS[f],
                alpha=0.5 + 0.5 * j,
                ls="-" if j else "--",
                label=f"W = {w}",
            )
        ax.set(title=FAMILY_LABELS[f], xlabel="Training cap R_max (cycles)")
        ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Val RMSE vs truth capped at 125 (cycles)")
    fig.suptitle("Sensitivity to target cap and window length (validation only, seed 0)", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def hparam_sensitivity(tunings: dict, path):
    fams = [f for f in ("ridge", "hgb", "lstm") if f in tunings and len(tunings[f]["rows"]) > 1]
    if not fams:
        return None
    fig, axes = plt.subplots(1, len(fams), figsize=(5 * len(fams), 3.8))
    for ax, f in zip(np.atleast_1d(axes), fams, strict=True):
        t = tunings[f]
        scores = [r["score"] for r in t["rows"]]
        abbrev = {
            "learning_rate": "lr",
            "max_leaf_nodes": "leaves",
            "min_samples_leaf": "min_leaf",
            "max_iter": "iters",
            "hidden_size": "H",
            "num_layers": "layers",
            "dropout": "p",
        }
        labels = [
            ", ".join(
                f"{abbrev.get(k, k)}={v}"
                for k, v in r["params"].items()
                if k != "l2_regularization"
            )
            for r in t["rows"]
        ]
        sel = [r["run_id"] == t["selected"]["run_id"] for r in t["rows"]]
        if f == "ridge":
            alphas = [r["params"]["alpha"] for r in t["rows"]]
            ax.plot(alphas, scores, marker="o", color=FAMILY_COLORS[f])
            ax.scatter(
                [a for a, s in zip(alphas, sel, strict=True) if s],
                [t["selected"]["score"]],
                s=90,
                facecolor="none",
                edgecolor=INK,
                zorder=3,
                label="selected",
            )
            ax.set(xscale="log", xlabel="alpha")
        else:
            y = np.arange(len(scores))
            ax.barh(y, scores, color=[INK if s else FAMILY_COLORS[f] for s in sel])
            ax.set_yticks(y, labels, fontsize=6)
            ax.set_xlim(min(scores) * 0.97, max(scores) * 1.01)
            ax.set_xlabel("Val RMSE (cycles)")
        ax.set_title(f"{FAMILY_LABELS[f]} ({len(scores)} configs; dark = selected)")
    np.atleast_1d(axes)[0].set_ylabel("Val RMSE, truncated points (cycles)")
    fig.tight_layout()
    return _save(fig, path)


def benchmarks(rows: list[dict], path):
    df = pd.DataFrame(rows)
    df = (
        df[df["device"] == "cpu"]
        .set_index("family")
        .reindex([f for f in FAMILY_ORDER if f in set(df["family"])])
    )
    metrics = [
        ("train_time_s", "Training time (s)"),
        ("latency_ms_median", "Latency, 1 engine (ms)"),
        ("artifact_bytes", "Artifact size (bytes)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (m, label) in zip(axes, metrics, strict=True):
        vals = df[m].clip(lower=1e-3)
        ax.barh(
            [FAMILY_LABELS[f] for f in df.index], vals, color=[FAMILY_COLORS[f] for f in df.index]
        )
        ax.set(xscale="log", xlabel=label + " (log scale)")
        for i, v in enumerate(df[m]):
            ax.text(vals.iloc[i] * 1.1, i, f"{v:,.3g}", va="center", fontsize=8)
    fig.suptitle("Computational cost on CPU (measured)", y=1.02)
    fig.tight_layout()
    return _save(fig, path)
