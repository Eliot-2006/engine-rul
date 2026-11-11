"""TEST-E2E-01: full pipeline on synthetic data (tune -> final -> locked test eval -> bench -> report),
plus the summary.json schema check and the EVAL-22 guard."""

import json
import re
import time

import pytest

from engine_rul.benchmark import run_benchmarks
from engine_rul.errors import ConfigError, TestSetLockError
from engine_rul.evaluate import evaluate_test_final
from engine_rul.report import build_report
from engine_rul.train import final, tune

FAMS = ["constant", "life_prior", "ridge", "hgb", "lstm"]
METRIC_KEYS = {"rmse", "mae", "nasa_sum", "nasa_mean", "bias", "late_fraction"}


def test_full_pipeline(synth_cfg, tmp_path):
    t0 = time.time()
    final_dir = tmp_path / "final_cfgs"
    for f in FAMS:
        tune(synth_cfg(f), None, out_dir=final_dir)
    results = synth_cfg("constant").results_dir
    final(FAMS, [0, 1], results_dir=results, final_dir=final_dir)
    log = tmp_path / "test_log.jsonl"
    summary_path = evaluate_test_final(
        FAMS, results, final_dir=final_dir, log_path=log, note="smoke"
    )
    s = json.loads(summary_path.read_text())
    # schema
    assert set(s["families"]) == set(FAMS)
    for fam in s["families"].values():
        assert METRIC_KEYS <= set(fam["test_primary"]) and METRIC_KEYS <= set(fam["test_capped"])
        assert len(fam["test_primary"]["rmse"]["per_seed"]) == len(fam["seeds"])
        lo, hi = fam["test_ci95_median_seed"]["rmse"]
        assert lo <= hi
    assert s["families"]["lstm"]["seeds"] == [0, 1] and s["families"]["ridge"]["seeds"] == [0]
    assert s["n_test_engines"] == 5 and s["paired_comparisons"]
    # one log line per (family, seed)
    entries = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(entries) == sum(len(fam["seeds"]) for fam in s["families"].values())
    assert all(e["note"] == "smoke" and e["config_hash"] for e in entries)
    run_benchmarks(FAMS, results)
    report = build_report(results, tmp_path / "reports", update_readme=False)
    text = report.read_text()
    assert not re.search(r"\bnan\b", text, flags=re.I)
    for section in (
        "Official test results",
        "Paired comparisons",
        "degradation stage",
        "Computational benchmarks",
    ):
        assert section in text
    for fig in (tmp_path / "reports" / "figures").rglob("*.png"):
        assert fig.stat().st_size > 1000
    assert time.time() - t0 < 120


def test_final_requires_frozen_config(synth_cfg, tmp_path):
    with pytest.raises(ConfigError, match="no frozen config"):
        final(["ridge"], [0], results_dir=str(tmp_path), final_dir=tmp_path / "empty")


def test_test_eval_rejects_non_final_runs(synth_cfg, tmp_path):
    from engine_rul.train import train_one

    cfg = synth_cfg("ridge")
    rec = train_one(cfg, "dev")
    final_dir = tmp_path / "f"
    tune(cfg, None, out_dir=final_dir)
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "final_runs.json").write_text(json.dumps({"ridge": [rec["run_id"]]}))
    with pytest.raises(TestSetLockError, match="stage=final"):
        evaluate_test_final(
            ["ridge"], cfg.results_dir, final_dir=final_dir, log_path=tmp_path / "l.jsonl"
        )
