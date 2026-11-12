"""Cross-artifact consistency: committed results, generated reports, README, dashboard data and the
dashboard page must all tell the same story. Reads committed files only (no dataset, no training)."""

from __future__ import annotations

import json
import math
import re
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from engine_rul import dashboard as DB
from engine_rul.plots import FAMILY_LABELS, FAMILY_ORDER

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUMMARY = json.loads((RESULTS / "final" / "summary.json").read_text())


@pytest.fixture(scope="module")
def payloads() -> dict[str, dict]:
    return DB.build_payloads(RESULTS)


def _row(markdown: str, label: str) -> list[str]:
    line = next(ln for ln in markdown.splitlines() if ln.startswith(f"| **{label}** |"))
    return [c.strip() for c in line.strip("|").split("|")]


def test_published_reports_match_the_test_summary():
    """reports/report.md and the README results block show the summary.json numbers."""
    report = (ROOT / "reports" / "report.md").read_text()
    readme = (ROOT / "README.md").read_text()
    for f in FAMILY_ORDER:
        rmse = SUMMARY["families"][f]["test_primary"]["rmse"]["mean"]
        assert _row(report, FAMILY_LABELS[f])[2].startswith(f"{rmse:.2f}"), f
        assert _row(readme, FAMILY_LABELS[f])[1].startswith(f"{rmse:.2f}"), f


def test_headline_agrees_across_readme_dashboard_and_link_previews(payloads):
    h = payloads["official_test.json"]["headline"]
    rmse, pct, n = f"{h['best_rmse']:.2f}", f"{h['rmse_reduction_pct']:.1f}", 100
    readme = (ROOT / "README.md").read_text()
    glance = readme.split("<!-- GLANCE:START -->")[1].split("<!-- GLANCE:END -->")[0]
    assert f"**{rmse} cycles**" in glance and f"**−{pct}% RMSE**" in glance
    assert f"**{n} test engines**" in glance
    # index.html's description / Open Graph / Twitter text is static: it must quote the same numbers
    html = (ROOT / "dashboard" / "index.html").read_text()
    previews = re.findall(r'(?:og|twitter):(?:description|image:alt)"\s+content="([^"]+)"', html)
    assert len(previews) == 4
    for text in previews:
        assert rmse in text, text
    assert all(f"{pct}%" in t for t in previews if "baseline" in t)
    assert "Simulated" in html and "not validated for real aircraft" in html


def test_official_test_predictions_are_complete_and_reproduce_the_metrics(payloads):
    o = payloads["official_test.json"]
    eng = o["engines"]
    assert o["n_engines"] == len(eng) == 100
    assert sorted(e["unit"] for e in eng) == list(range(1, 101))
    y = np.array([e["y_true"] for e in eng])
    assert np.all(np.isfinite(y)) and np.all(y >= 0) and np.all(y == np.round(y))
    assert o["n_true_rul_above_cap"] == int((y > o["training_target_cap"]).sum()) == 11
    for fam in o["families"]:
        p = np.array([e["pred"][fam["id"]] for e in eng])
        assert np.all(np.isfinite(p)) and np.all(p >= 0), fam["id"]  # predictions are clipped at 0
        rmse = math.sqrt(float(np.mean((p - y) ** 2)))
        mae = float(np.mean(np.abs(p - y)))
        lo, hi = fam["rmse_ci95"]
        assert lo <= rmse <= hi, fam["id"]  # median-seed RMSE lies inside its own bootstrap CI
        if (
            fam["seeds"] == 1 or fam["seeds_identical"]
        ):  # one distinct model: the mean is this model
            assert rmse == pytest.approx(fam["rmse"]["mean"], abs=0.01), fam["id"]
            assert mae == pytest.approx(fam["mae"]["mean"], abs=0.01), fam["id"]
        else:
            assert abs(rmse - fam["rmse"]["mean"]) <= 2 * fam["rmse"]["std"] + 0.01, fam["id"]


def _paired_ok(p: dict) -> None:
    lo, hi = p["ci95"]
    assert lo <= p["delta_rmse"] <= hi, p
    assert p["ci_excludes_0"] == (lo > 0 or hi < 0), p


def test_confidence_intervals_are_well_formed_and_read_correctly(payloads):
    """Every CI contains its estimate, and 'clear difference' means exactly 'CI excludes 0'."""
    o = payloads["official_test.json"]
    for fam in o["families"]:
        for key in ("rmse_ci95", "mae_ci95"):
            assert fam[key][0] <= fam[key][1], (fam["id"], key)
    for p in o["paired"]:
        _paired_ok(p)
    for st in payloads["cv.json"]["strategies"]:
        for p in st["paired"]:
            _paired_ok(p)
        for f in st["families"]:
            lo, hi = f["pooled_rmse_ci95"]
            assert lo <= f["pooled_rmse"] <= hi, f
            assert 0 <= f["rank1_frac"] <= 1 and 0 <= f["bootstrap_rank1_frac"] <= 1
    for r in payloads["ablation.json"]["rows"]:
        if r["variant"] != "base":
            _paired_ok(r)
    for r in payloads["conformal.json"]["summary"]:
        assert 0 <= r["coverage_fold_min"] <= r["coverage"] <= r["coverage_fold_max"] <= 1, r
        assert r["mean_width"] > 0


def test_dashboard_sources_are_committed(payloads):
    tracked = set(
        subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    )
    for name, p in payloads.items():
        for src in p.get("sources", []):
            assert src in tracked, f"{name} cites {src}, which is not committed"


def test_dashboard_export_is_deterministic(tmp_path):
    a = {p.name: p.read_bytes() for p in DB.export_dashboard(RESULTS, tmp_path / "a")}
    b = {p.name: p.read_bytes() for p in DB.export_dashboard(RESULTS, tmp_path / "b")}
    assert a == b


def test_link_preview_image_is_committed_at_the_large_card_size():
    png = ROOT / "dashboard" / "public" / "og.png"
    head = png.read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", head[16:24]) == DB.SOCIAL_SIZE
    assert png.stat().st_size < 300_000  # link-preview crawlers reject large images


@pytest.mark.skipif(
    subprocess.run(["which", "make"], capture_output=True).returncode, reason="no make"
)
def test_reproduction_writes_only_to_the_isolated_output_directory():
    """`make reproduce` on a fresh clone must not touch published artifacts: every output path it
    passes points under OUT, and the committed-artifact targets are not part of it."""
    out = subprocess.run(
        ["make", "-n", "reproduce", "repro-check"], cwd=ROOT, capture_output=True, text=True
    ).stdout
    cmds = [ln for ln in out.splitlines() if "engine-rul" in ln]
    assert cmds
    for ln in cmds:
        for flag in ("--results-dir", "--final-dir", "--root", "--out"):
            for value in re.findall(rf"{flag} (\S+)", ln):
                assert value.startswith("build/reproduce/"), ln
        assert "--split test" not in ln and "readme" not in ln.split()
        assert "export-dashboard" not in ln
    # tune / sensitivity / final / eda / phase-2 runs all carry an explicit output location
    for cmd in ("tune", "sensitivity", "final", "eda", "cv run", "cv ablation", "benchmark"):
        lines = [ln for ln in cmds if f"engine-rul {cmd}" in ln]
        assert lines and all(re.search(r"--(results-dir|root|out) build/", ln) for ln in lines), cmd
