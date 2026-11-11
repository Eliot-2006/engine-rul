"""Phase 3: dashboard export, README generation, provenance and reproduction-workflow isolation."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from engine_rul import dashboard as DB
from engine_rul.errors import DataValidationError
from engine_rul.plots import FAMILY_ORDER
from engine_rul.provenance import verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
VALIDATION_FILES = [f for f, p in DB.PROTOCOLS.items() if p not in ("official_test", "benchmark")]


@pytest.fixture(scope="module")
def payloads() -> dict[str, dict]:
    return DB.build_payloads(RESULTS)


@pytest.fixture
def results_copy(tmp_path) -> Path:
    """The artifacts the export reads (no bulky prediction archives), in a scratch tree."""
    dst = tmp_path / "results"
    shutil.copytree(RESULTS / "final", dst / "final")
    shutil.copytree(
        RESULTS / "phase2",
        dst / "phase2",
        ignore=shutil.ignore_patterns("predictions.csv.gz", "selection.csv", "runs.csv", "models"),
    )
    return dst


def test_export_writes_every_file_with_its_protocol(tmp_path):
    written = {p.name: p for p in DB.export_dashboard(RESULTS, tmp_path)}
    assert set(written) == set(DB.PROTOCOLS) | {"meta.json"}
    for name, protocol in DB.PROTOCOLS.items():
        data = json.loads(written[name].read_text())
        assert data["protocol"] == protocol
        assert data["schema_version"] == DB.SCHEMA_VERSION
    assert sum(p.stat().st_size for p in written.values()) < 400_000  # compact, not a data dump


def test_headline_is_derived_from_the_published_summary(payloads):
    s = json.loads((RESULTS / "final" / "summary.json").read_text())
    rmse = {f: v["test_primary"]["rmse"]["mean"] for f, v in s["families"].items()}
    best = min(rmse, key=rmse.get)
    h = payloads["official_test.json"]["headline"]
    assert h["best_family"] == best == "hgb"
    assert h["best_rmse"] == round(rmse[best], 2) == 13.26
    assert h["baseline_family"] == "life_prior"
    assert h["rmse_reduction_pct"] == round(100 * (1 - rmse[best] / rmse["life_prior"]), 1) == 63.9


def test_official_and_validation_results_never_share_a_file(payloads):
    official = payloads["official_test.json"]
    assert all(s.startswith("results/final/") for s in official["sources"])
    assert len(official["engines"]) == official["n_engines"] == 100
    assert "cycles" == official["unit"] and "uncapped" in official["target"]
    for name in VALIDATION_FILES:
        p = payloads[name]
        assert "official" not in p["protocol"]
        assert "capped" in p["target"].lower() or "125" in p["target"]
        if name != "phase1_validation.json":
            assert all(s.startswith("results/phase2/") for s in p["sources"])
            assert p["provenance"]["official_test_set_used"] is False
    # the Phase 1 validation file holds the selection data, never test predictions
    val = payloads["phase1_validation.json"]
    split = json.loads((ROOT / "splits" / "fd001_split.json").read_text())
    assert {t["unit"] for t in val["trajectories"]} == set(split["val_units"])


def test_model_identifiers_and_units_are_consistent(payloads):
    ids = set(FAMILY_ORDER)
    o = payloads["official_test.json"]
    assert {r["id"] for r in o["families"]} == ids
    assert all(set(e["pred"]) == ids for e in o["engines"])
    for st in payloads["cv.json"]["strategies"]:
        assert {r["id"] for r in st["families"]} <= ids
        assert {r["family"] for r in st["folds"]} <= ids
    assert {r["family"] for r in payloads["ablation.json"]["rows"]} <= ids
    assert {r["family"] for r in payloads["conformal.json"]["summary"]} <= ids
    for name in (
        "official_test.json",
        "phase1_validation.json",
        "cv.json",
        "ablation.json",
        "conformal.json",
    ):
        assert payloads[name]["unit"] == "cycles"


@pytest.mark.parametrize(
    "corrupt, match",
    [
        (lambda r: (r / "final" / "summary.json").unlink(), "missing artifact"),
        (lambda r: (r / "final" / "summary.json").write_text("{not json"), "malformed JSON"),
        (
            lambda r: _edit(r / "final" / "summary.json", lambda s: s.pop("paired_comparisons")),
            "required field",
        ),
        (
            lambda r: _edit(
                r / "final" / "summary.json",
                lambda s: s["families"]["hgb"]["test_primary"]["rmse"].update(mean=float("nan")),
            ),
            "non-finite",
        ),
        (
            lambda r: _edit(
                r / "phase2" / "cv" / "fd001_nested_cv" / "manifest.json",
                lambda s: s.update(official_test_set_used=True),
            ),
            "official_test_set_used",
        ),
        (
            lambda r: _edit(
                r / "phase2" / "ablation" / "fd001_ablation" / "summary.json",
                lambda s: s["families"].update(xgb=s["families"]["hgb"]),
            ),
            "unknown model identifiers",
        ),
    ],
)
def test_export_rejects_missing_or_malformed_artifacts(results_copy, tmp_path, corrupt, match):
    corrupt(results_copy)
    with pytest.raises(DataValidationError, match=match):
        DB.export_dashboard(results_copy, tmp_path / "out")
    assert not (tmp_path / "out").exists()  # nothing half-written


def _edit(path: Path, fn) -> None:
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d))


def test_dashboard_protocols_match_the_frontend():
    ts = (ROOT / "dashboard" / "src" / "data.ts").read_text()
    check = (ROOT / "dashboard" / "scripts" / "check-dist.mjs").read_text()
    expected = {**DB.PROTOCOLS, "meta.json": "meta"}
    for src in (ts, check):
        found = dict(re.findall(r'"([a-z0-9_]+\.json)": "([a-z0-9_]+)"', src))
        assert found == expected
    assert f"SCHEMA_VERSION = {DB.SCHEMA_VERSION}" in ts


def test_pages_base_path_matches_repository_name():
    vite = (ROOT / "dashboard" / "vite.config.ts").read_text()
    readme = (ROOT / "README.md").read_text()
    repo = re.search(r"github\.com/Eliot-2006/([\w-]+)", readme).group(1)
    assert f'"/{repo}/"' in vite
    assert f"https://eliot-2006.github.io/{repo}/" in readme


def test_readme_generated_blocks_are_current():
    readme = (ROOT / "README.md").read_text()
    for name, block in DB.readme_blocks(RESULTS).items():
        assert f"<!-- {name}:START -->\n{block}\n<!-- {name}:END -->" in readme, (
            f"README {name} block is stale: run `engine-rul readme`"
        )


@pytest.mark.parametrize(
    "doc", ["README.md", *sorted(p.name for p in (ROOT / "docs").glob("*.md"))]
)
def test_local_links_and_images_resolve(doc):
    path = ROOT / doc if doc == "README.md" else ROOT / "docs" / doc
    text = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
    targets = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'src="([^"]+)"', text)
    local = [
        t.split("#")[0] for t in targets if not re.match(r"[a-z]+:", t) and not t.startswith("#")
    ]
    missing = [t for t in local if t and not (path.parent / t).exists()]
    assert not missing, f"{doc}: broken links {missing}"


def test_published_experiments_reference_retrievable_source_snapshots():
    tracked = set(
        subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    )
    manifests = sorted((RESULTS / "phase2").glob("*/*/manifest.json"))
    assert manifests
    for m in manifests:
        prov = json.loads(m.read_text())["provenance"]
        snap = prov["source_snapshot"]
        if not prov["git_dirty"]:
            continue
        assert snap, f"{m}: dirty run without a source snapshot"
        assert snap["path"] in tracked, f"{snap['path']} is referenced by {m} but not committed"
        assert verify_snapshot(ROOT / snap["path"], snap["sha256"])


def _make(*args: str) -> str:
    out = subprocess.run(
        ["make", "-n", *args], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout


@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
def test_default_workflows_never_evaluate_on_the_test_set():
    for target in ("reproduce", "reproduce-phase1", "phase2", "reports"):
        assert "--split test" not in _make(target), target
    assert "--split test" in _make("official-test-eval", "CONFIRM=official-test")
    # without the explicit confirmation the guard stops before anything runs (RUN=echo: harmless if broken)
    run = subprocess.run(
        ["make", "official-test-eval", "RUN=echo SHOULD-NOT-RUN"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0 and "SHOULD-NOT-RUN" not in run.stdout
