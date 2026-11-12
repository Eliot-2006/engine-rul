"""`engine-rul` command line (ADR-10, PRD 7.3). Dispatch only; logic lives in the modules.

Exit codes: 0 ok, 1 user/config error, 2 data validation/checksum error, 3 test-set lock."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from engine_rul.errors import EngineRULError

log = logging.getLogger("engine_rul")
ALL = ["constant", "life_prior", "ridge", "hgb", "lstm"]


def _families(values: list[str]) -> list[str]:
    return ALL if values == ["all"] else values


def cmd_data(a):
    from engine_rul import data as D

    subsets = ("FD001", "FD002", "FD003", "FD004") if a.subsets == ["all"] else tuple(a.subsets)
    if a.action == "download":
        D.download(a.dest, subsets)
    if a.zip:
        res = D.verify_zip(a.zip)
        for k, v in res.items():
            print(f"{k}: {'MATCH' if v else 'MISSING' if v is None else 'MISMATCH'}")
        if not all(res.values()):
            return 2
    res = D.verify(a.dest, subsets)
    for k, ok in res.items():
        print(f"{k}: {'OK' if ok else 'FAILED'}")
    if not all(res.values()):
        return 2
    man = D.load_manifest()
    for s in subsets:
        inv = man["subsets"][s].get("invariants")
        if inv:
            D.check_train_invariants(D.load_train(a.dest, s), inv)
            D.check_test_invariants(
                D.load_test_features(a.dest, s), D.load_test_rul(a.dest, s), inv
            )
            print(f"{s}: structural invariants OK (DATA-10)")
    return 0


def cmd_split(a):
    from engine_rul import data as D
    from engine_rul.split import make_split, save_split
    from engine_rul.utils import sha256_file

    df = D.load_train(a.data_dir, a.subset)
    split = make_split(
        df,
        seed=a.seed,
        window=a.window,
        subset=a.subset,
        train_file_sha256=sha256_file(D.subset_path(a.data_dir, a.subset, "train")),
    )
    save_split(split, a.out, force=a.force)
    print(
        f"wrote {a.out}: {len(split['train_units'])} train / {len(split['val_units'])} val engines, "
        f"{len(split['val_eval_points'])} validation eval points"
    )
    return 0


def cmd_eda(a):
    from engine_rul.eda import run_eda

    run_eda(a.data_dir, a.subset, a.split_file, a.out)
    return 0


def _cfg(a, **over):
    from engine_rul.config import load_config

    o = {k: v for k, v in over.items() if v is not None}
    return load_config(a.config, o)


def cmd_train(a):
    from engine_rul.train import train_one

    rec = train_one(_cfg(a, seed=a.seed, device=a.device), stage="dev")
    print(
        json.dumps(
            {"run_id": rec["run_id"], "val_points_capped": rec["metrics"]["val_points_capped"]},
            indent=1,
        )
    )
    return 0


def cmd_tune(a):
    from pathlib import Path

    from engine_rul.train import FINAL_DIR, tune

    cfg = _cfg(a, device=a.device, results_dir=a.results_dir)
    out = tune(
        cfg,
        a.grid,
        out_dir=a.final_dir or FINAL_DIR,
        summary_dir=Path(cfg.results_dir) / "final",
        force=a.force,
    )
    print(f"wrote {out}")
    return 0


def cmd_sensitivity(a):
    from engine_rul.train import FINAL_DIR, sensitivity

    print(f"wrote {sensitivity(a.config, a.results_dir, a.final_dir or FINAL_DIR, force=a.force)}")
    return 0


def cmd_final(a):
    from engine_rul.train import FINAL_DIR, final

    out = final(
        _families(a.family), a.seeds, a.results_dir, a.final_dir or FINAL_DIR, force=a.force
    )
    print(f"wrote {out}")
    return 0


def cmd_evaluate(a):
    from engine_rul.evaluate import evaluate_test_final, evaluate_val, require_final

    require_final(a.split, a.final)
    if a.split == "val":
        if not a.run:
            raise EngineRULError("--run is required for --split val")
        print(json.dumps(evaluate_val(a.run), indent=1))
        return 0
    out = evaluate_test_final(_families(a.family), a.results_dir, note=a.note, force=a.force)
    print(f"wrote {out}")
    return 0


def cmd_benchmark(a):
    from engine_rul.benchmark import run_benchmarks, run_phase2_benchmarks

    if a.phase2:
        out = run_phase2_benchmarks(
            _families(a.family), a.results_dir, tuple(a.devices), a.root, force=a.force
        )
        print(f"wrote {out}")
        return 0
    out = run_benchmarks(_families(a.family), a.results_dir, tuple(a.devices), force=a.force)
    print(f"wrote {out}")
    return 0


def cmd_report(a):
    from engine_rul.report import build_report

    print(f"wrote {build_report(a.results_dir, a.out, update_readme=not a.no_readme)}")
    return 0


def cmd_export_dashboard(a):
    from engine_rul.dashboard import export_dashboard

    for p in export_dashboard(a.results_dir, a.out):
        print(f"wrote {p}")
    return 0


def cmd_readme(a):
    from engine_rul.dashboard import update_readme

    print(f"updated {update_readme(a.results_dir)}")
    return 0


def cmd_predict(a):
    from engine_rul import data as D
    from engine_rul.inference import predict_engines

    out = predict_engines(
        a.run,
        D.read_sensor_file(a.input),
        levels=a.levels,
        all_cycles=a.all_cycles,
        method=a.method,
        device=a.device,
    )
    if a.out:
        out.to_csv(a.out, index=False)
        print(f"wrote {a.out}")
    else:
        print(out.to_csv(index=False), end="")
    return 0


def cmd_repro_check(a):
    from engine_rul.train import FINAL_DIR, repro_check

    out = repro_check(
        _families(a.family),
        a.seeds,
        a.results_dir,
        a.final_dir or FINAL_DIR,
        a.root,
        a.name,
        a.tol,
        force=a.force,
    )
    for r in out["rows"]:
        print(
            f"{r['family']:<10} seed {r['seed']}: published {r['published_val_rmse']:.6f} "
            f"reproduced {r['reproduced_val_rmse']:.6f} |diff| {r['abs_diff']:.2e} "
            f"{'OK' if r['ok'] else 'MISMATCH'}"
        )
    return 0 if out["all_ok"] else 1


def cmd_cv(a):
    from engine_rul import cv

    if a.action == "make-folds":
        from engine_rul import data as D
        from engine_rul.split import save_split
        from engine_rul.utils import sha256_file

        spec = cv.make_cv_folds(
            D.load_train(a.data_dir, a.subset),
            n_folds=a.n_folds,
            seeds=a.seeds,
            subset=a.subset,
            train_file_sha256=sha256_file(D.subset_path(a.data_dir, a.subset, "train")),
        )
        save_split(spec, a.out, force=a.force)
        print(f"wrote {a.out}")
    elif a.action == "run":
        print(f"wrote {cv.run_cv(a.spec, a.root, force=a.force)}")
    elif a.action == "ablation":
        print(f"wrote {cv.run_ablation(a.spec, a.root, force=a.force)}")
    else:
        fn = cv.analyze_ablation if (Path(a.exp) / "runs.csv").exists() else cv.analyze_cv
        fn(a.exp)
        print(f"wrote {Path(a.exp) / 'summary.json'}")
    return 0


def cmd_report_phase2(a):
    from engine_rul.report_phase2 import build_phase2_report

    print(
        f"wrote {build_phase2_report(a.root, a.out, readme=None if a.no_readme else 'README.md')}"
    )
    return 0


def _isolation_args(p, results_default: str | None = "results") -> None:
    """Where outputs go. Published artifacts are never overwritten without --force."""
    p.add_argument(
        "--results-dir",
        default=results_default,
        help="results root (default: the config's results_dir / 'results')",
    )
    p.add_argument(
        "--final-dir", default=None, help="frozen-config directory (default: configs/final)"
    )
    p.add_argument(
        "--force", action="store_true", help="allow overwriting existing published artifacts"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="engine-rul", description="RUL prediction on NASA C-MAPSS")
    p.add_argument("--verbose", action="store_true", help="debug logging and full tracebacks")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="download / verify the pinned dataset")
    d.add_argument("action", choices=["download", "verify"])
    d.add_argument("--dest", default="data/raw")
    d.add_argument("--subsets", nargs="+", default=["FD001"])
    d.add_argument(
        "--zip", help="NASA CMAPSSData.zip to cross-check against the manifest (DATA-04)"
    )
    d.set_defaults(fn=cmd_data)

    s = sub.add_parser("split", help="create the engine-level split file")
    s.add_argument("action", choices=["make"])
    s.add_argument("--subset", default="FD001")
    s.add_argument("--data-dir", default="data/raw")
    s.add_argument("--seed", type=int, default=20261110)
    s.add_argument("--window", type=int, default=30)
    s.add_argument("--out", default="splits/fd001_split.json")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_split)

    e = sub.add_parser("eda", help="exploratory analysis figures and data-quality summary")
    e.add_argument("--subset", default="FD001")
    e.add_argument("--data-dir", default="data/raw")
    e.add_argument("--split-file", default="splits/fd001_split.json")
    e.add_argument("--out", default="reports")
    e.set_defaults(fn=cmd_eda)

    for name, fn, help_ in (
        ("train", cmd_train, "train one config"),
        ("tune", cmd_tune, "grid search on validation"),
    ):
        t = sub.add_parser(name, help=help_)
        t.add_argument("--config", required=True)
        t.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
        if name == "train":
            t.add_argument("--seed", type=int)
        else:
            t.add_argument("--grid", help="YAML {param: [values]}; omit for a single run")
            _isolation_args(t, results_default=None)
        t.set_defaults(fn=fn)

    se = sub.add_parser("sensitivity", help="R_max / window sensitivity on validation (EXP-04)")
    se.add_argument("--config", default="configs/sensitivity.yaml")
    _isolation_args(se)
    se.set_defaults(fn=cmd_sensitivity)

    f = sub.add_parser("final", help="train frozen configs/final/* with several seeds")
    f.add_argument("--family", nargs="+", default=["all"])
    f.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    _isolation_args(f)
    f.set_defaults(fn=cmd_final)

    ev = sub.add_parser(
        "evaluate", help="validation re-evaluation or the locked final test evaluation"
    )
    ev.add_argument("--split", choices=["val", "test"], required=True)
    ev.add_argument("--final", action="store_true", help="required for --split test (EVAL-21)")
    ev.add_argument("--run", help="run directory (for --split val)")
    ev.add_argument("--family", nargs="+", default=["all"])
    ev.add_argument("--results-dir", default="results")
    ev.add_argument("--note", default="", help="free-text note stored in test_evaluations.jsonl")
    ev.add_argument("--force", action="store_true", help="allow replacing an existing summary.json")
    ev.set_defaults(fn=cmd_evaluate)

    b = sub.add_parser("benchmark", help="latency / throughput / size benchmarks")
    b.add_argument("--family", nargs="+", default=["all"])
    b.add_argument("--results-dir", default="results")
    b.add_argument("--devices", nargs="+", default=["cpu"])
    b.add_argument(
        "--phase2",
        action="store_true",
        help="model-only vs end-to-end latency into results/phase2/benchmark/",
    )
    b.add_argument("--root", default="results/phase2")
    b.add_argument("--force", action="store_true")
    b.set_defaults(fn=cmd_benchmark)

    pr = sub.add_parser("predict", help="predict RUL from a raw sensor file (NASA text format)")
    pr.add_argument(
        "--run", required=True, help="run directory (Phase-1 run or Phase-2 fold model)"
    )
    pr.add_argument(
        "--input", required=True, help="sensor file, e.g. data/raw/FD001/test_FD001.txt"
    )
    pr.add_argument(
        "--levels",
        nargs="*",
        type=float,
        default=[],
        help="conformal interval levels, e.g. 0.8 0.9 (needs conformal.json in --run)",
    )
    pr.add_argument(
        "--method",
        default="global_one_per_engine",
        choices=["global_one_per_engine", "global_pooled", "mondrian_pooled"],
    )
    pr.add_argument(
        "--all-cycles", action="store_true", help="predict every cycle, not just the last"
    )
    pr.add_argument("--device", default="cpu")
    pr.add_argument("--out", help="CSV path (default: stdout)")
    pr.set_defaults(fn=cmd_predict)

    rc = sub.add_parser(
        "repro-check", help="retrain frozen configs in isolation; compare with published val RMSE"
    )
    rc.add_argument("--family", nargs="+", default=["all"])
    rc.add_argument("--seeds", nargs="+", type=int, default=[0])
    rc.add_argument(
        "--tol", type=float, default=1e-6, help="max |RMSE difference| (same machine: exact)"
    )
    rc.add_argument("--name", default="frozen_configs")
    rc.add_argument("--root", default="results/phase2")
    _isolation_args(rc)
    rc.set_defaults(fn=cmd_repro_check)

    c = sub.add_parser("cv", help="Phase 2 nested CV / ablations on training engines only")
    c.add_argument("action", choices=["make-folds", "run", "ablation", "analyze"])
    c.add_argument("--spec", default="configs/phase2/cv.yaml")
    c.add_argument("--root", default="results/phase2")
    c.add_argument("--exp", help="experiment directory (analyze)")
    c.add_argument("--subset", default="FD001")
    c.add_argument("--data-dir", default="data/raw")
    c.add_argument("--n-folds", type=int, default=5)
    c.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    c.add_argument("--out", default="splits/cv_fd001.json")
    c.add_argument("--force", action="store_true", help="allow overwriting an existing output")
    c.set_defaults(fn=cmd_cv)

    r = sub.add_parser("report", help="build reports/report.md, figures and README tables")
    r.add_argument("--results-dir", default="results")
    r.add_argument("--out", default="reports")
    r.add_argument("--no-readme", action="store_true")
    r.set_defaults(fn=cmd_report)

    r2 = sub.add_parser("report-phase2", help="build reports/phase2_report.md from results/phase2/")
    r2.add_argument("--root", default="results/phase2")
    r2.add_argument("--out", default="reports")
    r2.add_argument("--no-readme", action="store_true")
    r2.set_defaults(fn=cmd_report_phase2)

    ed = sub.add_parser(
        "export-dashboard",
        help="write validated dashboard JSON from committed results (no training)",
    )
    ed.add_argument("--results-dir", default="results")
    ed.add_argument("--out", default="dashboard/public/data")
    ed.set_defaults(fn=cmd_export_dashboard)

    rd = sub.add_parser(
        "readme", help="refresh README result blocks and hero figure from committed results"
    )
    rd.add_argument("--results-dir", default="results")
    rd.set_defaults(fn=cmd_readme)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    from engine_rul.utils import setup_logging

    setup_logging(a.verbose)
    try:
        return a.fn(a) or 0
    except EngineRULError as e:
        if a.verbose:
            raise
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())
