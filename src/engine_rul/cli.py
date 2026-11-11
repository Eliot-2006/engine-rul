"""`engine-rul` command line (ADR-10, PRD 7.3). Dispatch only; logic lives in the modules.

Exit codes: 0 ok, 1 user/config error, 2 data validation/checksum error, 3 test-set lock."""

from __future__ import annotations

import argparse
import json
import logging
import sys

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
    from engine_rul.train import tune

    print(f"wrote {tune(_cfg(a, device=a.device), a.grid)}")
    return 0


def cmd_sensitivity(a):
    from engine_rul.train import sensitivity

    print(f"wrote {sensitivity(a.config)}")
    return 0


def cmd_final(a):
    from engine_rul.train import final

    print(f"wrote {final(_families(a.family), a.seeds)}")
    return 0


def cmd_evaluate(a):
    from engine_rul.evaluate import evaluate_test_final, evaluate_val, require_final

    require_final(a.split, a.final)
    if a.split == "val":
        if not a.run:
            raise EngineRULError("--run is required for --split val")
        print(json.dumps(evaluate_val(a.run), indent=1))
        return 0
    print(f"wrote {evaluate_test_final(_families(a.family), a.results_dir, note=a.note)}")
    return 0


def cmd_benchmark(a):
    from engine_rul.benchmark import run_benchmarks

    print(f"wrote {run_benchmarks(_families(a.family), a.results_dir, devices=tuple(a.devices))}")
    return 0


def cmd_report(a):
    from engine_rul.report import build_report

    print(f"wrote {build_report(a.results_dir, a.out, update_readme=not a.no_readme)}")
    return 0


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
        t.set_defaults(fn=fn)

    se = sub.add_parser("sensitivity", help="R_max / window sensitivity on validation (EXP-04)")
    se.add_argument("--config", default="configs/sensitivity.yaml")
    se.set_defaults(fn=cmd_sensitivity)

    f = sub.add_parser("final", help="train frozen configs/final/* with several seeds")
    f.add_argument("--family", nargs="+", default=["all"])
    f.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
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
    ev.set_defaults(fn=cmd_evaluate)

    b = sub.add_parser("benchmark", help="latency / throughput / size benchmarks")
    b.add_argument("--family", nargs="+", default=["all"])
    b.add_argument("--results-dir", default="results")
    b.add_argument("--devices", nargs="+", default=["cpu"])
    b.set_defaults(fn=cmd_benchmark)

    r = sub.add_parser("report", help="build reports/report.md, figures and README tables")
    r.add_argument("--results-dir", default="results")
    r.add_argument("--out", default="reports")
    r.add_argument("--no-readme", action="store_true")
    r.set_defaults(fn=cmd_report)
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
