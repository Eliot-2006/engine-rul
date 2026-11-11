"""TEST-FAIR-01: every family is scored on identical (unit, cycle) points with identical clipping."""

import numpy as np
import pandas as pd

from engine_rul.train import train_one


def test_identical_eval_points(synth_cfg):
    frames = {}
    for fam in ("constant", "life_prior", "ridge", "hgb", "lstm"):
        cfg = synth_cfg(fam)
        rec = train_one(cfg)
        frames[fam] = pd.read_csv(f"{cfg.results_dir}/{rec['run_id']}/predictions_val.csv")
        assert rec["n_val_points"] == len(frames[fam])
    ref = frames["constant"]
    for f in frames.values():
        np.testing.assert_array_equal(
            f[["unit", "cycle", "y_true", "y_capped"]], ref[["unit", "cycle", "y_true", "y_capped"]]
        )
        assert (f["y_pred"] >= 0).all()  # same clip(·, 0) post-processing
