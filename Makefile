# Reproducible pipeline (REPRO-07). Each target is one CLI call; `make reproduce` runs them in order.
# Published artifacts (configs/final, results/final, final_runs.json) are never overwritten unless
# FORCE=--force is given; `make reproduce FORCE=--force` also appends a new round to
# results/test_evaluations.jsonl. Use `make repro-check` to verify reproducibility without that.
RUN := uv run engine-rul

.PHONY: setup data split eda tune sensitivity final test-eval bench report reproduce test lint cv-folds cv ablation conformal bench-phase2 report-phase2 phase2 repro-check

setup:        ; uv sync --frozen
data:         ; $(RUN) data download
split:        ; $(RUN) split make --subset FD001   # only needed to regenerate; the split is committed
eda:          ; $(RUN) eda
tune:
	$(RUN) tune --config configs/baseline_constant.yaml $(FORCE)
	$(RUN) tune --config configs/baseline_life_prior.yaml $(FORCE)
	$(RUN) tune --config configs/ridge.yaml --grid configs/grids/ridge.yaml $(FORCE)
	$(RUN) tune --config configs/hgb.yaml   --grid configs/grids/hgb.yaml $(FORCE)
	$(RUN) tune --config configs/lstm.yaml  --grid configs/grids/lstm.yaml $(FORCE)
sensitivity:  ; $(RUN) sensitivity --config configs/sensitivity.yaml $(FORCE)
final:        ; $(RUN) final --family all --seeds 0 1 2 3 4 $(FORCE)
test-eval:    ; $(RUN) evaluate --split test --final --family all --note "make reproduce" $(FORCE)
bench:        ; $(RUN) benchmark --family all --devices cpu mps
report:       ; $(RUN) report
reproduce: eda tune sensitivity final test-eval bench report
# Phase 2 (validation only: never reads the official test files). Outputs under results/phase2/;
# existing experiment directories are never overwritten (pass FORCE=--force to replace).
cv-folds:     ; $(RUN) cv make-folds   # only needed to regenerate; the fold file is committed
cv:           ; $(RUN) cv run --spec configs/phase2/cv.yaml $(FORCE)
ablation:     ; $(RUN) cv ablation --spec configs/phase2/ablation.yaml $(FORCE)
conformal:    ; $(RUN) cv run --spec configs/phase2/conformal.yaml $(FORCE)
bench-phase2: ; $(RUN) benchmark --phase2 --family all --devices cpu mps $(FORCE)
report-phase2: ; $(RUN) report-phase2
phase2: cv ablation conformal bench-phase2 report-phase2
repro-check:  ; $(RUN) repro-check --family all --seeds 0 $(FORCE)
test:         ; uv run pytest -q
lint:         ; uv run ruff check src tests
