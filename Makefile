# Reproducible pipeline. Each target is one CLI call. Workflows (see docs/experiments.md):
#   make reproduce-phase1   Phase 1 training + validation (EDA, tuning, sensitivity, 5-seed final, benchmarks)
#   make phase2             Phase 2 validation-only experiments (nested CV, ablations, conformal, latency)
#   make reports            reports, README blocks and dashboard data from committed artifacts only
#   make reproduce          all three of the above; never reads the official test labels
#   make official-test-eval CONFIRM=official-test   the original Phase 1 test evaluation (explicit opt-in)
# Published artifacts (configs/final, results/final, final_runs.json, results/phase2/<experiment>) are
# never overwritten unless FORCE=--force is given.
RUN := uv run engine-rul

.PHONY: setup data split eda tune sensitivity final official-test-eval bench report reproduce-phase1 reproduce reports readme dashboard-data dashboard test lint cv-folds cv ablation conformal bench-phase2 report-phase2 phase2 repro-check

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
bench:        ; $(RUN) benchmark --family all --devices cpu mps
reproduce-phase1: eda tune sensitivity final bench
# The official test labels are read only here. Evaluating again appends a round to
# results/test_evaluations.jsonl (the log is append-only) and needs FORCE=--force to replace
# results/final/summary.json. Not part of any other target.
official-test-eval:
	@if [ "$(CONFIRM)" != "official-test" ]; then \
	  echo "official-test-eval reads the official FD001 test labels and appends to results/test_evaluations.jsonl."; \
	  echo "Re-run as: make official-test-eval CONFIRM=official-test [FORCE=--force]"; exit 2; fi
	$(RUN) evaluate --split test --final --family all --note "make official-test-eval" $(FORCE)
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
# From committed artifacts only: no training and no test labels.
report:       ; $(RUN) report
readme:       ; $(RUN) readme
dashboard-data: ; $(RUN) export-dashboard
reports: report report-phase2 readme dashboard-data
reproduce: reproduce-phase1 phase2 reports
# Static results dashboard (dashboard/, Node >= 20.19): `make dashboard` builds dashboard/dist
dashboard: dashboard-data
	cd dashboard && npm ci && npm run check && npm run build
test:         ; uv run pytest -q
lint:         ; uv run ruff check src tests
