# Reproducible pipeline. Each target is one CLI call. Workflows (see docs/testing.md):
#   make reproduce-phase1   Phase 1 training + validation (EDA, tuning, sensitivity, 5-seed final, benchmarks)
#   make phase2             Phase 2 validation-only experiments (nested CV, ablations, conformal, latency)
#   make reproduce          both of the above; never reads the official test labels
#   make reports            reports, README blocks and dashboard data from committed artifacts only
#   make official-test-eval CONFIRM=official-test   the original Phase 1 test evaluation (explicit opt-in)
# Training and experiment targets write under $(OUT) (default build/reproduce/, git-ignored), laid out like
# the repository (results/, results/phase2/, configs/final/, reports/), so a fresh clone can rerun
# everything and compare without touching the published artifacts. A second run into the same OUT
# refuses to overwrite; pass FORCE=--force or another OUT. OUT=. targets the published locations,
# which are likewise never overwritten without FORCE=--force.
RUN := uv run engine-rul
OUT ?= build/reproduce
ISO := --results-dir $(OUT)/results --final-dir $(OUT)/configs/final
DEVICES ?= cpu$(if $(filter Darwin,$(shell uname -s)), mps)

.PHONY: setup data split eda tune sensitivity final official-test-eval bench report reproduce-phase1 reproduce reports readme dashboard-data dashboard dashboard-test test lint cv-folds cv ablation conformal bench-phase2 report-phase2 phase2 repro-check

setup:        ; uv sync --frozen
data:         ; $(RUN) data download
split:        ; $(RUN) split make --subset FD001   # only needed to regenerate; the split is committed
eda:          ; $(RUN) eda --out $(OUT)/reports
tune:
	$(RUN) tune --config configs/baseline_constant.yaml $(ISO) $(FORCE)
	$(RUN) tune --config configs/baseline_life_prior.yaml $(ISO) $(FORCE)
	$(RUN) tune --config configs/ridge.yaml --grid configs/grids/ridge.yaml $(ISO) $(FORCE)
	$(RUN) tune --config configs/hgb.yaml   --grid configs/grids/hgb.yaml $(ISO) $(FORCE)
	$(RUN) tune --config configs/lstm.yaml  --grid configs/grids/lstm.yaml $(ISO) $(FORCE)
sensitivity:  ; $(RUN) sensitivity --config configs/sensitivity.yaml $(ISO) $(FORCE)
final:        ; $(RUN) final --family all --seeds 0 1 2 3 4 $(ISO) $(FORCE)
bench:        ; $(RUN) benchmark --family all --devices $(DEVICES) --results-dir $(OUT)/results $(FORCE)
reproduce-phase1: eda tune sensitivity final bench
# The official test labels are read only here, always against the published final runs in results/
# (independent of OUT). Evaluating again appends a round to results/test_evaluations.jsonl (the log
# is append-only) and needs FORCE=--force to replace results/final/summary.json. Not part of any
# other target.
official-test-eval:
	@if [ "$(CONFIRM)" != "official-test" ]; then \
	  echo "official-test-eval reads the official FD001 test labels and appends to results/test_evaluations.jsonl."; \
	  echo "Re-run as: make official-test-eval CONFIRM=official-test [FORCE=--force]"; exit 2; fi
	$(RUN) evaluate --split test --final --family all --note "make official-test-eval" $(FORCE)
# Phase 2 (validation only: never reads the official test files). Outputs under $(OUT)/results/phase2/;
# existing experiment directories are never overwritten (pass FORCE=--force to replace). Phase 2
# uses the published frozen configs (configs/final/); the ablation's determinism check compares
# against the published CV run; bench-phase2 needs Phase 1 final runs in $(OUT)/results.
cv-folds:     ; $(RUN) cv make-folds   # only needed to regenerate; the fold file is committed
P2 := --root $(OUT)/results/phase2
cv:           ; $(RUN) cv run --spec configs/phase2/cv.yaml $(P2) $(FORCE)
ablation:     ; $(RUN) cv ablation --spec configs/phase2/ablation.yaml $(P2) $(FORCE)
conformal:    ; $(RUN) cv run --spec configs/phase2/conformal.yaml $(P2) $(FORCE)
bench-phase2: ; $(RUN) benchmark --phase2 --family all --devices $(DEVICES) --results-dir $(OUT)/results $(P2) $(FORCE)
phase2: cv ablation conformal bench-phase2
	$(RUN) report-phase2 $(P2) --out $(OUT)/reports --no-readme
# Retrain the frozen configs (seed 0) and compare with the published validation RMSE (<1 min CPU).
repro-check:  ; $(RUN) repro-check --family all --seeds 0 $(P2) $(FORCE)
# From committed artifacts only (results/, reports/, README.md): no training, no test labels, no OUT.
report:       ; $(RUN) report
report-phase2: ; $(RUN) report-phase2
readme:       ; $(RUN) readme
dashboard-data: ; $(RUN) export-dashboard
reports: report report-phase2 readme dashboard-data
reproduce: reproduce-phase1 phase2
# Static results dashboard (dashboard/, Node >= 20.19): `make dashboard` builds dashboard/dist,
# `make dashboard-test` runs the Playwright browser tests against that build (Chromium only).
dashboard: dashboard-data
	cd dashboard && npm ci && npm run check && npm run build
dashboard-test: dashboard
	cd dashboard && npx playwright install chromium && npm run e2e
test:         ; uv run pytest -q
lint:         ; uv run ruff check src tests
