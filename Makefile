# Reproducible pipeline (REPRO-07). Each target is one CLI call; `make reproduce` runs them in order.
RUN := uv run engine-rul

.PHONY: setup data split eda tune sensitivity final test-eval bench report reproduce test lint

setup:        ; uv sync --frozen
data:         ; $(RUN) data download
split:        ; $(RUN) split make --subset FD001   # only needed to regenerate; the split is committed
eda:          ; $(RUN) eda
tune:
	$(RUN) tune --config configs/baseline_constant.yaml
	$(RUN) tune --config configs/baseline_life_prior.yaml
	$(RUN) tune --config configs/ridge.yaml --grid configs/grids/ridge.yaml
	$(RUN) tune --config configs/hgb.yaml   --grid configs/grids/hgb.yaml
	$(RUN) tune --config configs/lstm.yaml  --grid configs/grids/lstm.yaml
sensitivity:  ; $(RUN) sensitivity --config configs/sensitivity.yaml
final:        ; $(RUN) final --family all --seeds 0 1 2 3 4
test-eval:    ; $(RUN) evaluate --split test --final --family all --note "make reproduce"
bench:        ; $(RUN) benchmark --family all --devices cpu mps
report:       ; $(RUN) report
reproduce: eda tune sensitivity final test-eval bench report
test:         ; uv run pytest -q
lint:         ; uv run ruff check src tests
