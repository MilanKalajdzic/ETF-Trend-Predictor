# ─────────────────────────────────────────────────────────────────────────────
# ETF Predictor — Makefile
# Usage:  make <target>
# ─────────────────────────────────────────────────────────────────────────────

PYTHON     := python3
RUNPY      := PYTHONPATH=src $(PYTHON)
SRC        := src/etf_predictor
TESTS      := tests
REPORTS    := reports
DOCS_SRC   := docs/source
DOCS_BUILD := docs/_build/html
IMAGE_NAME := etf-predictor

.PHONY: help install install-dev lint format test coverage \
        data eda analysis modeling docs clean clean-data docker-build docker-run docker-test

# ── Default target ───────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "ETF Predictor — available targets"
	@echo "──────────────────────────────────"
	@echo "  install       Install package (production deps)"
	@echo "  install-dev   Install package + dev deps + pre-commit hooks"
	@echo "  lint          Run ruff linter"
	@echo "  format        Run ruff formatter"
	@echo "  test          Run pytest"
	@echo "  coverage      Run pytest with HTML coverage report"
	@echo "  data          Download & process all ETF data"
	@echo "  eda           Generate all EDA figures to reports/figures/"
	@echo "  analysis      Run statistical analysis pipeline"
	@echo "  modeling      Train MLP + LSTM with walk-forward validation"
	@echo "  docs          Build Sphinx HTML documentation"
	@echo "  clean         Remove build/cache/data artefacts"
	@echo "  clean-data    Remove raw and processed data"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run pipeline inside Docker container"
	@echo "  docker-test   Run tests inside Docker container"
	@echo ""

# ── Installation ─────────────────────────────────────────────────────────────
install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,notebook]"
	pre-commit install

# ── Linting & formatting ─────────────────────────────────────────────────────
lint:
	ruff check $(SRC) $(TESTS)

format:
	ruff format $(SRC) $(TESTS)

# ── Testing ──────────────────────────────────────────────────────────────────
test:
	PYTHONPATH=src pytest $(TESTS) -v

coverage:
	PYTHONPATH=src pytest $(TESTS) --cov=etf_predictor --cov-report=html --cov-report=term
	@echo "HTML report: htmlcov/index.html"

# ── Data pipeline ────────────────────────────────────────────────────────────
data:
	$(RUNPY) -c "\
from etf_predictor.data.pipeline import DataPipeline; \
p = DataPipeline(); \
datasets = p.run(); \
print(p.summary(datasets))"

# ── EDA figures ──────────────────────────────────────────────────────────────
eda:
	$(RUNPY) scripts/generate_eda.py

# ── Statistical analysis ─────────────────────────────────────────────────────
analysis:
	$(RUNPY) scripts/run_analysis.py

# ── Modeling (MLP + LSTM + walk-forward backtest) ────────────────────────────
modeling:
	$(RUNPY) scripts/run_modeling.py
	$(RUNPY) scripts/run_modeling.py --top-k-features 10 --feature-suffix _top10
	$(RUNPY) scripts/cost_sensitivity.py

# ── Documentation ────────────────────────────────────────────────────────────
docs:
	sphinx-build -b html $(DOCS_SRC) $(DOCS_BUILD)
	@echo "Docs built: $(DOCS_BUILD)/index.html"

# ── Clean ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info .pytest_cache/ htmlcov/ .coverage
	rm -rf $(DOCS_BUILD)

clean-data:
	rm -rf data/raw/ data/processed/
	@echo "Raw and processed data removed."

# ── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker build -t $(IMAGE_NAME):latest .

docker-run:
	docker compose run --rm etf-predictor make data

docker-test:
	docker compose run --rm test




.PHONY: help install install-dev lint format test coverage \
        data eda analysis modeling report render publish docs clean clean-data \
        docker-build docker-run docker-test


# ── Full report ──────────────────────────────────────────────────────────────
# Runs the full pipeline (data → analysis → modeling) then renders the
# three Quarto reports and copies them to /output/ so they survive past
# the container lifetime when /output/ is bind-mounted by the caller.
report:
	$(RUNPY) -m etf_predictor.data.pipeline
	$(RUNPY) scripts/run_analysis.py
	$(RUNPY) scripts/run_modeling.py
	$(RUNPY) scripts/run_modeling.py --top-k-features 10 --feature-suffix _top10
	$(RUNPY) scripts/cost_sensitivity.py
	quarto render reports/eda_report.qmd
	quarto render reports/statistical_analysis.qmd
	quarto render reports/modeling_report.qmd
	@$(MAKE) --no-print-directory publish

# ── Render only ──────────────────────────────────────────────────────────────
# Re-renders the three Quarto reports from the pre-computed CSVs + figures
# already on disk (no Yahoo download, no statistical analysis, no model
# training). Use this for fast class demos when the image already ships
# the data/ and reports/results/ artefacts. Total runtime: ~30 s.
render:
	quarto render reports/eda_report.qmd
	quarto render reports/statistical_analysis.qmd
	quarto render reports/modeling_report.qmd
	@$(MAKE) --no-print-directory publish

# ── Publish rendered HTMLs to /output ────────────────────────────────────────
# Copies the three self-contained HTML files to /output/, which the caller
# is expected to bind-mount (`docker run -v $PWD/output:/output ...`).
# Safe to call when /output is not mounted — it just lands inside the
# container's filesystem and is discarded when the container exits.
publish:
	@mkdir -p /output
	@cp reports/eda_report.html              /output/ 2>/dev/null || true
	@cp reports/statistical_analysis.html    /output/ 2>/dev/null || true
	@cp reports/modeling_report.html         /output/ 2>/dev/null || true
	@echo "Published HTML reports to /output/"
	@ls -lh /output/*.html 2>/dev/null || true