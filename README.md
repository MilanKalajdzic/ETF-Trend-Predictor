<p align="center">
  <img src="assets/social-preview.png" alt="ETF Trend Predictor: next-day direction of European ETFs from technical indicators and neural networks" width="100%">
</p>

<p align="center">
  <a href="https://hub.docker.com/r/milankalajdzic/etf-predictor"><img src="https://img.shields.io/badge/Docker%20Hub-etf--predictor-2496ED?logo=docker&logoColor=white" alt="Docker Hub"></a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/reports-Quarto-75AADB?logo=quarto&logoColor=white" alt="Quarto reports">
  <img src="https://img.shields.io/badge/tests-39%20unit%20tests-0A9EDC?logo=pytest&logoColor=white" alt="39 unit tests">
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

<p align="center">
  <b>Can 260+ technical indicators and a neural network call tomorrow's direction for European ETFs?</b><br>
  A reproducible replication and extension of
  <a href="https://doi.org/10.1515/econ-2022-0073">Sagaceta-Mejía et al. (2024)</a>,
  from raw prices to rendered reports in a single <code>docker run</code>.
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#results">Results</a> ·
  <a href="#reports">Reports</a> ·
  <a href="#development">Development</a> ·
  <a href="#limitations">Limitations</a>
</p>

---

## At a glance

| | |
|---|---|
| **Universe** | IEUR, FEZ, EUFN (Europe) against IVV (US benchmark) |
| **Data** | Daily OHLCV from Yahoo Finance, 2010 to April 2026 |
| **Features** | ~260 technical indicators across 10 pandas-ta categories |
| **Target** | Γ(t) = +1 if Open(t) > Open(t−1), else −1 |
| **Models** | MLP signal classifier and LSTM next-close regressor (PyTorch) |
| **Validation** | Expanding-window walk-forward, ~22 folds per ticker, benchmarked against buy-and-hold |
| **Baselines** | Random Forest feature importance, ADF stationarity tests, ARIMA(1,1,1) |
| **Reproducibility** | One Docker image, Makefile automation, 39 unit tests, Sphinx API docs |

---

## Quick start

All you need is Docker.

```bash
docker pull milankalajdzic/etf-predictor:latest
mkdir output
docker run --rm -v "$PWD/output:/output" milankalajdzic/etf-predictor:latest
```

<details>
<summary>Windows PowerShell</summary>

```powershell
docker pull milankalajdzic/etf-predictor:latest
mkdir output
docker run --rm -v "${PWD}/output:/output" milankalajdzic/etf-predictor:latest
```

</details>

The default command (`make report`) runs the whole pipeline from scratch: download and process data, statistical analysis, MLP and LSTM walk-forward training, render all three Quarto reports and copy them to `output/`. Expect **about 10 minutes** on CPU.

> [!TIP]
> On Apple Silicon, if the pull fails, add `--platform linux/amd64` to both the `pull` and `run` commands.

---

## How it works

```mermaid
flowchart LR
    A["Yahoo Finance<br/>daily OHLCV"] --> B["Loader<br/>parquet cache"]
    B --> C["pandas-ta<br/>~260 indicators"]
    C --> D["Target Γ(t)<br/>open-to-open direction"]
    D --> E["Min-max scaling<br/>and cleaning"]
    E --> F[("Processed<br/>datasets")]
    F --> G["Statistical analysis<br/>variance · correlation<br/>RF · ADF · ARIMA"]
    F --> H["Walk-forward models<br/>MLP · LSTM"]
    G --> R["Quarto reports"]
    H --> R
```

| Stage | Package | What it does |
|---|---|---|
| **1. Data** | `etf_predictor.data` | Downloads and caches prices, computes ~260 indicators, builds the Γ(t) target, scales features to [0, 1], drops sparse columns and forward-fills warm-up gaps |
| **2. Analysis** | `etf_predictor.analysis` | Low-variance and correlation redundancy checks, Random Forest feature importance, ADF stationarity tests, ARIMA(1,1,1) baseline |
| **3. Modeling** | `etf_predictor.models` | MLP signal and LSTM next-close models trained in an expanding window, long/short equity curves against buy-and-hold, top-10 feature ablation |

---

## Results

Out-of-sample walk-forward results, trading long/short on each model's daily signal. Annualised Sharpe ratios, no transaction costs:

| Ticker | MLP | LSTM | Buy & hold |
|:---|---:|---:|---:|
| IEUR | −0.04 | 0.01 | 0.33 |
| FEZ | −0.08 | 1.67\* | 0.23 |
| EUFN | 1.00\* | 1.55\* | 0.19 |
| IVV | −0.27 | 0.06 | 0.66 |

On the broad indices (IEUR, IVV) neither network beats buy-and-hold, which is what weak-form efficiency would predict. The starred results do **not** hold up: retrained on only the top-10 features they fall to 0.20 (FEZ LSTM), 0.44 (EUFN MLP) and −0.17 (EUFN LSTM), and Sharpe ratios this high from a hit rate of roughly 52% are a classic sign of look-ahead leakage. See [Limitations](#limitations). Per-fold metrics and equity curves are in the modeling report.

---

## Reports

Running the pipeline produces three self-contained HTML reports:

| Report | Source | Contents |
|---|---|---|
| **EDA** | [`eda_report.qmd`](reports/eda_report.qmd) | Raw data, indicator construction, target definition, class balance, cleaning |
| **Statistical analysis** | [`statistical_analysis.qmd`](reports/statistical_analysis.qmd) | Feature variance, correlation redundancy, Random Forest importance, ADF tests, ARIMA baseline |
| **Modeling** | [`modeling_report.qmd`](reports/modeling_report.qmd) | MLP and LSTM walk-forward backtests, equity curves, per-fold metrics, top-10 feature ablation |

---

## Development

<details>
<summary><b>Build locally and run individual steps</b></summary>

```bash
docker compose build
docker compose run --rm etf-predictor bash
```

Inside the container:

| Target | Description |
|---|---|
| `make data` | Download and process all ETF data |
| `make eda` | Generate EDA figures to `reports/figures/` |
| `make analysis` | Run the statistical analysis |
| `make modeling` | Train MLP and LSTM with walk-forward validation |
| `make report` | Full pipeline, render all reports, publish to `/output` |
| `make render` | Re-render the Quarto reports from existing results (fast) |
| `make publish` | Copy `reports/*.html` to `/output` |
| `make test` | Run the unit tests |
| `make coverage` | Tests with an HTML coverage report |
| `make docs` | Build the Sphinx HTML docs to `docs/_build/html/` |
| `make lint` / `make format` | Ruff linter and formatter |

</details>

<details>
<summary><b>Project structure</b></summary>

```text
├── src/etf_predictor/
│   ├── data/          loader, indicators, targets, preprocessing, pipeline, visualization
│   ├── analysis/      variance, correlation, feature importance, time series (ADF, ARIMA)
│   └── models/        MLP signal, LSTM value, walk-forward validator, equity curves
├── tests/             data/, analysis/, models/
├── scripts/           generate_eda.py, run_analysis.py, run_modeling.py
├── reports/           Quarto sources (.qmd) and results/*.csv
├── docs/source/       Sphinx configuration
├── assets/            social preview image and its HTML source
├── Dockerfile         Python 3.12 slim, CPU-only PyTorch, Quarto
├── docker-compose.yml services: etf-predictor, test, notebook
├── Makefile           automation targets
└── pyproject.toml     dependencies and Ruff configuration
```

</details>

<details>
<summary><b>Dataset</b></summary>

| Ticker | Fund | Exposure | History |
|---|---|---|---|
| **IEUR** | iShares Core MSCI Europe ETF | Broad Europe | 2014-06-12 to 2026-04-30 |
| **FEZ** | SPDR EURO STOXX 50 ETF | Eurozone large caps | 2010-01-04 to 2026-04-30 |
| **EUFN** | iShares MSCI Europe Financials ETF | European financials | 2010-02-03 to 2026-04-30 |
| **IVV** | iShares Core S&P 500 ETF | US benchmark | 2010-01-04 to 2026-04-30 |

Raw prices are cached as parquet under `data/raw/` and are not committed. IEUR history starts in June 2014 because that is all Yahoo Finance provides.

</details>

<details>
<summary><b>Compatibility notes</b></summary>

- pandas-ta 0.4.x removed the `Strategy` API, so indicators are computed one by one through the `df.ta` accessor
- yfinance returns MultiIndex columns, which the loader flattens
- Python 3.12 is required by pandas-ta 0.4.x
- statsmodels provides the ADF tests and ARIMA baseline

</details>

---

## Limitations

> [!WARNING]
> Known issues, listed so the results above can be read correctly.

- **Non-causal indicators.** By default some pandas-ta indicators use future prices: DPO is centred and the Ichimoku chikou span (`ICS_26`) is the close shifted backwards in time. Both are in the current feature set and are the prime suspects for the starred results.
- **Scaling leakage.** Min-max scaling is fitted on the full sample before the walk-forward split, so every fold sees the range of later prices.
- **Same-day label.** Following the paper, Γ(t) is predicted from same-day indicators, so the Random Forest accuracy in the statistical report measures contemporaneous fit rather than forecasting skill.
- **No trading frictions.** Returns ignore transaction costs, spreads and shorting costs.

---

## Reference

Sagaceta-Mejía, A. R., Sánchez-Gutiérrez, M. E., & Fresán-Figueroa, J. A. (2024). An intelligent approach for predicting stock market movements in emerging markets using optimized technical indicators and neural networks. *Economics*, 18, 20220073. https://doi.org/10.1515/econ-2022-0073
