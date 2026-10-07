"""Generate all EDA figures and save them to reports/figures/.

Called via `make eda`.
"""

import logging
import sys
from pathlib import Path

# Ensure src/ is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from etf_predictor.data.loader import YahooFinanceLoader
from etf_predictor.data.pipeline import DataPipeline
from etf_predictor.data.visualization import EDAVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run pipeline and generate all EDA figures."""
    tickers = ["IEUR", "FEZ", "EUFN", "IVV"]

    # ── 1. Load raw data (for price plots) ───────────────────────────
    logger.info("Loading raw data …")
    loader = YahooFinanceLoader(
        tickers=tickers,
        start="2010-01-01",
        end="2020-01-01",
        include_benchmark=True,
    )
    raw_data = loader.load()

    # ── 2. Run pipeline ───────────────────────────────────────────────
    logger.info("Running DataPipeline …")
    pipeline = DataPipeline(
        tickers=["IEUR", "FEZ", "EUFN"],
        start="2010-01-01",
        end="2020-01-01",
        include_benchmark=True,
    )
    datasets = pipeline.run(save=True)

    logger.info("Pipeline summary:\n%s", pipeline.summary(datasets))

    # ── 3. Generate figures ───────────────────────────────────────────
    viz = EDAVisualizer()

    figures = {
        "01_open_prices": viz.plot_open_prices(raw_data),
        "02_cumulative_gamma": viz.plot_cumulative_gamma(datasets),
        "03_class_balance": viz.plot_class_balance(datasets),
        "04_sector_exposure": viz.plot_sector_exposure(),
    }

    # Per-ticker missing data and correlation heatmaps
    # (computed on raw indicator data before cleaning)
    for ticker in ["IEUR", "FEZ", "EUFN"]:
        df = datasets[ticker]
        figures[f"05_correlation_{ticker}"] = viz.plot_correlation_heatmap(
            df, ticker=ticker
        )

    viz.save_all(figures, output_dir="reports/figures")
    logger.info("All EDA figures saved to reports/figures/")


if __name__ == "__main__":
    main()
