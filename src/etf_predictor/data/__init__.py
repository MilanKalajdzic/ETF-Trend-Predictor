"""Data preparation subpackage.

Handles data acquisition, target construction, technical indicator
computation, preprocessing, and exploratory visualization.

Typical usage
-------------
>>> from etf_predictor.data.pipeline import DataPipeline
>>> pipeline = DataPipeline(
...     tickers=["IEUR", "FEZ", "EUFN"],
...     start="2010-01-01",
...     end="2020-01-01",
... )
>>> datasets = pipeline.run()
"""

from etf_predictor.data.pipeline import DataPipeline

__all__ = ["DataPipeline"]
