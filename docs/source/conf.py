# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add src/ to path so Sphinx can find the package
sys.path.insert(0, os.path.abspath("../../src"))

# ── Project information ───────────────────────────────────────────────────────
project = "ETF Trend Predictor"
author = "ETF Predictor Team"
copyright = "2026, ETF Predictor Team"
release = "0.1.0"

# ── General configuration ────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",       # pull docstrings from code
    "sphinx.ext.napoleon",      # Google / NumPy style docstrings
    "sphinx.ext.viewcode",      # add [source] links
    "myst_parser",              # Markdown support
]

autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ── HTML output ──────────────────────────────────────────────────────────────
html_theme = "sphinx_rtd_theme"
html_title = "ETF Trend Predictor API"
