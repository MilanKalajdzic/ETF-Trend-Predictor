# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: base image with system deps
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

# Prevent .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System packages needed by some Python deps (lxml, pandas, matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && ARCH="$(dpkg --print-architecture)" \
    && curl -LO "https://quarto.org/download/latest/quarto-linux-${ARCH}.deb" \
    && dpkg -i "quarto-linux-${ARCH}.deb" \
    && rm "quarto-linux-${ARCH}.deb" \
    && rm -rf /var/lib/apt/lists/*

RUN pip install jupyter

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS deps

COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install CPU-only torch first from the PyTorch CPU wheel index so the
# image stays small (no CUDA libs). The .[dev,notebook] install below
# then sees torch already satisfied and skips the GPU wheel.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"

RUN pip install ".[dev,notebook]"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: final image
# ─────────────────────────────────────────────────────────────────────────────
FROM deps AS final

# Source and config
COPY src/ ./src/
COPY tests/ ./tests/
COPY notebooks/ ./notebooks/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY Makefile ./
COPY .pre-commit-config.yaml ./

# Reports — qmd sources + pre-computed CSVs and figures (no .html)
COPY reports/ ./reports/

# Raw price cache from data/ is baked in if present, otherwise downloaded at run time
COPY data/ ./data/

LABEL maintainer="ETF Predictor Team" \
      description="ETF trend prediction: full pipeline and reports" \
      version="1.0.0" \
      org.opencontainers.image.source="https://github.com/MilanKalajdzic/ETF-Trend-Predictor"

# /output/ is where the published HTML reports land. Bind-mount this
# directory (`docker run -v $PWD/output:/output ...`) to retrieve the
# rendered reports on the host.
RUN mkdir -p /output

# Default command: run the full pipeline end-to-end — download/process
# data, statistical analysis, MLP + LSTM walk-forward training, render
# all three Quarto reports, and publish them to /output/. Takes
# ~10 minutes on CPU.
CMD ["make", "report"]