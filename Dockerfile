# ---------- Stage: Python backend ----------
FROM python:3.11-slim AS backend

# System deps required by XGBoost (libomp) and general build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends libomp-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Copy source code
COPY api/ api/
COPY backtest/ backtest/
COPY data/ data/
COPY models/ models/
COPY config/ config/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
