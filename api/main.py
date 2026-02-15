"""FastAPI application entry point for FinAI platform."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import backtest, dashboard, models, monitoring, pipeline, stocks

app = FastAPI(
    title="FinAI API",
    description="Taiwan Stock AI Analysis Platform API",
    version="0.1.0",
)

# CORS — allow the SvelteKit frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(stocks.router)
app.include_router(backtest.router)
app.include_router(models.router)
app.include_router(dashboard.router)
app.include_router(monitoring.router)
app.include_router(pipeline.router)


@app.get("/api/health")
def health_check() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}
