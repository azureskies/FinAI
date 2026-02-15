-- FinAI Platform Database Schema (SQLite)
-- Local equivalent of the PostgreSQL/Supabase schema

-- =============================================================
-- Stock Prices
-- =============================================================
CREATE TABLE IF NOT EXISTS stock_prices (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,
    stock_id    TEXT    NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      INTEGER,
    adj_close   REAL,
    created_at  TEXT DEFAULT (datetime('now')),
    UNIQUE (date, stock_id)
);

CREATE INDEX IF NOT EXISTS idx_prices_stock_date
    ON stock_prices (stock_id, date);
CREATE INDEX IF NOT EXISTS idx_prices_date
    ON stock_prices (date);

-- =============================================================
-- Stock Features (computed by FeatureEngine)
-- =============================================================
CREATE TABLE IF NOT EXISTS stock_features (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,
    stock_id    TEXT    NOT NULL,
    features    TEXT    NOT NULL,  -- JSON blob
    created_at  TEXT DEFAULT (datetime('now')),
    UNIQUE (date, stock_id)
);

CREATE INDEX IF NOT EXISTS idx_features_stock_date
    ON stock_features (stock_id, date);
CREATE INDEX IF NOT EXISTS idx_features_date
    ON stock_features (date);

-- =============================================================
-- Model Versions
-- =============================================================
CREATE TABLE IF NOT EXISTS model_versions (
    id              TEXT PRIMARY KEY,  -- UUID as text
    model_type      TEXT    NOT NULL,
    metrics         TEXT    NOT NULL,  -- JSON blob
    file_path       TEXT,
    storage_path    TEXT,
    is_active       INTEGER DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now')),
    description     TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_type_active
    ON model_versions (model_type, is_active);

-- =============================================================
-- Predictions
-- =============================================================
CREATE TABLE IF NOT EXISTS predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    date              TEXT    NOT NULL,
    stock_id          TEXT    NOT NULL,
    predicted_return  REAL,
    score             REAL,
    model_version     TEXT REFERENCES model_versions(id),
    created_at        TEXT DEFAULT (datetime('now')),
    UNIQUE (date, stock_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_date
    ON predictions (date DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model
    ON predictions (model_version, date);

-- =============================================================
-- Backtest Results
-- =============================================================
CREATE TABLE IF NOT EXISTS backtest_results (
    id              TEXT PRIMARY KEY,  -- UUID as text
    run_date        TEXT    NOT NULL,
    model_type      TEXT,
    period_start    TEXT,
    period_end      TEXT,
    metrics         TEXT    NOT NULL,  -- JSON blob
    config          TEXT,              -- JSON blob
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_backtest_date
    ON backtest_results (run_date DESC);

-- =============================================================
-- Pipeline Runs (monitoring)
-- =============================================================
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              TEXT PRIMARY KEY,  -- UUID as text
    pipeline_name   TEXT    NOT NULL,
    start_time      TEXT    NOT NULL,
    end_time        TEXT,
    status          TEXT    NOT NULL DEFAULT 'running',
    metrics         TEXT,   -- JSON blob
    error           TEXT,
    metadata        TEXT,   -- JSON blob
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_name_time
    ON pipeline_runs (pipeline_name, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status
    ON pipeline_runs (status);

-- =============================================================
-- Stock Universe
-- =============================================================
CREATE TABLE IF NOT EXISTS stock_universe (
    stock_id      TEXT PRIMARY KEY,
    stock_name    TEXT,
    market        TEXT,
    listing_date  TEXT,
    market_cap    REAL,
    avg_volume    REAL,
    updated_at    TEXT DEFAULT (datetime('now'))
);

-- =============================================================
-- Stock Scores
-- =============================================================
CREATE TABLE IF NOT EXISTS stock_scores (
    date              TEXT    NOT NULL,
    stock_id          TEXT    NOT NULL,
    composite_score   REAL,
    momentum_score    REAL,
    trend_score       REAL,
    volatility_score  REAL,
    volume_score      REAL,
    ai_score          REAL,
    risk_level        TEXT,
    max_drawdown      REAL,
    volatility_ann    REAL,
    win_rate          REAL,
    created_at        TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (date, stock_id)
);

CREATE INDEX IF NOT EXISTS idx_scores_date
    ON stock_scores (date DESC);
CREATE INDEX IF NOT EXISTS idx_scores_composite
    ON stock_scores (date, composite_score DESC);
