-- FinAI Platform Database Schema
-- Supabase (PostgreSQL) tables for Taiwan stock AI analysis

-- =============================================================
-- Stock Prices
-- =============================================================
CREATE TABLE IF NOT EXISTS stock_prices (
    id          BIGSERIAL PRIMARY KEY,
    date        DATE        NOT NULL,
    stock_id    VARCHAR(10) NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    adj_close   DOUBLE PRECISION,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
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
    id          BIGSERIAL PRIMARY KEY,
    date        DATE        NOT NULL,
    stock_id    VARCHAR(10) NOT NULL,
    features    JSONB       NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
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
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    model_type      VARCHAR(50)  NOT NULL,
    metrics         JSONB        NOT NULL,
    file_path       TEXT,
    storage_path    TEXT,
    is_active       BOOLEAN      DEFAULT FALSE,
    created_at      TIMESTAMPTZ  DEFAULT NOW(),
    description     TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_type_active
    ON model_versions (model_type, is_active);

-- =============================================================
-- Predictions
-- =============================================================
CREATE TABLE IF NOT EXISTS predictions (
    id                BIGSERIAL PRIMARY KEY,
    date              DATE         NOT NULL,
    stock_id          VARCHAR(10)  NOT NULL,
    predicted_return  DOUBLE PRECISION,
    score             DOUBLE PRECISION,
    model_version     UUID REFERENCES model_versions(id),
    created_at        TIMESTAMPTZ  DEFAULT NOW(),
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
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    run_date        DATE         NOT NULL,
    model_type      VARCHAR(50),
    period_start    DATE,
    period_end      DATE,
    metrics         JSONB        NOT NULL,
    config          JSONB,
    created_at      TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_backtest_date
    ON backtest_results (run_date DESC);

-- =============================================================
-- Pipeline Runs (monitoring)
-- =============================================================
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    pipeline_name   VARCHAR(50)  NOT NULL,
    start_time      TIMESTAMPTZ  NOT NULL,
    end_time        TIMESTAMPTZ,
    status          VARCHAR(20)  NOT NULL DEFAULT 'running',
    metrics         JSONB,
    error           TEXT,
    metadata        JSONB,
    created_at      TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_name_time
    ON pipeline_runs (pipeline_name, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status
    ON pipeline_runs (status);

-- =============================================================
-- Supabase Storage bucket (run via dashboard or API)
-- =============================================================
-- INSERT INTO storage.buckets (id, name, public)
-- VALUES ('models', 'models', false)
-- ON CONFLICT DO NOTHING;
