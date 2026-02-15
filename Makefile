.PHONY: help install install-dev setup test lint dev serve serve-api serve-frontend \
       fetch update retrain backtest pipeline pipeline-full health check-db docker-up docker-down clean

PYTHON ?= $(shell command -v python3 2>/dev/null || echo python)
PIP    ?= $(shell command -v pip3 2>/dev/null || echo pip)

# Default target
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------
# Environment
# ---------------------------------------------------------------
install: ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install all dependencies (including dev)
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov ruff httpx
	cd frontend && npm install

# ---------------------------------------------------------------
# Database Setup (SQLite by default, zero-config)
# ---------------------------------------------------------------
setup: ## Initialize database (create tables)
	$(PYTHON) -m scripts.setup_db

setup-check: ## Check database status only
	$(PYTHON) -m scripts.setup_db --check-only

setup-init: ## Setup + fetch initial data for top 5 stocks (1 year)
	$(PYTHON) -m scripts.setup_db --init-data

setup-init-custom: ## Setup + fetch initial data for custom stocks (e.g. make setup-init-custom STOCKS=2330,2317)
	$(PYTHON) -m scripts.setup_db --init-data --stocks $(STOCKS)

# ---------------------------------------------------------------
# Data Operations
# ---------------------------------------------------------------
fetch: ## Run daily data update (fetch prices, compute features, predict)
	$(PYTHON) -m scripts.daily_update

fetch-dry: ## Dry-run daily update (no writes to database)
	$(PYTHON) -m scripts.daily_update --dry-run

fetch-stock: ## Update a single stock (e.g. make fetch-stock STOCK=2330)
	$(PYTHON) -m scripts.daily_update --stock-id $(STOCK)

retrain: ## Run weekly model retrain with Optuna tuning
	$(PYTHON) -m scripts.weekly_retrain

backtest: ## Run backtest for a specific stock (e.g. make backtest STOCK=2330)
	$(PYTHON) -m scripts.daily_update --stock-id $(STOCK) --dry-run

pipeline: ## Trigger incremental daily update via API
	@curl -s -X POST http://localhost:8000/api/pipeline/daily-update \
		-H 'Content-Type: application/json' \
		-d '{"mode":"incremental"}' | $(PYTHON) -m json.tool

pipeline-full: ## Trigger full daily update via API
	@curl -s -X POST http://localhost:8000/api/pipeline/daily-update \
		-H 'Content-Type: application/json' \
		-d '{"mode":"full"}' | $(PYTHON) -m json.tool

# ---------------------------------------------------------------
# Testing
# ---------------------------------------------------------------
test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v

test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --cov --cov-report=term-missing

test-data: ## Run data module tests only
	$(PYTHON) -m pytest tests/test_cleaning.py tests/test_features.py tests/test_price_collector.py -v

test-models: ## Run model tests only
	$(PYTHON) -m pytest tests/test_baseline.py tests/test_tree_models.py tests/test_ensemble.py tests/test_training.py -v

test-backtest: ## Run backtest tests only
	$(PYTHON) -m pytest tests/test_costs.py tests/test_metrics.py tests/test_portfolio.py tests/test_engine.py -v

test-api: ## Run API tests only
	$(PYTHON) -m pytest tests/test_api.py -v

lint: ## Run ruff linter
	ruff check .

lint-fix: ## Auto-fix lint issues
	ruff check --fix .

# ---------------------------------------------------------------
# Servers
# ---------------------------------------------------------------
dev: ## Start frontend (auto-starts backend via Vite plugin)
	cd frontend && npm run dev

serve: ## Start both API and frontend in parallel
	@trap 'kill 0' EXIT; \
	$(PYTHON) -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 & \
	cd frontend && npm run dev

serve-api: ## Start FastAPI backend on port 8000
	$(PYTHON) -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

serve-frontend: ## Start SvelteKit frontend on port 5173
	cd frontend && npm run dev

# ---------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------
health: ## Run health check
	$(PYTHON) -m scripts.health_check

# ---------------------------------------------------------------
# Docker
# ---------------------------------------------------------------
docker-up: ## Start all services with Docker Compose
	docker compose up -d --build

docker-down: ## Stop all Docker services
	docker compose down

docker-logs: ## Tail Docker Compose logs
	docker compose logs -f

# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov
	@echo "Cleaned."
