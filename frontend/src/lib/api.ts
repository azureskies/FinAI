/**
 * API client for the FinAI FastAPI backend.
 * All fetch calls go through /api which is proxied to the backend in dev.
 */

const BASE = '/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const res = await fetch(`${BASE}${path}`, init);
	if (!res.ok) {
		const detail = await res.text();
		throw new Error(`API ${res.status}: ${detail}`);
	}
	return res.json() as Promise<T>;
}

// ------------------------------------------------------------------ //
//  Dashboard
// ------------------------------------------------------------------ //

export interface DashboardSummary {
	stocks_count: number;
	latest_prediction_date: string | null;
	predictions_count: number;
	active_models: number;
	backtest_runs: number;
	message: string | null;
}

export interface TopPick {
	stock_id: string;
	predicted_return: number;
	score: number;
	date: string | null;
}

export interface ScoredStock {
	stock_id: string;
	stock_name: string | null;
	composite_score: number | null;
	momentum_score: number | null;
	trend_score: number | null;
	volatility_score: number | null;
	volume_score: number | null;
	ai_score: number | null;
	risk_level: string | null;
	max_drawdown: number | null;
	volatility_ann: number | null;
	win_rate: number | null;
	predicted_return: number | null;
	date: string | null;
}

export interface TopPicksResponse {
	picks: ScoredStock[];
	message: string | null;
}

export interface StockScoreResponse {
	stock_id: string;
	stock_name: string | null;
	composite_score: number | null;
	momentum_score: number | null;
	trend_score: number | null;
	volatility_score: number | null;
	volume_score: number | null;
	ai_score: number | null;
	risk_level: string | null;
	max_drawdown: number | null;
	volatility_ann: number | null;
	win_rate: number | null;
	date: string | null;
	message: string | null;
}

export function getDashboardSummary() {
	return request<DashboardSummary>('/dashboard/summary');
}

export function getTopPicks(n = 10) {
	return request<TopPicksResponse>(`/dashboard/top-picks?n=${n}`);
}

// ------------------------------------------------------------------ //
//  Stocks
// ------------------------------------------------------------------ //

export interface StockItem {
	stock_id: string;
	name: string | null;
}

export interface StockListResponse {
	stocks: StockItem[];
	message: string | null;
}

export interface PriceRecord {
	date: string;
	stock_id: string;
	open: number | null;
	high: number | null;
	low: number | null;
	close: number | null;
	volume: number | null;
	adj_close: number | null;
}

export interface PriceResponse {
	prices: PriceRecord[];
	message: string | null;
}

export interface FeatureRecord {
	date: string;
	stock_id: string;
	features: Record<string, number>;
}

export interface FeatureResponse {
	features: FeatureRecord[];
	message: string | null;
}

export interface PredictionRecord {
	date: string;
	stock_id: string;
	predicted_return: number;
	score: number;
	model_version: string | null;
}

export interface PredictionResponse {
	predictions: PredictionRecord[];
	message: string | null;
}

export function listStocks() {
	return request<StockListResponse>('/stocks');
}

export function getStockPrices(stockId: string, startDate?: string, endDate?: string) {
	const params = new URLSearchParams();
	if (startDate) params.set('start_date', startDate);
	if (endDate) params.set('end_date', endDate);
	const qs = params.toString();
	return request<PriceResponse>(`/stocks/${stockId}/prices${qs ? '?' + qs : ''}`);
}

export function getStockFeatures(stockId: string, startDate?: string, endDate?: string) {
	const params = new URLSearchParams();
	if (startDate) params.set('start_date', startDate);
	if (endDate) params.set('end_date', endDate);
	const qs = params.toString();
	return request<FeatureResponse>(`/stocks/${stockId}/features${qs ? '?' + qs : ''}`);
}

export function getStockPredictions(stockId: string) {
	return request<PredictionResponse>(`/stocks/${stockId}/predictions`);
}

export function getStockScore(stockId: string) {
	return request<StockScoreResponse>(`/stocks/${stockId}/score`);
}

export function getTopScored(n = 10) {
	return request<TopPicksResponse>(`/dashboard/top-picks?n=${n}`);
}

// ------------------------------------------------------------------ //
//  Backtest
// ------------------------------------------------------------------ //

export interface BacktestSummary {
	id: string;
	run_date: string | null;
	model_type: string | null;
	period_start: string | null;
	period_end: string | null;
	metrics: Record<string, number> | null;
	config: Record<string, unknown> | null;
	created_at: string | null;
}

export interface BacktestListResponse {
	results: BacktestSummary[];
	message: string | null;
}

export function listBacktestResults(limit = 10) {
	return request<BacktestListResponse>(`/backtest/results?limit=${limit}`);
}

export function getBacktestResult(id: string) {
	return request<BacktestSummary>(`/backtest/${id}`);
}

export function deleteBacktestResult(id: string) {
	return request<{ status: string; message: string }>(`/backtest/${id}`, {
		method: 'DELETE'
	});
}

// ------------------------------------------------------------------ //
//  Models
// ------------------------------------------------------------------ //

export interface ModelVersion {
	id: string;
	model_type: string | null;
	metrics: Record<string, number> | null;
	file_path: string | null;
	storage_path: string | null;
	is_active: boolean | null;
	description: string | null;
	created_at: string | null;
}

export interface ModelListResponse {
	models: ModelVersion[];
	message: string | null;
}

export function listModels(modelType?: string) {
	const qs = modelType ? `?model_type=${modelType}` : '';
	return request<ModelListResponse>(`/models${qs}`);
}

export function getActiveModels(modelType?: string) {
	const qs = modelType ? `?model_type=${modelType}` : '';
	return request<ModelListResponse>(`/models/active${qs}`);
}

// ------------------------------------------------------------------ //
//  Pipeline
// ------------------------------------------------------------------ //

export interface PipelineRunRequest {
	mode?: 'full' | 'incremental';
	stock_ids?: string[];
}

export interface PipelineRunResponse {
	status: string;
	message: string;
	task_id: string;
}

export interface PipelineTaskStatus {
	task_id: string;
	status: string;
	started_at: string | null;
	finished_at: string | null;
	progress: string | null;
	error: string | null;
}

export interface PipelineStatusResponse {
	tasks: PipelineTaskStatus[];
}

export function runPipeline(req: PipelineRunRequest) {
	return request<PipelineRunResponse>('/pipeline/daily-update', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(req)
	});
}

export function getPipelineStatus() {
	return request<PipelineStatusResponse>('/pipeline/status');
}

// ------------------------------------------------------------------ //
//  Backtest — trigger
// ------------------------------------------------------------------ //

export interface BacktestRunRequest {
	model_type?: string;
	mode?: 'run' | 'walk_forward';
	period_start?: string;
	period_end?: string;
	initial_capital?: number;
}

export interface BacktestRunResponse {
	status: string;
	message: string;
	task_id: string | null;
}

export interface BacktestTaskStatus {
	task_id: string;
	status: string;
	progress: string | null;
	error: string | null;
}

export function runBacktest(req: BacktestRunRequest) {
	return request<BacktestRunResponse>('/backtest/run', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(req)
	});
}

export function getBacktestTaskStatus(taskId: string) {
	return request<BacktestTaskStatus>(`/backtest/status/${taskId}`);
}

// ------------------------------------------------------------------ //
//  Health
// ------------------------------------------------------------------ //

export function checkHealth() {
	return request<{ status: string }>('/health');
}

// ------------------------------------------------------------------ //
//  Monitoring
// ------------------------------------------------------------------ //

export interface PipelineRunItem {
	id: string;
	pipeline_name: string;
	start_time: string | null;
	end_time: string | null;
	status: string;
	metrics: Record<string, unknown> | null;
	error: string | null;
}

export interface PipelineRunsResponse {
	runs: PipelineRunItem[];
	message: string | null;
}

export function getPipelineRuns(limit = 10) {
	return request<PipelineRunsResponse>(`/monitoring/pipeline-runs?limit=${limit}`);
}
