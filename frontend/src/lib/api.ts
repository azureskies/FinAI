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

export interface TopPicksResponse {
	picks: TopPick[];
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
