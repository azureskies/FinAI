<script lang="ts">
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import Chart from '$lib/components/Chart.svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import ScoreGauge from '$lib/components/ScoreGauge.svelte';
	import ScoreRadar from '$lib/components/ScoreRadar.svelte';
	import RiskBadge from '$lib/components/RiskBadge.svelte';
	import MetricCard from '$lib/components/MetricCard.svelte';
	import {
		getStockPrices,
		getStockFeatures,
		getStockPredictions,
		getStockScore,
		type PriceRecord,
		type FeatureRecord,
		type PredictionRecord,
		type StockScoreResponse
	} from '$lib/api';

	const stockId = $derived(page.params.id ?? '');

	let prices: PriceRecord[] = $state([]);
	let features: FeatureRecord[] = $state([]);
	let predictions: PredictionRecord[] = $state([]);
	let scoreData: StockScoreResponse | null = $state(null);
	let loading = $state(true);
	let error = $state('');

	onMount(async () => {
		try {
			const [priceRes, featRes, predRes, scoreRes] = await Promise.all([
				getStockPrices(stockId),
				getStockFeatures(stockId),
				getStockPredictions(stockId),
				getStockScore(stockId)
			]);
			prices = priceRes.prices;
			features = featRes.features;
			predictions = predRes.predictions;
			scoreData = scoreRes;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	});

	// Price chart data
	let priceLabels = $derived(prices.map((p) => p.date));
	let priceData = $derived(prices.map((p) => p.close ?? 0));

	// Prediction columns
	const predColumns = [
		{ key: 'date', label: '日期' },
		{
			key: 'predicted_return',
			label: '預測報酬率',
			align: 'right' as const,
			format: (v: unknown) => ((v as number) * 100).toFixed(2) + '%'
		},
		{
			key: 'score',
			label: '評分',
			align: 'right' as const,
			format: (v: unknown) => (v as number).toFixed(4)
		},
		{ key: 'model_version', label: '模型版本' }
	];

	// Latest feature values
	let latestFeature = $derived(features.length > 0 ? features[features.length - 1] : null);
	let featureEntries = $derived(
		latestFeature ? Object.entries(latestFeature.features).sort(([a], [b]) => a.localeCompare(b)) : []
	);
</script>

<svelte:head>
	<title>FinAI - {stockId}</title>
</svelte:head>

<div class="space-y-6">
	<div class="flex items-center gap-4">
		<a href="/stocks" class="text-sm hover:underline" style="color: var(--color-accent);">&larr; 返回列表</a>
		<h1 class="text-2xl font-bold" style="color: var(--text-primary);">{stockId}</h1>
		{#if scoreData?.risk_level}
			<RiskBadge level={scoreData.risk_level} />
		{/if}
	</div>

	{#if loading}
		<p style="color: var(--text-secondary);">載入中...</p>
	{:else if error}
		<p style="color: var(--color-danger);">錯誤: {error}</p>
	{:else}
		<!-- Score overview -->
		{#if scoreData?.composite_score != null}
			<section>
				<h2 class="mb-3 text-lg font-semibold" style="color: var(--text-primary);">評分總覽</h2>
				<div class="rounded-lg border p-6" style="background-color: var(--bg-secondary); border-color: var(--border-color);">
					<div class="flex flex-wrap items-center justify-around gap-6">
						<!-- Composite gauge -->
						<ScoreGauge score={scoreData.composite_score} size={140} label="綜合評分" />

						<!-- Radar chart -->
						<ScoreRadar
							momentum={scoreData.momentum_score ?? 50}
							trend={scoreData.trend_score ?? 50}
							volatility={scoreData.volatility_score ?? 50}
							volume={scoreData.volume_score ?? 50}
							ai={scoreData.ai_score ?? 50}
							size={280}
						/>

						<!-- Risk metrics -->
						<div class="grid grid-cols-1 gap-3">
							<MetricCard
								title="最大回撤"
								value={scoreData.max_drawdown != null ? (scoreData.max_drawdown * 100).toFixed(1) + '%' : '-'}
								color="red"
							/>
							<MetricCard
								title="年化波動率"
								value={scoreData.volatility_ann != null ? (scoreData.volatility_ann * 100).toFixed(1) + '%' : '-'}
								color="yellow"
							/>
							<MetricCard
								title="勝率"
								value={scoreData.win_rate != null ? (scoreData.win_rate * 100).toFixed(1) + '%' : '-'}
								color="green"
							/>
						</div>
					</div>
				</div>
			</section>
		{/if}

		<!-- Price chart -->
		<section>
			<h2 class="mb-3 text-lg font-semibold" style="color: var(--text-primary);">價格走勢</h2>
			<div class="rounded-lg border p-4" style="background-color: var(--bg-secondary); border-color: var(--border-color);">
				{#if prices.length > 0}
					<Chart
						labels={priceLabels}
						datasets={[
							{
								label: '收盤價',
								data: priceData,
								borderColor: '#58a6ff',
								backgroundColor: 'rgba(88,166,255,0.08)',
								fill: true,
								tension: 0.2
							}
						]}
						height={350}
					/>
				{:else}
					<p class="py-8 text-center" style="color: var(--text-secondary);">暫無價格資料</p>
				{/if}
			</div>
		</section>

		<!-- Technical indicators / features -->
		<section>
			<h2 class="mb-3 text-lg font-semibold" style="color: var(--text-primary);">技術指標 (最新)</h2>
			<div class="rounded-lg border p-4" style="background-color: var(--bg-secondary); border-color: var(--border-color);">
				{#if featureEntries.length > 0}
					<div class="grid grid-cols-2 gap-3 md:grid-cols-4">
						{#each featureEntries as [key, val]}
							<div class="rounded border px-3 py-2" style="border-color: var(--border-color); background-color: var(--bg-tertiary);">
								<p class="text-xs" style="color: var(--text-secondary);">{key}</p>
								<p class="font-mono text-sm font-medium" style="color: var(--text-primary);">
									{typeof val === 'number' ? val.toFixed(4) : val}
								</p>
							</div>
						{/each}
					</div>
				{:else}
					<p class="py-4 text-center" style="color: var(--text-secondary);">暫無特徵資料</p>
				{/if}
			</div>
		</section>

		<!-- Prediction history -->
		<section>
			<h2 class="mb-3 text-lg font-semibold" style="color: var(--text-primary);">預測歷史</h2>
			<DataTable
				columns={predColumns}
				rows={predictions as unknown as Record<string, unknown>[]}
				emptyText="暫無預測資料"
			/>
		</section>
	{/if}
</div>
