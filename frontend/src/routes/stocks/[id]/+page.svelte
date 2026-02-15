<script lang="ts">
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import Chart from '$lib/components/Chart.svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import {
		getStockPrices,
		getStockFeatures,
		getStockPredictions,
		type PriceRecord,
		type FeatureRecord,
		type PredictionRecord
	} from '$lib/api';

	const stockId = $derived(page.params.id);

	let prices: PriceRecord[] = $state([]);
	let features: FeatureRecord[] = $state([]);
	let predictions: PredictionRecord[] = $state([]);
	let loading = $state(true);
	let error = $state('');

	onMount(async () => {
		try {
			const [priceRes, featRes, predRes] = await Promise.all([
				getStockPrices(stockId),
				getStockFeatures(stockId),
				getStockPredictions(stockId)
			]);
			prices = priceRes.prices;
			features = featRes.features;
			predictions = predRes.predictions;
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
			label: '预测报酬率',
			align: 'right' as const,
			format: (v: unknown) => ((v as number) * 100).toFixed(2) + '%'
		},
		{
			key: 'score',
			label: '评分',
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
		<a href="/stocks" class="text-sm text-blue-600 hover:underline">&larr; 返回列表</a>
		<h1 class="text-2xl font-bold text-gray-800">{stockId}</h1>
	</div>

	{#if loading}
		<p class="text-gray-500">载入中...</p>
	{:else if error}
		<p class="text-red-500">错误: {error}</p>
	{:else}
		<!-- Price chart -->
		<section>
			<h2 class="mb-3 text-lg font-semibold text-gray-700">价格走势</h2>
			<div class="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
				{#if prices.length > 0}
					<Chart
						labels={priceLabels}
						datasets={[
							{
								label: '收盘价',
								data: priceData,
								borderColor: '#2563eb',
								backgroundColor: 'rgba(37,99,235,0.08)',
								fill: true,
								tension: 0.2
							}
						]}
						height={350}
					/>
				{:else}
					<p class="py-8 text-center text-gray-400">暂无价格资料</p>
				{/if}
			</div>
		</section>

		<!-- Technical indicators / features -->
		<section>
			<h2 class="mb-3 text-lg font-semibold text-gray-700">技术指标 (最新)</h2>
			<div class="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
				{#if featureEntries.length > 0}
					<div class="grid grid-cols-2 gap-3 md:grid-cols-4">
						{#each featureEntries as [key, val]}
							<div class="rounded border border-gray-100 bg-gray-50 px-3 py-2">
								<p class="text-xs text-gray-500">{key}</p>
								<p class="font-mono text-sm font-medium text-gray-800">
									{typeof val === 'number' ? val.toFixed(4) : val}
								</p>
							</div>
						{/each}
					</div>
				{:else}
					<p class="py-4 text-center text-gray-400">暂无特征资料</p>
				{/if}
			</div>
		</section>

		<!-- Prediction history -->
		<section>
			<h2 class="mb-3 text-lg font-semibold text-gray-700">预测历史</h2>
			<DataTable
				columns={predColumns}
				rows={predictions as unknown as Record<string, unknown>[]}
				emptyText="暂无预测资料"
			/>
		</section>
	{/if}
</div>
