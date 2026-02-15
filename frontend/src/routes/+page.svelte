<script lang="ts">
	import { onMount } from 'svelte';
	import MetricCard from '$lib/components/MetricCard.svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import Chart from '$lib/components/Chart.svelte';
	import {
		getDashboardSummary,
		getTopPicks,
		type DashboardSummary,
		type TopPick
	} from '$lib/api';

	let summary: DashboardSummary | null = $state(null);
	let picks: TopPick[] = $state([]);
	let loading = $state(true);
	let error = $state('');

	onMount(async () => {
		try {
			const [s, p] = await Promise.all([getDashboardSummary(), getTopPicks(10)]);
			summary = s;
			picks = p.picks;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	});

	const pickColumns = [
		{ key: 'stock_id', label: '股票代码' },
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
		{ key: 'date', label: '日期' }
	];

	// Equity curve placeholder data (will be replaced with real API data)
	const equityLabels = Array.from({ length: 30 }, (_, i) => `D${i + 1}`);
	const equityData = Array.from({ length: 30 }, (_, i) => 100 + i * 0.5 + Math.sin(i) * 2);
</script>

<svelte:head>
	<title>FinAI - 首页</title>
</svelte:head>

<div class="space-y-6">
	<h1 class="text-2xl font-bold text-gray-800">市场总览</h1>

	{#if loading}
		<p class="text-gray-500">载入中...</p>
	{:else if error}
		<p class="text-red-500">错误: {error}</p>
	{:else}
		<!-- Metric cards -->
		<div class="grid grid-cols-2 gap-4 md:grid-cols-4">
			<MetricCard
				title="股票数量"
				value={summary?.stocks_count ?? 0}
				color="blue"
			/>
			<MetricCard
				title="最新预测日"
				value={summary?.latest_prediction_date ?? '-'}
				color="green"
			/>
			<MetricCard
				title="启用模型数"
				value={summary?.active_models ?? 0}
				color="yellow"
			/>
			<MetricCard
				title="回测次数"
				value={summary?.backtest_runs ?? 0}
				color="gray"
			/>
		</div>

		<!-- Top 10 stock picks -->
		<section>
			<h2 class="mb-3 text-lg font-semibold text-gray-700">Top 10 推荐股票</h2>
			<DataTable columns={pickColumns} rows={picks as unknown as Record<string, unknown>[]} />
		</section>

		<!-- Equity curve -->
		<section>
			<h2 class="mb-3 text-lg font-semibold text-gray-700">投资组合表现</h2>
			<div class="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
				<Chart
					labels={equityLabels}
					datasets={[
						{
							label: '权益曲线',
							data: equityData,
							borderColor: '#2563eb',
							backgroundColor: 'rgba(37,99,235,0.1)',
							fill: true,
							tension: 0.3
						}
					]}
					height={320}
				/>
			</div>
		</section>

		<!-- Recent predictions summary -->
		<section>
			<h2 class="mb-3 text-lg font-semibold text-gray-700">最新预测摘要</h2>
			<div class="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
				<p class="text-sm text-gray-600">
					共 <span class="font-semibold">{summary?.predictions_count ?? 0}</span> 笔预测，
					最新日期:
					<span class="font-semibold">{summary?.latest_prediction_date ?? '-'}</span>
				</p>
			</div>
		</section>
	{/if}
</div>
