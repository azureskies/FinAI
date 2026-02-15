<script lang="ts">
	import { onMount } from 'svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import Chart from '$lib/components/Chart.svelte';
	import MetricCard from '$lib/components/MetricCard.svelte';
	import { listBacktestResults, type BacktestSummary } from '$lib/api';

	let results: BacktestSummary[] = $state([]);
	let selected: BacktestSummary | null = $state(null);
	let loading = $state(true);
	let error = $state('');

	onMount(async () => {
		try {
			const res = await listBacktestResults(20);
			results = res.results;
			if (results.length > 0) selected = results[0];
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	});

	const listColumns = [
		{ key: 'run_date', label: '执行日期' },
		{ key: 'model_type', label: '模型类型' },
		{ key: 'period_start', label: '起始日' },
		{ key: 'period_end', label: '结束日' }
	];

	// Extract metrics from selected result for display
	let metricEntries = $derived(
		selected?.metrics ? Object.entries(selected.metrics) : []
	);

	function handleRowClick(row: Record<string, unknown>) {
		selected = results.find((r) => r.id === row.id) ?? null;
	}

	// Metric formatting
	function fmtMetric(key: string, val: number): string {
		if (key.toLowerCase().includes('sharpe') || key.toLowerCase().includes('ic')) {
			return val.toFixed(4);
		}
		if (key.toLowerCase().includes('return') || key.toLowerCase().includes('dd')) {
			return (val * 100).toFixed(2) + '%';
		}
		return val.toFixed(4);
	}

	function metricColor(key: string): 'blue' | 'green' | 'red' | 'yellow' | 'gray' {
		if (key.toLowerCase().includes('sharpe')) return 'blue';
		if (key.toLowerCase().includes('return')) return 'green';
		if (key.toLowerCase().includes('dd') || key.toLowerCase().includes('drawdown')) return 'red';
		return 'gray';
	}
</script>

<svelte:head>
	<title>FinAI - 回测结果</title>
</svelte:head>

<div class="space-y-6">
	<h1 class="text-2xl font-bold text-gray-800">回测结果</h1>

	{#if loading}
		<p class="text-gray-500">载入中...</p>
	{:else if error}
		<p class="text-red-500">错误: {error}</p>
	{:else}
		<!-- Results list -->
		<section>
			<h2 class="mb-3 text-lg font-semibold text-gray-700">历史回测</h2>
			<DataTable
				columns={listColumns}
				rows={results as unknown as Record<string, unknown>[]}
				emptyText="暂无回测记录"
				onRowClick={handleRowClick}
			/>
		</section>

		<!-- Selected result metrics -->
		{#if selected}
			<section>
				<h2 class="mb-3 text-lg font-semibold text-gray-700">
					回测详情 — {selected.model_type ?? '未知模型'}
					({selected.period_start} ~ {selected.period_end})
				</h2>

				{#if metricEntries.length > 0}
					<div class="grid grid-cols-2 gap-4 md:grid-cols-4">
						{#each metricEntries as [key, val]}
							<MetricCard
								title={key}
								value={fmtMetric(key, val as number)}
								color={metricColor(key)}
							/>
						{/each}
					</div>
				{:else}
					<p class="text-gray-400">该回测无指标资料</p>
				{/if}
			</section>
		{/if}
	{/if}
</div>
