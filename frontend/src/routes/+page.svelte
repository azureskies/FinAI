<script lang="ts">
	import { onMount } from 'svelte';
	import MetricCard from '$lib/components/MetricCard.svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import Chart from '$lib/components/Chart.svelte';
	import ScoreBar from '$lib/components/ScoreBar.svelte';
	import RiskBadge from '$lib/components/RiskBadge.svelte';
	import {
		getDashboardSummary,
		getTopPicks,
		type DashboardSummary,
		type ScoredStock
	} from '$lib/api';

	let summary: DashboardSummary | null = $state(null);
	let picks: ScoredStock[] = $state([]);
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

	// Equity curve placeholder data
	const equityLabels = Array.from({ length: 30 }, (_, i) => `D${i + 1}`);
	const equityData = Array.from({ length: 30 }, (_, i) => 100 + i * 0.5 + Math.sin(i) * 2);
</script>

<svelte:head>
	<title>FinAI - 儀表板</title>
</svelte:head>

<div class="space-y-6">
	<h1 class="text-2xl font-bold" style="color: var(--text-primary);">市場總覽</h1>

	{#if loading}
		<p style="color: var(--text-secondary);">載入中...</p>
	{:else if error}
		<p style="color: var(--color-danger);">錯誤: {error}</p>
	{:else}
		<!-- Metric cards -->
		<div class="grid grid-cols-2 gap-4 md:grid-cols-4">
			<MetricCard title="股票數量" value={summary?.stocks_count ?? 0} color="blue" />
			<MetricCard title="最新預測日" value={summary?.latest_prediction_date ?? '-'} color="green" />
			<MetricCard title="啟用模型數" value={summary?.active_models ?? 0} color="yellow" />
			<MetricCard title="回測次數" value={summary?.backtest_runs ?? 0} color="gray" />
		</div>

		<!-- Top 10 scored stocks -->
		<section>
			<h2 class="mb-3 text-lg font-semibold" style="color: var(--text-primary);">Top 10 推薦股票</h2>
			<div class="rounded-lg border p-4" style="background-color: var(--bg-secondary); border-color: var(--border-color);">
				{#if picks.length === 0}
					<p class="py-4 text-center" style="color: var(--text-secondary);">暫無評分資料</p>
				{:else}
					<div class="space-y-3">
						{#each picks as pick, i}
							<div class="flex items-center gap-4 rounded-lg p-3" style="background-color: var(--bg-tertiary);">
								<span class="flex h-8 w-8 items-center justify-center rounded-full text-sm font-bold" style="background-color: var(--bg-primary); color: var(--color-accent);">
									{i + 1}
								</span>
								<a href="/stocks/{pick.stock_id}" class="w-16 font-mono font-bold hover:underline" style="color: var(--color-accent);">
									{pick.stock_id}
								</a>
								<div class="flex-1">
									<ScoreBar score={pick.composite_score ?? 0} label="總分" />
								</div>
								{#if pick.risk_level}
									<RiskBadge level={pick.risk_level} />
								{/if}
								<span class="w-20 text-right font-mono text-sm" style="color: {(pick.predicted_return ?? 0) >= 0 ? 'var(--color-up)' : 'var(--color-down)'};">
									{pick.predicted_return != null ? (pick.predicted_return * 100).toFixed(2) + '%' : '-'}
								</span>
							</div>
						{/each}
					</div>
				{/if}
			</div>
		</section>

		<!-- Equity curve -->
		<section>
			<h2 class="mb-3 text-lg font-semibold" style="color: var(--text-primary);">投資組合表現</h2>
			<div class="rounded-lg border p-4" style="background-color: var(--bg-secondary); border-color: var(--border-color);">
				<Chart
					labels={equityLabels}
					datasets={[
						{
							label: '權益曲線',
							data: equityData,
							borderColor: '#58a6ff',
							backgroundColor: 'rgba(88,166,255,0.08)',
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
			<h2 class="mb-3 text-lg font-semibold" style="color: var(--text-primary);">最新預測摘要</h2>
			<div class="rounded-lg border p-4" style="background-color: var(--bg-secondary); border-color: var(--border-color);">
				<p class="text-sm" style="color: var(--text-secondary);">
					共 <span class="font-semibold" style="color: var(--text-primary);">{summary?.predictions_count ?? 0}</span> 筆預測，
					最新日期:
					<span class="font-semibold" style="color: var(--text-primary);">{summary?.latest_prediction_date ?? '-'}</span>
				</p>
			</div>
		</section>
	{/if}
</div>
