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
		{ key: 'run_date', label: '執行日期' },
		{ key: 'model_type', label: '模型類型' },
		{ key: 'period_start', label: '起始日' },
		{ key: 'period_end', label: '結束日' }
	];

	// Extract metrics from selected result for display (only numeric values)
	let metricEntries = $derived.by(() => {
		const m = selected?.metrics;
		if (!m) return [] as [string, number][];
		return Object.entries(m).filter(([, v]) => typeof v === 'number' && isFinite(v)) as [string, number][];
	});

	function handleRowClick(row: Record<string, unknown>) {
		selected = results.find((r) => r.id === row.id) ?? null;
	}

	// Metric Chinese labels and descriptions
	const metricLabels: Record<string, { label: string; desc: string }> = {
		sharpe_ratio: { label: '夏普比率', desc: '每單位風險的超額報酬，>1 為佳' },
		sortino_ratio: { label: '索提諾比率', desc: '僅考慮下行風險的報酬比，>1 為佳' },
		calmar_ratio: { label: '卡瑪比率', desc: '年化報酬 / 最大回撤，>1 為佳' },
		max_drawdown: { label: '最大回撤', desc: '期間內從高點到低點的最大跌幅' },
		win_rate: { label: '勝率', desc: '獲利交易佔全部交易的比例' },
		annualized_return: { label: '年化報酬率', desc: '換算成年度的投資報酬率' },
		annualized_volatility: { label: '年化波動率', desc: '報酬率的年化標準差，越低越穩定' },
		total_return: { label: '總報酬率', desc: '整段回測期間的累計報酬' },
		trading_days: { label: '交易天數', desc: '回測期間的總交易日數' },
		total_trading_costs: { label: '總交易成本', desc: '手續費與稅費等交易費用總和' },
		num_trades: { label: '交易次數', desc: '期間內執行的買賣交易總數' },
		initial_capital: { label: '初始資金', desc: '回測起始的投入金額' },
		final_value: { label: '最終價值', desc: '回測結束時的投資組合總值' },
		profit_factor: { label: '獲利因子', desc: '總獲利 / 總虧損，>1 表示整體獲利' },
		avg_return: { label: '平均報酬', desc: '每筆交易的平均報酬率' },
		max_consecutive_wins: { label: '最大連勝', desc: '連續獲利交易的最大次數' },
		max_consecutive_losses: { label: '最大連敗', desc: '連續虧損交易的最大次數' },
		information_ratio: { label: '資訊比率', desc: '超額報酬 / 追蹤誤差，衡量主動管理能力' },
		beta: { label: 'Beta', desc: '相對大盤的系統性風險，1 表示與大盤同步' },
		alpha: { label: 'Alpha', desc: '超越大盤的超額報酬' }
	};

	function getMetricLabel(key: string): string {
		return metricLabels[key]?.label ?? key;
	}

	function getMetricDesc(key: string): string {
		return metricLabels[key]?.desc ?? '';
	}

	// Metric formatting
	function fmtMetric(key: string, val: number): string {
		const k = key.toLowerCase();
		// Percentage metrics
		if (
			k.includes('return') ||
			k.includes('drawdown') ||
			k.includes('volatility') ||
			k === 'win_rate'
		) {
			return (val * 100).toFixed(2) + '%';
		}
		// Ratio metrics
		if (k.includes('ratio') || k === 'alpha' || k === 'beta' || k === 'profit_factor') {
			return val.toFixed(2);
		}
		// Currency metrics
		if (k.includes('capital') || k.includes('value') || k.includes('cost')) {
			return val.toLocaleString('zh-TW', { maximumFractionDigits: 0 });
		}
		// Integer metrics
		if (k.includes('days') || k.includes('trades') || k.includes('consecutive') || k === 'num_trades') {
			return Math.round(val).toLocaleString('zh-TW');
		}
		return val.toFixed(4);
	}

	function metricColor(key: string): 'blue' | 'green' | 'red' | 'yellow' | 'gray' {
		const k = key.toLowerCase();
		if (k.includes('sharpe') || k.includes('sortino') || k.includes('calmar') || k.includes('information')) return 'blue';
		if (k.includes('return') || k === 'alpha' || k === 'win_rate' || k === 'profit_factor') return 'green';
		if (k.includes('drawdown') || k.includes('volatility') || k.includes('loss')) return 'red';
		if (k.includes('capital') || k.includes('value')) return 'yellow';
		return 'gray';
	}
</script>

<svelte:head>
	<title>FinAI - 回測結果</title>
</svelte:head>

<div class="space-y-6">
	<h1 class="text-2xl font-bold" style="color: var(--text-primary);">回測結果</h1>

	{#if loading}
		<p style="color: var(--text-secondary);">載入中...</p>
	{:else if error}
		<p style="color: var(--color-danger);">錯誤: {error}</p>
	{:else}
		<!-- Results list -->
		<section>
			<h2 class="mb-3 text-lg font-semibold" style="color: var(--text-primary);">歷史回測</h2>
			<DataTable
				columns={listColumns}
				rows={results as unknown as Record<string, unknown>[]}
				emptyText="暫無回測紀錄"
				onRowClick={handleRowClick}
			/>
		</section>

		<!-- Selected result metrics -->
		{#if selected}
			<section>
				<h2 class="mb-3 text-lg font-semibold" style="color: var(--text-primary);">
					回測詳情 — {selected.model_type ?? '未知模型'}
					({selected.period_start} ~ {selected.period_end})
				</h2>

				{#if metricEntries.length > 0}
					<div class="grid grid-cols-2 gap-4 md:grid-cols-4">
						{#each metricEntries as [key, val]}
							<MetricCard
								title={getMetricLabel(key)}
								value={fmtMetric(key, val as number)}
								subtitle={getMetricDesc(key)}
								color={metricColor(key)}
							/>
						{/each}
					</div>
				{:else}
					<p style="color: var(--text-secondary);">該回測無指標資料</p>
				{/if}
			</section>
		{/if}
	{/if}
</div>
