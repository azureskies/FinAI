<script lang="ts">
	import { onMount } from 'svelte';
	import MetricCard from '$lib/components/MetricCard.svelte';
	import {
		listBacktestResults,
		runBacktest,
		deleteBacktestResult,
		getBacktestTaskStatus,
		type BacktestSummary,
		type BacktestRunRequest
	} from '$lib/api';

	// --- Form state ---
	let btModelType = $state('ensemble');
	let btMode = $state<'run' | 'walk_forward'>('run');
	let btCapital = $state(10000000);
	let btStart = $state(defaultStartDate());
	let btEnd = $state(todayStr());
	let btLoading = $state(false);
	let btMessage = $state('');
	let btError = $state('');
	let btProgress = $state('');
	let btTaskStatus = $state<'pending' | 'running' | 'success' | 'failed' | ''>('');

	// --- Results state ---
	let results: BacktestSummary[] = $state([]);
	let selected: BacktestSummary | null = $state(null);
	let loading = $state(true);
	let error = $state('');

	// --- Compare state ---
	let compareIds = $state(new Set<string>());

	function todayStr(): string {
		return new Date().toISOString().slice(0, 10);
	}

	function defaultStartDate(): string {
		const d = new Date();
		d.setMonth(d.getMonth() - 6);
		return d.toISOString().slice(0, 10);
	}

	onMount(async () => {
		await fetchResults();
	});

	async function fetchResults() {
		loading = true;
		error = '';
		try {
			const res = await listBacktestResults(20);
			results = res.results;
			if (results.length > 0 && !selected) selected = results[0];
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	async function handleRunBacktest() {
		btLoading = true;
		btMessage = '';
		btError = '';
		btProgress = '提交中...';
		btTaskStatus = 'pending';
		try {
			const req: BacktestRunRequest = {
				model_type: btModelType,
				mode: btMode,
				initial_capital: btCapital
			};
			if (btStart) req.period_start = btStart;
			if (btEnd) req.period_end = btEnd;
			const res = await runBacktest(req);
			if (res.task_id) {
				await pollBacktestProgress(res.task_id);
			} else {
				btMessage = res.message;
				btLoading = false;
			}
		} catch (e) {
			btError = e instanceof Error ? e.message : String(e);
			btLoading = false;
			btTaskStatus = '';
		}
	}

	async function pollBacktestProgress(taskId: string) {
		const poll = async () => {
			try {
				const status = await getBacktestTaskStatus(taskId);
				btProgress = status.progress ?? '';
				btTaskStatus = status.status as typeof btTaskStatus;

				if (status.status === 'running' || status.status === 'pending') {
					setTimeout(poll, 1500);
				} else if (status.status === 'success') {
					btMessage = btProgress;
					btLoading = false;
					selected = null;
					await fetchResults();
				} else {
					btError = status.error ?? '回測執行失敗';
					btLoading = false;
				}
			} catch {
				// Status endpoint might not be ready yet, retry
				setTimeout(poll, 2000);
			}
		};
		poll();
	}

	function handleRowClick(r: BacktestSummary) {
		selected = r;
		// Clear compare selection when clicking a row directly
		compareIds = new Set<string>();
	}

	function handleCheckbox(id: string) {
		const next = new Set(compareIds);
		if (next.has(id)) {
			next.delete(id);
		} else {
			next.add(id);
		}
		compareIds = next;
	}

	function clearCompare() {
		compareIds = new Set<string>();
	}

	let allSelected = $derived(results.length > 0 && compareIds.size === results.length);

	function toggleSelectAll() {
		if (allSelected) {
			compareIds = new Set<string>();
		} else {
			compareIds = new Set(results.map((r) => r.id));
		}
	}

	let deleting = $state(false);

	async function handleDeleteSelected() {
		const ids = [...compareIds];
		if (ids.length === 0) return;
		deleting = true;
		try {
			await Promise.all(ids.map((id) => deleteBacktestResult(id)));
			// Clear selection and refresh
			compareIds = new Set<string>();
			if (selected && ids.includes(selected.id)) selected = null;
			await fetchResults();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			deleting = false;
		}
	}

	// --- Model descriptions ---
	const modelDescriptions: Record<string, string> = {
		ensemble: '綜合多個模型的預測，通常表現最穩定',
		ridge: 'Ridge 迴歸，適合線性關係明顯的市場',
		xgboost: '梯度提升樹，擅長捕捉非線性特徵',
		random_forest: '隨機森林，對異常值較為穩健',
		lightgbm: '輕量級梯度提升，訓練速度快'
	};

	const modeDescriptions: Record<string, string> = {
		run: '靜態回測：使用固定訓練集對測試期間進行回測',
		walk_forward: 'Walk Forward：滾動視窗逐步推進，更貼近實際交易情境'
	};

	// --- Metric labels & formatting (reuse from previous) ---
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

	function fmtMetric(key: string, val: number): string {
		const k = key.toLowerCase();
		if (
			k.includes('return') ||
			k.includes('drawdown') ||
			k.includes('volatility') ||
			k === 'win_rate'
		) {
			return (val * 100).toFixed(2) + '%';
		}
		if (k.includes('ratio') || k === 'alpha' || k === 'beta' || k === 'profit_factor') {
			return val.toFixed(2);
		}
		if (k.includes('capital') || k.includes('value') || k.includes('cost')) {
			return val.toLocaleString('zh-TW', { maximumFractionDigits: 0 });
		}
		if (
			k.includes('days') ||
			k.includes('trades') ||
			k.includes('consecutive') ||
			k === 'num_trades'
		) {
			return Math.round(val).toLocaleString('zh-TW');
		}
		return val.toFixed(4);
	}

	function metricColor(key: string): 'blue' | 'green' | 'red' | 'yellow' | 'gray' {
		const k = key.toLowerCase();
		if (
			k.includes('sharpe') ||
			k.includes('sortino') ||
			k.includes('calmar') ||
			k.includes('information')
		)
			return 'blue';
		if (k.includes('return') || k === 'alpha' || k === 'win_rate' || k === 'profit_factor')
			return 'green';
		if (k.includes('drawdown') || k.includes('volatility') || k.includes('loss')) return 'red';
		if (k.includes('capital') || k.includes('value')) return 'yellow';
		return 'gray';
	}

	// --- Metric grouping ---
	interface MetricGroup {
		label: string;
		color: 'green' | 'blue' | 'gray';
		keys: string[];
	}

	const metricGroups: MetricGroup[] = [
		{
			label: '報酬',
			color: 'green',
			keys: ['total_return', 'annualized_return', 'initial_capital', 'final_value']
		},
		{
			label: '風險',
			color: 'blue',
			keys: [
				'sharpe_ratio',
				'sortino_ratio',
				'calmar_ratio',
				'max_drawdown',
				'annualized_volatility'
			]
		},
		{
			label: '交易',
			color: 'gray',
			keys: ['num_trades', 'win_rate', 'total_trading_costs', 'trading_days']
		}
	];

	const groupedKeys = new Set(metricGroups.flatMap((g) => g.keys));

	function getGroupedMetrics(metrics: Record<string, number>) {
		const entries = Object.entries(metrics).filter(
			([, v]) => typeof v === 'number' && isFinite(v)
		);
		const groups = metricGroups.map((g) => ({
			...g,
			entries: g.keys
				.filter((k) => metrics[k] !== undefined && typeof metrics[k] === 'number' && isFinite(metrics[k]))
				.map((k) => [k, metrics[k]] as [string, number])
		}));
		// Collect "other" metrics not in predefined groups
		const otherEntries = entries.filter(([k]) => !groupedKeys.has(k));
		if (otherEntries.length > 0) {
			groups.push({
				label: '其他',
				color: 'gray',
				keys: otherEntries.map(([k]) => k),
				entries: otherEntries as [string, number][]
			});
		}
		return groups.filter((g) => g.entries.length > 0);
	}

	// --- Compare logic ---
	let compareResults = $derived.by(() => {
		if (compareIds.size < 2) return [];
		return results.filter((r) => compareIds.has(r.id));
	});

	let isCompareMode = $derived(compareIds.size >= 2);

	// Metrics where higher is better
	const higherIsBetter = new Set([
		'total_return',
		'annualized_return',
		'sharpe_ratio',
		'sortino_ratio',
		'calmar_ratio',
		'win_rate',
		'final_value',
		'profit_factor',
		'alpha',
		'avg_return',
		'max_consecutive_wins',
		'information_ratio'
	]);

	// Metrics where lower is better
	const lowerIsBetter = new Set([
		'max_drawdown',
		'annualized_volatility',
		'total_trading_costs',
		'max_consecutive_losses',
		'beta'
	]);

	function getAllCompareKeys(items: BacktestSummary[]): string[] {
		const keySet = new Set<string>();
		for (const item of items) {
			if (item.metrics) {
				for (const [k, v] of Object.entries(item.metrics)) {
					if (typeof v === 'number' && isFinite(v)) keySet.add(k);
				}
			}
		}
		// Sort: grouped first in order, then others
		const allGroupedKeys = metricGroups.flatMap((g) => g.keys);
		const sorted: string[] = [];
		for (const k of allGroupedKeys) {
			if (keySet.has(k)) sorted.push(k);
		}
		for (const k of keySet) {
			if (!sorted.includes(k)) sorted.push(k);
		}
		return sorted;
	}

	function getBestIndex(key: string, items: BacktestSummary[]): number {
		const values = items.map((r) => r.metrics?.[key] ?? NaN);
		if (values.every((v) => isNaN(v))) return -1;

		if (higherIsBetter.has(key)) {
			let bestIdx = -1;
			let bestVal = -Infinity;
			for (let i = 0; i < values.length; i++) {
				if (!isNaN(values[i]) && values[i] > bestVal) {
					bestVal = values[i];
					bestIdx = i;
				}
			}
			return bestIdx;
		}

		if (lowerIsBetter.has(key)) {
			let bestIdx = -1;
			let bestVal = Infinity;
			for (let i = 0; i < values.length; i++) {
				if (!isNaN(values[i]) && values[i] < bestVal) {
					bestVal = values[i];
					bestIdx = i;
				}
			}
			return bestIdx;
		}

		return -1; // No best for neutral metrics
	}

	function fmtTotalReturn(r: BacktestSummary): string {
		const tr = r.metrics?.total_return;
		if (typeof tr !== 'number' || !isFinite(tr)) return '-';
		return (tr * 100).toFixed(2) + '%';
	}

	function fmtDateTime(r: BacktestSummary): string {
		const raw = r.created_at ?? r.run_date;
		if (!raw) return '-';
		const d = new Date(raw);
		if (isNaN(d.getTime())) return raw;
		// If created_at has time info, show date+time; otherwise just date
		if (r.created_at) {
			return d.toLocaleString('zh-TW', {
				month: '2-digit',
				day: '2-digit',
				hour: '2-digit',
				minute: '2-digit'
			});
		}
		return d.toLocaleDateString('zh-TW', {
			year: 'numeric',
			month: '2-digit',
			day: '2-digit'
		});
	}

	function fmtCapitalInput(val: number): string {
		return val.toLocaleString('zh-TW');
	}

	function handleCapitalInput(e: Event) {
		const input = e.target as HTMLInputElement;
		const raw = input.value.replace(/[^\d]/g, '');
		btCapital = raw ? parseInt(raw, 10) : 0;
	}
</script>

<svelte:head>
	<title>FinAI - 回測分析</title>
</svelte:head>

<div class="space-y-6">
	<h1 class="text-2xl font-bold" style="color: var(--text-primary);">回測分析</h1>

	<!-- Section 1: Backtest Settings -->
	<section
		class="rounded-lg border p-5"
		style="background-color: var(--bg-secondary); border-color: var(--border-color);"
	>
		<h2 class="mb-4 text-lg font-semibold" style="color: var(--text-primary);">回測設定</h2>
		<div class="grid grid-cols-2 gap-4 md:grid-cols-3">
			<div>
				<label class="mb-1 block text-xs" style="color: var(--text-secondary);">模型類型</label>
				<select
					bind:value={btModelType}
					class="w-full rounded border px-3 py-2 text-sm"
					style="background-color: var(--bg-tertiary); border-color: var(--border-color); color: var(--text-primary);"
				>
					<option value="ensemble">Ensemble</option>
					<option value="ridge">Ridge</option>
					<option value="xgboost">XGBoost</option>
					<option value="random_forest">Random Forest</option>
					<option value="lightgbm">LightGBM</option>
				</select>
				<p class="mt-1 text-xs" style="color: var(--text-secondary);">
					{modelDescriptions[btModelType] ?? ''}
				</p>
			</div>
			<div>
				<label class="mb-1 block text-xs" style="color: var(--text-secondary);">回測模式</label>
				<select
					bind:value={btMode}
					class="w-full rounded border px-3 py-2 text-sm"
					style="background-color: var(--bg-tertiary); border-color: var(--border-color); color: var(--text-primary);"
				>
					<option value="run">靜態回測</option>
					<option value="walk_forward">Walk Forward</option>
				</select>
				<p class="mt-1 text-xs" style="color: var(--text-secondary);">
					{modeDescriptions[btMode] ?? ''}
				</p>
			</div>
			<div>
				<label class="mb-1 block text-xs" style="color: var(--text-secondary);">初始資金</label>
				<input
					type="text"
					value={fmtCapitalInput(btCapital)}
					oninput={handleCapitalInput}
					class="w-full rounded border px-3 py-2 text-sm"
					style="background-color: var(--bg-tertiary); border-color: var(--border-color); color: var(--text-primary);"
				/>
				<p class="mt-1 text-xs" style="color: var(--text-secondary);">回測起始投入金額</p>
			</div>
			<div>
				<label class="mb-1 block text-xs" style="color: var(--text-secondary);">起始日期</label>
				<input
					type="date"
					bind:value={btStart}
					class="w-full rounded border px-3 py-2 text-sm"
					style="background-color: var(--bg-tertiary); border-color: var(--border-color); color: var(--text-primary);"
				/>
				<p class="mt-1 text-xs" style="color: var(--text-secondary);">預設 6 個月前</p>
			</div>
			<div>
				<label class="mb-1 block text-xs" style="color: var(--text-secondary);">結束日期</label>
				<input
					type="date"
					bind:value={btEnd}
					class="w-full rounded border px-3 py-2 text-sm"
					style="background-color: var(--bg-tertiary); border-color: var(--border-color); color: var(--text-primary);"
				/>
				<p class="mt-1 text-xs" style="color: var(--text-secondary);">預設今天</p>
			</div>
			<div class="flex items-end">
				<button
					onclick={handleRunBacktest}
					disabled={btLoading}
					class="rounded px-4 py-2 text-sm font-medium text-white transition-colors disabled:opacity-50"
					style="background-color: var(--color-accent);"
				>
					{#if btLoading}
						<span class="inline-flex items-center gap-2">
							<svg class="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
								<circle
									class="opacity-25"
									cx="12"
									cy="12"
									r="10"
									stroke="currentColor"
									stroke-width="4"
								/>
								<path
									class="opacity-75"
									fill="currentColor"
									d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
								/>
							</svg>
							執行中...
						</span>
					{:else}
						執行回測
					{/if}
				</button>
			</div>
		</div>

		{#if btLoading && btProgress}
			<div
				class="mt-3 flex items-center gap-3 rounded border px-3 py-2 text-sm"
				style="border-color: var(--border-color); background-color: var(--bg-tertiary);"
			>
				<svg class="h-4 w-4 shrink-0 animate-spin" viewBox="0 0 24 24" fill="none" style="color: var(--color-accent);">
					<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
					<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
				</svg>
				<span style="color: var(--text-primary);">{btProgress}</span>
			</div>
		{/if}
		{#if btMessage && !btLoading}
			<p class="mt-3 text-sm" style="color: var(--color-success);">{btMessage}</p>
		{/if}
		{#if btError}
			<p class="mt-3 text-sm" style="color: var(--color-danger);">{btError}</p>
		{/if}
	</section>

	<!-- Section 2: Results List -->
	{#if loading}
		<p style="color: var(--text-secondary);">載入中...</p>
	{:else if error}
		<p style="color: var(--color-danger);">錯誤: {error}</p>
	{:else}
		<section
			class="rounded-lg border p-5"
			style="background-color: var(--bg-secondary); border-color: var(--border-color);"
		>
			<div class="mb-4 flex items-center justify-between">
				<h2 class="text-lg font-semibold" style="color: var(--text-primary);">歷史回測</h2>
				{#if compareIds.size > 0}
					<div class="flex gap-2">
						<button
							onclick={handleDeleteSelected}
							disabled={deleting}
							class="rounded px-3 py-1 text-xs font-medium transition-colors"
							style="background-color: var(--color-danger); color: white;"
						>
							{deleting ? '刪除中...' : `刪除所選 (${compareIds.size})`}
						</button>
						<button
							onclick={clearCompare}
							class="rounded px-3 py-1 text-xs font-medium transition-colors"
							style="background-color: var(--bg-tertiary); color: var(--text-secondary); border: 1px solid var(--border-color);"
						>
							清除比較
						</button>
					</div>
				{/if}
			</div>

			{#if results.length === 0}
				<p style="color: var(--text-secondary);">暫無回測紀錄</p>
			{:else}
				<div class="overflow-x-auto">
					<table class="w-full text-sm" style="color: var(--text-primary);">
						<thead>
							<tr style="border-color: var(--border-color);" class="border-b">
								<th class="w-10 px-3 py-2 text-center text-xs font-medium" style="color: var(--text-secondary);">
									<input
										type="checkbox"
										checked={allSelected}
										onchange={toggleSelectAll}
										class="cursor-pointer"
										title="全選"
									/>
								</th>
								<th class="px-3 py-2 text-left text-xs font-medium" style="color: var(--text-secondary);">
									執行時間
								</th>
								<th class="px-3 py-2 text-left text-xs font-medium" style="color: var(--text-secondary);">
									模型
								</th>
								<th class="px-3 py-2 text-left text-xs font-medium" style="color: var(--text-secondary);">
									期間
								</th>
								<th class="px-3 py-2 text-right text-xs font-medium" style="color: var(--text-secondary);">
									總報酬率
								</th>
							</tr>
						</thead>
						<tbody>
							{#each results as r (r.id)}
								<tr
									class="cursor-pointer border-b transition-colors"
									style="border-color: var(--border-color); {selected?.id === r.id && !isCompareMode ? 'background-color: var(--bg-tertiary);' : ''}"
									onclick={() => handleRowClick(r)}
								>
									<td class="px-3 py-2" onclick={(e) => e.stopPropagation()}>
										<input
											type="checkbox"
											checked={compareIds.has(r.id)}
											onchange={() => handleCheckbox(r.id)}
											class="cursor-pointer"
										/>
									</td>
									<td class="px-3 py-2" style="color: var(--text-secondary);">
										{fmtDateTime(r)}
									</td>
									<td class="px-3 py-2">{r.model_type ?? '-'}</td>
									<td class="px-3 py-2" style="color: var(--text-secondary);">
										{r.period_start ?? '-'} ~ {r.period_end ?? '-'}
									</td>
									<td
										class="px-3 py-2 text-right font-medium"
										style="color: {(r.metrics?.total_return ?? 0) >= 0 ? 'var(--color-success)' : 'var(--color-danger)'};"
									>
										{fmtTotalReturn(r)}
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</section>

		<!-- Section 3: Detail or Compare -->
		{#if isCompareMode}
			<!-- Compare mode -->
			<section
				class="rounded-lg border p-5"
				style="background-color: var(--bg-secondary); border-color: var(--border-color);"
			>
				<h2 class="mb-4 text-lg font-semibold" style="color: var(--text-primary);">
					結果比較 ({compareResults.length} 筆)
				</h2>
				<div class="overflow-x-auto">
					<table class="w-full text-sm" style="color: var(--text-primary);">
						<thead>
							<tr style="border-color: var(--border-color);" class="border-b">
								<th class="px-3 py-2 text-left text-xs font-medium" style="color: var(--text-secondary);">
									指標
								</th>
								{#each compareResults as cr}
									<th class="px-3 py-2 text-right text-xs font-medium" style="color: var(--text-secondary);">
										{cr.model_type ?? '未知'}
										<br />
										<span class="font-normal">{fmtDateTime(cr)}</span>
									</th>
								{/each}
							</tr>
						</thead>
						<tbody>
							{#each getAllCompareKeys(compareResults) as key}
								{@const bestIdx = getBestIndex(key, compareResults)}
								<tr class="border-b" style="border-color: var(--border-color);">
									<td class="px-3 py-2 text-xs" style="color: var(--text-secondary);">
										{getMetricLabel(key)}
									</td>
									{#each compareResults as cr, i}
										{@const val = cr.metrics?.[key]}
										<td
											class="px-3 py-2 text-right font-medium"
											style="{i === bestIdx ? 'color: var(--color-accent); font-weight: 700;' : ''}"
										>
											{typeof val === 'number' && isFinite(val)
												? fmtMetric(key, val)
												: '-'}
										</td>
									{/each}
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</section>
		{:else if selected}
			<!-- Detail mode -->
			<section
				class="rounded-lg border p-5"
				style="background-color: var(--bg-secondary); border-color: var(--border-color);"
			>
				<h2 class="mb-4 text-lg font-semibold" style="color: var(--text-primary);">
					回測詳情 — {selected.model_type ?? '未知模型'}
					({selected.period_start} ~ {selected.period_end})
				</h2>

				{#if selected.metrics}
					{@const groups = getGroupedMetrics(selected.metrics)}
					{#each groups as group}
						<div class="mb-4">
							<h3 class="mb-2 text-sm font-semibold" style="color: var(--text-secondary);">
								{group.label}
							</h3>
							<div class="grid grid-cols-2 gap-4 md:grid-cols-4">
								{#each group.entries as [key, val]}
									<MetricCard
										title={getMetricLabel(key)}
										value={fmtMetric(key, val)}
										subtitle={getMetricDesc(key)}
										color={metricColor(key)}
									/>
								{/each}
							</div>
						</div>
					{/each}
				{:else}
					<p style="color: var(--text-secondary);">該回測無指標資料</p>
				{/if}
			</section>
		{/if}
	{/if}
</div>
