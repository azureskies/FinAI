<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import DataTable from '$lib/components/DataTable.svelte';
	import RiskBadge from '$lib/components/RiskBadge.svelte';
	import { listStocks, getTopPicks, type StockItem, type ScoredStock } from '$lib/api';

	let stocks: StockItem[] = $state([]);
	let picks: ScoredStock[] = $state([]);
	let search = $state('');
	let riskFilter = $state('all');
	let sortBy = $state('score');
	let loading = $state(true);
	let error = $state('');

	onMount(async () => {
		try {
			const [stockRes, pickRes] = await Promise.all([listStocks(), getTopPicks(5000)]);
			stocks = stockRes.stocks;
			picks = pickRes.picks;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	});

	// Merge stock list with score data
	let mergedRows = $derived.by(() => {
		const pickMap = new Map(picks.map((p) => [p.stock_id, p]));
		let rows = stocks.map((s) => {
			const pick = pickMap.get(s.stock_id);
			return {
				stock_id: s.stock_id,
				name: s.name ?? '-',
				composite_score: pick?.composite_score ?? null,
				predicted_return: pick?.predicted_return ?? null,
				risk_level: pick?.risk_level ?? null
			};
		});

		// Search filter
		if (search.trim()) {
			const q = search.trim().toLowerCase();
			rows = rows.filter(
				(r) =>
					r.stock_id.toLowerCase().includes(q) ||
					r.name.toLowerCase().includes(q)
			);
		}

		// Risk filter
		if (riskFilter !== 'all') {
			rows = rows.filter((r) => r.risk_level === riskFilter);
		}

		// Sort
		if (sortBy === 'score') {
			rows.sort((a, b) => (b.composite_score ?? 0) - (a.composite_score ?? 0));
		} else if (sortBy === 'return') {
			rows.sort((a, b) => (b.predicted_return ?? 0) - (a.predicted_return ?? 0));
		} else {
			rows.sort((a, b) => a.stock_id.localeCompare(b.stock_id));
		}

		return rows;
	});

	const columns = [
		{ key: 'stock_id', label: '股票代碼' },
		{ key: 'name', label: '名稱' },
		{
			key: 'composite_score',
			label: '綜合評分',
			align: 'right' as const,
			format: (v: unknown) => (v != null ? (v as number).toFixed(1) : '-')
		},
		{
			key: 'predicted_return',
			label: '預測報酬率',
			align: 'right' as const,
			format: (v: unknown) => (v != null ? ((v as number) * 100).toFixed(2) + '%' : '-')
		},
		{
			key: 'risk_level',
			label: '風險等級',
			format: (v: unknown) => (v != null ? String(v) : '-')
		}
	];

	function handleRowClick(row: Record<string, unknown>) {
		goto(`/stocks/${row.stock_id}`);
	}
</script>

<svelte:head>
	<title>FinAI - 股票列表</title>
</svelte:head>

<div class="space-y-4">
	<h1 class="text-2xl font-bold" style="color: var(--text-primary);">股票列表</h1>

	<!-- Search and filters -->
	<div class="flex flex-wrap items-center gap-3">
		<input
			type="text"
			placeholder="搜尋股票代碼或名稱..."
			bind:value={search}
			class="w-full max-w-md rounded-lg border px-4 py-2 text-sm focus:outline-none"
			style="background-color: var(--bg-secondary); border-color: var(--border-color); color: var(--text-primary);"
		/>
		<select
			bind:value={riskFilter}
			class="rounded-lg border px-3 py-2 text-sm"
			style="background-color: var(--bg-secondary); border-color: var(--border-color); color: var(--text-primary);"
		>
			<option value="all">全部風險等級</option>
			<option value="低風險">低風險</option>
			<option value="中風險">中風險</option>
			<option value="高風險">高風險</option>
		</select>
		<select
			bind:value={sortBy}
			class="rounded-lg border px-3 py-2 text-sm"
			style="background-color: var(--bg-secondary); border-color: var(--border-color); color: var(--text-primary);"
		>
			<option value="score">依評分排序</option>
			<option value="return">依報酬率排序</option>
			<option value="id">依代碼排序</option>
		</select>
	</div>

	{#if loading}
		<p style="color: var(--text-secondary);">載入中...</p>
	{:else if error}
		<p style="color: var(--color-danger);">錯誤: {error}</p>
	{:else}
		<p class="text-sm" style="color: var(--text-secondary);">共 {mergedRows.length} 檔股票</p>
		<DataTable
			{columns}
			rows={mergedRows as unknown as Record<string, unknown>[]}
			onRowClick={handleRowClick}
		/>
	{/if}
</div>
