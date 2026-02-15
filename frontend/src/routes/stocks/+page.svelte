<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import DataTable from '$lib/components/DataTable.svelte';
	import { listStocks, getTopPicks, type StockItem, type TopPick } from '$lib/api';

	let stocks: StockItem[] = $state([]);
	let picks: TopPick[] = $state([]);
	let search = $state('');
	let loading = $state(true);
	let error = $state('');

	onMount(async () => {
		try {
			const [stockRes, pickRes] = await Promise.all([listStocks(), getTopPicks(50)]);
			stocks = stockRes.stocks;
			picks = pickRes.picks;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	});

	// Merge stock list with prediction data
	let mergedRows = $derived.by(() => {
		const pickMap = new Map(picks.map((p) => [p.stock_id, p]));
		let rows = stocks.map((s) => {
			const pick = pickMap.get(s.stock_id);
			return {
				stock_id: s.stock_id,
				name: s.name ?? '-',
				predicted_return: pick?.predicted_return ?? null,
				score: pick?.score ?? null
			};
		});
		if (search.trim()) {
			const q = search.trim().toLowerCase();
			rows = rows.filter(
				(r) =>
					r.stock_id.toLowerCase().includes(q) ||
					r.name.toLowerCase().includes(q)
			);
		}
		return rows;
	});

	const columns = [
		{ key: 'stock_id', label: '股票代码' },
		{ key: 'name', label: '名称' },
		{
			key: 'predicted_return',
			label: '预测报酬率',
			align: 'right' as const,
			format: (v: unknown) => (v != null ? ((v as number) * 100).toFixed(2) + '%' : '-')
		},
		{
			key: 'score',
			label: '评分',
			align: 'right' as const,
			format: (v: unknown) => (v != null ? (v as number).toFixed(4) : '-')
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
	<h1 class="text-2xl font-bold text-gray-800">股票列表</h1>

	<!-- Search -->
	<input
		type="text"
		placeholder="搜索股票代码或名称..."
		bind:value={search}
		class="w-full max-w-md rounded-lg border border-gray-300 px-4 py-2 text-sm focus:border-blue-500 focus:outline-none"
	/>

	{#if loading}
		<p class="text-gray-500">载入中...</p>
	{:else if error}
		<p class="text-red-500">错误: {error}</p>
	{:else}
		<p class="text-sm text-gray-500">共 {mergedRows.length} 档股票</p>
		<DataTable
			{columns}
			rows={mergedRows as unknown as Record<string, unknown>[]}
			onRowClick={handleRowClick}
		/>
	{/if}
</div>
