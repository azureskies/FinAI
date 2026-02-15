<script lang="ts">
	import { onMount } from 'svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import MetricCard from '$lib/components/MetricCard.svelte';
	import { listModels, type ModelVersion } from '$lib/api';

	let models: ModelVersion[] = $state([]);
	let loading = $state(true);
	let error = $state('');

	onMount(async () => {
		try {
			const res = await listModels();
			models = res.models;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	});

	let activeCount = $derived(models.filter((m) => m.is_active).length);

	const columns = [
		{ key: 'model_type', label: '模型類型' },
		{ key: 'description', label: '說明' },
		{
			key: 'is_active',
			label: '狀態',
			format: (v: unknown) => (v ? '啟用中' : '未啟用')
		},
		{ key: 'created_at', label: '建立時間' }
	];

	let selected: ModelVersion | null = $state(null);
	let metricEntries = $derived.by(() => {
		const m = selected?.metrics;
		if (!m) return [] as [string, number][];
		return Object.entries(m) as [string, number][];
	});

	function handleRowClick(row: Record<string, unknown>) {
		selected = models.find((m) => m.id === row.id) ?? null;
	}
</script>

<svelte:head>
	<title>FinAI - 模型管理</title>
</svelte:head>

<div class="space-y-6">
	<h1 class="text-2xl font-bold" style="color: var(--text-primary);">模型管理</h1>

	{#if loading}
		<p style="color: var(--text-secondary);">載入中...</p>
	{:else if error}
		<p style="color: var(--color-danger);">錯誤: {error}</p>
	{:else}
		<!-- Summary -->
		<div class="grid grid-cols-2 gap-4 md:grid-cols-3">
			<MetricCard title="模型總數" value={models.length} color="blue" />
			<MetricCard title="啟用中" value={activeCount} color="green" />
			<MetricCard
				title="模型類型"
				value={new Set(models.map((m) => m.model_type).filter(Boolean)).size}
				color="gray"
			/>
		</div>

		<!-- Model list -->
		<section>
			<h2 class="mb-3 text-lg font-semibold" style="color: var(--text-primary);">模型版本列表</h2>
			<DataTable
				{columns}
				rows={models as unknown as Record<string, unknown>[]}
				emptyText="暫無模型版本"
				onRowClick={handleRowClick}
			/>
		</section>

		<!-- Selected model metrics -->
		{#if selected}
			<section>
				<h2 class="mb-3 text-lg font-semibold" style="color: var(--text-primary);">
					模型效能 — {selected.model_type ?? selected.id}
					{#if selected.is_active}
						<span class="ml-2 rounded px-2 py-0.5 text-xs font-medium" style="background-color: rgba(38,166,65,0.2); color: var(--color-success);">啟用中</span>
					{/if}
				</h2>

				{#if metricEntries.length > 0}
					<div class="grid grid-cols-2 gap-4 md:grid-cols-4">
						{#each metricEntries as [key, val]}
							<MetricCard
								title={key}
								value={typeof val === 'number' ? (val as number).toFixed(4) : String(val)}
								color="blue"
							/>
						{/each}
					</div>
				{:else}
					<p style="color: var(--text-secondary);">該模型無效能指標</p>
				{/if}
			</section>
		{/if}
	{/if}
</div>
