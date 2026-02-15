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
		{ key: 'model_type', label: '模型类型' },
		{ key: 'description', label: '说明' },
		{
			key: 'is_active',
			label: '状态',
			format: (v: unknown) => (v ? '启用中' : '未启用')
		},
		{ key: 'created_at', label: '建立时间' }
	];

	let selected: ModelVersion | null = $state(null);
	let metricEntries = $derived(
		selected?.metrics ? Object.entries(selected.metrics) : []
	);

	function handleRowClick(row: Record<string, unknown>) {
		selected = models.find((m) => m.id === row.id) ?? null;
	}
</script>

<svelte:head>
	<title>FinAI - 模型管理</title>
</svelte:head>

<div class="space-y-6">
	<h1 class="text-2xl font-bold text-gray-800">模型管理</h1>

	{#if loading}
		<p class="text-gray-500">载入中...</p>
	{:else if error}
		<p class="text-red-500">错误: {error}</p>
	{:else}
		<!-- Summary -->
		<div class="grid grid-cols-2 gap-4 md:grid-cols-3">
			<MetricCard title="模型总数" value={models.length} color="blue" />
			<MetricCard title="启用中" value={activeCount} color="green" />
			<MetricCard
				title="模型类型"
				value={new Set(models.map((m) => m.model_type).filter(Boolean)).size}
				color="gray"
			/>
		</div>

		<!-- Model list -->
		<section>
			<h2 class="mb-3 text-lg font-semibold text-gray-700">模型版本列表</h2>
			<DataTable
				{columns}
				rows={models as unknown as Record<string, unknown>[]}
				emptyText="暂无模型版本"
				onRowClick={handleRowClick}
			/>
		</section>

		<!-- Selected model metrics -->
		{#if selected}
			<section>
				<h2 class="mb-3 text-lg font-semibold text-gray-700">
					模型效能 — {selected.model_type ?? selected.id}
					{#if selected.is_active}
						<span class="ml-2 rounded bg-green-100 px-2 py-0.5 text-xs text-green-700">启用中</span>
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
					<p class="text-gray-400">该模型无效能指标</p>
				{/if}
			</section>
		{/if}
	{/if}
</div>
