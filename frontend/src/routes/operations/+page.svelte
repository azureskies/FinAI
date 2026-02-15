<script lang="ts">
	import { onMount } from 'svelte';
	import {
		checkHealth,
		getDashboardSummary,
		runPipeline,
		getPipelineRuns,
		getPipelineStatus,
		type DashboardSummary,
		type PipelineRunItem,
		type PipelineTaskStatus
	} from '$lib/api';

	// --- System status ---
	let healthStatus = $state('checking');
	let summary: DashboardSummary | null = $state(null);

	// --- Pipeline ---
	let pipelineMode = $state<'incremental' | 'full'>('incremental');
	let pipelineLoading = $state(false);
	let pipelineMessage = $state('');
	let pipelineError = $state('');
	let activeTasks = $state<PipelineTaskStatus[]>([]);

	// --- Pipeline history ---
	let pipelineRuns = $state<PipelineRunItem[]>([]);
	let historyLoading = $state(true);

	onMount(async () => {
		// Fetch health, summary, and pipeline history in parallel
		const [healthRes, summaryRes, runsRes] = await Promise.allSettled([
			checkHealth(),
			getDashboardSummary(),
			getPipelineRuns(10)
		]);

		healthStatus =
			healthRes.status === 'fulfilled' && healthRes.value.status === 'ok'
				? 'ok'
				: 'error';

		if (summaryRes.status === 'fulfilled') {
			summary = summaryRes.value;
		}

		if (runsRes.status === 'fulfilled') {
			pipelineRuns = runsRes.value.runs;
		}
		historyLoading = false;
	});

	async function handleRunPipeline() {
		pipelineLoading = true;
		pipelineMessage = '';
		pipelineError = '';
		try {
			const res = await runPipeline({ mode: pipelineMode });
			pipelineMessage = res.message;
			pollTaskStatus();
		} catch (e) {
			pipelineError = e instanceof Error ? e.message : String(e);
		} finally {
			pipelineLoading = false;
		}
	}

	async function pollTaskStatus() {
		// Poll pipeline task status every 3 seconds until no running tasks
		const poll = async () => {
			try {
				const res = await getPipelineStatus();
				activeTasks = res.tasks;
				const hasRunning = res.tasks.some(
					(t) => t.status === 'running' || t.status === 'pending'
				);
				if (hasRunning) {
					setTimeout(poll, 3000);
				} else {
					// Refresh pipeline history
					const runsRes = await getPipelineRuns(10);
					pipelineRuns = runsRes.runs;
				}
			} catch {
				// Silently stop polling on error
			}
		};
		poll();
	}

	function statusBadgeClass(status: string): string {
		switch (status) {
			case 'success':
				return 'bg-green-900/50 text-green-400';
			case 'failed':
				return 'bg-red-900/50 text-red-400';
			case 'running':
				return 'bg-blue-900/50 text-blue-400';
			default:
				return 'bg-gray-700/50 text-gray-400';
		}
	}
</script>

<svelte:head>
	<title>FinAI - 系統操作</title>
</svelte:head>

<div class="space-y-8">
	<h1 class="text-2xl font-bold" style="color: var(--text-primary);">系統操作</h1>

	<!-- Section 1: System status -->
	<section
		class="rounded-lg border p-5"
		style="background-color: var(--bg-secondary); border-color: var(--border-color);"
	>
		<h2 class="mb-4 text-lg font-semibold" style="color: var(--text-primary);">系統狀態</h2>
		<div class="grid grid-cols-2 gap-4 md:grid-cols-4">
			<div class="rounded-md border p-3" style="border-color: var(--border-color);">
				<p class="text-xs" style="color: var(--text-secondary);">後端狀態</p>
				<p class="mt-1 text-sm font-semibold" style="color: {healthStatus === 'ok' ? 'var(--color-success)' : healthStatus === 'checking' ? 'var(--color-warning)' : 'var(--color-danger)'};">
					{healthStatus === 'ok' ? '正常' : healthStatus === 'checking' ? '檢查中...' : '無法連線'}
				</p>
			</div>
			<div class="rounded-md border p-3" style="border-color: var(--border-color);">
				<p class="text-xs" style="color: var(--text-secondary);">股票數量</p>
				<p class="mt-1 text-sm font-semibold" style="color: var(--text-primary);">
					{summary?.stocks_count ?? '-'}
				</p>
			</div>
			<div class="rounded-md border p-3" style="border-color: var(--border-color);">
				<p class="text-xs" style="color: var(--text-secondary);">最新預測日</p>
				<p class="mt-1 text-sm font-semibold" style="color: var(--text-primary);">
					{summary?.latest_prediction_date ?? '-'}
				</p>
			</div>
			<div class="rounded-md border p-3" style="border-color: var(--border-color);">
				<p class="text-xs" style="color: var(--text-secondary);">回測次數</p>
				<p class="mt-1 text-sm font-semibold" style="color: var(--text-primary);">
					{summary?.backtest_runs ?? '-'}
				</p>
			</div>
		</div>
	</section>

	<!-- Section 2: Daily Update Pipeline -->
	<section
		class="rounded-lg border p-5"
		style="background-color: var(--bg-secondary); border-color: var(--border-color);"
	>
		<h2 class="mb-4 text-lg font-semibold" style="color: var(--text-primary);">每日更新 Pipeline</h2>
		<div class="flex flex-wrap items-end gap-4">
			<div>
				<label class="mb-1 block text-xs" style="color: var(--text-secondary);">更新模式</label>
				<select
					bind:value={pipelineMode}
					class="rounded border px-3 py-2 text-sm"
					style="background-color: var(--bg-tertiary); border-color: var(--border-color); color: var(--text-primary);"
				>
					<option value="incremental">增量更新</option>
					<option value="full">全量更新</option>
				</select>
			</div>
			<button
				onclick={handleRunPipeline}
				disabled={pipelineLoading}
				class="rounded px-4 py-2 text-sm font-medium text-white transition-colors disabled:opacity-50"
				style="background-color: var(--color-accent);"
			>
				{#if pipelineLoading}
					<span class="inline-flex items-center gap-2">
						<svg class="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
							<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
							<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
						</svg>
						執行中...
					</span>
				{:else}
					執行更新
				{/if}
			</button>
		</div>

		{#if pipelineMessage}
			<p class="mt-3 text-sm" style="color: var(--color-success);">{pipelineMessage}</p>
		{/if}
		{#if pipelineError}
			<p class="mt-3 text-sm" style="color: var(--color-danger);">{pipelineError}</p>
		{/if}

		{#if activeTasks.length > 0}
			<div class="mt-4 space-y-2">
				{#each activeTasks as task}
					<div
						class="flex items-center gap-3 rounded border px-3 py-2 text-sm"
						style="border-color: var(--border-color);"
					>
						<span class="rounded px-2 py-0.5 text-xs font-medium {statusBadgeClass(task.status)}">
							{task.status}
						</span>
						<span style="color: var(--text-secondary);">{task.progress ?? ''}</span>
						{#if task.error}
							<span style="color: var(--color-danger);">{task.error}</span>
						{/if}
					</div>
				{/each}
			</div>
		{/if}
	</section>

	<!-- Section 3: Pipeline history -->
	<section
		class="rounded-lg border p-5"
		style="background-color: var(--bg-secondary); border-color: var(--border-color);"
	>
		<h2 class="mb-4 text-lg font-semibold" style="color: var(--text-primary);">Pipeline 執行歷史</h2>

		{#if historyLoading}
			<p style="color: var(--text-secondary);">載入中...</p>
		{:else if pipelineRuns.length === 0}
			<p style="color: var(--text-secondary);">暫無執行紀錄</p>
		{:else}
			<div class="overflow-x-auto">
				<table class="w-full text-sm" style="color: var(--text-primary);">
					<thead>
						<tr style="border-color: var(--border-color);" class="border-b">
							<th class="px-3 py-2 text-left text-xs font-medium" style="color: var(--text-secondary);">Pipeline</th>
							<th class="px-3 py-2 text-left text-xs font-medium" style="color: var(--text-secondary);">開始時間</th>
							<th class="px-3 py-2 text-left text-xs font-medium" style="color: var(--text-secondary);">結束時間</th>
							<th class="px-3 py-2 text-left text-xs font-medium" style="color: var(--text-secondary);">狀態</th>
							<th class="px-3 py-2 text-left text-xs font-medium" style="color: var(--text-secondary);">錯誤</th>
						</tr>
					</thead>
					<tbody>
						{#each pipelineRuns as run}
							<tr class="border-b" style="border-color: var(--border-color);">
								<td class="px-3 py-2">{run.pipeline_name}</td>
								<td class="px-3 py-2" style="color: var(--text-secondary);">{run.start_time ?? '-'}</td>
								<td class="px-3 py-2" style="color: var(--text-secondary);">{run.end_time ?? '-'}</td>
								<td class="px-3 py-2">
									<span class="rounded px-2 py-0.5 text-xs font-medium {statusBadgeClass(run.status)}">
										{run.status}
									</span>
								</td>
								<td class="px-3 py-2 text-xs" style="color: var(--color-danger);">{run.error ?? ''}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</section>
</div>
