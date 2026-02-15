<script lang="ts">
	interface Props {
		score: number;
		label?: string;
		maxScore?: number;
	}

	let { score, label = '', maxScore = 100 }: Props = $props();

	let pct = $derived(Math.min(100, Math.max(0, (score / maxScore) * 100)));
	let color = $derived(
		pct >= 75 ? 'bg-red-500' : pct >= 50 ? 'bg-yellow-500' : pct >= 25 ? 'bg-blue-500' : 'bg-gray-500'
	);
</script>

<div class="flex items-center gap-2">
	{#if label}
		<span class="w-16 text-xs text-[var(--text-secondary)] shrink-0">{label}</span>
	{/if}
	<div class="flex-1 h-2 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
		<div
			class="h-full rounded-full transition-all duration-500 {color}"
			style="width: {pct}%"
		></div>
	</div>
	<span class="w-8 text-right text-xs font-mono text-[var(--text-primary)]">{score.toFixed(0)}</span>
</div>
