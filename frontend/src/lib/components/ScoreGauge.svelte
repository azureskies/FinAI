<script lang="ts">
	interface Props {
		score: number;
		size?: number;
		label?: string;
	}

	let { score, size = 120, label = '' }: Props = $props();

	const radius = 45;
	const circumference = 2 * Math.PI * radius;
	let offset = $derived(circumference - (score / 100) * circumference);

	let color = $derived(
		score >= 75 ? '#f85149' : score >= 50 ? '#d29922' : score >= 25 ? '#58a6ff' : '#8b949e'
	);
</script>

<div class="flex flex-col items-center gap-1">
	<svg width={size} height={size} viewBox="0 0 100 100">
		<circle cx="50" cy="50" r={radius} fill="none" stroke="#30363d" stroke-width="8" />
		<circle
			cx="50"
			cy="50"
			r={radius}
			fill="none"
			stroke={color}
			stroke-width="8"
			stroke-linecap="round"
			stroke-dasharray={circumference}
			stroke-dashoffset={offset}
			transform="rotate(-90 50 50)"
			class="transition-all duration-700"
		/>
		<text
			x="50"
			y="48"
			text-anchor="middle"
			dominant-baseline="middle"
			fill={color}
			font-size="22"
			font-weight="bold"
			font-family="monospace"
		>
			{score.toFixed(0)}
		</text>
		<text
			x="50"
			y="65"
			text-anchor="middle"
			fill="#8b949e"
			font-size="9"
		>
			/ 100
		</text>
	</svg>
	{#if label}
		<span class="text-xs text-[var(--text-secondary)]">{label}</span>
	{/if}
</div>
