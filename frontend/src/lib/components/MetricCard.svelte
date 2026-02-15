<script lang="ts">
	interface Props {
		title: string;
		value: string | number;
		subtitle?: string;
		color?: 'blue' | 'green' | 'red' | 'yellow' | 'gray';
	}

	let { title, value, subtitle = '', color = 'blue' }: Props = $props();

	const colorMap: Record<string, string> = {
		blue: 'border-l-blue-500',
		green: 'border-l-green-500',
		red: 'border-l-red-500',
		yellow: 'border-l-yellow-500',
		gray: 'border-l-gray-500'
	};
</script>

<div
	class="rounded-lg border-l-4 p-5 {colorMap[color]}"
	style="background-color: var(--bg-secondary); border-color: var(--border-color); border-right: 1px solid var(--border-color); border-top: 1px solid var(--border-color); border-bottom: 1px solid var(--border-color);"
>
	<p class="text-sm" style="color: var(--text-secondary);">{title}</p>
	<p class="mt-1 text-2xl font-bold" style="color: var(--text-primary);">{value}</p>
	{#if subtitle}
		<p class="mt-1 text-xs" style="color: var(--text-secondary);">{subtitle}</p>
	{/if}
</div>
