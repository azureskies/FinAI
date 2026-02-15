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
		gray: 'border-l-gray-400'
	};
</script>

<div class="rounded-lg border border-gray-200 bg-white p-5 shadow-sm border-l-4 {colorMap[color]}">
	<p class="text-sm text-gray-500">{title}</p>
	<p class="mt-1 text-2xl font-bold text-gray-900">{value}</p>
	{#if subtitle}
		<p class="mt-1 text-xs text-gray-400">{subtitle}</p>
	{/if}
</div>
