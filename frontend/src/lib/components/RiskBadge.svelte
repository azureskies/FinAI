<script lang="ts">
	interface Props {
		level: string;
	}

	let { level }: Props = $props();

	const colorMap: Record<string, string> = {
		'低風險': 'bg-green-900/40 text-green-400 border-green-700/50',
		'中風險': 'bg-yellow-900/40 text-yellow-400 border-yellow-700/50',
		'高風險': 'bg-red-900/40 text-red-400 border-red-700/50'
	};

	let classes = $derived(colorMap[level] ?? 'bg-gray-800 text-gray-400 border-gray-600');
</script>

<span class="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium {classes}">
	{level}
</span>
