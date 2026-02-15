<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Column {
		key: string;
		label: string;
		align?: 'left' | 'center' | 'right';
		format?: (value: unknown) => string;
	}

	interface Props {
		columns: Column[];
		rows: Record<string, unknown>[];
		emptyText?: string;
		onRowClick?: (row: Record<string, unknown>) => void;
	}

	let { columns, rows, emptyText = '暂无资料', onRowClick }: Props = $props();

	function formatCell(col: Column, row: Record<string, unknown>): string {
		const val = row[col.key];
		if (val == null) return '-';
		if (col.format) return col.format(val);
		return String(val);
	}

	const alignClass: Record<string, string> = {
		left: 'text-left',
		center: 'text-center',
		right: 'text-right'
	};
</script>

<div class="overflow-x-auto rounded-lg border border-gray-200 shadow-sm">
	<table class="min-w-full divide-y divide-gray-200">
		<thead class="bg-gray-50">
			<tr>
				{#each columns as col}
					<th
						class="px-4 py-3 text-xs font-medium tracking-wider text-gray-500 uppercase {alignClass[col.align ?? 'left']}"
					>
						{col.label}
					</th>
				{/each}
			</tr>
		</thead>
		<tbody class="divide-y divide-gray-100 bg-white">
			{#if rows.length === 0}
				<tr>
					<td colspan={columns.length} class="px-4 py-8 text-center text-gray-400">
						{emptyText}
					</td>
				</tr>
			{:else}
				{#each rows as row}
					<tr
						class="{onRowClick ? 'cursor-pointer hover:bg-gray-50' : ''}"
						onclick={() => onRowClick?.(row)}
					>
						{#each columns as col}
							<td class="whitespace-nowrap px-4 py-3 text-sm text-gray-700 {alignClass[col.align ?? 'left']}">
								{formatCell(col, row)}
							</td>
						{/each}
					</tr>
				{/each}
			{/if}
		</tbody>
	</table>
</div>
