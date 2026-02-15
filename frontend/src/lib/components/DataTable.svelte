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

	let { columns, rows, emptyText = '暫無資料', onRowClick }: Props = $props();

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

<div class="overflow-x-auto rounded-lg border" style="border-color: var(--border-color);">
	<table class="min-w-full divide-y" style="--tw-divide-opacity: 1; --tw-divide-color: var(--border-color);">
		<thead style="background-color: var(--bg-tertiary);">
			<tr>
				{#each columns as col}
					<th
						class="px-4 py-3 text-xs font-medium tracking-wider uppercase {alignClass[col.align ?? 'left']}"
						style="color: var(--text-secondary);"
					>
						{col.label}
					</th>
				{/each}
			</tr>
		</thead>
		<tbody class="divide-y" style="background-color: var(--bg-secondary); --tw-divide-color: var(--border-color);">
			{#if rows.length === 0}
				<tr>
					<td colspan={columns.length} class="px-4 py-8 text-center" style="color: var(--text-secondary);">
						{emptyText}
					</td>
				</tr>
			{:else}
				{#each rows as row}
					<tr
						class="{onRowClick ? 'cursor-pointer' : ''}"
						style="{onRowClick ? '--hover-bg: var(--bg-tertiary);' : ''}"
						onmouseenter={(e) => { if (onRowClick) (e.currentTarget as HTMLElement).style.backgroundColor = 'var(--bg-tertiary)'; }}
						onmouseleave={(e) => { if (onRowClick) (e.currentTarget as HTMLElement).style.backgroundColor = ''; }}
						onclick={() => onRowClick?.(row)}
					>
						{#each columns as col}
							<td class="whitespace-nowrap px-4 py-3 text-sm {alignClass[col.align ?? 'left']}" style="color: var(--text-primary);">
								{formatCell(col, row)}
							</td>
						{/each}
					</tr>
				{/each}
			{/if}
		</tbody>
	</table>
</div>
