<script lang="ts">
	import '../app.css';
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
	}

	let { children }: Props = $props();

	const navItems = [
		{ href: '/', label: '儀表板' },
		{ href: '/stocks', label: '股票列表' },
		{ href: '/backtest', label: '回測結果' },
		{ href: '/models', label: '模型管理' }
	];
</script>

<div class="flex min-h-screen flex-col" style="background-color: var(--bg-primary);">
	<!-- Navigation -->
	<nav class="border-b shadow-sm" style="background-color: var(--bg-secondary); border-color: var(--border-color);">
		<div class="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
			<a href="/" class="text-xl font-bold" style="color: var(--color-accent);">FinAI</a>
			<div class="flex gap-6">
				{#each navItems as item}
					<a
						href={item.href}
						class="text-sm font-medium transition-colors hover:text-[var(--color-accent)]"
						style="color: var(--text-secondary);"
					>
						{item.label}
					</a>
				{/each}
			</div>
		</div>
	</nav>

	<!-- Main content -->
	<main class="mx-auto w-full max-w-7xl flex-1 px-4 py-6">
		{@render children()}
	</main>

	<!-- Footer -->
	<footer class="border-t py-4 text-center text-xs" style="border-color: var(--border-color); color: var(--text-secondary); background-color: var(--bg-secondary);">
		FinAI — 台灣股票 AI 分析平台
	</footer>
</div>
