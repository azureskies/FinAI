<script lang="ts">
	import '../app.css';
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
	}

	let { children }: Props = $props();

	const navItems = [
		{ href: '/', label: '首页' },
		{ href: '/stocks', label: '股票列表' },
		{ href: '/backtest', label: '回测结果' },
		{ href: '/models', label: '模型管理' }
	];
</script>

<div class="flex min-h-screen flex-col bg-gray-50">
	<!-- Navigation -->
	<nav class="border-b border-gray-200 bg-white shadow-sm">
		<div class="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
			<a href="/" class="text-xl font-bold text-blue-600">FinAI</a>
			<div class="flex gap-6">
				{#each navItems as item}
					<a
						href={item.href}
						class="text-sm font-medium text-gray-600 transition-colors hover:text-blue-600"
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
	<footer class="border-t border-gray-200 bg-white py-4 text-center text-xs text-gray-400">
		FinAI — 台湾股票 AI 分析平台
	</footer>
</div>
