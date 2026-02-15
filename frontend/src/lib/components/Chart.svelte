<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		Chart as ChartJS,
		CategoryScale,
		LinearScale,
		PointElement,
		LineElement,
		LineController,
		BarElement,
		BarController,
		Title,
		Tooltip,
		Legend,
		Filler
	} from 'chart.js';

	ChartJS.register(
		CategoryScale,
		LinearScale,
		PointElement,
		LineElement,
		LineController,
		BarElement,
		BarController,
		Title,
		Tooltip,
		Legend,
		Filler
	);

	interface Dataset {
		label: string;
		data: number[];
		borderColor?: string;
		backgroundColor?: string;
		fill?: boolean;
		tension?: number;
	}

	interface Props {
		type?: 'line' | 'bar';
		labels: string[];
		datasets: Dataset[];
		title?: string;
		height?: number;
	}

	let { type = 'line', labels, datasets, title = '', height = 300 }: Props = $props();

	let canvas: HTMLCanvasElement;
	let chart: ChartJS | null = null;

	function createChart() {
		if (chart) chart.destroy();
		chart = new ChartJS(canvas, {
			type,
			data: { labels, datasets },
			options: {
				responsive: true,
				maintainAspectRatio: false,
				plugins: {
					title: {
						display: !!title,
						text: title,
						color: '#e6edf3'
					},
					legend: {
						position: 'top',
						labels: { color: '#8b949e' }
					}
				},
				scales: {
					x: {
						grid: { color: '#21262d' },
						ticks: { color: '#8b949e' }
					},
					y: {
						beginAtZero: false,
						grid: { color: '#21262d' },
						ticks: { color: '#8b949e' }
					}
				}
			}
		});
	}

	onMount(() => {
		createChart();
	});

	onDestroy(() => {
		chart?.destroy();
	});

	$effect(() => {
		// Re-render when data changes
		if (chart && labels && datasets) {
			chart.data.labels = labels;
			chart.data.datasets = datasets;
			chart.update();
		}
	});
</script>

<div style="height: {height}px;">
	<canvas bind:this={canvas}></canvas>
</div>
