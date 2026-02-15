<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		Chart as ChartJS,
		RadarController,
		RadialLinearScale,
		PointElement,
		LineElement,
		Filler,
		Tooltip,
		Legend
	} from 'chart.js';

	ChartJS.register(RadarController, RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

	interface Props {
		momentum: number;
		trend: number;
		volatility: number;
		volume: number;
		ai: number;
		size?: number;
	}

	let { momentum, trend, volatility, volume, ai, size = 250 }: Props = $props();

	let canvas: HTMLCanvasElement;
	let chart: ChartJS | null = null;

	const labels = ['動能', '趨勢', '波動', '成交量', 'AI預測'];

	function createChart() {
		if (chart) chart.destroy();
		chart = new ChartJS(canvas, {
			type: 'radar',
			data: {
				labels,
				datasets: [
					{
						label: '因子分數',
						data: [momentum, trend, volatility, volume, ai],
						backgroundColor: 'rgba(88, 166, 255, 0.15)',
						borderColor: '#58a6ff',
						borderWidth: 2,
						pointBackgroundColor: '#58a6ff',
						pointBorderColor: '#0d1117',
						pointBorderWidth: 2,
						pointRadius: 4
					}
				]
			},
			options: {
				responsive: true,
				maintainAspectRatio: false,
				plugins: {
					legend: { display: false }
				},
				scales: {
					r: {
						beginAtZero: true,
						max: 100,
						ticks: {
							stepSize: 25,
							color: '#8b949e',
							backdropColor: 'transparent',
							font: { size: 10 }
						},
						grid: { color: '#30363d' },
						angleLines: { color: '#30363d' },
						pointLabels: {
							color: '#e6edf3',
							font: { size: 12 }
						}
					}
				}
			}
		});
	}

	onMount(() => createChart());
	onDestroy(() => chart?.destroy());

	$effect(() => {
		if (chart) {
			chart.data.datasets[0].data = [momentum, trend, volatility, volume, ai];
			chart.update();
		}
	});
</script>

<div style="height: {size}px; width: {size}px;">
	<canvas bind:this={canvas}></canvas>
</div>
