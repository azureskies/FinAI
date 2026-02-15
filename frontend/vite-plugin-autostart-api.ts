/**
 * Vite plugin that auto-starts the FastAPI backend when the dev server starts.
 *
 * - Checks if the backend is already running by fetching /api/health
 * - If not, spawns `python -m uvicorn api.main:app` as a child process
 * - Waits up to 15 seconds for the backend to become ready
 * - Kills the child process when the dev server shuts down
 */

import type { Plugin, ViteDevServer } from 'vite';
import { spawn, type ChildProcess } from 'child_process';

const API_URL = 'http://localhost:8000/api/health';
const MAX_WAIT_MS = 15_000;
const POLL_INTERVAL_MS = 500;

async function isBackendRunning(): Promise<boolean> {
	try {
		const res = await fetch(API_URL, { signal: AbortSignal.timeout(2000) });
		return res.ok;
	} catch {
		return false;
	}
}

async function waitForBackend(timeoutMs: number): Promise<boolean> {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		if (await isBackendRunning()) return true;
		await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
	}
	return false;
}

export default function autostartApi(): Plugin {
	let child: ChildProcess | null = null;

	return {
		name: 'autostart-api',
		configureServer(server: ViteDevServer) {
			const startup = async () => {
				if (await isBackendRunning()) {
					console.log('[autostart-api] Backend already running.');
					return;
				}

				console.log('[autostart-api] Backend not detected, starting uvicorn...');
				child = spawn(
					'python',
					['-m', 'uvicorn', 'api.main:app', '--host', '0.0.0.0', '--port', '8000'],
					{
						cwd: process.cwd().replace(/\/frontend$/, ''),
						stdio: 'pipe',
						detached: false
					}
				);

				child.stdout?.on('data', (data: Buffer) => {
					console.log(`[api] ${data.toString().trimEnd()}`);
				});
				child.stderr?.on('data', (data: Buffer) => {
					console.log(`[api] ${data.toString().trimEnd()}`);
				});
				child.on('error', (err) => {
					console.error('[autostart-api] Failed to start backend:', err.message);
				});
				child.on('exit', (code) => {
					if (code !== null && code !== 0) {
						console.error(`[autostart-api] Backend exited with code ${code}`);
					}
					child = null;
				});

				const ready = await waitForBackend(MAX_WAIT_MS);
				if (ready) {
					console.log('[autostart-api] Backend is ready.');
				} else {
					console.warn(
						'[autostart-api] Backend did not become ready within timeout. Continuing anyway.'
					);
				}
			};

			// Run startup and attach cleanup
			startup();

			const cleanup = () => {
				if (child && !child.killed) {
					console.log('[autostart-api] Stopping backend...');
					child.kill('SIGTERM');
					child = null;
				}
			};

			server.httpServer?.on('close', cleanup);
			process.on('exit', cleanup);
			process.on('SIGINT', () => {
				cleanup();
				process.exit(0);
			});
			process.on('SIGTERM', () => {
				cleanup();
				process.exit(0);
			});
		}
	};
}
