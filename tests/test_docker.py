"""Tests for Docker deployment configuration files."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _read(rel_path: str) -> str:
    """Read a file relative to the project root."""
    return (ROOT / rel_path).read_text()


# ------------------------------------------------------------------ #
#  Dockerfile (backend)
# ------------------------------------------------------------------ #

class TestBackendDockerfile:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = _read("Dockerfile")

    def test_file_exists(self):
        assert (ROOT / "Dockerfile").is_file()

    def test_has_from_instruction(self):
        assert "FROM" in self.content

    def test_base_image_is_python(self):
        assert "python:3.11-slim" in self.content

    def test_installs_libomp(self):
        assert "libomp" in self.content

    def test_exposes_port(self):
        assert "EXPOSE 8000" in self.content

    def test_has_cmd(self):
        assert "CMD" in self.content
        assert "uvicorn" in self.content

    def test_copies_source_dirs(self):
        for dirname in ("api/", "models/", "data/"):
            assert dirname in self.content


# ------------------------------------------------------------------ #
#  Dockerfile (frontend)
# ------------------------------------------------------------------ #

class TestFrontendDockerfile:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = _read("frontend/Dockerfile")

    def test_file_exists(self):
        assert (ROOT / "frontend" / "Dockerfile").is_file()

    def test_has_build_stage(self):
        assert "AS build" in self.content

    def test_uses_node_image(self):
        assert "node:" in self.content

    def test_uses_nginx_image(self):
        assert "nginx:" in self.content

    def test_runs_npm_build(self):
        assert "npm run build" in self.content


# ------------------------------------------------------------------ #
#  docker-compose.yml
# ------------------------------------------------------------------ #

class TestDockerCompose:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = _read("docker-compose.yml")

    def test_file_exists(self):
        assert (ROOT / "docker-compose.yml").is_file()

    def test_defines_backend_service(self):
        assert "backend:" in self.content

    def test_defines_frontend_service(self):
        assert "frontend:" in self.content

    def test_backend_port(self):
        assert "8000:8000" in self.content

    def test_frontend_port(self):
        assert "3000:3000" in self.content

    def test_healthcheck(self):
        assert "healthcheck" in self.content

    def test_restart_policy(self):
        assert "restart:" in self.content


# ------------------------------------------------------------------ #
#  nginx.conf
# ------------------------------------------------------------------ #

class TestNginxConf:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = _read("nginx.conf")

    def test_file_exists(self):
        assert (ROOT / "nginx.conf").is_file()

    def test_listens_on_3000(self):
        assert "listen 3000" in self.content

    def test_proxies_api(self):
        assert "proxy_pass" in self.content
        assert "backend:8000" in self.content


# ------------------------------------------------------------------ #
#  .dockerignore
# ------------------------------------------------------------------ #

class TestDockerignore:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = _read(".dockerignore")
        self.lines = {line.strip() for line in self.content.splitlines() if line.strip()}

    def test_file_exists(self):
        assert (ROOT / ".dockerignore").is_file()

    def test_excludes_venv(self):
        assert ".venv" in self.lines

    def test_excludes_pycache(self):
        assert "__pycache__" in self.lines

    def test_excludes_git(self):
        assert ".git" in self.lines

    def test_excludes_node_modules(self):
        assert "node_modules" in self.lines

    def test_excludes_env(self):
        assert ".env" in self.lines

    def test_excludes_pkl_files(self):
        assert "*.pkl" in self.lines
