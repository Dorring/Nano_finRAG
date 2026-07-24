"""Tests for Phase 7 start/stop/restart shell scripts.

Verify all 8 deployment shell scripts exist, use strict mode
(``set -euo pipefail``), pass ``bash -n`` syntax validation, and
that start/stop/restart orchestration calls sub-scripts in the
correct order.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "deploy"

EXPECTED_SCRIPTS = [
    "load_env.sh",
    "start_model.sh",
    "start_backend.sh",
    "start_frontend.sh",
    "start_all.sh",
    "stop_all.sh",
    "restart_all.sh",
    "status.sh",
]


@pytest.fixture
def scripts_dir() -> Path:
    if not SCRIPTS_DIR.is_dir():
        pytest.skip(f"Scripts directory not found: {SCRIPTS_DIR}")
    return SCRIPTS_DIR


def _read_script(name: str) -> str:
    path = SCRIPTS_DIR / name
    if not path.is_file():
        pytest.skip(f"Script not found: {path}")
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("script_name", EXPECTED_SCRIPTS)
def test_script_exists(scripts_dir, script_name):
    # Verify each of the 8 shell scripts exists
    path = scripts_dir / script_name
    assert path.is_file(), f"Missing script: {script_name}"


@pytest.mark.parametrize("script_name", EXPECTED_SCRIPTS)
def test_script_strict_mode(script_name):
    # Verify each script starts with `set -euo pipefail`
    text = _read_script(script_name)
    lines = text.splitlines()
    # Check the first 10 lines for the strict-mode directive
    head = "\n".join(lines[:10])
    assert "set -euo pipefail" in head, \
        f"{script_name} does not start with 'set -euo pipefail' in its first 10 lines"


@pytest.mark.parametrize("script_name", EXPECTED_SCRIPTS)
def test_script_syntax_valid(script_name):
    # Verify each script passes `bash -n` syntax check
    if not shutil.which("bash"):
        pytest.skip("bash not available on PATH")
    path = SCRIPTS_DIR / script_name
    if not path.is_file():
        pytest.skip(f"Script not found: {path}")
    result = subprocess.run(
        ["bash", "-n", str(path)],
        capture_output=True, text=True, errors="replace", timeout=30,
    )
    # On Windows, bash may be WSL which can't translate drive paths (Y:\).
    # Skip if the failure is due to environment, not syntax.
    stderr = result.stderr or ""
    stdout = result.stdout or ""
    combined = stderr + stdout
    if result.returncode != 0 and (
        "WSL" in combined
        or "Failed to translate" in combined
        or "execvpe" in combined
        or not combined  # stderr empty due to encoding crash
    ):
        pytest.skip(f"bash environment issue for {script_name}")
    assert result.returncode == 0, \
        f"bash -n failed for {script_name}: {result.stderr}"


def test_start_all_calls_subscripts_in_order():
    # Verify start_all.sh calls start_model.sh, start_backend.sh, start_frontend.sh in order
    text = _read_script("start_all.sh")
    model_pos = text.find("start_model.sh")
    backend_pos = text.find("start_backend.sh")
    frontend_pos = text.find("start_frontend.sh")
    assert model_pos != -1, "start_all.sh does not call start_model.sh"
    assert backend_pos != -1, "start_all.sh does not call start_backend.sh"
    assert frontend_pos != -1, "start_all.sh does not call start_frontend.sh"
    assert model_pos < backend_pos < frontend_pos, \
        "start_all.sh must call start_model.sh before start_backend.sh before start_frontend.sh"


def test_stop_all_stops_in_reverse_order():
    # Verify stop_all.sh stops frontend, backend, model (reverse order)
    text = _read_script("stop_all.sh")
    # stop_all.sh calls stop_service with "Frontend", "Backend", "Model" labels
    frontend_pos = text.find("Frontend")
    backend_pos = text.find("Backend")
    model_pos = text.find("Model")
    assert frontend_pos != -1, "stop_all.sh does not reference Frontend"
    assert backend_pos != -1, "stop_all.sh does not reference Backend"
    assert model_pos != -1, "stop_all.sh does not reference Model"
    assert frontend_pos < backend_pos < model_pos, \
        "stop_all.sh must stop Frontend before Backend before Model"


def test_restart_all_calls_stop_then_start():
    # Verify restart_all.sh calls stop_all.sh then start_all.sh
    text = _read_script("restart_all.sh")
    stop_pos = text.find("stop_all.sh")
    start_pos = text.find("start_all.sh")
    assert stop_pos != -1, "restart_all.sh does not call stop_all.sh"
    assert start_pos != -1, "restart_all.sh does not call start_all.sh"
    assert stop_pos < start_pos, \
        "restart_all.sh must call stop_all.sh before start_all.sh"
