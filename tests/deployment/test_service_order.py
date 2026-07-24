"""Tests verifying Phase 7 service startup and shutdown ordering.

Verify start_all.sh starts model -> backend -> frontend, stop_all.sh
stops in reverse (frontend -> backend -> model), and that start_backend.sh
checks model availability before starting, while start_frontend.sh checks
backend availability before starting.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "deploy"


def _read_script(name: str) -> str:
    path = SCRIPTS_DIR / name
    if not path.is_file():
        pytest.skip(f"Script not found: {path}")
    return path.read_text(encoding="utf-8")


def test_start_order_model_before_backend_before_frontend():
    # Parse start_all.sh and verify start_model.sh is called before
    # start_backend.sh which is called before start_frontend.sh
    text = _read_script("start_all.sh")
    model_pos = text.find("start_model.sh")
    backend_pos = text.find("start_backend.sh")
    frontend_pos = text.find("start_frontend.sh")
    assert model_pos != -1, "start_all.sh does not call start_model.sh"
    assert backend_pos != -1, "start_all.sh does not call start_backend.sh"
    assert frontend_pos != -1, "start_all.sh does not call start_frontend.sh"
    assert model_pos < backend_pos, \
        "start_all.sh must call start_model.sh before start_backend.sh"
    assert backend_pos < frontend_pos, \
        "start_all.sh must call start_backend.sh before start_frontend.sh"


def test_stop_order_frontend_before_backend_before_model():
    # Parse stop_all.sh and verify it stops frontend before backend before model
    text = _read_script("stop_all.sh")
    # stop_all.sh calls stop_service with labels "Frontend", "Backend", "Model"
    frontend_pos = text.find("Frontend")
    backend_pos = text.find("Backend")
    model_pos = text.find("Model")
    assert frontend_pos != -1, "stop_all.sh does not reference Frontend"
    assert backend_pos != -1, "stop_all.sh does not reference Backend"
    assert model_pos != -1, "stop_all.sh does not reference Model"
    assert frontend_pos < backend_pos, \
        "stop_all.sh must stop Frontend before Backend"
    assert backend_pos < model_pos, \
        "stop_all.sh must stop Backend before Model"


def test_start_backend_checks_model_availability():
    # Verify start_backend.sh checks model service availability before starting
    text = _read_script("start_backend.sh")
    # Look for model health check (curl to model /health endpoint)
    assert "MODEL_HOST" in text or "MODEL_PORT" in text, \
        "start_backend.sh does not reference model host/port"
    assert "curl" in text or "wait_for_http" in text, \
        "start_backend.sh does not perform an HTTP check before starting"
    assert "/health" in text, \
        "start_backend.sh does not check model /health endpoint"
    # Verify there is a pre-flight check that exits if model is unavailable
    assert "exit 1" in text or "write_status" in text, \
        "start_backend.sh does not exit on model unavailability"


def test_start_frontend_checks_backend_availability():
    # Verify start_frontend.sh checks backend availability before starting
    text = _read_script("start_frontend.sh")
    assert "BACKEND_HOST" in text or "BACKEND_PORT" in text, \
        "start_frontend.sh does not reference backend host/port"
    assert "curl" in text or "wait_for_http" in text, \
        "start_frontend.sh does not perform an HTTP check before starting"
    assert "/healthz" in text, \
        "start_frontend.sh does not check backend /healthz endpoint"
    assert "exit 1" in text or "write_status" in text, \
        "start_frontend.sh does not exit on backend unavailability"
