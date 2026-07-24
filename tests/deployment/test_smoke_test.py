"""Tests for scripts/deploy/smoke_test.py.

Verify smoke_test.py exists, is importable, covers all 12 test cases,
writes a JSON report, and does not log full questions/answers.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_TEST = REPO_ROOT / "scripts" / "deploy" / "smoke_test.py"

# The 12 smoke test names expected in the report
EXPECTED_TEST_NAMES = [
    "model_accessible",
    "backend_healthz",
    "frontend_root",
    "backend_calls_model",
    "frontend_reaches_backend",
    "query_normal",
    "query_calculation",
    "query_unanswerable_safe",
    "sse_terminates",
    "trace_id_present",
    "no_path_leak_in_errors",
    "restart_recovery",
]


def _load_module(module_name: str, path: Path):
    """Import a Python module from a file path without running main()."""
    if not path.is_file():
        pytest.skip(f"Script not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        pytest.skip(f"Cannot create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def smoke_source() -> str:
    if not SMOKE_TEST.is_file():
        pytest.skip(f"smoke_test.py not found: {SMOKE_TEST}")
    return SMOKE_TEST.read_text(encoding="utf-8")


def test_smoke_test_exists():
    # Verify smoke_test.py exists
    assert SMOKE_TEST.is_file(), f"Expected {SMOKE_TEST} to exist"


def test_smoke_test_importable():
    # Verify smoke_test.py can be imported without errors
    module = _load_module("deploy_smoke_test", SMOKE_TEST)
    assert hasattr(module, "main"), "smoke_test.py missing main() function"
    assert hasattr(module, "run_all_tests"), "smoke_test.py missing run_all_tests() function"


def test_smoke_test_covers_12_cases(smoke_source):
    # Verify all 12 test cases are present (by function name or TestResult name)
    for test_name in EXPECTED_TEST_NAMES:
        assert test_name in smoke_source, \
            f"smoke_test.py does not cover test case: {test_name}"


def test_smoke_test_count_is_12(smoke_source):
    # Verify the test count is exactly 12
    module = _load_module("deploy_smoke_test_count", SMOKE_TEST)
    env = {
        "MODEL_HOST": "127.0.0.1", "MODEL_PORT": "18001",
        "BACKEND_HOST": "127.0.0.1", "BACKEND_PORT": "18002",
        "FRONTEND_HOST": "127.0.0.1", "FRONTEND_PORT": "18003",
        "MODEL_NAME": "test",
    }
    report = module.run_all_tests(env)
    assert report["total"] == 12, f"Expected 12 smoke tests, got {report['total']}"


def test_smoke_test_writes_json_report(smoke_source):
    # Verify smoke_test.py writes a JSON report
    assert "SMOKE_REPORT_PATH" in smoke_source, \
        "smoke_test.py should define a report path (SMOKE_REPORT_PATH)"
    assert "json.dump" in smoke_source, \
        "smoke_test.py should write JSON (json.dump)"
    assert "write_report" in smoke_source, \
        "smoke_test.py should have a write_report function"


def test_smoke_test_does_not_log_full_questions():
    # Verify TestResult only stores name, status, detail (not question/answer text)
    module = _load_module("deploy_smoke_test_slots", SMOKE_TEST)
    tr_cls = getattr(module, "TestResult", None)
    assert tr_cls is not None, "smoke_test.py missing TestResult class"
    # Check __slots__ does not include question or answer
    slots = getattr(tr_cls, "__slots__", None)
    if slots is not None:
        slot_names = [s.lower() for s in slots]
        assert "question" not in slot_names, \
            "TestResult __slots__ includes 'question' — may log full questions"
        assert "answer" not in slot_names, \
            "TestResult __slots__ includes 'answer' — may log full answers"


def test_smoke_test_to_dict_excludes_question_answer():
    # Verify TestResult.to_dict() only produces name, status, detail keys
    module = _load_module("deploy_smoke_test_todict", SMOKE_TEST)
    tr_cls = getattr(module, "TestResult", None)
    assert tr_cls is not None, "smoke_test.py missing TestResult class"
    instance = tr_cls("test_name", "pass", "some detail")
    result = instance.to_dict()
    result_keys = set(k.lower() for k in result.keys())
    assert "question" not in result_keys, \
        "TestResult.to_dict() includes 'question' key — may log full questions"
    assert "answer" not in result_keys, \
        "TestResult.to_dict() includes 'answer' key — may log full answers"
    # Only expected keys: name, status, detail
    assert result_keys.issubset({"name", "status", "detail"}), \
        f"Unexpected keys in to_dict(): {result_keys}"


def test_smoke_test_detail_fields_are_generic(smoke_source):
    # Verify detail messages are generic (not full question/answer text)
    # Check that TestResult construction doesn't embed QUESTION_* constants in detail
    detail_pattern = re.compile(r'TestResult\([^)]*"([^"]*)"[^)]*\)')
    for m in detail_pattern.finditer(smoke_source):
        detail_text = m.group(1)
        assert "QUESTION_" not in detail_text, \
            f"TestResult detail references QUESTION_ constant: {detail_text}"
