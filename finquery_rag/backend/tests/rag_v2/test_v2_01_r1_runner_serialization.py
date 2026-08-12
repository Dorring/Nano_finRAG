"""Regression coverage for the NF-V2-01 R1 formal runner result contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _load_runner():
    import importlib

    return importlib.import_module("scripts.evaluation.run_nf_v2_01_r1_formal_72")


def test_attempt_one_unpack_failure_is_reproduced_by_old_contract() -> None:
    """Keep the exact failure executable while the production path is typed."""

    legacy_producer_value = ([{"question_id": "synthetic-1"}], None, 1.0)
    with pytest.raises(ValueError, match=r"too many values to unpack \(expected 2\)"):
        records, failure = legacy_producer_value
        del records, failure


def test_formal_run_result_flows_through_real_prediction_serializer(tmp_path: Path) -> None:
    runner = _load_runner()
    records = [{
        "question_id": "synthetic-1",
        "intent": "DIRECT_FACT",
        "required_slots": [],
        "operation": None,
        "next_action": "RETRIEVE",
        "plan_valid": True,
        "provider": "bailian",
        "model": "qwen3.7-max-2026-06-08",
    }]
    result = runner.FormalRunResult(records=records, failure=None, elapsed_ms=12.5)
    assert result.records == records
    assert result.failure is None
    assert result.elapsed_ms == 12.5

    path = tmp_path / "predictions.jsonl.gz"
    digest = runner.serialize_prediction_records(result.records, path)
    loaded = runner.load_prediction_records(path)
    assert loaded == records
    assert digest == runner.sha256_file(path)
    assert json.loads(json.dumps(loaded[0]))["question_id"] == "synthetic-1"
