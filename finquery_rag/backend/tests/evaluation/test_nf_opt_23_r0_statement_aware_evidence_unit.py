"""CPU-safe contract tests for NF-OPT-23 R0."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.evaluation.run_nf_opt_23_r0_statement_aware_evidence_unit import (
    build_statement_unit,
    classify_structured_view,
)


def _graph() -> dict:
    return {
        "logical_tables": {"table:t": {"statement_type": "income_statement", "table_title": "Operations"}},
        "semantic_rows": {"row:r": {"raw_label": "Revenue"}},
        "axis_bindings": {("table:t", "row:r"): [{"column_index": 0, "normalized_period": "FY2024", "cell_id": "cell:1"}, {"column_index": 1, "normalized_period": "FY2023", "cell_id": "cell:2"}]},
        "row_matrices": {"row:r": {"currency_code": "USD", "scale_unit": "million", "dimensions": [{"column_index": 0, "normalized_period": "FY2024", "value_raw": "120"}, {"column_index": 1, "normalized_period": "FY2023", "value_raw": "100"}]}},
    }


def _view() -> dict:
    return {"candidate_key": "candidate:v1:test", "document_id": "acme_fy2024", "pdf_page": 9, "candidate_type": "table_row", "raw_content": "Revenue | 120 | 100", "table_title": "Operations", "metric_paths": ["Revenue"], "facts": [], "row_ids": ["row:r"], "source_traceback": [{"table_fragment_id": "table:t", "row_id": "row:r", "document_id": "acme_fy2024", "pdf_page": 9}]}


def test_statement_unit_is_deterministic_and_query_independent() -> None:
    a = build_statement_unit("candidate:v1:test", "[DOCUMENT]\n[CONTENT]\nold", _view(), _graph())
    b = build_statement_unit("candidate:v1:test", "[DOCUMENT]\n[CONTENT]\nold", _view(), _graph())
    assert a["serialization"] == b["serialization"]
    assert a["serialization_sha256"] == b["serialization_sha256"]
    assert "FY2024 = 120" in a["serialization"] and "FY2023 = 100" in a["serialization"]
    assert a["serialization"].index("FY2024") < a["serialization"].index("FY2023")
    assert a["relational_structure_available"] is True


def test_missing_structure_fails_closed_to_baseline() -> None:
    baseline = "[DOCUMENT]\nDocument: acme\n\n[CONTENT]\nraw text"
    unit = build_statement_unit("candidate:v1:unresolved", baseline, None, _graph())
    assert unit["candidate_type"] == "unresolved"
    assert unit["serialization"] == baseline
    assert unit["fallback"] == "baseline_document_view"
    assert unit["relational_structure_available"] is False


def test_type_classification_is_deterministic() -> None:
    assert classify_structured_view(_view()) == "table_row_backed"
    assert classify_structured_view(None) == "unresolved"


def test_sealed_artifacts_preserve_candidate_universe_and_top100() -> None:
    backend = Path(__file__).resolve().parents[2]
    out = backend / "artifacts" / "evaluation" / "nf-opt-23-r0-statement-aware-evidence-unit"
    if not (out / "decision.json").exists():
        return
    decision = json.loads((out / "decision.json").read_text())
    strict = json.loads((out / "strict-metrics.json").read_text())
    seal = json.loads((out / "prediction-seal.json").read_text())
    assert decision["retrieval_rerun"] is False
    assert decision["candidate_identity_unchanged"] is True
    assert decision["production_switch_allowed"] is False
    assert strict["statement_aware"]["@100"]["hits"] == 68
    assert strict["qwen"]["@100"]["hits"] == 68
    assert seal["gold_reads_before_prediction_seal"] == 0
    with gzip.open(out / "predictions.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    assert len(rows) == 72
    assert all(len(row["ranked_candidates"]) == 100 for row in rows)
