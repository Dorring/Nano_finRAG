from __future__ import annotations

from src.pdf_retrieval_v4.candidate_field_query import build_field_queries
from src.pdf_retrieval_v4.candidate_field_view import FIELD_NAMES, field_view_id, project_candidate_fields
from src.pdf_retrieval_v4.query_plan_models import OperandSlot, QueryPlan, RetrievalConstraints


def _plan() -> QueryPlan:
    slot = OperandSlot("current", "current", "cloud revenue", ("revenue",), "FY2025", "duration", None, "Americas", "atomic_fact")
    return QueryPlan("p", "v", "What was cloud revenue?", ("doc",), "table_single_fact", None, None, ("cloud revenue",), ("FY2025",), ("atomic_fact",), (slot,), (), RetrievalConstraints(), True, True, statement_hint="income statement")


def test_field_projection_contract() -> None:
    record = {"candidate_key": "k", "document_id": "d", "metric_paths": ["Intelligent Cloud / Revenue"], "periods": ["FY2025"], "temporal_types": ["duration"], "segments": ["Americas"], "buckets": [], "section_path": ["MD&A"], "table_title": "Operations", "candidate_type": "table_row", "facts": [{"type": "atomic", "metric": "Revenue", "period": "FY2025", "value": "123"}], "row_matrix": None}
    views = project_candidate_fields(record)
    assert set(views) == set(FIELD_NAMES)
    assert "123" not in views["evidence"].retrieval_text
    assert "Intelligent Cloud Revenue" in views["metric"].retrieval_text
    assert "FY2025" in views["axis"].retrieval_text
    assert len({view.field_view_id for view in views.values()}) == 4


def test_field_view_identity_stable() -> None:
    assert field_view_id("k", "metric") == field_view_id("k", "metric")
    assert field_view_id("k", "metric") != field_view_id("k", "axis")


def test_field_query_preserves_raw_metric_and_axis() -> None:
    plan = _plan()
    queries = build_field_queries(plan, plan.operand_slots[0])
    assert "cloud revenue" in queries["metric"]
    assert "revenue" in queries["metric"]
    assert "FY2025" in queries["axis"]
    assert "duration" in queries["axis"]
    assert "Americas" in queries["axis"]
    assert "income statement" in queries["context"]
    assert "atomic_fact" in queries["evidence"]


def test_empty_axis_does_not_fallback() -> None:
    plan = _plan()
    empty = OperandSlot("x", "x", "revenue", (), None, None, None, None, "atomic_fact")
    queries = build_field_queries(plan, empty)
    assert queries["axis"] == ""
    assert plan.raw_question not in queries["context"]
