"""Regression tests for the production Binder fact-view boundary."""

from __future__ import annotations

from rag_v2.contracts.plan import SupervisorPlan
from rag_v2.evidence.binder_fact_view import build_runtime_binder_fact_view
from rag_v2.evidence.binder_service import BinderRequest


def _plan() -> SupervisorPlan:
    return SupervisorPlan.from_dict(
        {
            "intent": "DIRECT_FACT",
            "required_slots": [
                {
                    "slot_id": "revenue_fy2024",
                    "metric": "Revenue",
                    "period": "FY2024",
                    "role": "value",
                    "value_type": "numeric",
                    "unit": None,
                }
            ],
            "operation": None,
            "next_action": "RETRIEVE",
        }
    )


def _fact() -> dict[str, object]:
    return {
        "fact_id": "fact-001",
        "evidence_id": "evidence-001",
        "candidate_id": "candidate-001",
        "citation_id": "citation-001",
        "physical_source_id": "physical-page-3-row-8",
        "document_id": "apple-10k-2024",
        "pdf_page": 3,
        "metric": "Revenue",
        "period": "FY2024",
        "value": "391035",
        "currency": "USD",
        "scale": "million",
        "provenance_complete": True,
        "content": "unbounded raw retrieval text",
        "raw_content": "unbounded raw content",
        "source_text": "source text must not cross the provider boundary",
        "evidence_text": "more raw text",
        "assistant_text": "The assistant claims revenue was $999B.",
        "model_generated_summary": "model summary must remain semantic context only",
        "conversation_history": [{"role": "assistant", "content": "fake fact"}],
        "reasoning_content": "private reasoning must never be retained",
        "chain_of_thought": "private reasoning must never be retained",
        "opaque_debug": {"raw": "not a binder field"},
        "structural_context": {
            "scope": "consolidated",
            "row_label": "Net sales",
            "metric_paths": ["Statements", "Net sales"],
            "table_title": "Consolidated Statements of Operations",
            "unknown_blob": {"text": "must not pass through"},
        },
    }


def test_runtime_binder_view_is_source_derived_and_excludes_untrusted_text() -> None:
    view = build_runtime_binder_fact_view(_fact())

    assert view == {
        "fact_id": "fact-001",
        "evidence_id": "evidence-001",
        "candidate_id": "candidate-001",
        "citation_id": "citation-001",
        "physical_source_id": "physical-page-3-row-8",
        "document_id": "apple-10k-2024",
        "pdf_page": 3,
        "metric": "Revenue",
        "period": "FY2024",
        "value": "391035",
        "currency": "USD",
        "scale": "million",
        "scope": "consolidated",
        "row_label": "Net sales",
        "metric_paths": ["Statements", "Net sales"],
        "table_title": "Consolidated Statements of Operations",
        "provenance_complete": True,
    }
    forbidden = {
        "content",
        "raw_content",
        "source_text",
        "evidence_text",
        "assistant_text",
        "model_generated_summary",
        "conversation_history",
        "reasoning_content",
        "chain_of_thought",
        "opaque_debug",
        "structural_context",
    }
    assert not (forbidden & set(view))


def test_binder_request_sends_safe_view_but_keeps_full_fact_internal() -> None:
    fact = _fact()
    request = BinderRequest("request-1", "What was Apple revenue?", _plan(), (fact,))

    provider_payload = request.to_dict()

    assert provider_payload["financial_facts"] == [build_runtime_binder_fact_view(fact)]
    assert request.facts[0]["source_text"] == "source text must not cross the provider boundary"
    assert request.facts[0]["assistant_text"] == "The assistant claims revenue was $999B."
