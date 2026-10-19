"""Unit tests for interactive reflective follow-up clarification (Interactive Disambiguation)."""

from __future__ import annotations

from typing import Any
import pytest

from src.conversation.contracts import DialogueState, PendingClarification, ReasonCode
from src.conversation.service import ConversationContextManager
from src.finance.interactive_disambiguation import (
    InteractiveDisambiguator,
    DisambiguationOption,
    DisambiguationResult,
)
from src.finance.source_label_grounding import SourceLabelGrounding


def test_extract_options_deduplicates_values() -> None:
    conflicting = [
        {"value": "604", "page": 43, "content": "Deferred | 604 | 347 | 669", "candidate_id": "c1"},
        {"value": "604", "page": 44, "content": "Deferred | 604 | duplicate", "candidate_id": "c2"},
        {"value": "(1,804)", "page": 43, "content": "Deferred | (1,804) | (3,080)", "candidate_id": "c3"},
        {"value": "(139)", "page": 43, "content": "Deferred | (139) | (298)", "candidate_id": "c4"},
    ]
    options = InteractiveDisambiguator.extract_options(conflicting)
    assert len(options) == 3
    assert options[0].value == "604"
    assert options[0].page == 43
    assert options[1].value == "(1,804)"
    assert options[2].value == "(139)"


def test_build_clarification_prompt_structure() -> None:
    ambiguities = [
        {
            "entity": "Apple",
            "metric": "Deferred",
            "period": "FY2025",
            "conflicting_facts": [
                {"value": "(1,804)", "page": 43, "content": "Deferred | (1,804) | (3,080) | (3,644)"},
                {"value": "(139)", "page": 43, "content": "Deferred | (139) | (298) | (49)"},
                {"value": "604", "page": 43, "content": "Deferred | 604 | 347 | 669"},
            ],
        }
    ]
    res = InteractiveDisambiguator.build_clarification(
        "According to Apple's FY2025 10-K filing, what was the reported Deferred?",
        ambiguities,
    )
    assert res is not None
    assert "Apple" in res.prompt
    assert "Deferred" in res.prompt
    assert "(1,804)" in res.prompt
    assert "604" in res.prompt
    assert len(res.options) == 3
    assert "CLARIFICATION_REQUIRED" in res.reason_codes

    dict_repr = res.to_clarification_dict()
    assert dict_repr["question"] == res.prompt
    assert len(dict_repr["structured_options"]) == 3
    assert dict_repr["structured_options"][2]["value"] == "604"


def test_build_clarification_empty_ambiguities_returns_none() -> None:
    assert InteractiveDisambiguator.build_clarification("test query", []) is None


def test_source_label_grounding_fact_value_matching() -> None:
    fact_604 = {"value": "604"}
    fact_neg = {"value": "(1,804)"}

    assert SourceLabelGrounding._fact_value_matches_query(fact_604, "what was Deferred of 604?")
    assert SourceLabelGrounding._fact_value_matches_query(fact_neg, "what was Deferred of (1,804)?")
    assert SourceLabelGrounding._fact_value_matches_query(fact_neg, "what was Deferred of 1804?")
    assert not SourceLabelGrounding._fact_value_matches_query(fact_604, "what was Deferred?")


class DummyFactStore:
    def __init__(self, facts: list[dict[str, Any]]) -> None:
        self._facts = facts

    def facts_at_coordinate(self, entity: str, metric: str, period: str) -> list[dict[str, Any]]:
        return list(self._facts)

    def facts_for_label(self, metric: str, period: str) -> list[dict[str, Any]]:
        return list(self._facts)


class DummySlot:
    def __init__(self, metric: str, entity: str = "Apple", period: str = "FY2025") -> None:
        self.metric = metric
        self.entity = entity
        self.period = period
        self.scope = None
        self.slot_id = "slot_1"


class DummyPlan:
    def __init__(self, slots: list[DummySlot]) -> None:
        self.required_slots = slots


def test_source_label_grounding_disambiguates_with_query() -> None:
    facts = [
        {"value": "(1,804)", "entity": "Apple", "metric": "Deferred", "period": "FY2025"},
        {"value": "(139)", "entity": "Apple", "metric": "Deferred", "period": "FY2025"},
        {"value": "604", "entity": "Apple", "metric": "Deferred", "period": "FY2025"},
    ]
    store = DummyFactStore(facts)
    grounding = SourceLabelGrounding(store)
    plan = DummyPlan([DummySlot("Deferred")])

    # Turn 1: Without query value, detected as ambiguous conflict
    res1 = grounding(plan, query="what was Deferred?")
    assert res1 == {}
    assert len(grounding.last_detected_ambiguities) == 1
    assert grounding.last_detected_ambiguities[0]["metric"] == "Deferred"

    # Turn 2: Query specifies 604 -> uniquely resolved and grounded!
    res2 = grounding(plan, query="what was Deferred of 604?")
    assert "Deferred" in res2
    assert len(grounding.last_detected_ambiguities) == 0


def test_conversation_service_resolves_coordinate_choices() -> None:
    class DummyContextManager(ConversationContextManager):
        def __init__(self) -> None:
            pass  # omit full initialization

    mgr = DummyContextManager()
    state = DialogueState(
        conversation_id="conv_1",
        active_entity="Apple",
        active_period="FY2025",
        active_topic="Deferred",
    )
    pending = PendingClarification(
        reason_codes=["COORDINATE_CONFLICT", "CLARIFICATION_REQUIRED"],
        candidates=[
            "数值 (1,804) (第 43 页) [上下文: Deferred | (1,804) | (3,080)]",
            "数值 (139) (第 43 页) [上下文: Deferred | (139) | (298)]",
            "数值 604 (第 43 页) [上下文: Deferred | 604 | 347]",
        ],
        unresolved_fields=["coordinate_value"],
        entity="Apple",
        period="FY2025",
        topic="Deferred",
    )
    state.pending_clarification = pending

    # Test choice by exact value: "604"
    res_val = mgr._resolve_pending_choice("604", state)
    assert res_val is not None
    assert "604" in res_val.standalone_query
    assert "Apple" in res_val.standalone_query
    assert "Deferred" in res_val.standalone_query
    assert ReasonCode.PENDING_CLARIFICATION_RESOLVED in res_val.reason_codes

    # Test choice by Chinese ordinal: "第三项"
    res_ord = mgr._resolve_pending_choice("第三项", state)
    assert res_ord is not None
    assert "604" in res_ord.standalone_query

    # Test choice by digit index: "1" (corresponding to (1,804))
    res_idx = mgr._resolve_pending_choice("1", state)
    assert res_idx is not None
    assert "(1,804)" in res_idx.standalone_query

    # Test choice by English ordinals
    res_en1 = mgr._resolve_pending_choice("first", state)
    assert res_en1 is not None
    assert "(1,804)" in res_en1.standalone_query

    res_en2 = mgr._resolve_pending_choice("option 2", state)
    assert res_en2 is not None
    assert "(139)" in res_en2.standalone_query

    res_en3 = mgr._resolve_pending_choice("3rd", state)
    assert res_en3 is not None
    assert "604" in res_en3.standalone_query


def test_candidate_value_substring_collision_safety() -> None:
    class DummyContextManager(ConversationContextManager):
        def __init__(self) -> None:
            pass

    mgr = DummyContextManager()
    state = DialogueState(
        conversation_id="conv_2",
        active_entity="Apple",
        active_period="FY2025",
        active_topic="Expenses",
    )
    pending = PendingClarification(
        reason_codes=["COORDINATE_CONFLICT"],
        candidates=[
            "数值 10 (第 1 页)",
            "数值 100 (第 2 页)",
        ],
        unresolved_fields=["coordinate_value"],
        entity="Apple",
        period="FY2025",
        topic="Expenses",
    )
    state.pending_clarification = pending

    # When user inputs "100", it must match 100, not 10!
    res = mgr._resolve_pending_choice("100", state)
    assert res is not None
    assert "100" in res.standalone_query
    assert "Target value for Expenses: 100" in res.standalone_query


def test_trusted_v2_binder_target_value_and_sign_matching() -> None:
    from src.runtime.trusted_v2_binder import SemanticEvidenceEvaluationCapability

    # Positive target matches positive fact
    assert SemanticEvidenceEvaluationCapability._fact_value_matches_target("604", "604")
    assert SemanticEvidenceEvaluationCapability._fact_value_matches_target("1,804", "1804")

    # Negative target matches negative fact
    assert SemanticEvidenceEvaluationCapability._fact_value_matches_target("(1,804)", "(1,804)")
    assert SemanticEvidenceEvaluationCapability._fact_value_matches_target("-1,804", "(1,804)")

    # Negative target must NOT match positive fact
    assert not SemanticEvidenceEvaluationCapability._fact_value_matches_target("1,804", "(1,804)")
    assert not SemanticEvidenceEvaluationCapability._fact_value_matches_target("1804", "-1804")


def test_trusted_v2_binder_multi_slot_target_value_isolation() -> None:
    from src.runtime.trusted_v2_binder import SemanticEvidenceEvaluationCapability

    class DummySlotObj:
        def __init__(self, metric: str) -> None:
            self.metric = metric

    query = "According to Apple's FY2025 filing, compare Operating Income and Deferred? Target value for Deferred: 604."
    
    # Slot 1: Deferred should extract 604
    val_deferred = SemanticEvidenceEvaluationCapability._extract_target_value_for_slot(
        query, DummySlotObj("Deferred")
    )
    assert val_deferred == "604"

    # Slot 2: Operating Income must NOT be constrained by Deferred's target value
    val_oi = SemanticEvidenceEvaluationCapability._extract_target_value_for_slot(
        query, DummySlotObj("Operating Income")
    )
    assert val_oi is None


def test_conversation_service_semantic_context_and_page_matching() -> None:
    class DummyContextManager(ConversationContextManager):
        def __init__(self) -> None:
            pass

    mgr = DummyContextManager()
    state = DialogueState(
        conversation_id="conv_context_match",
        active_entity="Apple",
        active_period="FY2025",
        active_topic="Deferred",
    )
    pending = PendingClarification(
        reason_codes=["COORDINATE_CONFLICT"],
        candidates=[
            "数值 (1,804) (第 43 页) [上下文: Deferred | (1,804) | (3,080) | (3,644)]",
            "数值 (139) (第 43 页) [上下文: Deferred | (139) | (298) | (49)]",
            "数值 604 (第 44 页) [上下文: Deferred | 604 | 347 | 669]",
        ],
        unresolved_fields=["coordinate_value"],
        entity="Apple",
        period="FY2025",
        topic="Deferred",
    )
    state.pending_clarification = pending

    # 1. Match by unique cell content in context: "3080"
    res_ctx = mgr._resolve_pending_choice("有3080的那行", state)
    assert res_ctx is not None
    assert "(1,804)" in res_ctx.standalone_query

    # 2. Match by unique page: "第44页"
    res_page = mgr._resolve_pending_choice("第44页那项", state)
    assert res_page is not None
    assert "604" in res_page.standalone_query


def test_conversation_service_history_backtracking_and_reselection() -> None:
    class DummyBudgetManager:
        def update_compressed_history(self, state: Any, turns: Any) -> None:
            return None

    class DummyContextManager(ConversationContextManager):
        def __init__(self) -> None:
            self.budget_manager = DummyBudgetManager()

    mgr = DummyContextManager()
    state = DialogueState(
        conversation_id="conv_backtrack",
        active_entity="Apple",
        active_period="FY2025",
        active_topic="Deferred",
    )
    pending = PendingClarification(
        reason_codes=["COORDINATE_CONFLICT"],
        candidates=[
            "数值 (1,804) (第 43 页) [上下文: Deferred | (1,804) | (3,080)]",
            "数值 (139) (第 43 页) [上下文: Deferred | (139) | (298)]",
            "数值 604 (第 43 页) [上下文: Deferred | 604 | 347]",
        ],
        unresolved_fields=["coordinate_value"],
        entity="Apple",
        period="FY2025",
        topic="Deferred",
    )
    state.pending_clarification = pending

    # Turn 2: User chose option 3 (604)
    res_turn2 = mgr._resolve_pending_choice("604", state)
    assert res_turn2 is not None
    mgr._update_state_on_resolution(state, "604", res_turn2)

    # State now has pending_clarification=None, but last_clarification is preserved
    assert state.pending_clarification is None
    assert state.last_clarification is not None
    assert len(state.last_clarification.candidates) == 3

    # Turn 3: User backtracks: "不对，换成第1项"
    res_turn3 = mgr._resolve_pending_choice("不对，换成第1项", state)
    assert res_turn3 is not None
    assert "(1,804)" in res_turn3.standalone_query
    assert "Apple" in res_turn3.standalone_query


def test_interactive_disambiguator_multi_slot_cascade() -> None:
    ambiguities = [
        {
            "slot_id": "slot_deferred",
            "entity": "Apple",
            "metric": "Deferred",
            "period": "FY2025",
            "conflicting_facts": [
                {"value": "(1,804)", "page": 43, "content": "Deferred | (1,804)"},
                {"value": "604", "page": 43, "content": "Deferred | 604"},
            ],
        },
        {
            "slot_id": "slot_other_income",
            "entity": "Apple",
            "metric": "Other Income",
            "period": "FY2025",
            "conflicting_facts": [
                {"value": "250", "page": 44, "content": "Other Income | 250"},
                {"value": "180", "page": 44, "content": "Other Income | 180"},
            ],
        },
    ]

    res = InteractiveDisambiguator.build_clarification("compare Deferred and Other Income", ambiguities)
    assert res is not None
    assert "Deferred" in res.prompt
    assert "Other Income" in res.prompt
    assert "【科目 1: Deferred】" in res.prompt
    assert "【科目 2: Other Income】" in res.prompt
    assert len(res.options) == 4
    assert len(res.structured_options) == 4
    assert res.structured_options[0]["metric"] == "Deferred"
    assert res.structured_options[2]["metric"] == "Other Income"


def test_footnote_anchor_detector() -> None:
    from src.finance.footnote_anchor import FootnoteAnchorDetector

    assert FootnoteAnchorDetector.extract_marker("Services (1)") == "(1)"
    assert FootnoteAnchorDetector.extract_marker("Term debt [a]") == "[a]"
    assert FootnoteAnchorDetector.extract_marker("Cash equivalents *") == "*"
    assert FootnoteAnchorDetector.extract_marker("Net Income") is None

    assert FootnoteAnchorDetector.strip_marker("Services (1)") == "Services"

    page_text = (
        "Consolidated Statements of Operations\n"
        "Services (1) | 109,158 | 85,200\n"
        "(1) Includes revenue recognized from cloud services, App Store, and digital content.\n"
        "* Reflects rounding adjustments."
    )
    anchor = FootnoteAnchorDetector.link_metric_footnotes("Services (1)", page_text, page=43)
    assert anchor is not None
    assert anchor.marker == "(1)"
    assert anchor.metric == "Services"
    assert "Includes revenue recognized" in anchor.annotation
    assert anchor.page == 43


def test_edgar_feed_monitor_db_tracking(tmp_path: Any) -> None:
    from src.finance.edgar_feed_monitor import EdgarFeedMonitor

    db_path = tmp_path / "test_facts.sqlite"
    monitor = EdgarFeedMonitor(db_path=db_path)

    # Empty initially
    assert monitor.get_ingested_accession_numbers() == set()

    # Mark filing ingested
    monitor.mark_filing_ingested(
        ticker="AAPL",
        accession_number="0000320193-25-000106",
        form="10-K",
        filing_date="2025-10-31",
        facts_count=42,
    )

    seen = monitor.get_ingested_accession_numbers()
    assert "0000320193-25-000106" in seen


