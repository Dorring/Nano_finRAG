"""Unit tests for SourceRouter dynamic multi-source routing."""

from rag_v2.contracts.plan import Action, Intent, RequiredSlot, SupervisorPlan
from rag_v2.routing.source_router import DataSourceType, SourceRouter


def _make_slot(metric: str, slot_id: str = "s1", scope: str | None = None) -> RequiredSlot:
    return RequiredSlot(
        slot_id=slot_id,
        metric=metric,
        period="FY2025",
        role="value",
        value_type="numeric",
        unit="USD",
        entity="Apple",
        entity_id="aapl",
        scope=scope,
    )


def test_market_api_routing_by_query() -> None:
    res = SourceRouter.route_query("What is Tesla's current PE ratio and stock price?")
    assert res.target_source is DataSourceType.MARKET_API
    assert "Market Data Provider" in (res.section_hint or "")
    assert res.confidence >= 0.9


def test_narrative_routing_by_query() -> None:
    res = SourceRouter.route_query("Why did Apple's Greater China revenue decline according to MD&A?")
    assert res.target_source is DataSourceType.NARRATIVE_DISCLOSURES
    assert "Item 7" in (res.section_hint or "")


def test_footnote_routing_for_segment_slot() -> None:
    slot = _make_slot(metric="Services revenue", scope="segment")
    res = SourceRouter.route_slot(slot)
    assert res.target_source is DataSourceType.FOOTNOTE_DISCLOSURES
    assert "Segment Information" in (res.section_hint or "")

    # When query specifically names a note (e.g. Note 19 or Note 21)
    res_explicit = SourceRouter.route_slot(slot, query="Details in Note 19")
    assert res_explicit.section_hint == "Item 8. Note 19"


def test_structured_store_routing_for_topline_metrics() -> None:
    slot_rev = _make_slot(metric="Total Net Sales")
    res_rev = SourceRouter.route_slot(slot_rev)
    assert res_rev.target_source is DataSourceType.STRUCTURED_STORE
    assert "Statements of Operations" in (res_rev.section_hint or "")

    slot_asset = _make_slot(metric="Total Assets")
    res_asset = SourceRouter.route_slot(slot_asset)
    assert res_asset.target_source is DataSourceType.STRUCTURED_STORE
    assert "Balance Sheets" in (res_asset.section_hint or "")


def test_annotate_plan_with_routing_decisions() -> None:
    plan = SupervisorPlan(
        intent=Intent.DIRECT_FACT,
        required_slots=(
            _make_slot(metric="Total Net Sales", slot_id="s1"),
            _make_slot(metric="Intelligent Cloud revenue", slot_id="s2", scope="segment"),
        ),
        operation=None,
        next_action=Action.RETRIEVE,
    )

    annotated = SourceRouter.annotate_plan(plan)
    assert len(annotated.required_slots) == 2
    slot0 = annotated.required_slots[0]
    slot1 = annotated.required_slots[1]

    assert slot0.source_preference == DataSourceType.STRUCTURED_STORE.value
    assert "Statements of Operations" in (slot0.section_hint or "")

    assert slot1.source_preference == DataSourceType.FOOTNOTE_DISCLOSURES.value
    assert "Segment Information" in (slot1.section_hint or "")

