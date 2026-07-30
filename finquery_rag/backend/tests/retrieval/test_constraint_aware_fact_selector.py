from decimal import Decimal

from src.evaluation.nf41_fact_selection import (
    StructuredFactCandidate,
    build_constraints,
    has_explicit_conflict,
    select_constraint_aware_fact,
)


def _fact(*, fact_id: str, value: str, period: str | None, unit: str = "percentage", rank: int = 1):
    return StructuredFactCandidate(
        fact_id=fact_id, candidate_key=fact_id, candidate_rank=rank,
        document_id="report.pdf", page=1, text=f"Revenue margin was {value}% in {period or 'an unknown period'}.",
        value=Decimal(value), source_expression=f"{value}%", unit=unit, period=period,
        metric_text="Revenue margin", extraction_source="numeric_span", extraction_confidence=1.0,
    )


def test_unknown_period_is_not_filtered():
    constraints = build_constraints("What percentage was reported in 2025?")
    assert not has_explicit_conflict(_fact(fact_id="unknown", value="72", period=None), constraints)


def test_explicit_period_conflict_is_filtered():
    constraints = build_constraints("What percentage was reported in 2025?")
    assert has_explicit_conflict(_fact(fact_id="old", value="70", period="2024"), constraints)


def test_compatible_unit_is_retained():
    constraints = build_constraints("What percentage was reported in 2025?")
    assert not has_explicit_conflict(_fact(fact_id="rate", value="72", period="2025"), constraints)


def test_candidate_rank_is_stable_tiebreak():
    constraints = build_constraints("What percentage was reported in 2025?")
    first = _fact(fact_id="a", value="72", period="2025", rank=1)
    later = _fact(fact_id="b", value="70", period="2025", rank=2)
    assert select_constraint_aware_fact([later, first], constraints) == first


def test_selector_is_deterministic():
    constraints = build_constraints("What percentage was reported in 2025?")
    facts = [_fact(fact_id="a", value="72", period="2025"), _fact(fact_id="b", value="70", period="2024")]
    assert select_constraint_aware_fact(facts, constraints) == select_constraint_aware_fact(facts, constraints)
