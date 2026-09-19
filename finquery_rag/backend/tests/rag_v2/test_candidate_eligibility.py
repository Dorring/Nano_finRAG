"""Slot-aware candidate eligibility: the slot's own structural minimum.

A retrieval packet is allowed to be broad.  It carries rows for audit and
recovery that no particular slot can use, and it is meant to.  What it must not
do is put those rows in front of a model and ask the model to rule them out --
which is what happened until this existed, and the model did it inconsistently:
`pctshare-004` bound its denominator on one run and reported both slots missing
on the next, over a byte-identical packet.

Eligibility is therefore a deterministic reading of the *slot's* declared
requirements, applied before the Binder is shown anything.  It decides what
cannot be a candidate; the model still decides what is the right one.

The last three tests are the ones that keep this honest.  A rule that quietly
became "drop rows with missing fields" would pass everything above them and fail
those, and it is the wrong change: period-less and value-less rows are real rows
that other work will want.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_v2.contracts.plan import RequiredSlot
from src.runtime.trusted_v2_binder import (
    CANDIDATE_PERIOD_ABSENT,
    CANDIDATE_PERIOD_MISMATCH,
    CANDIDATE_VALUE_NOT_NUMERIC,
    _metric_is_admissible,
    candidate_eligible_for_slot,
)


def _slot(
    period: str | None = "FY2025",
    *,
    metric: str = "Cost of sales",
    value_type: str = "numeric",
) -> RequiredSlot:
    return RequiredSlot(
        slot_id="s1",
        metric=metric,
        period=period,
        role="value",
        value_type=value_type,
    )


def _fact(value: str | None = "(220,960)", period: str | None = "FY2025") -> dict:
    return {"metric": "Cost of sales", "period": period, "value": value, "entity": "Apple"}


# --- what the rule is for -----------------------------------------------------


def test_a_complete_candidate_is_eligible() -> None:
    eligible, reasons = candidate_eligible_for_slot(_fact(), _slot())

    assert eligible is True
    assert reasons == ()


def test_a_period_mismatch_is_ineligible() -> None:
    """Nowhere near a conflict: the row is simply not for this slot."""

    eligible, reasons = candidate_eligible_for_slot(_fact(period="FY2024"), _slot())

    assert eligible is False
    assert CANDIDATE_PERIOD_MISMATCH in reasons


def test_the_period_less_sibling_is_ineligible() -> None:
    """The row that made the 004 packet ambiguous.

    ``period=None, value=None`` is not a second opinion about the FY2025 figure;
    it is not a quantitative fact at all.  Ruling it out is a reading of the
    slot's contract, not a judgement about meaning.
    """

    eligible, reasons = candidate_eligible_for_slot(
        _fact(value=None, period=None), _slot()
    )

    assert eligible is False
    assert CANDIDATE_PERIOD_ABSENT in reasons
    assert CANDIDATE_VALUE_NOT_NUMERIC in reasons


def test_a_numeric_slot_rejects_a_missing_value() -> None:
    eligible, reasons = candidate_eligible_for_slot(_fact(value=None), _slot())

    assert eligible is False
    assert CANDIDATE_VALUE_NOT_NUMERIC in reasons


def test_a_non_numeric_value_is_not_a_value() -> None:
    eligible, reasons = candidate_eligible_for_slot(_fact(value="not a number"), _slot())

    assert eligible is False
    assert CANDIDATE_VALUE_NOT_NUMERIC in reasons


@pytest.mark.parametrize("value", ["21%", "(486)", "$ 62,151", "1,500", "(0.3)%"])
def test_a_stated_financial_value_counts_including_a_percentage(value: str) -> None:
    """``21%`` is what an income-tax-rate row states.

    A rule that leaned on ``canonical_decimal`` would call it a non-value and
    exclude the very candidates these fixtures exist to bind.
    """

    assert candidate_eligible_for_slot(_fact(value=value), _slot())[0] is True


def test_the_004_shape_leaves_exactly_one_candidate() -> None:
    """Three rows for one metric, one slot: what the Binder should be handed."""

    slot = _slot(metric="Impact of the State Aid Decision")
    packet = [
        _fact(value="10,246", period="FY2024"),
        _fact(value="(486)", period="FY2025"),
        _fact(value=None, period=None),
    ]

    eligible = [fact for fact in packet if candidate_eligible_for_slot(fact, slot)[0]]

    assert eligible == [packet[1]]


# --- the three that keep it from becoming a global tidy-up --------------------


def test_a_slot_that_asks_for_no_period_does_not_reject_a_period_less_row() -> None:
    """Silence is not a requirement.  Only what the slot states can be unmet.

    A stand-in rather than a ``RequiredSlot``, because the contract already
    forbids the case this guards: ``RequiredSlot.period`` is validated as a
    non-empty string, so no plan can ever reach the Binder with a period-less
    slot.  The guard is here so the function cannot over-reject if that ever
    relaxes *or* if a caller passes something slot-shaped that is not a
    ``RequiredSlot`` -- and it is tested so the guard is not mistaken for dead
    code and deleted.
    """

    period_less_slot = SimpleNamespace(metric="Cost of sales", period=None, value_type="numeric")

    eligible, reasons = candidate_eligible_for_slot(
        _fact(value="604", period=None), period_less_slot
    )

    assert eligible is True, reasons


def test_a_non_numeric_slot_is_not_judged_by_the_numeric_rule() -> None:
    """A qualitative slot has no numeric requirement to violate.

    This is the rule that would otherwise turn a real fix into a blanket cleanup
    of every row the tabular extractor left incomplete -- and those rows are
    exactly what a qualitative, heading or metric-discovery task is for.
    """

    eligible, reasons = candidate_eligible_for_slot(
        _fact(value=None), _slot(value_type="qualitative")
    )

    assert eligible is True, reasons


def test_the_period_rule_still_applies_to_a_qualitative_slot() -> None:
    """Exempting a slot from one requirement does not exempt it from the others."""

    eligible, reasons = candidate_eligible_for_slot(
        _fact(value=None, period="FY2024"),
        _slot(period="FY2025", value_type="qualitative"),
    )

    assert eligible is False
    assert CANDIDATE_PERIOD_MISMATCH in reasons
    assert CANDIDATE_VALUE_NOT_NUMERIC not in reasons


# --- the metric prefilter, which drops what it can identify and nothing else ---


def test_a_metric_the_ontology_cannot_name_is_admissible() -> None:
    """Unnamed is not irrelevant, and treating it as such caused two regressions.

    A plan naming `Total operating expenses` and `Leasehold improvements` makes
    the prefilter fire on the first and, before this, dropped every fact for the
    second -- because an unnamed metric is not in a set of names.  The Binder was
    then asked to fill a slot whose only fact it had never been shown.
    """

    assert _metric_is_admissible(
        {"metric": "Leasehold improvements"}, {"total_operating_expenses"}
    )
    assert _metric_is_admissible(
        {"metric": "Impact of the State Aid Decision"}, {"cost_of_revenue"}
    )


def test_a_metric_the_ontology_names_and_the_plan_does_not_want_is_dropped() -> None:
    """The prefilter still does its job -- it is a filter, not a no-op."""

    assert not _metric_is_admissible({"metric": "Net income"}, {"total_operating_expenses"})
    assert _metric_is_admissible(
        {"metric": "Total operating expenses"}, {"total_operating_expenses"}
    )


def test_a_fact_with_no_metric_at_all_is_admissible() -> None:
    """Nothing to identify means nothing to reject."""

    assert _metric_is_admissible({"metric": None}, {"total_operating_expenses"})
    assert _metric_is_admissible({}, {"total_operating_expenses"})
