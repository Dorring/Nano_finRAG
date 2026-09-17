"""The harness fixtures must be able to express the sealed labels they drive.

The sealed gold (`tests/fixtures/tv2_07_production_readiness/`) and the fixture
specs (`tv2_readiness_cases.py`) were authored independently: the labels came
first, the fixtures were written later to make them executable.  That is the
right order -- it is what makes the labels an oracle -- but it also means a
fixture can be *wrong* in a way that looks like a runtime failure.

H2A-2C found exactly that for `multi-evidence-001`:

    sealed label wants    release, route MULTI, evidence [Q1, Q2],
                          and the answer to contain "Revenue"
    fixture supplied      one operating_margin slot, Q1 = 100, Q2 = 80,
                          and no Revenue fact at all

No runtime could satisfy both, because the fixture cannot produce the term the
label requires and its two facts contradict each other for one slot.  The
mismatch was recorded as a runtime gap and it was not one.

These tests state the invariants a fixture must satisfy to be a fair test of the
runtime, so the next mis-wiring is caught as a fixture defect rather than
reported as a capability gap.  They read the sealed labels and the fixture specs
directly, and derive nothing from the production binding path: a fixture check
that asked the binder what it thought would have exactly the circularity this
file exists to remove.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_v2.contracts.financial_semantics import quantity_identity
from tests.benchmark.tv2_readiness_cases import cases

#: Facts whose quantitiy identity a case needs to be equivalent for them to be
#: corroborating rather than competing.
CORROBORATING_FACT_PAIRS: dict[str, tuple[str, str]] = {
    "multi_evidence": ("Q1", "Q2"),
}


def _case_by_id(case_id: str) -> Any:
    for case in cases():
        if case.case_id == case_id:
            return case
    raise AssertionError(f"no sealed case {case_id!r}")


# --- the general invariant ---------------------------------------------------


def test_cases_presenting_identical_evidence_agree_on_whether_it_answers() -> None:
    """A tripwire, not a theorem: identical evidence, opposite labels.

    `multi_evidence` and `qualitative` are *separate* fixture objects -- not one
    shared one -- but they carry identical slots, identical facts and identical
    retrieval routes, and their labels ask for opposite outcomes (release versus
    abstain).  Only the query text differs.

    This runtime's release decision is driven by whether the required slot is
    supported by admissible evidence, so identical evidence cannot produce
    opposite outcomes: at most one of the two labels is reachable, and the
    mismatch count was guaranteed to be non-zero before any runtime behaviour
    was considered.  If this fires, a human has to say which label the evidence
    is meant to demonstrate -- it is not something a runtime can be asked to
    reconcile.
    """

    by_evidence: dict[tuple, list[Any]] = {}
    for case in cases():
        spec = case.fixture
        key = (
            tuple(tuple(sorted(dict(slot).items())) for slot in spec.slots),
            tuple(sorted((name, tuple(sorted(dict(fact).items()))) for name, fact in spec.facts.items())),
            tuple(tuple(route) for route in spec.routes),
        )
        by_evidence.setdefault(key, []).append(case)

    for shared in by_evidence.values():
        if len(shared) < 2:
            continue
        outcomes = {(case.answerable, case.expected_release) for case in shared}
        assert len(outcomes) == 1, (
            "these cases present identical evidence but their labels disagree, so "
            "at most one is reachable: "
            + ", ".join(f"{c.case_id}={(c.answerable, c.expected_release)}" for c in shared)
        )


def test_a_released_case_can_produce_its_required_answer_terms() -> None:
    """A required term the fixture cannot express makes the case unpassable.

    Only cases expected to release are checked: the harness rates answer terms
    on release (`tv2_readiness_scoring.py:104`), and an abstaining case is not
    asked to produce an answer at all.
    """

    unproducible: list[str] = []
    for case in cases():
        if not case.expected_release or not case.required_answer_terms:
            continue
        spec = case.fixture
        # Everything the case can put into an answer: the metric names it
        # declares, the values the facts carry, and the query itself.
        available = " ".join(
            [str(slot.get("metric") or "") for slot in spec.slots]
            + [str(fact.get("value") or "") for fact in spec.facts.values()]
            + [str(spec.query)]
        ).casefold()
        for term in case.required_answer_terms:
            if str(term).casefold() not in available:
                unproducible.append(f"{case.case_id}: {term!r}")
    assert unproducible == [], (
        "required answer term(s) no fixture field can produce: " + "; ".join(unproducible)
    )


# --- the specific defect -----------------------------------------------------


def test_the_multi_evidence_fixture_supplies_corroboration_not_a_conflict() -> None:
    """The label asks for two *supporting* evidence items for one claim.

    Supporting evidence for one claim states one quantity.  Q1 = 100 and
    Q2 = 80 for the same slot is a conflict, and the correct runtime response to
    a conflict is to abstain -- which the sealed `multi-evidence-002` asks for
    with M1/M2 and already gets.  So a fixture that disagrees here cannot be
    testing corroboration; it is a second copy of the case next door.
    """

    spec = _case_by_id("multi-evidence-001").fixture
    left, right = (spec.facts[name] for name in CORROBORATING_FACT_PAIRS["multi_evidence"])

    keys = [
        quantity_identity(
            fact.get("value"),
            scale=fact.get("scale"),
            unit=fact.get("unit"),
            currency=fact.get("currency"),
        )
        for fact in (left, right)
    ]
    assert keys[0] == keys[1], f"the two supports state different quantities: {keys}"


def test_the_multi_evidence_fixture_uses_independent_sources() -> None:
    """Two rows from one physical source are not corroboration.

    This is the distinction H2A-2B separated: semantic content identity says
    whether two items make the same *claim*, provenance identity says whether
    they are independent *witnesses* to it.  Corroboration needs the second.
    """

    spec = _case_by_id("multi-evidence-001").fixture
    left, right = (spec.facts[name] for name in CORROBORATING_FACT_PAIRS["multi_evidence"])

    left_source = left.get("physical_source_id") or left.get("source_id") or left.get("fact_id")
    right_source = right.get("physical_source_id") or right.get("source_id") or right.get("fact_id")

    assert left_source != right_source, (
        "the shared fixture's two supports come from one physical source, so they "
        "cannot demonstrate corroboration"
    )
