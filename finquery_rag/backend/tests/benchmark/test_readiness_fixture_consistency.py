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

import json
from typing import Any

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


def _evidence_key(spec: Any) -> str:
    """What the runtime is given, ignoring the question text."""

    return json.dumps(
        {"slots": spec.slots, "facts": spec.facts, "routes": spec.routes},
        sort_keys=True,
        default=str,
    )


def _metric_vocabulary() -> set[str]:
    """Every metric name any fixture declares, lower-cased.

    Used to tell an answer term that *names a metric* from one that is a value
    or a word the deterministic renderer contributes on its own.  `Revenue` is
    a metric name; `100` and `Growth Rate` are not.
    """

    names: set[str] = set()
    for case in cases():
        for slot in case.fixture.slots:
            metric = slot.get("metric")
            if metric:
                names.add(str(metric).casefold())
    return names


def test_cases_presenting_identical_evidence_agree_on_whether_it_answers() -> None:
    """A tripwire, not a theorem: identical evidence, opposite labels.

    `multi_evidence` and `qualitative` carried identical slots, identical facts
    and identical retrieval routes under opposite labels -- abstain versus
    release -- with only the query text differing.  This runtime's release
    decision is driven by whether the required slot is supported by admissible
    evidence, so identical evidence cannot produce opposite outcomes: at most
    one of the two labels is reachable, and the mismatch count was guaranteed
    non-zero before any runtime behaviour was considered.

    If this fires, a human has to say which label the evidence is meant to
    demonstrate.  It is not something a runtime can be asked to reconcile.
    """

    by_evidence: dict[str, list[Any]] = {}
    for case in cases():
        by_evidence.setdefault(_evidence_key(case.fixture), []).append(case)

    for shared in by_evidence.values():
        if len(shared) < 2:
            continue
        outcomes = {(case.answerable, case.expected_release) for case in shared}
        assert len(outcomes) == 1, (
            "these cases present identical evidence but their labels disagree, so "
            "at most one is reachable: "
            + ", ".join(f"{c.case_id}={(c.answerable, c.expected_release)}" for c in shared)
        )


def test_a_released_case_declares_every_metric_its_answer_terms_name() -> None:
    """A required term naming a metric needs a fact carrying that metric.

    `multi-evidence-001` required the answer to contain "Revenue" while its
    fixture declared one `operating_margin` slot and no Revenue fact: the term
    was unreachable by construction and no binding change could produce it.

    Scoped to terms that name a metric, because a required term may instead be a
    value the fixture carries or a word the deterministic renderer emits on its
    own (`calculation-growth-001` requires "Growth", which the calculation
    renderer writes as "Growth Rate:").  Only the metric case is a statement
    about the fixture's ability to express its label.
    """

    vocabulary = _metric_vocabulary()
    unproducible: list[str] = []
    for case in cases():
        if not case.expected_release or not case.required_answer_terms:
            continue
        declared = {
            str(slot.get("metric") or "").casefold() for slot in case.fixture.slots
        } | {
            str(fact.get("metric") or "").casefold()
            for fact in case.fixture.facts.values()
        }
        for term in case.required_answer_terms:
            name = str(term).casefold()
            if name in vocabulary and name not in declared:
                unproducible.append(f"{case.case_id}: {term!r}")
    assert unproducible == [], (
        "required answer term(s) name a metric this fixture does not declare: "
        + "; ".join(unproducible)
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
