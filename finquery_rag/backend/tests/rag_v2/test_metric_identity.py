"""Metric identity: the ontology's id, or the metric's own exact surface.

The ontology cannot name every metric a filing reports, and refusing to compare
what it cannot name made those slots unsatisfiable while the fact existed --
``Impact of the State Aid Decision`` is a real Apple row, and no amount of
aliasing makes it ``revenue``.  The fallback is the weakest thing that works: the
normalized surface and nothing else.

The two halves are tested together on purpose.  Widening the *comparison* while
keeping the *vocabulary* narrow is the whole design -- the gate must go on saying
it does not recognise a metric while the binder is allowed to bind it -- so a
change that widens both, or neither, has to fail something here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_v2.supervisor import canonical_metric_id, metric_identity

FIXTURE_V5 = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/tv2_canonical_v1/plan-fixtures-v5.jsonl"
)


# --- the vocabulary layer -----------------------------------------------------


@pytest.mark.parametrize(
    "written",
    [
        "Total operating expenses",
        "total operating expenses",
        "  Total   Operating   Expenses  ",
        "Total Operating Expenses.",
        "TOTAL OPERATING EXPENSES",
        "Operating expenses",
    ],
)
def test_the_new_canonical_metric_resolves_stably(written: str) -> None:
    """Case, padding, interior runs and trailing punctuation are all the same
    metric, and none of them is a literal."""

    assert canonical_metric_id(written) == "total_operating_expenses"
    assert metric_identity(written) == "total_operating_expenses"


def test_a_known_metric_keeps_its_ontology_id() -> None:
    """The literal layer never shadows a definition."""

    assert metric_identity("Cost of sales") == "cost_of_revenue"
    assert metric_identity("Total net sales") == "revenue"


# --- the literal layer --------------------------------------------------------


@pytest.mark.parametrize(
    "written",
    [
        "Impact of the State Aid Decision",
        "impact of the state aid decision",
        "  Impact  of  the  State  Aid  Decision ",
        "Impact of the State Aid Decision.",
        "IMPACT OF THE STATE AID DECISION",
    ],
)
def test_an_unnamed_metric_takes_its_own_normalized_surface(written: str) -> None:
    assert metric_identity(written) == "literal:impact_of_the_state_aid_decision"


def test_a_literal_is_exact_only() -> None:
    """No synonyms and no fuzzy distance -- that is the point of calling it exact.

    A nearer string is a *different* literal, not a match.  If this ever starts
    passing by similarity, a metric the corpus holds once could bind a fact it
    has nothing to do with.
    """

    assert metric_identity("Impact of the State Aid") == "literal:impact_of_the_state_aid"
    assert metric_identity("Impact of the State Aid") != metric_identity(
        "Impact of the State Aid Decision"
    )
    assert metric_identity("Impact of State Aid Decision") != metric_identity(
        "Impact of the State Aid Decision"
    )


def test_unrelated_metrics_do_not_share_an_identity() -> None:
    identities = {
        metric_identity(metric)
        for metric in (
            "Cost of sales",
            "Total operating expenses",
            "Impact of the State Aid Decision",
            "iPhone",
            "Statutory federal income tax rate",
        )
    }
    assert len(identities) == 5


def test_an_empty_metric_has_no_identity() -> None:
    for value in (None, "", "   ", "..."):
        assert metric_identity(value) is None


# --- the two layers stay two layers -------------------------------------------


def test_the_vocabulary_still_declines_what_it_cannot_name() -> None:
    """The gate's signal.  Widening the comparison must not silence it.

    ``align_query_to_plan`` reports ``unrecognized_plan_metric`` from
    ``canonical_metric_id``.  If the literal layer were folded into that
    function, every plan metric would resolve and the gate would stop being able
    to say it does not recognise one -- including for an adversarial case whose
    metric was invented by the question.
    """

    assert canonical_metric_id("Impact of the State Aid Decision") is None
    assert canonical_metric_id("Number of Employees") is None
    assert metric_identity("Number of Employees") == "literal:number_of_employees"


# --- what the change was for, checked against the built fixture ----------------


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE_V5.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _by_id() -> dict[str, dict]:
    return {row["id"]: row for row in _rows()}


@pytest.mark.parametrize(
    "case_id,expected_identity",
    [
        # 002's numerator is in the ontology; 004's is a literal.  Testing one
        # of each is the point -- they reach a unique fact by different routes.
        ("tv2f01-s2-pctshare-002", "total_operating_expenses"),
        ("tv2f01-s2-pctshare-004", "literal:impact_of_the_state_aid_decision"),
    ],
)
def test_the_two_named_cases_have_a_unique_fact_for_both_slots(
    case_id: str, expected_identity: str
) -> None:
    """Entity and metric together, which is what the coordinate needed.

    The counts are recorded by the build against the real store, so this asserts
    the property the whole round was for without opening a 50MB file: one
    candidate per slot, and the metric identity that gets there.
    """

    row = _by_id()[case_id]
    assert row["coordinate_candidates"] == {"s1": 1, "s2": 1}

    metrics = [slot["metric"] for slot in row["plan"]["required_slots"]]
    identities = [metric_identity(metric) for metric in metrics]
    assert all(identity is not None for identity in identities)
    assert expected_identity in identities
    assert len(set(identities)) == 2, "the two slots must not name one metric"


def test_every_abstention_row_still_selects_nothing() -> None:
    """The safety property the literal layer could have broken.

    A literal identity is only ever compared, so it cannot conjure a fact.  If
    that were not true, ``Number of Employees`` would resolve to something and an
    unanswerable question would be treated as answerable.
    """

    abstention = [row for row in _rows() if row["stratum"] == "adversarial_abstention"]
    assert len(abstention) == 25
    for row in abstention:
        assert row["expects_no_candidate"] is True
        assert set(row["coordinate_candidates"].values()) == {0}, row["id"]


def test_every_answerable_row_selects_at_least_one_fact() -> None:
    for row in _rows():
        if row["stratum"] == "adversarial_abstention":
            continue
        counts = row["coordinate_candidates"]
        assert counts and all(count >= 1 for count in counts.values()), row["id"]
