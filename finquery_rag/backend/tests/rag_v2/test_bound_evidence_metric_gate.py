"""The bound-evidence gate must not judge a metric the ontology cannot name.

The gate builds its query-side and plan-side metric sets by scanning for
ontology *aliases*, so those sets can only ever hold metrics the ontology names.
A fact whose metric is a literal identity is therefore never in them -- and two
branches read that as "this fact is about something the query never asked
about".

The failure needs a *mix* to appear, which is why it stayed hidden for so long:
with every metric in a question unnamed, the frame set is empty, the check is
skipped, and the case passes.  Naming one metric makes the set non-empty, the
check fires, and the fact for the metric that is still unnamed is rejected --
over a query that names both.  That is `pctshare-001`, and `pctshare-004` is the
same shape one branch over.
"""

from __future__ import annotations

from rag_v2.contracts.plan import Action, Intent, RequiredSlot, SupervisorPlan
from rag_v2.supervisor import align_bound_evidence_to_query


def _plan(metric: str, *, intent: Intent = Intent.DIRECT_FACT) -> SupervisorPlan:
    return SupervisorPlan(
        intent=intent,
        required_slots=(
            RequiredSlot(
                slot_id="s1",
                metric=metric,
                period="FY2025",
                role="value",
                value_type="numeric",
                entity="Apple",
            ),
        ),
        operation=None,
        next_action=Action.RETRIEVE,
    )


def _fact(metric: str, value: str = "15,091") -> dict:
    return {
        "fact_id": "atomic:x",
        "metric": metric,
        "period": "FY2025",
        "value": value,
        "entity": "Apple",
    }


def _mismatches(query: str, plan: SupervisorPlan, metric: str) -> tuple[str, ...]:
    check = align_bound_evidence_to_query(
        query,
        plan,
        [_fact(metric)],
        {"s1": ["atomic:x"]},
    )
    return check.mismatches


def test_a_named_metric_in_the_query_passes() -> None:
    """"...Apple's Total operating expenses..." -- named on both sides."""

    mismatches = _mismatches(
        "What was Apple's Total operating expenses in FY2025?",
        _plan("Total operating expenses"),
        "Total operating expenses",
    )

    assert not any("fact_metric_not_in" in item for item in mismatches), mismatches


def test_an_unnamed_metric_is_not_rejected_by_a_frame_that_names_another() -> None:
    """`pctshare-001`: the query names both, the ontology names one.

    The Leasehold fact matches its slot exactly; it is refused only because the
    query-side set, built from aliases, cannot hold a metric the ontology has
    never heard of.
    """

    mismatches = _mismatches(
        "What percentage of Apple's Total operating expenses was Leasehold improvements in FY2025?",
        _plan("Leasehold improvements"),
        "Leasehold improvements",
    )

    assert not any("fact_metric_not_in" in item for item in mismatches), mismatches


def test_an_unnamed_metric_is_not_rejected_in_the_plan_branch_either() -> None:
    """The same defect one branch over, which is `pctshare-004`'s shape."""

    mismatches = _mismatches(
        "For Apple in FY2025, what is Cost of sales as a share of Leasehold improvements?",
        _plan("Leasehold improvements", intent=Intent.CALCULATION),
        "Leasehold improvements",
    )

    assert not any("fact_metric_not_in" in item for item in mismatches), mismatches


def test_a_named_metric_the_query_never_asked_about_is_still_rejected() -> None:
    """The check still does its job.  This is a guard, not a removal."""

    mismatches = _mismatches(
        "What was Apple's Total operating expenses in FY2025?",
        _plan("Total operating expenses"),
        "Net income",
    )

    assert any("fact_metric_not_in_query" in item for item in mismatches), mismatches
