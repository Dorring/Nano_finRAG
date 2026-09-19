"""P1.5-1: `RequiredSlot[]` is the only retrieval cardinality authority.

`SupervisorPlan.required_slots` says how many facts a question needs.  The
retrieval layer used to re-derive that number from the question text --
`route_question` -> `profile.task_type` -> `build_operand_slots` -> the length of
`QueryPlan.operand_slots` -- and for every cross-entity comparison and ranking
that re-derivation answered 1.  A four-company ranking therefore shared one
retrieval lane and one top-40, and 0/21 gold facts reached the packet while
157/157 sat in the index.

These tests are structural, in the sense `tests/architecture/
test_harness_boundaries.py` means it: **absence is asserted on the syntax tree,
never on the source text**, because a comment naming what was removed must not
be able to satisfy a guard.

The defect this prevents is not "the classifier was wrong".  It is that two
components both claimed the right to decide how many facts a question needs.
Fixing the classifier would teach it one more case and leave the next one --
three-period average, two-entity ratio, mixed entity+period comparison -- to be
taught again, because the answer was already known upstream.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from rag_v2.contracts.plan import Action, Intent, RequiredSlot, SupervisorPlan
from src.pdf_retrieval_v4.retrieval_demand import SlotRetrievalRequestV1
from src.runtime.trusted_v2_r4 import build_slot_retrieval_requests

BACKEND_ROOT = Path(__file__).resolve().parents[2]

#: Names that used to decide the lane count, and must not be able to again.
#: `operand_slots` is the field, `is_multi_slot` the local derived from it, and
#: `task_type` the classifier upstream of both.
CARDINALITY_NAMES = frozenset({"operand_slots", "is_multi_slot", "task_type"})

#: The functions that produce lanes.  Scoped to these rather than the whole
#: module because the legacy `CandidateDirectRetriever.retrieve(plan)` and the
#: policy's slot-id helper are deliberately kept for callers that still use
#: them; what must not read a QueryPlan is the path that decides lane count.
#:
#: `CandidateDirectR4Policy.retrieve` is listed with its class because the
#: module also defines `R4RetrievalCapability.retrieve`, and `ast.walk` would
#: silently pick the first one named `retrieve` -- a scan that reads the wrong
#: function is the vacuous kind this file is written to avoid.
CARDINALITY_PRODUCERS = (
    (BACKEND_ROOT / "src" / "pdf_retrieval_v4" / "retrieval_demand.py", "query", None),
    (
        BACKEND_ROOT / "src" / "pdf_retrieval_v4" / "candidate_direct_retriever.py",
        "retrieve_for_requests",
        None,
    ),
    (
        BACKEND_ROOT / "src" / "runtime" / "trusted_v2_r4.py",
        "build_slot_retrieval_requests",
        None,
    ),
    (
        BACKEND_ROOT / "src" / "runtime" / "trusted_v2_r4.py",
        "retrieve",
        "CandidateDirectR4Policy",
    ),
)


def _slot(
    slot_id: str,
    *,
    metric: str = "Comprehensive income",
    period: str = "FY2025",
    entity: str | None = None,
    entity_id: str | None = None,
) -> RequiredSlot:
    return RequiredSlot(
        slot_id=slot_id,
        metric=metric,
        period=period,
        role="value",
        value_type="numeric",
        entity=entity,
        entity_id=entity_id,
    )


def _plan(*slots: RequiredSlot) -> SupervisorPlan:
    return SupervisorPlan(
        intent=Intent.MULTI_EVIDENCE,
        required_slots=tuple(slots),
        operation=None,
        next_action=Action.RETRIEVE,
    )


# --- the invariant ------------------------------------------------------------


def test_two_slots_give_two_requests() -> None:
    plan = _plan(_slot("s1", entity="Visa"), _slot("s2", entity="Mastercard"))

    requests = build_slot_retrieval_requests(plan)

    assert len(requests) == len(plan.required_slots) == 2


def test_four_slots_give_four_requests() -> None:
    """`rank-003`'s shape: four companies, four lanes.

    This is the case that measured `operand_slots == 1` in production.
    """

    companies = ("The Coca-Cola Company", "Visa", "Tesla", "Microsoft")
    plan = _plan(*[_slot(f"s{i}", entity=name) for i, name in enumerate(companies, 1)])

    requests = build_slot_retrieval_requests(plan)

    assert len(requests) == 4
    assert [item.entity for item in requests] == list(companies)


def test_the_request_order_follows_the_plan() -> None:
    plan = _plan(_slot("a", entity="Visa"), _slot("b", entity="Tesla"))

    requests = build_slot_retrieval_requests(plan)

    assert [item.slot_id for item in requests] == ["a", "b"]


def test_slot_order_does_not_change_the_lane_set() -> None:
    """Reordering the plan reorders the lanes; it does not add or drop one."""

    first = build_slot_retrieval_requests(
        _plan(_slot("s1", entity="Visa"), _slot("s2", entity="Tesla"))
    )
    reversed_ = build_slot_retrieval_requests(
        _plan(_slot("s2", entity="Tesla"), _slot("s1", entity="Visa"))
    )

    assert sorted(item.slot_id for item in first) == sorted(
        item.slot_id for item in reversed_
    )
    assert {item.query for item in first} == {item.query for item in reversed_}


def test_slots_that_differ_only_by_slot_id_stay_separate_lanes() -> None:
    """Two identical coordinates are two demands, not one collapsed demand.

    Asserted as *behaviour that holds* rather than as a rejection, because the
    plan contract does not currently reject them -- `SupervisorPlan.__post_init__`
    enforces unique `slot_id` values and nothing else.  A guard that refused two
    semantically identical slots is a reasonable later addition; what must not
    happen meanwhile is the retrieval layer quietly deduping them, which would
    answer a question the plan asked twice.
    """

    plan = _plan(_slot("s1", entity="Visa"), _slot("s2", entity="Visa"))

    requests = build_slot_retrieval_requests(plan)

    assert len(requests) == 2
    assert requests[0].query == requests[1].query


# --- what must not reach the lane count ---------------------------------------


def test_a_wrong_task_type_cannot_reduce_lane_count() -> None:
    """No retrieval-side classification can arrive here to change the count.

    Asserted on the signature rather than on a value: there is no parameter
    through which a `task_type`, a `QueryPlan` or an `operand_slots` could be
    passed, so a future edit that wanted to consult one would have to change
    this signature -- which is the point at which it becomes visible.
    """

    parameters = list(inspect.signature(build_slot_retrieval_requests).parameters)

    assert parameters == ["plan"], parameters


def test_a_slot_query_carries_no_entity_at_all() -> None:
    """It used to carry its own entity, and that was the P1.5-R regression.

    The lanes tokenise a query as an OR expression, so a company name adds
    tokens matching every row for that company and dilutes the ranking until the
    specific fact falls outside `slot_top_k`.  Five single-entity cases went
    from correct releases to fail-closed because of it, and removing it restored
    all five while *improving* cross-entity reachability (P1.5-R1).

    The property this test was originally written for -- that no slot's query
    names *another* slot's entity -- is now satisfied by every query naming
    none, which is why it is asserted in that form rather than deleted.
    """

    companies = ("Tesla", "JPMorganChase", "Visa")
    plan = _plan(*[_slot(f"s{i}", entity=name) for i, name in enumerate(companies, 1)])

    for request in build_slot_retrieval_requests(plan):
        assert not any(name in request.query for name in companies), request.query
        assert request.query == f"{request.metric} {request.period}"


def test_a_slot_without_an_entity_still_queries() -> None:
    """A single-company question has no entity to add, and must not fail."""

    request = build_slot_retrieval_requests(_plan(_slot("s1")))[0]

    assert request.query == "Comprehensive income FY2025"


def test_an_unnamed_entity_is_still_a_constraint() -> None:
    """`entity_id is None` means the ontology cannot name the mention.

    It does not mean there is no entity constraint, and a lane that read it that
    way would match every company reporting the metric.
    """

    request = build_slot_retrieval_requests(
        _plan(_slot("s1", entity="Pfizer", entity_id=None))
    )[0]

    assert request.entity == "Pfizer"
    assert request.entity_id is None
    # The constraint lives on the slot, not in its query text.  This asserted
    # `query.startswith("Pfizer ")` until P1.5-R1 removed the entity from the
    # query; the constraint it was protecting is unchanged and still asserted
    # above, and the query is now the metric and period whoever the slot is for.
    assert request.query == "Comprehensive income FY2025"


# --- the tripwire -------------------------------------------------------------


def _referenced_names(path: Path, function: str, in_class: str | None = None) -> set[str]:
    """Every attribute/name identifier read inside one function.

    Syntax tree, not source text: a comment or a docstring naming a removed
    field must not satisfy this, and must not fail it either.

    ``in_class`` disambiguates modules that define the same method name twice;
    without it the first match wins and the scan can read a function nobody
    asked about.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    scopes: list[ast.AST] = list(ast.walk(tree))
    if in_class is not None:
        owners = [
            node
            for node in scopes
            if isinstance(node, ast.ClassDef) and node.name == in_class
        ]
        if not owners:
            raise AssertionError(f"class {in_class} not found in {path}")
        scopes = [inner for owner in owners for inner in ast.walk(owner)]

    for node in scopes:
        if isinstance(node, ast.FunctionDef) and node.name == function:
            found: set[str] = set()
            for inner in ast.walk(node):
                if isinstance(inner, ast.Attribute):
                    found.add(inner.attr)
                elif isinstance(inner, ast.Name):
                    found.add(inner.id)
            return found
    raise AssertionError(f"{function} not found in {path}")


def test_the_scan_finds_a_function_that_does_read_them(tmp_path: Path) -> None:
    """Vacuity guard: the tripwire must be able to fail.

    Without this, `_referenced_names` returning an empty or truncated set would
    make the assertion below pass for the wrong reason -- which is the failure
    mode every other guard in this suite is written to avoid.
    """

    offending = tmp_path / "offending.py"
    offending.write_text(
        "def retrieve_for_requests(plan):\n"
        "    return len(plan.operand_slots)\n",
        encoding="utf-8",
    )

    assert "operand_slots" in _referenced_names(offending, "retrieve_for_requests")


def test_the_scan_reads_the_class_it_was_asked_for(tmp_path: Path) -> None:
    """The disambiguation guard: two methods, one name, the right one scanned."""

    offending = tmp_path / "twinned.py"
    offending.write_text(
        "class A:\n"
        "    def retrieve(self, plan):\n"
        "        return len(plan.operand_slots)\n"
        "\n"
        "class B:\n"
        "    def retrieve(self, plan):\n"
        "        return 1\n",
        encoding="utf-8",
    )

    assert "operand_slots" in _referenced_names(offending, "retrieve", "A")
    assert "operand_slots" not in _referenced_names(offending, "retrieve", "B")


@pytest.mark.parametrize(
    ("path", "function", "in_class"),
    CARDINALITY_PRODUCERS,
    ids=[
        f"{path.name}::{in_class or ''}{function}"
        for path, function, in_class in CARDINALITY_PRODUCERS
    ],
)
def test_lane_count_does_not_read_the_query_plan(
    path: Path, function: str, in_class: str | None
) -> None:
    """The tripwire.

    A `QueryPlan` field becoming load-bearing here means the retrieval layer has
    gone back to deciding how many facts the question needs.
    """

    found = _referenced_names(path, function, in_class)

    assert found, f"{path.name}::{function} yielded nothing to scan"
    assert not (found & CARDINALITY_NAMES), (
        f"{path.name}::{in_class or ''}{function} reads "
        f"{sorted(found & CARDINALITY_NAMES)}; lane count must come from RequiredSlot[]"
    )


def test_the_policy_retrieve_is_the_one_being_scanned() -> None:
    """Proves the scanned `retrieve` is the policy's, not the capability's.

    The capability's `retrieve` is a different method in the same module and
    does not call `build_slot_retrieval_requests`.  If the scan had picked it,
    the test above would be asserting about a function that never decides
    anything.
    """

    policy = _referenced_names(
        BACKEND_ROOT / "src" / "runtime" / "trusted_v2_r4.py",
        "retrieve",
        "CandidateDirectR4Policy",
    )
    capability = _referenced_names(
        BACKEND_ROOT / "src" / "runtime" / "trusted_v2_r4.py",
        "retrieve",
        "R4RetrievalCapability",
    )

    assert "build_slot_retrieval_requests" in policy
    assert "build_slot_retrieval_requests" not in capability
    assert policy != capability


def test_the_scan_covers_every_producer() -> None:
    """A producer added without being listed here is not covered by the tripwire."""

    assert len(CARDINALITY_PRODUCERS) == 4
    for path, function, in_class in CARDINALITY_PRODUCERS:
        assert path.exists(), path
        assert _referenced_names(path, function, in_class)


def test_the_request_type_carries_the_slot_coordinates() -> None:
    """The demand is a copy of the slot, so it must be able to hold all of it.

    `OperandSlot` could not hold an entity, which is what forced the plan-wide
    re-attachment.  This pins the replacement to the coordinates that matter.
    """

    fields = set(SlotRetrievalRequestV1.__dataclass_fields__)

    assert {
        "slot_id",
        "metric",
        "period",
        "role",
        "value_type",
        "unit",
        "entity",
        "entity_id",
    } <= fields
