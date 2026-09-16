"""NF-V3 H1.1: a budget field that is declared but not enforced.

``AdaptiveRAGBudgetV1.max_identical_query_retry`` is configurable, is read from
``V2_MAX_IDENTICAL_QUERY_RETRIES`` in production, and is asserted in the V2-16
contract test -- but no control-flow decision reads it.

H1 must not start enforcing it.  ``BoundedReplannerV1`` reuses the same query
text for ``MISSING_SLOT``, so enforcing the field at its default of 0 would
forbid a retry the legacy runtime has always performed.  That is a retrieval
behaviour change arriving through a budget, in an ablation whose only claim is
that ``harness_v3`` matches ``legacy``.

These tests are tripwires, not behaviour tests.  The moment the field becomes
load-bearing they fail, which is the signal to enforce it deliberately, move it
out of ``RESERVED_FIELDS`` and update the note in ``adaptive_budget.py``.
"""

from __future__ import annotations

from pathlib import Path

from rag_v2.adaptive import AdaptiveRAGBudgetV1, BoundedAdaptiveRAGV1
from tests.harness.harness_support import loop_state

BACKEND_ROOT = Path(__file__).resolve().parents[2]
#: The control plane: the loop and the code that drives it.  Anything here that
#: reads a reserved field has started enforcing it.
CONTROL_PLANE = (
    BACKEND_ROOT / "rag_v2" / "adaptive",
    BACKEND_ROOT / "src" / "runtime" / "trusted_v2_coordinator.py",
    BACKEND_ROOT / "src" / "runtime" / "trusted_v2_factory.py",
)


def _run_with_identical_query_retry(limit: int) -> tuple[list[str], str, int]:
    """Drive the loop into retrying one byte-identical query."""

    queries: list[str] = []

    def empty_tool(query: str, state: object) -> list[dict]:
        queries.append(query)
        return []

    budget = AdaptiveRAGBudgetV1(
        max_replan_rounds=3,
        max_total_tool_calls=4,
        max_same_tool_retry=3,
        max_identical_query_retry=limit,
    )
    result = BoundedAdaptiveRAGV1(budget=budget).run(
        loop_state(), {"SEMANTIC_RETRIEVAL": empty_tool}
    )
    return queries, result.state.status, result.state.tool_calls


def test_the_fixture_actually_repeats_an_identical_query() -> None:
    """Without this the comparison below would be vacuous."""

    queries, _, tool_calls = _run_with_identical_query_retry(0)

    assert tool_calls > 1, "the loop never retried, so the field was untested"
    assert len(set(queries)) == 1, queries


def test_the_bound_does_not_change_what_the_loop_does() -> None:
    default = _run_with_identical_query_retry(0)
    permissive = _run_with_identical_query_retry(5)

    assert default == permissive


def test_no_adaptive_decision_reads_a_reserved_budget_field() -> None:
    """The enforcement tripwire.

    A field that is read anywhere in the control plane stops being reserved.
    """

    sources: dict[str, str] = {}
    for entry in CONTROL_PLANE:
        paths = sorted(entry.rglob("*.py")) if entry.is_dir() else [entry]
        for path in paths:
            if path.name != "adaptive_budget.py":
                sources[path.name] = path.read_text(encoding="utf-8")
    assert sources, "the tripwire found nothing to scan"

    for name in AdaptiveRAGBudgetV1.RESERVED_FIELDS:
        readers = sorted(path for path, text in sources.items() if name in text)
        assert not readers, f"{name} is read by {readers}; it is no longer reserved"


def test_reserved_settings_are_reported_when_configured_away_from_default() -> None:
    default = AdaptiveRAGBudgetV1()
    assert default.unenforced_settings() == {}

    configured = AdaptiveRAGBudgetV1(max_identical_query_retry=5)
    assert configured.unenforced_settings() == {"max_identical_query_retry": 5}
