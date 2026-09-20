"""A3-W1 acceptance: the contract holds, and it changes nothing.

The behaviour-neutrality half is the part that matters. W1 introduces types; if it also
moves a number, introduction and migration were mixed and a later 47-slot regression could
not be attributed to either.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from src.pdf_retrieval_v4.period_binding import (  # noqa: E402
    DIRECT_DECLARATIONS,
    INHERITED_CONTEXT,
    Conflict,
    PeriodBindingMethod,
    PeriodBindingStatus,
    PeriodBindingV2,
    PeriodGranularity,
    PeriodTargetScope,
    SourceCell,
    TemporalKind,
    TemporalKindEvidence,
    TemporalKindMethod,
    periods_are_compatible,
    resolve_period_evidence,
)


def _cell(row: int, column: int, text: str = "") -> SourceCell:
    return SourceCell(document_id="ko_fy2025", table_id="table_x", row=row,
                      column=column, text=text)


def _year_binding(year: str = "2025") -> PeriodBindingV2:
    return PeriodBindingV2(
        normalized_period=year,
        granularity=PeriodGranularity.YEAR,
        status=PeriodBindingStatus.PARTIAL,
        method=PeriodBindingMethod.YEAR_ONLY_PERIOD,
        target_scope=PeriodTargetScope.CELL_GROUP,
        source_cells=(_cell(1, 3, year),),
        temporal=TemporalKindEvidence(kind=TemporalKind.UNKNOWN,
                                      method=TemporalKindMethod.PERIOD_BINDING),
    )


def _day_binding(period: str = "2022-12-31") -> PeriodBindingV2:
    return PeriodBindingV2(
        normalized_period=period,
        granularity=PeriodGranularity.DAY,
        status=PeriodBindingStatus.RESOLVED,
        method=PeriodBindingMethod.INLINE_PERIOD_DATA_ROW,
        target_scope=PeriodTargetScope.ROW,
        source_cells=(_cell(15, 0, "Balance, December 31, 2022"),),
        temporal=TemporalKindEvidence(kind=TemporalKind.POINT,
                                      method=TemporalKindMethod.PERIOD_BINDING),
    )


# --- the two enums are different enums ---------------------------------------------

def test_status_and_temporal_kind_do_not_share_a_word_for_a_failure():
    """`UNRESOLVED` is a binding that found nothing; `UNKNOWN` is a kind with no shape."""
    assert PeriodBindingStatus.UNRESOLVED.value == "UNRESOLVED"
    assert TemporalKind.UNKNOWN.value == "UNKNOWN"
    assert "UNRESOLVED" not in {k.value for k in TemporalKind}
    assert "UNKNOWN" not in {s.value for s in PeriodBindingStatus}


def test_partial_is_a_usable_binding_and_not_a_failure():
    """Coca-Cola's equity statement: identity resolved, shape unstated."""
    binding = _year_binding()
    assert binding.status is PeriodBindingStatus.PARTIAL
    assert binding.temporal.kind is TemporalKind.UNKNOWN
    assert binding.is_usable


# --- direct declaration shadows inherited context ----------------------------------

def test_direct_declaration_shadows_inherited_context():
    declared, inherited = _day_binding(), _year_binding()
    chosen = resolve_period_evidence(declared, [inherited])
    assert chosen is declared
    assert chosen.target_scope is PeriodTargetScope.ROW


def test_shadowing_is_scope_semantics_not_an_ordering_over_methods():
    """No function here may rank method values; the enum is provenance, not authority."""
    assert DIRECT_DECLARATIONS.isdisjoint(INHERITED_CONTEXT)
    assert set(DIRECT_DECLARATIONS) | set(INHERITED_CONTEXT) == set(PeriodBindingMethod)


def test_two_inherited_bindings_that_agree_are_merged_not_chosen_between():
    a = PeriodBindingV2(
        normalized_period="2025-12-31", granularity=PeriodGranularity.DAY,
        status=PeriodBindingStatus.RESOLVED, method=PeriodBindingMethod.DIRECT_HEADER,
        target_scope=PeriodTargetScope.COLUMN, source_cells=(_cell(1, 3),))
    b = PeriodBindingV2(
        normalized_period="2025-12-31", granularity=PeriodGranularity.DAY,
        status=PeriodBindingStatus.RESOLVED,
        method=PeriodBindingMethod.ADJACENT_YEAR_JOIN,
        target_scope=PeriodTargetScope.COLUMN, source_cells=(_cell(1, 4),))
    merged = resolve_period_evidence(None, [a, b])
    assert isinstance(merged, PeriodBindingV2)
    assert len(merged.source_cells) == 2


def test_two_inherited_bindings_that_disagree_are_a_conflict():
    a = PeriodBindingV2(
        normalized_period="2025-12-31", granularity=PeriodGranularity.DAY,
        status=PeriodBindingStatus.RESOLVED, method=PeriodBindingMethod.DIRECT_HEADER,
        target_scope=PeriodTargetScope.COLUMN)
    b = PeriodBindingV2(
        normalized_period="2024-12-31", granularity=PeriodGranularity.DAY,
        status=PeriodBindingStatus.RESOLVED, method=PeriodBindingMethod.ADJACENT_YEAR_JOIN,
        target_scope=PeriodTargetScope.COLUMN)
    outcome = resolve_period_evidence(None, [a, b])
    assert isinstance(outcome, Conflict)
    assert len(outcome.candidates) == 2


def test_a_conflict_is_a_binding_outcome_and_carries_no_single_certain_period():
    conflicted = PeriodBindingV2(status=PeriodBindingStatus.CONFLICT,
                                 normalized_period=None)
    assert not conflicted.is_usable


# --- compatibility is granularity-aware, not string equality ------------------------

def test_a_year_answers_a_fiscal_year_request():
    assert periods_are_compatible(_year_binding("2025"), PeriodGranularity.YEAR, "2025")


def test_a_year_does_not_answer_a_day_request():
    """The rule the seal recorded: a year does not pin a day."""
    assert not periods_are_compatible(_year_binding("2025"), PeriodGranularity.DAY,
                                      "2025-12-31")


def test_a_day_answers_a_coarser_request():
    assert periods_are_compatible(_day_binding("2025-12-31"), PeriodGranularity.YEAR,
                                  "2025")


def test_an_unusable_binding_answers_nothing():
    assert not periods_are_compatible(None, PeriodGranularity.YEAR, "2025")
    assert not periods_are_compatible(PeriodBindingV2(), PeriodGranularity.YEAR, "2025")


# --- serialisation round-trip -------------------------------------------------------

@pytest.mark.parametrize("binding", [_year_binding(), _day_binding(),
                                     PeriodBindingV2(status=PeriodBindingStatus.CONFLICT,
                                                     conflict_candidates=(_day_binding(),))])
def test_round_trip_preserves_the_provenance(binding):
    assert PeriodBindingV2.from_dict(binding.to_dict()) == binding


def test_round_trip_keeps_the_two_enums_apart():
    restored = PeriodBindingV2.from_dict(_year_binding().to_dict())
    assert restored.status is PeriodBindingStatus.PARTIAL
    assert restored.temporal.kind is TemporalKind.UNKNOWN


# --- W1 is behaviour-neutral --------------------------------------------------------

def test_nothing_in_the_pipeline_imports_this_module_yet():
    """W1 lands the contract; W2 starts populating it.

    If a pipeline module already imports this, W1 stopped being a contract commit and
    the 47-slot result can no longer be attributed between introduction and migration.
    """
    roots = [_BACKEND_DIR / "src" / "pdf_retrieval_v4",
             _BACKEND_DIR / "src" / "runtime",
             _BACKEND_DIR / "scripts" / "evaluation"]
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            if path.name == "period_binding.py":
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if "period_binding import" in text or "from .period_binding" in text:
                offenders.append(str(path.relative_to(_BACKEND_DIR)))
    assert offenders == [], f"W1 must not be imported yet: {offenders}"


def test_the_sealed_counts_are_untouched_by_this_commit():
    """The commit that introduced this module must not have touched a behaviour file.

    An earlier version of this test diffed `HEAD~1..HEAD`, which passes for the wrong
    reason in any checkout where HEAD is not that commit -- the diff is empty and the
    assertion holds without checking anything.  A test that is green because it looked at
    nothing is worse than no test, because the green gets read as evidence.

    So the commit is *found*, not assumed, and the test skips with a stated reason when it
    cannot be found rather than passing quietly.
    """
    module = _BACKEND_DIR / "src" / "pdf_retrieval_v4" / "period_binding.py"
    if not module.is_file():
        pytest.skip("W1 module is not in this checkout")

    found = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%H", "--",
         str(module.relative_to(_BACKEND_DIR))],
        cwd=_BACKEND_DIR, capture_output=True, text=True,
    )
    if found.returncode != 0 or not found.stdout.strip():
        pytest.skip("cannot find the commit that added period_binding.py: "
                    "no git history for it in this checkout")

    commit = found.stdout.split()[0]
    changed = subprocess.run(
        ["git", "show", "--name-only", "--format=", commit],
        cwd=_BACKEND_DIR, capture_output=True, text=True,
    ).stdout.split()
    assert changed, f"commit {commit} reports no files; the check would be vacuous"

    behaviour_paths = ("run_nf_v2_17a4_parse", "build_store_v2",
                       "resolve_store_v2_slot", "trusted_v2_canonical_fact_store",
                       "html_semantic_adapter", "temporal_axis_graph",
                       "typed_evidence_emitters")
    touched = [p for p in changed if any(b in p for b in behaviour_paths)]
    assert touched == [], f"W1 touched behaviour-bearing files: {touched}"
