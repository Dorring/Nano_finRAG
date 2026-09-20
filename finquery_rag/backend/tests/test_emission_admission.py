"""A3-W4-A acceptance: admission and temporal kind are now separable questions.

Pure contract tests -- no corpus, no filing on disk.  W4-A is behaviour-neutral by
construction: the legacy rule in `typed_evidence_emitters` is still the authoritative one
and nothing here is imported by it, so what these tests pin is the *meaning* of the new
decision, not its effect.  The effect is measured separately, as a delta.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from src.pdf_retrieval_v4.period_binding import (  # noqa: E402
    DISAGGREGATION_KINDS,
    AdmissionOutcome,
    AdmissionReason,
    AdmissionRequest,
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
    decide_emission_admission,
    periods_are_compatible,
    temporal_kind_of,
)


def _cell(row: int = 15, text: str = "Balance, December 31, 2022") -> SourceCell:
    return SourceCell("ko_fy2025", "table_x", row, 3, text)


def _resolved(period: str = "2025-12-31") -> PeriodBindingV2:
    return PeriodBindingV2(
        normalized_period=period, granularity=PeriodGranularity.DAY,
        status=PeriodBindingStatus.RESOLVED,
        method=PeriodBindingMethod.DIRECT_HEADER,
        target_scope=PeriodTargetScope.COLUMN, source_cells=(_cell(),))


def _year_only(year: str = "2025") -> PeriodBindingV2:
    """Coca-Cola's equity statement: the source states a year and no shape."""
    return PeriodBindingV2(
        normalized_period=year, granularity=PeriodGranularity.YEAR,
        status=PeriodBindingStatus.PARTIAL,
        method=PeriodBindingMethod.YEAR_ONLY_PERIOD,
        target_scope=PeriodTargetScope.CELL_GROUP, source_cells=(_cell(),),
        temporal=TemporalKindEvidence(TemporalKind.UNKNOWN,
                                      TemporalKindMethod.PERIOD_BINDING))


def _request(binding, kind: TemporalKind = TemporalKind.DURATION, **overrides
             ) -> AdmissionRequest:
    kwargs = {"binding": binding, "temporal_kind": kind, "metric_path": "net_income",
              "metric_status": "resolved", "value_normalized": "1234",
              "cell_id": "cell_1"}
    kwargs.update(overrides)
    return AdmissionRequest(**kwargs)


# --- the decoupling -------------------------------------------------------------------

def test_an_unstated_shape_is_not_a_reason_to_withhold():
    """The defect W4 exists for, stated as a contract.

    Coca-Cola's column resolves to `YEAR(2025)` with no shape the source ever stated.
    The legacy dropped it because `UNKNOWN` is not one of `point`/`duration`/`comparison`
    -- a completeness verdict reached from a shape that was simply never declared.
    """
    admission = decide_emission_admission(_request(_year_only(), TemporalKind.UNKNOWN))
    assert admission.admitted, admission.reason
    assert admission.reason is None
    assert admission.normalized_period == "2025"


def test_a_kind_alone_never_moves_the_decision():
    """Same binding, same provenance, every non-disaggregation kind: same outcome.

    If this ever fails, admission has started reading the shape again and the two layers
    have re-welded.
    """
    kinds = [TemporalKind.POINT, TemporalKind.DURATION, TemporalKind.YEAR,
             TemporalKind.UNKNOWN, None]
    outcomes = {decide_emission_admission(_request(_resolved(), kind)).outcome
                for kind in kinds}
    assert outcomes == {AdmissionOutcome.ADMIT}


def test_admission_cannot_see_table_role():
    """Authority is not completeness.

    The field is absent rather than merely unused, and this asserts the exact field set:
    adding `table_role` later is precisely how the separation would be undone, and
    nothing else in the suite would notice.
    """
    assert set(AdmissionRequest.__dataclass_fields__) == {
        "binding", "temporal_kind", "metric_path", "metric_status",
        "value_normalized", "cell_id",
    }
    assert not hasattr(_request(_resolved()), "table_role")


def test_the_legacy_vocabulary_bridges_to_the_enum():
    """`unknown` and `UNKNOWN` are the same column shape, spelled by two layers.

    `TemporalKind("unknown")` raises, so a caller that did not normalise would read every
    legacy `unknown` as "no kind at all" -- the population W4 exists for, silently looking
    like a missing axis.  Nothing else in the suite would have caught it.
    """
    for name in ("point", "duration", "comparison", "segment", "bucket", "category",
                 "non_temporal", "unknown", "Unknown", "UNKNOWN", "year", "YEAR"):
        assert temporal_kind_of(name) is not None, name
    assert temporal_kind_of("unknown") is TemporalKind.UNKNOWN
    assert temporal_kind_of("UNKNOWN") is TemporalKind.UNKNOWN
    assert temporal_kind_of("non_temporal") is TemporalKind.NON_TEMPORAL

    # `None` in, `None` out; and an unmappable string is `None` too, which callers must
    # read as fail-closed rather than as permission.
    assert temporal_kind_of(None) is None
    assert temporal_kind_of("") is None
    assert temporal_kind_of("something_new") is None


def test_the_new_rule_differs_from_the_old_one_on_exactly_unknown():
    """W4's rule change is one kind wide, and this is what makes that checkable.

    The old expression is `kind in {point, duration, comparison}`; the new one is `kind
    not in {segment, bucket, category, non_temporal, comparison}`.  Reading the two sets
    against each other leaves exactly one kind on each side that the other does not have:

        only the old eligible    point, duration        still eligible
        only the new ineligible  segment, bucket, ...   still ineligible
        in both                  comparison             eligible  -> ineligible
        in neither               unknown                ineligible -> eligible

    `unknown` is the cell A3-1d diagnosed; `comparison` is the one genuine removal, and
    it is here so the removal is visible rather than discovered by a count going down.
    """
    from src.pdf_retrieval_v4.typed_evidence_emitters import ATOMIC_ELIGIBLE_KINDS

    legacy = set(ATOMIC_ELIGIBLE_KINDS)
    new_ineligible = ({k.value for k in DISAGGREGATION_KINDS}
                      | {TemporalKind.NON_TEMPORAL.value,
                         TemporalKind.COMPARISON.value})

    assert legacy - new_ineligible == {"point", "duration"}
    assert new_ineligible - legacy == {"segment", "bucket", "category", "non_temporal"}
    assert legacy & new_ineligible == {"comparison"}

    # Both expressions are written over the legacy classifier's vocabulary, and `unknown`
    # is the only value of it that neither side claims.  `YEAR` is W3's kind and belongs
    # to neither expression -- it is not a value the legacy classifier can produce.
    legacy_vocabulary = {k.value for k in TemporalKind} - {TemporalKind.YEAR.value,
                                                           TemporalKind.UNKNOWN.value}
    assert (legacy | new_ineligible) == legacy_vocabulary - {TemporalKind.UNKNOWN.value}
    assert TemporalKind.UNKNOWN.value not in (legacy | new_ineligible)


# --- routing: what the column is ------------------------------------------------------

def test_a_segment_column_is_routed_not_judged_incomplete():
    """A segment column is a different fact, not a defective one."""
    for kind in (TemporalKind.SEGMENT, TemporalKind.BUCKET, TemporalKind.CATEGORY):
        admission = decide_emission_admission(_request(_resolved(), kind))
        assert admission.outcome is AdmissionOutcome.WITHHOLD
        assert admission.reason is AdmissionReason.DISAGGREGATION_AXIS


def test_a_comparison_column_routes_to_its_own_emitter():
    admission = decide_emission_admission(_request(_resolved(), TemporalKind.COMPARISON))
    assert admission.reason is AdmissionReason.COMPARISON_AXIS


def test_a_non_temporal_column_is_withheld_as_having_no_period():
    admission = decide_emission_admission(_request(_resolved(), TemporalKind.NON_TEMPORAL))
    assert admission.reason is AdmissionReason.NON_PERIOD_AXIS


def test_routing_is_asked_before_completeness():
    """Order is the attribution: the reason names where the cell stopped.

    A segment column whose metric is also missing is withheld for being a segment
    column.  Reporting the missing metric would make the routing tally unreadable.
    """
    admission = decide_emission_admission(
        _request(_resolved(), TemporalKind.SEGMENT, metric_path=None))
    assert admission.reason is AdmissionReason.DISAGGREGATION_AXIS


# --- the period -----------------------------------------------------------------------

def test_partial_is_admitted_and_marked_grain_limited():
    admission = decide_emission_admission(_request(_year_only()))
    assert admission.outcome is AdmissionOutcome.ADMIT_GRAIN_LIMITED
    assert admission.grain_limited
    assert admission.admitted


def test_a_conflict_is_withheld_and_keeps_both_candidates():
    """`不猜` -- the disagreement is preserved, not resolved into a stored fact."""
    left, right = _resolved("2022-12-31"), _resolved("2023-12-31")
    conflict = Conflict(key="2022-12-31|2023-12-31", candidates=(left, right))
    admission = decide_emission_admission(_request(conflict))
    assert admission.outcome is AdmissionOutcome.WITHHOLD
    assert admission.reason is AdmissionReason.CONFLICTED_PERIOD
    assert len(admission.conflict_candidates) == 2


def test_a_withheld_fact_has_no_determinate_period():
    """The conflict rule made enforceable rather than advisory.

    Whatever else a withheld fact carries, it must not carry one period a consumer could
    read: that is the shape in which a guess becomes a stored fact.
    """
    conflict = Conflict(key="a|b", candidates=(_resolved("2022-12-31"),
                                               _resolved("2023-12-31")))
    withheld = [
        decide_emission_admission(_request(conflict)),
        decide_emission_admission(_request(PeriodBindingV2())),
        decide_emission_admission(_request(None)),
        decide_emission_admission(_request(_resolved(), TemporalKind.SEGMENT)),
        decide_emission_admission(_request(_resolved(), metric_path=None)),
    ]
    assert all(a.outcome is AdmissionOutcome.WITHHOLD for a in withheld)
    assert all(a.normalized_period is None for a in withheld)


def test_an_unresolved_binding_does_not_gain_admission_from_the_new_path():
    """Recovering more periods is not a reason for a cell with no period to be admitted."""
    admission = decide_emission_admission(_request(PeriodBindingV2()))
    assert admission.reason is AdmissionReason.UNRESOLVED_PERIOD
    assert decide_emission_admission(
        _request(None)).reason is AdmissionReason.NO_BINDING


def test_a_binding_with_no_source_cells_is_not_provenance():
    """A period that cannot be resolved back to a cell is an assertion, not evidence."""
    bare = PeriodBindingV2(normalized_period="2025-12-31",
                           granularity=PeriodGranularity.DAY,
                           status=PeriodBindingStatus.RESOLVED,
                           method=PeriodBindingMethod.DIRECT_HEADER)
    admission = decide_emission_admission(_request(bare))
    assert admission.reason is AdmissionReason.INCOMPLETE_PROVENANCE


# --- the fact's own coordinate --------------------------------------------------------

def test_missing_metric_value_or_cell_withholds_as_incomplete():
    for override in ({"metric_path": None}, {"metric_status": "missing"},
                     {"value_normalized": None}, {"value_normalized": ""},
                     {"cell_id": None}):
        admission = decide_emission_admission(_request(_resolved(), **override))
        assert admission.reason is AdmissionReason.INCOMPLETE_PROVENANCE, override


# --- what a grain-limited admission may be used for -----------------------------------

def test_a_year_only_fact_answers_a_year_request_and_not_a_day_one():
    """`FY2025` and `YEAR(2025)` are compatible; `2025-12-31` is not automatically.

    The restriction carried by `ADMIT_GRAIN_LIMITED` in one assertion.
    """
    year = _year_only("2025")
    assert periods_are_compatible(year, PeriodGranularity.YEAR, "2025")
    assert not periods_are_compatible(year, PeriodGranularity.DAY, "2025-12-31")
    assert not periods_are_compatible(year, PeriodGranularity.DAY)


def test_a_dated_fact_still_answers_a_coarser_request():
    """Finer may answer coarser; the restriction is one-directional."""
    assert periods_are_compatible(_resolved("2025-12-31"), PeriodGranularity.YEAR)


def test_the_admitted_period_is_the_sources_own_string():
    """No fabricated day, even at the point of admission."""
    admission = decide_emission_admission(_request(_year_only("2025")))
    assert admission.normalized_period == "2025"
    assert not admission.normalized_period.endswith("-12-31")
    assert admission.binding.granularity is PeriodGranularity.YEAR
