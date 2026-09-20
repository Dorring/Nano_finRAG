"""A3-W2 acceptance: the producers are right, not merely running.

The contract-level half is pure and fast.  The corpus half needs a filing on disk and
skips with a reason when there is none -- a test that passes because it looked at nothing
is worse than no test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from src.pdf_retrieval_v4.period_binding import (  # noqa: E402
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
    resolve_period_evidence,
)

CORPUS = Path("/disk/qh/nano-finrag/data/financial_corpus_v2/raw/SEC")

SHADOW = _BACKEND_DIR / "scripts/evaluation/period_binding_shadow.py"
PARSER = _BACKEND_DIR / "scripts/evaluation/run_nf_v2_17a4_parse.py"


def _shadow():
    spec = importlib.util.spec_from_file_location("pb_shadow", SHADOW)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell(method_row: int) -> SourceCell:
    return SourceCell("ko_fy2025", "table_x", method_row, 0, "Balance, December 31, 2022")


def _direct(period: str = "2022-12-31", row: int = 15) -> PeriodBindingV2:
    return PeriodBindingV2(
        normalized_period=period, granularity=PeriodGranularity.DAY,
        status=PeriodBindingStatus.RESOLVED,
        method=PeriodBindingMethod.INLINE_PERIOD_DATA_ROW,
        target_scope=PeriodTargetScope.ROW, source_cells=(_cell(row),))


def _inherited(period: str = "2025-12-31",
               method: PeriodBindingMethod = PeriodBindingMethod.DIRECT_HEADER
               ) -> PeriodBindingV2:
    return PeriodBindingV2(
        normalized_period=period, granularity=PeriodGranularity.DAY,
        status=PeriodBindingStatus.RESOLVED, method=method,
        target_scope=PeriodTargetScope.COLUMN, source_cells=(_cell(1),))


def _year_only(year: str = "2025") -> PeriodBindingV2:
    return PeriodBindingV2(
        normalized_period=year, granularity=PeriodGranularity.YEAR,
        status=PeriodBindingStatus.PARTIAL,
        method=PeriodBindingMethod.YEAR_ONLY_PERIOD,
        target_scope=PeriodTargetScope.CELL_GROUP, source_cells=(_cell(1),),
        temporal=TemporalKindEvidence(TemporalKind.UNKNOWN,
                                      TemporalKindMethod.PERIOD_BINDING))


# --- contract semantics --------------------------------------------------------------

def test_direct_declaration_shadows_a_disagreeing_inherited_binding():
    """Not a conflict: the row states its own period, and that is scope, not precedence."""
    declared, inherited = _direct("2022-12-31"), _inherited("2025-12-31")
    assert resolve_period_evidence(declared, [inherited]) is declared


def test_two_disagreeing_direct_declarations_are_a_conflict():
    outcome = resolve_period_evidence(None, [_direct("2022-12-31", 15),
                                             _direct("2023-12-31", 26)])
    assert isinstance(outcome, Conflict)
    assert len(outcome.candidates) == 2


def test_two_disagreeing_inherited_bindings_are_a_conflict():
    outcome = resolve_period_evidence(None, [_inherited("2025-12-31"), _inherited("2024-12-31",
                                             PeriodBindingMethod.ADJACENT_YEAR_JOIN)])
    assert isinstance(outcome, Conflict)


def test_agreeing_evidence_merges_rather_than_conflicting():
    """Repeated evidence is not disagreement; a false conflict would be as wrong."""
    merged = resolve_period_evidence(None, [_inherited("2025-12-31"),
                                            _inherited("2025-12-31",
                                                       PeriodBindingMethod.ADJACENT_YEAR_JOIN)])
    assert isinstance(merged, PeriodBindingV2)
    assert merged.normalized_period == "2025-12-31"
    assert len(merged.source_cells) == 2


def test_year_only_never_fabricates_a_calendar_date():
    binding = _year_only("2025")
    assert binding.normalized_period == "2025"
    assert binding.granularity is PeriodGranularity.YEAR
    assert not binding.normalized_period.endswith("-12-31")


def test_partial_does_not_collapse_into_unresolved():
    binding = _year_only("2025")
    assert binding.status is PeriodBindingStatus.PARTIAL
    assert binding.is_usable
    assert binding.temporal.kind is TemporalKind.UNKNOWN
    assert binding.status is not PeriodBindingStatus.UNRESOLVED


# --- the producers, against the filings ----------------------------------------------

def _bind(document_id: str, ticker: str, accession: str, order: int):
    if not CORPUS.is_dir():
        pytest.skip("corpus not present on this host")
    from lxml import etree, html
    nf_spec = importlib.util.spec_from_file_location("nf17a4", PARSER)
    nf = importlib.util.module_from_spec(nf_spec)
    nf_spec.loader.exec_module(nf)
    root = html.parse(str(CORPUS / ticker / accession / "primary.html"),
                      etree.HTMLParser(recover=True, no_network=True, huge_tree=True,
                                       remove_comments=True)).getroot()
    blocks, lookup, _prior = nf.make_blocks(
        root, {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"})
    block = next(b for b in blocks
                 if b["block_type"] == "TABLE" and b["source_order"] == order)
    return _shadow().bind_table(nf, nf.grid_rows(nf.direct_rows(lookup[block["table_id"]])),
                                document_id, block["table_id"])


@pytest.mark.parametrize("document_id,ticker,accession,order", [
    ("jpm_fy2025", "JPM", "SEC_19617_000162828026008131", 63193),
    ("ko_fy2025", "KO", "SEC_21344_000162828026010047", 11388),
])
def test_adjacent_year_join_keeps_each_column_its_own_year(document_id, ticker,
                                                           accession, order):
    """The core correctness property of the join: no column eats a sibling's year."""
    bound = _bind(document_id, ticker, accession, order)
    joins = {c: b for c, b in bound["columns"].items()
             if b.method is PeriodBindingMethod.ADJACENT_YEAR_JOIN}
    assert joins, "no ADJACENT_YEAR_JOIN columns; the fixture has moved"

    for column, binding in joins.items():
        target_year = binding.normalized_period[:4]
        cited = [cell.text for cell in binding.source_cells]
        own_years = {y for text in cited for y in __import__("re").findall(r"(?:19|20)\d{2}",
                                                                          text)}
        assert target_year in own_years, (column, binding.normalized_period, cited)
        for other in own_years:
            assert other == target_year, (
                f"column {column} bound {binding.normalized_period} but cites {other}: "
                f"{cited}")

    bound_years = {b.normalized_period[:4] for b in joins.values()}
    assert bound_years == {"2024", "2025"}, bound_years


def test_visa_row_39_is_not_a_direct_declaration():
    """`… shares issued and outstanding as of September 30, 2025 and 2024` is a sentence.

    It names two years, so it declares no single period.  Taking it would bind a date to a
    row that never claimed one -- the over-reach a producer must not have.
    """
    bound = _bind("v_fy2025", "V", "SEC_1403161_000140316125000089", 9951)
    assert 39 not in bound["rows"], (
        f"row 39 was bound as {bound['rows'].get(39)}")
    assert not any(b.method is PeriodBindingMethod.INLINE_PERIOD_DATA_ROW
                   for b in bound["rows"].values())


def test_pfizer_inline_rows_are_scoped_to_their_own_row_and_do_not_inherit():
    bound = _bind("pfe_fy2024", "PFE", "SEC_78003_000007800325000054", 24395)
    inline = {r: b for r, b in bound["rows"].items()
              if b.method is PeriodBindingMethod.INLINE_PERIOD_DATA_ROW}
    assert inline, "no INLINE_PERIOD_DATA_ROW rows; the fixture has moved"
    for row, binding in inline.items():
        assert binding.target_scope is PeriodTargetScope.ROW
        for cell in binding.source_cells:
            assert cell.row == row, f"row {row} cites row {cell.row}"


# --- month names, abbreviated and not -------------------------------------------------

def test_an_abbreviated_month_is_the_same_claim_as_the_full_name():
    """`Jan 26, 2025` and `January 26, 2025` are one date.

    W4-A measured 352 oracle-PRIMARY cells lost purely because NVIDIA's fiscal calendar is
    written `Jan 26, 2025` and the pattern listed full names only.
    """
    shadow = _shadow()
    assert shadow._iso("Jan 26,", "2025") == "2025-01-26"
    assert shadow._iso("January 26,", "2025") == "2025-01-26"
    assert shadow._iso("Sept 27", "2025") == "2025-09-27"
    assert shadow._iso("Sep 27", "2025") == "2025-09-27"
    assert shadow._iso("September 27,", "2025") == "2025-09-27"
    assert shadow._iso("Dec 31", "2024") == "2024-12-31"


def test_every_month_resolves_from_its_first_three_letters():
    shadow = _shadow()
    assert [shadow._month_number(n) for n in
            ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
             "Nov", "Dec")] == list(range(1, 13))
    assert shadow._month_number("January") == 1
    assert shadow._month_number("Sept") == 9
    # Not months, and not silently read as one.
    assert shadow._month_number("Nope") is None
    assert shadow._month_number("Ju") is None
    assert shadow._month_number("") is None
    assert shadow._iso("Nope 5", "2025") is None


def test_an_abbreviation_cannot_match_inside_a_longer_word():
    """`Mar` must not fire on `Marketing`, or widening the pattern would admit prose."""
    shadow = _shadow()
    for text in ("Marketing 5", "Junction 7", "Decembering 3", "Mayo 2", "Augment 9"):
        assert not shadow._MONTH_DAY.search(text), text
    assert shadow._MONTH_DAY.search("Mar 5")


def test_the_year_capture_moved_with_the_unnamed_month():
    """`_MONTH_DAY_YEAR`'s month is no longer a group, so the year is group 1.

    Both call sites read `whole.group(1)`; this pins the arity so a later edit cannot
    quietly index the wrong group and produce a period with the day as its year.
    """
    shadow = _shadow()
    match = shadow._MONTH_DAY_YEAR.search("as of Jan 26, 2025")
    assert match and match.group(1) == "2025"
    assert shadow._MONTH_DAY_YEAR.groups == 1


def test_nvda_abbreviated_headers_now_bind():
    """The measured case, end to end: NVIDIA's income statement columns get a date.

    Before the widening this returned no DAY-granularity column at all -- `Year Ended /
    Jan 26, 2025` matched nothing -- which is what put 352 primary cells in W4-A's loss
    column.
    """
    bound = _bind("nvda_fy2025", "NVDA", "SEC_1045810_000104581025000023", 6754)
    days = {c: b for c, b in bound["columns"].items()
            if b.granularity is PeriodGranularity.DAY}
    assert days, "no DAY-granularity column; the fixture has moved"
    assert {b.normalized_period for b in days.values()} == {
        "2025-01-26", "2024-01-28", "2023-01-29"}, (
        {b.normalized_period for b in days.values()})
    for binding in days.values():
        assert binding.status is PeriodBindingStatus.RESOLVED
        assert binding.method is PeriodBindingMethod.DIRECT_HEADER


# --- W4-A7: the period-header predicate, against the filings' own families ------------
#
# Both lists are copied from real cells, read in full during W4-A6.  They are the oracle
# because they are the two sides of the distinction the old predicate could not make: the
# positives are the 876 cells the twelve-character rule refused, the negatives are the 28
# it correctly refused.  A rule that accepts the first list and refuses the second is the
# whole requirement, and no rule that fails either half is good enough.

#: Must be accepted: each is a source-grounded claim about when the column is reported.
PERIOD_HEADERS = (
    "Fair value at Jan. 1, 2025",
    "For the Year Ended September 30, 2025",
    "Change in unrealized gains/(losses) related to financial instruments held at "
    "Dec. 31, 2025",
    "Central case assumptions at December 31, 2025",
    "Contractual rate in effect at December 31, 2025",
    "As of or for the year ended December 31, 2025 (in millions, except ratios)",
    "Fair value purchase price allocation as of May 1, 2023",
    "December 31, 2025",
    "(in millions) September 27, 2025",
    "Year Ended June 30, 2025",
    "Age (at December 31, 2025)",
)

#: Must be refused: each contains a date that is not the column's reporting period.
NOT_PERIOD_HEADERS = (
    "Chairman of the Board of Directors and Chief Executive Officer February 20, 2026",
    "President and Chief Financial Officer February 20, 2026",
    'Recovery of Erroneously Awarded Incentive-Based Compensation Policy - Firmwide, '
    "effective October 10, 2025 . (b)",
    '50% of the net issued shares received as a result of Performance Share Units '
    '("PSUs") vesting on March 25, 2026',
    "Indenture, dated as of October 28, 2021, between the Company and the Trustee",
    "Class C common stock, and 9 shares issued and outstanding as of September 30, "
    "2025 and 2024",
    "Amounts Recognized as of Acquisition Date (as previously reported as of "
    "December 31, 2023)",
    "Date: October 31, 2025",
    "Dated: February 27, 2025",
)


def _accepts(text: str) -> bool:
    shadow = _shadow()
    match = shadow._MONTH_DAY_YEAR.search(text)
    assert match, f"the fixture names no date the pattern can see: {text!r}"
    return shadow._is_period_header_cell(text, match)


def test_every_period_header_in_the_filings_is_accepted():
    """W4-A6's class A, cell for cell.  The twelve-character rule refused all of these."""
    refused = [t for t in PERIOD_HEADERS if not _accepts(t)]
    assert refused == [], f"period headers still refused: {refused}"


def test_no_date_that_is_not_a_reporting_period_is_accepted():
    """W4-A6's class B.  A predicate that accepts the first list by accepting everything
    would pass that test and fail this one, which is the whole point of keeping them."""
    accepted = [t for t in NOT_PERIOD_HEADERS if _accepts(t)]
    assert accepted == [], f"not reporting periods, but accepted: {accepted}"


def test_a_state_and_an_event_have_the_same_shape():
    """The distinction the length rule could never make, stated as a test.

    `... instruments held at Dec. 31, 2025` and `... PSUs vesting on March 25, 2026` are
    both clause, preposition, date.  One names the point the column is measured at, the
    other names when something happened, and a predicate that only counts punctuation
    cannot tell them apart.  If someone later replaces the -ing rule with a length rule,
    this fails first.
    """
    assert _accepts("financial instruments held at Dec. 31, 2025")
    assert not _accepts("Performance Share Units vesting on March 25, 2026")
