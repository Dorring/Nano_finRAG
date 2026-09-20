"""A3-W2: the five period-binding producers, as dual output.

**Shadow only.**  The legacy period path stays the only authoritative one.  This runs
beside it, produces a `PeriodBindingV2` per target, and changes nothing that any consumer
reads -- the guard test asserts the production path does not import it.

Method is **attribution, not selection**: which of the five describes where this period
came from.  Nothing here ranks them, and `resolve_period_evidence` is what combines
evidence -- a direct declaration shadowing inherited context is scope semantics, and
disagreement inside a class is a `Conflict`, never a choice.

    DIRECT_HEADER                 a complete date in the column's own header cell
    ADJACENT_YEAR_JOIN            a month-day fragment joined to that column's own year
    HEADER_ROW_SELECTION_EXTEND   the same, where the header row was one the legacy
                                  selector had missed
    INLINE_PERIOD_DATA_ROW        the target row states its own period
    YEAR_ONLY_PERIOD              the source states a year and no date

A3-W4-A named two mechanical gaps in the date patterns, and this widens the first of them:
month names were spelled out in full, so `Jan 26, 2025` -- NVIDIA's whole fiscal calendar
-- matched nothing.  The second (`as of December 31` with the year in a sibling cell) is
untouched here and is measured separately.

W4-A7 replaces the period-header predicate.  It used to be "a date, and no more than twelve
characters of anything else", which W4-A6 falsified: the limit refused 876 source-grounded
cells, `For the Year Ended September 30, 2025` among them, and it had no semantic content
to begin with.  It is replaced by what precedes and follows the date and what that means --
see `_is_period_header_cell`.  The A and B families W4-A6 read out of the filings are the
oracle, in `tests/test_period_binding_shadow.py`.

  python period_binding_shadow.py --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from src.pdf_retrieval_v4.period_binding import (  # noqa: E402
    PeriodBindingMethod,
    PeriodBindingStatus,
    PeriodBindingV2,
    PeriodGranularity,
    PeriodTargetScope,
    SourceCell,
    TemporalKind,
    TemporalKindEvidence,
    TemporalKindMethod,
)

PARSER = _BACKEND_DIR / "scripts/evaluation/run_nf_v2_17a4_parse.py"
BUILDER = _BACKEND_DIR / "scripts/evaluation/build_store_v2.py"

#: The four shapes A3 diagnosed, each with the oracle value it must reproduce.
ORACLES = (
    ("ko_fy2025", 12533, "EQUITY"),
    ("pfe_fy2024", 24395, "EQUITY"),
    ("v_fy2025", 9951, "BALANCE_SHEET"),
    ("jpm_fy2025", 63193, "BALANCE_SHEET"),
    ("ko_fy2025", 11388, "BALANCE_SHEET"),
)

_MONTHS = ("january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december")

#: `Jan 26, 2025` and `September 27, 2025` are the same claim.  The full names alone were
#: not enough: NVIDIA's fiscal calendar is written `Jan 26, 2025` throughout, so every one
#: of its primary statements had no column binding at all under the old pattern -- 352 of
#: the 708 primary-table cells W4-A measured as lost.
#:
#: Each alternative is the full name with a trailing optional run, so `Mar` cannot match
#: inside `Marketing`: the alternation is followed by `\s+\d`, and `keting` is not that.
_MONTH_NAME = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
               r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
               r"Dec(?:ember)?)")

_MONTH_DAY_YEAR = re.compile(
    rf"\b{_MONTH_NAME}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+((?:19|20)\d{{2}})\b", re.I)
_MONTH_DAY = re.compile(rf"\b{_MONTH_NAME}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?", re.I)
_BARE_YEAR = re.compile(r"^\s*(?:FY\s*)?((?:19|20)\d{2})\s*$", re.I)
_VALUE_LIKE = re.compile(r"^\(?\$?\s*-?[\d,]+(?:\.\d+)?\s*\)?%?$")
_YEAR_ONLY_TEXT = re.compile(r"^\s*(?:FY\s*)?(?:19|20)\d{2}\s*$", re.I)

#: `Jan`, `Sept`, `September` -> the same number.  Keyed by the first three letters,
#: which is the shortest prefix all twelve names are distinct on.
_MONTH_BY_PREFIX = {name[:3]: index + 1 for index, name in enumerate(_MONTHS)}


def _month_number(text: str) -> int | None:
    word = re.match(r"[A-Za-z]+", text.strip())
    if not word:
        return None
    return _MONTH_BY_PREFIX.get(word.group(0).lower()[:3])


def _iso(month_day: str, year: str) -> str | None:
    """`December 31,` + `2025` -> `2025-12-31`, or None if the month is unreadable.

    The month is read off the matched text rather than off a capture group, so an
    abbreviation and a full name go through the same path.
    """
    match = _MONTH_DAY.search(month_day)
    if not match:
        return None
    month = _month_number(match.group(0))
    day_match = re.search(r"(\d{1,2})", match.group(0))
    if month is None or not day_match:
        return None
    return f"{year}-{month:02d}-{int(day_match.group(1)):02d}"


def _value_like(text: str) -> bool:
    t = " ".join(str(text or "").split())
    if not t or _YEAR_ONLY_TEXT.match(t):
        return False
    return bool(_VALUE_LIKE.match(t))


#: Constructions that introduce *when* a column is reported at.  Matched at the end of
#: whatever precedes the date, so `For the Year Ended` and `... held at` both qualify.
_PERIOD_LOCATOR = re.compile(
    r"(?:"
    r"\bas\s+of"
    r"|\bat"
    r"|\bon"
    r"|\bended"
    r"|\bending"
    r"|\bthrough"
    r"|\bfor\s+the\s+(?:year|quarter|period|month|week|three|six|nine|twelve)\b"
    r"|\bfiscal\s+(?:year|quarter|period)\b"
    r"|\bperiod\s+ended"
    r"|\byear\s+to\s+date"
    r")\s*$", re.I)

#: A present participle in front of the locator turns the date into the date of an *event*
#: rather than the point the column is measured at:
#:
#:     financial instruments held    at  Dec. 31, 2025      a state at a time
#:     PSUs                vesting   on  March 25, 2026     an event that happens
#:
#: Same lexical shape -- clause, preposition, date -- and opposite meanings.  That is why
#: the twelve-character rule could not separate them and why no length rule ever will, and
#: it is the one place in this predicate where the decision is about what the words mean
#: rather than about how the cell is punctuated.
_EVENT_BEFORE_LOCATOR = re.compile(r"[A-Za-z]+ing\s*$", re.I)


def _is_period_header_cell(text: str, match: re.Match) -> bool:
    """Whether the date *is* the cell's claim about when the column is reported.

    Three things have to hold, and the length of what precedes the date is **not one of
    them**.  A twelve-character limit on the remainder was the previous rule and it was
    falsified by measurement: it refused 876 source-grounded cells (W4-A6), among them
    `For the Year Ended September 30, 2025` -- nineteen characters, and as plain a period
    header as a filing contains.  The limit had no semantic content; it was tuned to catch
    prose and it caught period headers instead.

      what follows the date   must be nothing but a units caption, or the cell is a
                              sentence that mentions a date
      what precedes it        must *introduce* a period -- `For the Year Ended`, `held at`,
                              `as of`.  `... Chief Executive Officer February 20, 2026` and
                              `... effective October 10, 2025` do not, and are refused.
      and not an event        a present participle before the locator makes the date the
                              date of a happening, not a reporting point

    The Visa case this replaces is still refused, by the first rule rather than the length
    one: `Class C common stock, and 9 shares issued and outstanding as of September 30,
    2025 and 2024` carries a second year, so it declares no single period.  Taking it bound
    a date to a column that never claimed one.
    """
    # Captions are dropped from *both* sides.  Dropping them from the suffix alone would
    # refuse `(in millions) September 27, 2025`, which the length rule accepted and which
    # is plainly a period header -- a repair that unbinds a column is not a repair.
    prefix = re.sub(r"\([^)]*\)", " ", text[:match.start()])
    suffix = re.sub(r"[\s,;:—–-]+", " ",
                    re.sub(r"\([^)]*\)", " ", text[match.end():])).strip()

    if re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", prefix + " " + suffix):
        return False                     # a second year -> not one period
    if len(suffix) > 12:
        return False                     # a sentence continues past the date
    # A parenthesis opened before the date and not yet closed puts the date inside an
    # aside -- unless the aside is *only* a locator.  The two cases look alike and mean
    # opposite things:
    #
    #     Age (at December 31, 2025)                                        a period
    #     Amounts Recognized as of Acquisition Date (as previously          a qualification
    #       reported as of December 31, 2023)
    #
    # so what is between the innermost open bracket and the date has to be a locator and
    # nothing else.  W4-A6 filed the second family as ambiguous; the first was binding
    # before the repair and must go on binding after it.
    raw_prefix = text[:match.start()]
    if raw_prefix.count("(") > raw_prefix.count(")"):
        inner = raw_prefix[raw_prefix.rfind("(") + 1:].strip(" \t,;:—–-")
        if inner and not _PERIOD_LOCATOR.fullmatch(inner):
            return False

    stripped = prefix.strip(" \t,;:—–-")
    if not stripped:
        return True                      # the cell *is* the date; nothing qualifies it
    locator = _PERIOD_LOCATOR.search(stripped)
    if not locator:
        return False
    return not _EVENT_BEFORE_LOCATOR.search(stripped[:locator.start()].rstrip())


def extended_header_idx(nf, grid) -> tuple[list[int], set[int]]:
    """The legacy header rows unioned with the structurally discovered ones.

    Returns `(union, added)`.  The union is what the binding may read; `added` is what the
    legacy selector had missed, which is what makes the method
    `HEADER_ROW_SELECTION_EXTEND` rather than `DIRECT_HEADER`.
    """
    legacy = nf.header_idx(grid)
    discovered: list[int] = []
    for i, row in enumerate(grid):
        cells = [c for c in row if c]
        if not cells:
            continue
        if sum(1 for c in row[1:] if c and _value_like(c["raw_text"])) >= 2:
            continue
        texts = [nf.ws(c["raw_text"]) for c in cells]
        row_text = nf.ws(" ".join(texts))
        if not row_text:
            continue
        if any(c["header"] for c in cells):
            discovered.append(i)
        elif (nf.parse_date_text(row_text) or _MONTH_DAY.search(row_text)
              or len(set(nf.year_tokens(row_text))) >= 2):
            discovered.append(i)
        elif i < 4 and not any(nf.has_num(t) for t in texts):
            discovered.append(i)
    union = sorted(set(legacy) | set(discovered))
    return union, set(union) - set(legacy)


def bind_table(nf, grid, doc_id: str, table_id: str) -> dict:
    """Every period binding this table would produce, keyed by target."""
    union, added = extended_header_idx(nf, grid)
    width = max((len(r) for r in grid), default=0)

    raw: dict[int, list[tuple[int, str]]] = {}
    for col in range(width):
        cells = []
        for i in union:
            if i < len(grid) and col < len(grid[i]) and grid[i][col]:
                text = nf.ws(grid[i][col]["raw_text"])
                if text:
                    cells.append((i, text))
        raw[col] = cells

    # A month-day that opens a header group applies to the bare-year cells that follow it.
    # **`pending` is carried across header rows, not reset per row.**  Visa puts
    # `September 30,` in row 1 spanning columns 3-11 and its years in row 2; resetting per
    # row meant the month-day never met a year and every data column fell back to
    # YEAR-only while three label columns picked up a bogus DAY from prose.  Each column
    # still keeps its own year, and no column consumes another's.
    group_month_day: dict[int, tuple[int, str]] = {}
    pending: tuple[int, str] | None = None
    for i in union:
        if i >= len(grid):
            continue
        seen_cells: set[int] = set()
        for col in range(width):
            cell = grid[i][col] if col < len(grid[i]) else None
            if not cell or id(cell) in seen_cells:
                continue
            seen_cells.add(id(cell))
            text = nf.ws(cell["raw_text"])
            if not text:
                continue
            month = _MONTH_DAY.search(text)
            if month and _is_period_header_cell(text, month):
                pending = (i, month.group(0))
                continue
            if _BARE_YEAR.match(text) and pending is not None:
                # The year cell spans its columns, so every column it covers inherits the
                # month-day -- not only the first.  Visa's `2025` has colspan 3 over
                # columns 3-5, and registering only column 3 left 4 and 5 as YEAR-only
                # inside a group the source declared as one date.
                covers = [c for c in range(len(grid[i]))
                          if grid[i][c] is not None and id(grid[i][c]) == id(cell)]
                for covered in covers:
                    group_month_day.setdefault(covered, pending)

    columns: dict[int, PeriodBindingV2] = {}
    for col in range(width):
        cells = raw[col]
        if not cells:
            continue
        source = tuple(SourceCell(doc_id, table_id, i, col, t) for i, t in cells)

        # 1. a complete date in this column's own header cell
        for i, text in cells:
            whole = _MONTH_DAY_YEAR.search(text)
            if not whole or not _is_period_header_cell(text, whole):
                continue
            resolved = _iso(whole.group(0), whole.group(1))
            # `_MONTH_DAY_YEAR` matched, so `_iso` should always resolve; if it somehow
            # does not, fall through rather than build a `RESOLVED` binding whose
            # `normalized_period` is None -- an incoherent state the contract forbids and
            # one that would read downstream as "resolved, but to nothing".
            if resolved is None:
                continue
            columns[col] = PeriodBindingV2(
                normalized_period=resolved,
                granularity=PeriodGranularity.DAY,
                status=PeriodBindingStatus.RESOLVED,
                method=(PeriodBindingMethod.HEADER_ROW_SELECTION_EXTEND
                        if i in added else PeriodBindingMethod.DIRECT_HEADER),
                target_scope=PeriodTargetScope.COLUMN,
                source_cells=source,
                temporal=TemporalKindEvidence(
                    kind=TemporalKind.POINT,
                    method=TemporalKindMethod.PERIOD_BINDING,
                    source_cells=source),
            )
            break
        if col in columns:
            continue

        # 2. a month-day fragment joined to this column's own bare year
        year = next((m.group(1) for _, text in cells
                     if (m := _BARE_YEAR.match(text))), None)
        if year and col in group_month_day:
            fragment_row, fragment = group_month_day[col]
            joined = _iso(fragment, year)
            if joined:
                columns[col] = PeriodBindingV2(
                    normalized_period=joined,
                    granularity=PeriodGranularity.DAY,
                    status=PeriodBindingStatus.RESOLVED,
                    method=(PeriodBindingMethod.HEADER_ROW_SELECTION_EXTEND
                            if fragment_row in added else
                            PeriodBindingMethod.ADJACENT_YEAR_JOIN),
                    target_scope=PeriodTargetScope.COLUMN,
                    source_cells=source + (SourceCell(doc_id, table_id,
                                                      fragment_row, col, fragment),),
                    temporal=TemporalKindEvidence(
                        kind=TemporalKind.DURATION
                        if re.search(r"year(?:s)?\s+ended", " ".join(t for _, t in cells), re.I)
                        else TemporalKind.POINT,
                        method=TemporalKindMethod.PERIOD_BINDING,
                        source_cells=source),
                )
                continue

        # 3. the source states a year and nothing finer
        if year:
            columns[col] = PeriodBindingV2(
                normalized_period=year,
                granularity=PeriodGranularity.YEAR,
                status=PeriodBindingStatus.PARTIAL,
                method=PeriodBindingMethod.YEAR_ONLY_PERIOD,
                target_scope=PeriodTargetScope.CELL_GROUP,
                source_cells=source,
                temporal=TemporalKindEvidence(kind=TemporalKind.UNKNOWN,
                                              method=TemporalKindMethod.PERIOD_BINDING,
                                              source_cells=source),
            )

    rows: dict[int, PeriodBindingV2] = {}
    for ri, row in enumerate(grid):
        seen: set[int] = set()
        for ci, cell in enumerate(row):
            if not cell or id(cell) in seen:
                continue
            seen.add(id(cell))
            text = nf.ws(cell["raw_text"])
            whole = _MONTH_DAY_YEAR.search(text)
            if not whole:
                continue
            # A row that states its own period, and is not merely a year header: it must
            # also carry values, or it is a header row wearing a label.
            numeric = sum(1 for c in row if c and _value_like(nf.ws(c["raw_text"])))
            if numeric < 2:
                continue
            # A row naming more than one year is not declaring one period.  Visa's
            # balance sheet has `… shares issued and outstanding as of September 30,
            # 2025 and 2024`, a disclosure sentence that contains a date; taking it would
            # bind a second date to a row that never claimed one.  The single-year rule
            # is objective, and it is what separates `Balance, December 31, 2022` from a
            # sentence that happens to mention a date.
            years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text))
            if len(years) != 1:
                continue
            resolved = _iso(whole.group(0), whole.group(1))
            if resolved is None:
                continue
            rows[ri] = PeriodBindingV2(
                normalized_period=resolved,
                granularity=PeriodGranularity.DAY,
                status=PeriodBindingStatus.RESOLVED,
                method=PeriodBindingMethod.INLINE_PERIOD_DATA_ROW,
                target_scope=PeriodTargetScope.ROW,
                source_cells=(SourceCell(doc_id, table_id, ri, ci, text),),
                temporal=TemporalKindEvidence(kind=TemporalKind.POINT,
                                              method=TemporalKindMethod.PERIOD_BINDING),
            )
            break

    return {"columns": columns, "rows": rows, "union_header_idx": union,
            "added_header_rows": sorted(added)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    spec = importlib.util.spec_from_file_location("nf17a4", PARSER)
    nf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nf)
    bspec = importlib.util.spec_from_file_location("store_builder", BUILDER)
    builder = importlib.util.module_from_spec(bspec)
    bspec.loader.exec_module(builder)

    from lxml import etree, html

    report = {"phase": "P1.6-A3-W2", "mutation": "none", "shadow_only": True,
              "tables": {}}
    print("=== PeriodBindingV2 dual output (shadow) ===")
    print()

    for document_id, order, family in ORACLES:
        ticker, accession = builder.DOCUMENTS[document_id]
        raw_path = (Path("/disk/qh/nano-finrag/data/financial_corpus_v2/raw/SEC")
                    / ticker / accession / "primary.html")
        root = html.parse(str(raw_path), etree.HTMLParser(
            recover=True, no_network=True, huge_tree=True,
            remove_comments=True)).getroot()
        blocks, lookup, _prior = nf.make_blocks(
            root, {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"})
        block = next(b for b in blocks
                     if b["block_type"] == "TABLE" and b["source_order"] == order)
        grid = nf.grid_rows(nf.direct_rows(lookup[block["table_id"]]))
        bound = bind_table(nf, grid, document_id, block["table_id"])

        methods: dict[str, int] = {}
        for binding in list(bound["columns"].values()) + list(bound["rows"].values()):
            name = binding.method.value if binding.method else "NONE"
            methods[name] = methods.get(name, 0) + 1

        key = f"{document_id}#{order}"
        report["tables"][key] = {
            "family": family,
            "columns": {str(c): b.to_dict() for c, b in bound["columns"].items()},
            "rows": {str(r): b.to_dict() for r, b in bound["rows"].items()},
            "union_header_idx": bound["union_header_idx"],
            "added_header_rows": bound["added_header_rows"],
            "methods": methods,
        }
        print(f"--- {key}  {family}")
        print(f"    methods {methods}")
        print(f"    added header rows {bound['added_header_rows']}")
        for ri, binding in sorted(bound["rows"].items()):
            print(f"    row {ri:<3} {binding.method.value:<24} "
                  f"{binding.normalized_period} scope={binding.target_scope.value} "
                  f"src={[(c.row, c.column) for c in binding.source_cells]}")
        sample = next((b for c, b in sorted(bound["columns"].items()) if c > 0), None)
        if sample:
            print(f"    col sample {sample.normalized_period} {sample.status.value} "
                  f"{sample.method.value if sample.method else None} "
                  f"granularity={sample.granularity.value}")
        print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "period-binding-shadow.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'period-binding-shadow.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
