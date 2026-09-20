"""P1.6-A2B-11: resolve a slot against Store V2, and refuse when it cannot.

Step 3, scoped to what one filing can demonstrate. Store V2 holds JPMorganChase,
so no benchmark case is complete here -- every case needs its other entities too.
What one filing can settle is the **slot**, and the slot is where the question has
been all along: given `(entity, metric, period)`, does it resolve to the company's
figure, or to a segment cell that shares its coordinate?

The legacy store could not answer that. `compare-009`'s `4,520` and `57,048` sit
in one table, one row, one coordinate there, and nothing in the record separates
them. Store V2 carries `column_header`, so the two are distinguishable facts -- and
this resolves a slot to the company-level one, or says why it will not.

**Refusing is a result.** A slot whose candidates disagree on scope and cannot be
ordered is reported rather than resolved to the first, because a confident wrong
pick is the failure this entire line of work exists to remove.

  python resolve_store_v2_slot.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

DEFAULT_STORE = Path(
    "/disk/qh/nano-finrag/artifacts/evaluation/p1-6-a2b9-store-v2/store-v2.jsonl"
)

#: How a company-level column announces itself.  `Total` and `Consolidated` are
#: what a filing heads the firm-wide column; a segment or a jurisdiction is
#: anything else.  Matched as a whole word so `Total assets` in a row label
#: cannot be mistaken for a column.
_COMPANY_LEVEL = re.compile(r"(?:^|/)\s*(total|consolidated|firm)\b", re.I)

#: Where a filing states the company's own figures.  In these tables the columns
#: are periods -- 2025, 2024, 2023 -- because the table as a whole is the company,
#: so there is no `Total` column to look for.  This is the signal that separates a
#: primary statement from a breakdown, and the earlier rule, which looked for a
#: column named `Total`, failed on 40 of 47 slots for want of it.
_PRIMARY_STATEMENTS = frozenset({"INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"})

#: Which field decides that a table may speak for the company.
#:
#: `statement_type` is the rule this resolver has used: the table's family, as the
#: parse layer classified it.  `table_role` is the field A2B-19A added, and it
#: answers the authority question directly rather than inferring it from the family.
#:
#: **Both are kept, and the default does not move.**  A2B-20 is a single-variable
#: experiment: the store gains `table_role`, and the resolver is asked for it
#: explicitly, so the run under the old rule is reproducible from the same store and
#: any difference between the two runs is attributable to this one choice.
AUTHORITY_STATEMENT_TYPE = "statement_type"
AUTHORITY_TABLE_ROLE = "table_role"

#: What a table's role must be to count as the company speaking for itself.
_AUTHORITATIVE_ROLE = "PRIMARY_FINANCIAL_STATEMENT"


def _has_authority(record: dict, authority: str) -> bool:
    if authority == AUTHORITY_TABLE_ROLE:
        return str(record.get("table_role")) == _AUTHORITATIVE_ROLE
    return str(record.get("statement_type")) in _PRIMARY_STATEMENTS

#: Columns that carry no figure of their own -- change columns, percentages,
#: basis comparisons.  Excluded from company-level candidacy rather than treated
#: as segments, because they are not scope claims at all.
_NON_SCOPE = re.compile(r"\b(change|%|percent|versus|increase|decrease)\b", re.I)

#: A column header that is only a period, a unit note or both.  In a breakdown
#: these say nothing about scope, so a row carrying one is neither a company
#: figure nor a named segment -- it is unlabelled, and is reported as such.
#:
#: Written as a **token test rather than one nested pattern**.  The first version
#: was `^(?:…|\s)*)+$`, whose inner `*` and outer `+` range over overlapping
#: alternatives -- catastrophic backtracking, and the acceptance run hung with no
#: output rather than failing. Splitting on separators and testing each token
#: keeps the same meaning with no nesting.
_PERIOD_TOKENS = frozenset({
    "in", "except", "millions", "thousands", "billions", "dollars", "shares",
    "per", "share", "amounts", "amount", "ratios", "ratio", "data", "and",
    "year", "years", "ended", "end", "as", "of", "fiscal", "quarter", "quarterly",
    "three", "six", "nine", "twelve", "months", "month", "ytd", "to", "date",
    "december", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november",
})
_PERIOD_SPLIT = re.compile(r"[^a-z0-9]+")


def _is_period_or_unit(column: str) -> bool:
    """Whether a column header says only *when*, or only what units it is in."""

    tokens = [t for t in _PERIOD_SPLIT.split(column.casefold()) if t]
    if not tokens:
        return True
    return all(t in _PERIOD_TOKENS or t.isdigit() for t in tokens)


def _norm(text: object) -> str:
    return " ".join(str(text or "").split()).casefold()


#: A trailing unit caption on a row label: `Net income (in millions)`.
_ROW_CAPTION = re.compile(r"\s*\((?:in|except|dollars in|amounts in)[^)]*\)\s*$", re.I)


def _matches_metric(label: object, metric: str) -> bool:
    """Whether the row label names the metric and nothing further.

    **Equality after stripping a unit caption**, not a prefix and not a
    substring.  Two defects have now arrived through this one function:

    - substring: `interest expense` is a substring of `Total noninterest
      expense`, so the slot resolved to `95,640`, a different line item;
    - word-boundary prefix: `net income` is a prefix of `Net income per share`
      and of `Net income attributable to …`, so a primary statement -- which
      holds all of those -- reported six competing values and the slot went
      AMBIGUOUS rather than resolving to the row the filing calls `Net income`.

    Each was found by an acceptance check rather than by reading the code, and
    each looked like a working resolver from the inside.
    """

    folded = _ROW_CAPTION.sub("", _norm(label)).strip()
    return folded == metric


def resolve(records: list[dict], entity: str, metric: str, period: str,
            authority: str = AUTHORITY_STATEMENT_TYPE) -> dict:
    """The company-level fact for a slot, or why one cannot be named.

    Exact on entity and period; the metric is matched against the row label, not
    the breadcrumb path, because the path is a table-level string and the label
    is what the filing calls the row.

    `authority` chooses only which field decides that a table may speak for the
    company.  Every other step -- the entity, metric and period matching, the
    company-level column rule, the refusal -- is the same under both.
    """

    want_entity, want_metric = _norm(entity), _norm(metric)
    want_year = "".join(ch for ch in str(period) if ch.isdigit())[:4]

    pool = [
        r for r in records
        if _norm(r.get("entity")) == want_entity
        and _matches_metric(r.get("row_label"), want_metric)
        and str(r.get("period_end") or "")[:4] == want_year
    ]
    if not pool:
        return {"status": "NO_FACT", "entity": entity, "metric": metric,
                "period": period, "candidates": 0}

    # A row in a primary statement *is* the company's figure: those tables have
    # period columns, not scope columns.  Tried first, because it is the direct
    # statement of the thing being asked and does not depend on a table having
    # chosen to head a column `Total`.
    primary = [r for r in pool if _has_authority(r, authority)]
    if primary:
        values = {str(r.get("value")) for r in primary}
        if len(values) == 1:
            record = primary[0]
            return {
                "status": "RESOLVED_COMPANY_LEVEL",
                "entity": entity, "metric": metric, "period": period,
                "value": record.get("value"), "value_raw": record.get("value_raw"),
                "column_header": record.get("column_header"),
                "statement_type": record.get("statement_type"),
                "table_role": record.get("table_role"),
                "table_fragment_id": record.get("table_fragment_id"),
                "cell_id": record.get("cell_id"),
                "segments_seen": [],
                "basis": f"primary statement ({authority})",
                "candidates": len(pool),
            }
        return {"status": "AMBIGUOUS_COMPANY_LEVEL", "entity": entity,
                "metric": metric, "period": period,
                "values": sorted(values), "candidates": len(pool),
                "basis": f"primary statement ({authority})",
                "authority_tables": sorted({
                    str(r.get("table_fragment_id")) for r in primary})}

    scoped: dict[str, list[dict]] = defaultdict(list)
    unscoped = []
    for record in pool:
        column = str(record.get("column_header") or "")
        if _NON_SCOPE.search(column):
            continue
        if _COMPANY_LEVEL.search(column):
            scoped["company"].append(record)
        elif column.strip() and not _is_period_or_unit(column):
            scoped[column].append(record)
        else:
            unscoped.append(record)

    company = scoped.pop("company", [])
    values = {str(r.get("value")) for r in company}
    if company and len(values) == 1:
        record = company[0]
        return {
            "status": "RESOLVED_COMPANY_LEVEL",
            "entity": entity, "metric": metric, "period": period,
            "value": record.get("value"), "value_raw": record.get("value_raw"),
            "column_header": record.get("column_header"),
            "table_fragment_id": record.get("table_fragment_id"),
            "cell_id": record.get("cell_id"),
            "segments_seen": sorted(scoped),
            "candidates": len(pool),
        }
    if company and len(values) > 1:
        return {"status": "AMBIGUOUS_COMPANY_LEVEL", "entity": entity,
                "metric": metric, "period": period,
                "values": sorted(values), "candidates": len(pool)}

    # No company-level column among the candidates.  If the scopes disagree the
    # slot cannot be settled here, and saying so is the honest answer.
    segment_values = {
        column: sorted({str(r.get("value")) for r in rows})
        for column, rows in scoped.items()
    }
    return {
        "status": "NO_COMPANY_LEVEL_FACT",
        "entity": entity, "metric": metric, "period": period,
        "segment_columns": segment_values,
        "unscoped": len(unscoped),
        "candidates": len(pool),
    }


#: Slots the store can be asked about, taken from the cases JPMorganChase appears
#: in.  Only its own side of each -- the other entities are not in this store.
SLOTS = (
    ("JPMorganChase", "Net income", "FY2025"),
    ("JPMorganChase", "Diluted earnings per share", "FY2025"),
    ("JPMorganChase", "Comprehensive income", "FY2025"),
    ("JPMorganChase", "Interest expense", "FY2025"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = parser.parse_args(argv)

    if not args.store.is_file():
        print(f"store not found: {args.store}")
        return 1
    records = [
        json.loads(line)
        for line in args.store.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"=== slot resolution against Store V2 ({len(records)} records) ===")
    print()
    ok = refused = 0
    for entity, metric, period in SLOTS:
        result = resolve(records, entity, metric, period)
        status = result["status"]
        if status == "RESOLVED_COMPANY_LEVEL":
            ok += 1
            print(f"  ok   {entity:15} {metric:28} {period}")
            print(f"         -> {result['value_raw']}  under {result['column_header'][:58]!r}")
            if result["segments_seen"]:
                print(f"         (segments also present, not chosen: "
                      f"{len(result['segments_seen'])})")
        elif status == "NO_COMPANY_LEVEL_FACT":
            refused += 1
            print(f"  --   {entity:15} {metric:28} {period}  {status}")
            for column, values in list(result["segment_columns"].items())[:3]:
                print(f"         segment {column[:46]!r} -> {values}")
        else:
            refused += 1
            print(f"  --   {entity:15} {metric:28} {period}  {status}  {result}")
    print()
    print(f"  resolved {ok} | refused or unresolved {refused}")
    print("  a refusal is a result: no confident pick was made")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
