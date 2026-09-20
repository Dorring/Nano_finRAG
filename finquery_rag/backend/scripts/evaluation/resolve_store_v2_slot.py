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

#: Columns that carry no figure of their own -- change columns, percentages,
#: basis comparisons.  Excluded from company-level candidacy rather than treated
#: as segments, because they are not scope claims at all.
_NON_SCOPE = re.compile(r"\b(change|%|percent|versus|increase|decrease)\b", re.I)


def _norm(text: object) -> str:
    return " ".join(str(text or "").split()).casefold()


def _matches_metric(label: object, metric: str) -> bool:
    """Whether the row label names the metric, as a whole phrase.

    Word-boundary rather than substring: `interest expense` is a substring of
    `Total noninterest expense`, so a naive `in` matched the wrong row and
    resolved the slot to `95,640` -- a different line item -- with no sign that
    anything was wrong.  That is the failure this whole line of work is about,
    arriving from a direction nothing had checked.
    """

    return re.search(rf"(?<![a-z]){re.escape(metric)}(?![a-z])", _norm(label)) is not None


def resolve(records: list[dict], entity: str, metric: str, period: str) -> dict:
    """The company-level fact for a slot, or why one cannot be named.

    Exact on entity and period; the metric is matched against the row label, not
    the breadcrumb path, because the path is a table-level string and the label
    is what the filing calls the row.
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

    scoped: dict[str, list[dict]] = defaultdict(list)
    unscoped = []
    for record in pool:
        column = str(record.get("column_header") or "")
        if _NON_SCOPE.search(column):
            continue
        if _COMPANY_LEVEL.search(column):
            scoped["company"].append(record)
        elif column.strip():
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
