"""P1.6-0C: is the cross-entity gold itself a valid correctness oracle?

P1.6-0's coordinate-ambiguity guard refuses an operand whose coordinate does not
identify one value.  On the cross-entity stratum that guard fired on three
cases, and one of them -- `compare-009` -- was *scored correct* before the
guard existed, and its bound operand was **bit-identical to the gold value**.
So the guard was not catching a wrong operand there.  It was catching something
about the gold.

This audit asks that question directly, per gold operand, and freezes the
evidence before any fixture is touched:

    VALID_GOLD               the gold value is the one the source supports;
                             the *store* is what is defective (it lacks the
                             scope/column dimension that separates it from its
                             siblings) -> belongs to the P1.6 extraction fix
    WRONG_SCOPE_GOLD         the gold took a cell from a flattened multi-column
                             row -- a segment or a single year -- where the
                             question asks for the Firm -> fixture fix
    UNRESOLVED_GROUND_TRUTH  the source cannot decide from the store alone ->
                             drop from the scored denominator; it cannot judge
                             the system either way

**The discriminator.**  A filing states its headline figure many times: the
consolidated statement, the financial highlights, the MD&A summary, the notes.
A segment cell appears once, in the table that breaks the concept down.  So for
each competing value the audit records how many *independent rows* and *pages*
carry it.  The recurring value is the consolidated one; the singleton is the
breakdown.  This is a property of the document, not of any fixture, so it does
not assume the answer it is checking.

Facts sharing one `row_id` are reported together: when a single row yields
several comparable values under one coordinate, that is the flattened-table
signature and the row text is quoted so it can be read.

  python audit_cross_entity_gold_validity.py --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

STRATUM_PREFIX = "tv2f01-s3-"

#: Text that marks a row as the *header* of a segment/breakdown table rather
#: than a data row.  A filing's segment table names its columns with these; a
#: consolidated statement does not.  Used only to surface candidates for a
#: reader -- the audit does not decide a column's meaning from them.
_HEADER_MARKERS = (
    "as of or for the year ended",
    "community banking",
    "investment bank",
    "wealth management",
    "reportable business segments",
)


def _header_candidates(store, record, window: int = 8) -> list[dict]:
    """Header-like rows near the fact, for a reader to resolve columns by.

    The store keeps a row's text but not its column geometry -- `row_bbox`
    spans the full table width -- so the column a cell belongs to cannot be
    recovered from the cell itself.  The header row is the nearest thing to
    that evidence and is quoted here rather than interpreted.
    """

    page = record.get("page")
    if page is None:
        return []
    found = []
    for other in store.iter_records():
        if other.get("document_name") != record.get("document_name"):
            continue
        other_page = other.get("page")
        if other_page is None or not (page - window <= other_page <= page):
            continue
        text = str(other.get("source_text") or "")
        lowered = text.casefold()
        if not any(marker in lowered for marker in _HEADER_MARKERS):
            continue
        found.append({"page": other_page, "text": text[:600]})
    # One entry per distinct text, nearest page first.
    unique: dict[str, dict] = {}
    for item in sorted(found, key=lambda item: (-item["page"], item["text"])):
        unique.setdefault(item["text"], item)
    return list(unique.values())[:4]


def _gold_rows(backend: Path) -> dict[str, dict]:
    path = backend / "benchmarks/tv2_canonical_v1/gold-evidence-v1.jsonl"
    return {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _comparable_siblings(store, record) -> list:
    """Facts at the record's coordinate whose quantity is comparable with its own.

    Reuses the shared contract's predicate rather than a local rule: a
    percentage and an amount are not competing answers, and counting them as
    such would invent conflicts the guard is careful not to invent.
    """

    from rag_v2.contracts.financial_semantics import quantities_are_comparable

    from src.finance.operand_ambiguity import quantity_of

    bound = quantity_of(record)
    siblings = store.facts_at_coordinate(
        record.get("entity"), record.get("metric"), record.get("period")
    )
    kept = []
    for sibling in siblings:
        quantity = quantity_of(sibling)
        if quantity is None:
            continue
        if bound is None or quantities_are_comparable(bound, quantity):
            kept.append(sibling)
    return kept


def _build_recurrence_index(store) -> dict[tuple, dict]:
    """``(document, entity, metric, period, value) -> how widely it is stated``.

    Scoped to the **coordinate**, not the document.  A first version keyed on
    ``(document, value)`` alone and was wrong in a way worth recording: small
    round numbers recur all over a filing for unrelated reasons, so `(2)` looked
    like the most widely-stated value at a coordinate where it actually appears
    once.  What the rule needs to know is how often *this quantity* is restated,
    and only facts sharing the concept can answer that.

    Built in one pass so a coordinate with nine competing values does not
    rescan the store nine times.
    """

    rows: dict[tuple, set[str]] = collections.defaultdict(set)
    pages: dict[tuple, set[int]] = collections.defaultdict(set)
    for record in store.iter_records():
        key = (
            str(record.get("document_name")),
            str(record.get("entity")),
            str(record.get("metric")),
            str(record.get("period")),
            str(record.get("value")),
        )
        rows[key].add(str(record.get("row_id")))
        pages[key].add(record.get("page"))

    return {
        key: {"rows": len(rows[key]), "pages": len(pages[key]),
              "pages_list": sorted(p for p in pages[key] if p is not None)}
        for key in rows
    }


def _recurrence(index, record, value: str) -> dict:
    """How widely the filing restates this quantity at this coordinate.

    The consolidated figure recurs across the report -- statement, highlights,
    MD&A, notes.  A segment cell appears once, in the table that breaks the
    concept down.
    """

    key = (
        str(record.get("document_name")),
        str(record.get("entity")),
        str(record.get("metric")),
        str(record.get("period")),
        str(value),
    )
    return index.get(key, {"rows": 0, "pages": 0, "pages_list": []})


def _classify(record, comparable, recurrence) -> tuple[str, str]:
    """A first-pass classification, with the reason it was reached.

    One comparable value at the coordinate needs no judgement: the store
    already identifies it.  Otherwise the recurrence rule decides, and it is
    recorded as the reason so a source read can overrule it.
    """

    from src.finance.operand_ambiguity import quantity_of

    distinct = {quantity_of(item).identity for item in comparable}
    if len(distinct) <= 1:
        return "VALID_GOLD", "one comparable value at the coordinate"

    gold_value = str(record.get("value"))
    gold_rec = recurrence[gold_value]
    rivals = {v: r for v, r in recurrence.items() if v != gold_value}
    if not rivals:
        return "UNRESOLVED_GROUND_TRUTH", "no rival value to compare against"

    rival_value, rival_rec = max(
        rivals.items(), key=lambda item: (item[1]["pages"], item[1]["rows"])
    )
    same_row = [item for item in comparable if item.get("row_id") == record.get("row_id")]
    same_row_values = {str(item.get("value")) for item in same_row}
    flattened = "flattened row" if len(same_row_values) > 1 else "across rows"

    if gold_rec["pages"] > rival_rec["pages"]:
        return (
            "VALID_GOLD",
            f"gold {gold_value} is restated on {gold_rec['pages']} page(s) at this "
            f"coordinate; the widest rival {rival_value} on {rival_rec['pages']} -- "
            f"the conflict comes from {flattened}, but the gold is the value the "
            f"filing restates",
        )
    if rival_rec["pages"] > gold_rec["pages"]:
        return (
            "WRONG_SCOPE_GOLD",
            f"gold {gold_value} recurs on {gold_rec['pages']} page(s); rival "
            f"{rival_value} recurs on {rival_rec['pages']} -- the gold took one cell "
            f"of a {flattened}, the rival is the figure the filing restates",
        )
    return (
        "UNRESOLVED_GROUND_TRUTH",
        f"gold {gold_value} and rival {rival_value} are both restated on "
        f"{gold_rec['pages']} page(s); recurrence does not separate them",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, default=_BACKEND_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fact-store", type=Path, default=None)
    args = parser.parse_args(argv)

    import os

    from src.runtime.trusted_v2_production import StructuredFactStore

    store_path = args.fact_store or Path(os.environ["TRUSTED_V2_FACT_STORE_PATH"])
    store = StructuredFactStore(store_path)
    recurrence_index = _build_recurrence_index(store)

    gold = {
        case_id: row
        for case_id, row in _gold_rows(args.backend).items()
        if case_id.startswith(STRATUM_PREFIX)
    }

    cases: list[dict] = []
    tally: collections.Counter = collections.Counter()

    for case_id in sorted(gold):
        row = gold[case_id]
        slots: list[dict] = []
        for fact_id in row.get("fact_ids") or []:
            record = store._by_candidate.get(fact_id)
            if record is None:
                slots.append({"gold_fact_id": fact_id, "classification": "MISSING_FACT"})
                tally["MISSING_FACT"] += 1
                continue

            comparable = _comparable_siblings(store, record)
            values = sorted({str(item.get("value")) for item in comparable})
            recurrence = {
                value: _recurrence(recurrence_index, record, value)
                for value in values
            }

            label, reason = _classify(record, comparable, recurrence)
            same_row = [
                {
                    "value": item.get("value"),
                    "page": item.get("page"),
                    "row_id": item.get("row_id"),
                    "evidence_id": item.get("evidence_id"),
                }
                for item in comparable
                if item.get("row_id") == record.get("row_id")
            ]
            slots.append(
                {
                    "gold_fact_id": fact_id,
                    "entity": record.get("entity"),
                    "metric": record.get("metric"),
                    "period": record.get("period"),
                    "gold_value": record.get("value"),
                    "page": record.get("page"),
                    "document": record.get("document_name"),
                    "row_id": record.get("row_id"),
                    "table_fragment_id": record.get("table_fragment_id"),
                    "source_text": record.get("source_text"),
                    "comparable_distinct_values": values,
                    "distinct_comparable": len(values),
                    "competing_values": recurrence,
                    "same_row_cells": same_row,
                    "table_header_candidates": _header_candidates(store, record),
                    "classification": label,
                    "classification_reason": reason,
                }
            )
            tally[label] += 1

        # A case is only as good as its weakest gold operand: one wrong-scope
        # gold under a question makes the whole case unable to judge the system.
        labels = {slot["classification"] for slot in slots}
        if "WRONG_SCOPE_GOLD" in labels:
            case_label = "WRONG_SCOPE_GOLD"
        elif "UNRESOLVED_GROUND_TRUTH" in labels:
            case_label = "UNRESOLVED_GROUND_TRUTH"
        elif labels == {"VALID_GOLD"}:
            case_label = "VALID_GOLD"
        else:
            case_label = "OTHER"

        cases.append(
            {
                "case_id": case_id,
                "question": row.get("question"),
                "operation": row.get("operation"),
                "gold_values": row.get("values"),
                "case_classification": case_label,
                "slots": slots,
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    artifact = {
        "phase": "P1.6-0C",
        "purpose": "cross-entity gold validity audit",
        "fact_store": str(store_path),
        "stratum": STRATUM_PREFIX,
        "slot_tally": dict(tally),
        "case_tally": dict(
            collections.Counter(case["case_classification"] for case in cases)
        ),
        "cases": cases,
    }
    (args.out / "cross-entity-gold-validity.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"=== P1.6-0C cross-entity gold validity  ({len(cases)} cases) ===")
    print(f"  case tally: {artifact['case_tally']}")
    print(f"  slot tally: {artifact['slot_tally']}")
    print()
    for case in cases:
        print(f"  {case['case_id']:28} {case['case_classification']}")
        for slot in case["slots"]:
            if slot["classification"] == "VALID_GOLD":
                continue
            print(
                f"      {str(slot['entity'])[:18]:18} gold={str(slot['gold_value'])[:12]:12} "
                f"{slot['classification']}"
            )
            print(f"          {slot['classification_reason']}")
    print()
    print(f"  written to {args.out / 'cross-entity-gold-validity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
