"""P1.6-A3-W5 acceptance: does the *why* survive the store boundary, unchanged?

W5 is a persistence migration and nothing else, so it has two things to prove and they pull
in opposite directions.  The provenance it adds has to be **complete and resolvable**, and
the store it adds it to has to be **otherwise untouched**.  A phase that got the first
without the second would be a behaviour change wearing a schema badge.

Four checks:

  1. provenance integrity   every stored fact that was admitted on a binding carries the
                            method, scope, cells and granularity that binding had
  2. resolvability          every source cell points at a filing, table, row and column
                            that exists -- a provenance record that cannot be followed
                            back is a string that resembles provenance
  3. round-trip             producer object -> AtomicFact -> store record -> reloaded
                            record, field by field, on the four cases this line spent
                            W3 and W4 verifying
  4. behaviour invariance   the count, and every field that existed before, are identical
                            to the W4-B2 store; only the new fields differ

  python accept_provenance_persistence.py --store <w5>/store-v2.jsonl --previous <w4b2>/store-v2.jsonl --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

BUILDER = _BACKEND_DIR / "scripts/evaluation/build_store_v2.py"

#: Every field W5 adds.  Anything outside this set differing between the W4-B2 store and
#: the W5 store would mean the phase did more than persist provenance.
W5_FIELDS = {
    "period_binding_method", "period_target_scope", "period_source_cells",
    "period_conflict_candidates", "temporal_kind", "legacy_temporal_kind",
    "temporal_kind_method", "temporal_kind_source_cells", "temporal_kind_matched_text",
}

#: The four cases this line verified in W3 and W4, selected by the evidence they must
#: carry rather than by a cell coordinate, so a re-parse that moves a row does not silently
#: turn the check into a no-op.  Each is (document, method, target_scope, what it proves).
ORACLES = (
    ("ko_fy2025", "YEAR_ONLY_PERIOD", "CELL_GROUP",
     "a bare year, the case that made PARTIAL a status of its own"),
    ("pfe_fy2024", "INLINE_PERIOD_DATA_ROW", "ROW",
     "the row states its own period and overrides the column above it"),
    ("v_fy2025", "HEADER_ROW_SELECTION_EXTEND", "COLUMN",
     "the fiscal year end is declared in the header row, not the column"),
    ("jpm_fy2025", "ADJACENT_YEAR_JOIN", "COLUMN",
     "a month-day fragment joined to the column's own year"),
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_store(path: Path) -> list[dict]:
    return [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical(view: dict) -> dict:
    """Project any of the four views onto the same seven provenance facts.

    The four do not use the same names -- the producer's payload says `method`, the fact
    says `period_binding_method` because `temporal_kind` was taken, the store says
    `temporal_kind` where the fact says `binding_temporal_kind`.  Normalizing here is what
    lets the comparison be field-by-field rather than shape-by-shape, and it is stated
    rather than implied for the same reason the round-trip exists at all.
    """
    # `temporal_kind` is the one name that means different things in different views: the
    # fact carries the legacy axis kind under it and the A3 kind under
    # `binding_temporal_kind`, while the store has only the A3 kind and calls it
    # `temporal_kind`.  Keyed on presence, not truthiness -- a fact whose A3 kind is absent
    # must not fall through to the axis kind and compare equal by accident.
    kind = (view["binding_temporal_kind"] if "binding_temporal_kind" in view
            else view.get("temporal_kind"))
    return {
        "method": view.get("period_binding_method", view.get("method")),
        "target_scope": view.get("period_target_scope", view.get("target_scope")),
        "source_cells": [dict(c) for c in view.get("period_source_cells")
                         or view.get("source_cells") or ()],
        "temporal_kind": kind,
        "temporal_kind_method": view.get("temporal_kind_method"),
        "temporal_kind_source_cells": [dict(c) for c in
                                       view.get("temporal_kind_source_cells") or ()],
        "temporal_kind_matched_text": view.get("temporal_kind_matched_text"),
    }


def from_payload(payload: dict | None) -> dict:
    """The producer's own binding payload, in the same canonical shape."""
    if not payload or "binding" not in payload:
        return {}
    binding = payload["binding"]
    temporal = binding.get("temporal") or {}
    return {
        "method": binding.get("method"),
        "target_scope": binding.get("target_scope"),
        "source_cells": [dict(c) for c in binding.get("source_cells") or ()],
        "temporal_kind": temporal.get("kind"),
        "temporal_kind_method": temporal.get("method"),
        "temporal_kind_source_cells": [dict(c) for c in temporal.get("source_cells") or ()],
        "temporal_kind_matched_text": temporal.get("matched_text"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    builder = _module(BUILDER, "builder")
    records = read_store(args.store)
    previous = {str(r["cell_id"]): r for r in read_store(args.previous)}
    by_cell = {str(r["cell_id"]): r for r in records}

    failures: list[str] = []
    report: dict = {"phase": "P1.6-A3-W5-accept", "records": len(records)}

    # --- 1. provenance integrity -------------------------------------------------------
    print("=== 1. provenance integrity ===")
    missing = collections.Counter()
    checked = 0
    conflicts = 0
    for record in records:
        status = str(record.get("period_binding_status") or "")
        if status not in ("RESOLVED", "PARTIAL"):
            missing[f"status {status!r} in the store at all"] += 1
            continue
        checked += 1
        for field, want in (("period_binding_method", "present"),
                            ("period_target_scope", "present"),
                            ("period_granularity", "present"),
                            ("normalized_period", "present")):
            if not record.get(field):
                missing[f"{field} {want}"] += 1
            elif field == "period_granularity" and record[field] == "UNKNOWN":
                missing["period_granularity is UNKNOWN"] += 1
        if not record.get("period_source_cells"):
            missing["period_source_cells non-empty"] += 1
        if record.get("period_conflict_candidates"):
            conflicts += 1
        # The kind condition, on the A3 kind the method and cells belong to.
        if str(record.get("temporal_kind") or "UNKNOWN") != "UNKNOWN":
            if not record.get("temporal_kind_method"):
                missing["temporal_kind_method present"] += 1
            if not record.get("temporal_kind_source_cells"):
                missing["temporal_kind_source_cells non-empty"] += 1

    print(f"    facts admitted on a binding: {checked}")
    for name, count in missing.most_common():
        print(f"    {count:>6}  missing {name}")
    if not missing:
        print("    every one carries method, scope, cells, granularity and period")
    print(f"    facts carrying conflict candidates: {conflicts}")
    if conflicts:
        failures.append(f"CONFLICT_CANDIDATES_ON_A_STORED_FACT = {conflicts}")
    if missing:
        failures.append(f"PROVENANCE_INCOMPLETE = {dict(missing)}")
    report["integrity"] = {"checked": checked, "missing": dict(missing),
                           "conflicts": conflicts}
    print()

    # --- 2. resolvability ---------------------------------------------------------------
    #
    # A source cell is only provenance if the coordinates can be followed.  The check is
    # the strong one -- parse the filing again and confirm the table exists and the cell
    # is inside its grid -- rather than "the fields are non-empty", which a tuple of
    # plausible-looking numbers would also satisfy.
    print("=== 2. every source cell resolves to a filing, table, row and column ===")
    grids: dict[tuple[str, str], tuple[int, int]] = {}
    parsed_by_doc: dict[str, dict] = {}
    for document_id in sorted(builder.DOCUMENTS):
        ticker, accession = builder.DOCUMENTS[document_id]
        parsed = builder.parse_filing(ticker, accession, document_id)
        parsed_by_doc[document_id] = parsed
        for table in parsed["tables"]:
            extent = [0, 0]
            for cell in table.get("cells") or ():
                extent[0] = max(extent[0], int(cell.get("row_index") or 0))
                extent[1] = max(extent[1], int(cell.get("column_index") or 0))
            grids[(document_id, str(table["table_id"]))] = (extent[0], extent[1])

    problems = collections.Counter()
    sampled = 0
    for record in records:
        for cell in record.get("period_source_cells") or ():
            sampled += 1
            document_id = str(cell.get("document_id") or "")
            table_id = str(cell.get("table_id") or "")
            if document_id not in parsed_by_doc:
                problems["document_id names no filing"] += 1
                continue
            extent = grids.get((document_id, table_id))
            if extent is None:
                problems["table_id names no table in that filing"] += 1
                continue
            row, column = int(cell.get("row") or 0), int(cell.get("column") or 0)
            if not (0 <= row <= extent[0] and 0 <= column <= extent[1]):
                problems[f"row/column outside the grid {extent}"] += 1
    print(f"    source cells checked: {sampled}")
    for name, count in problems.most_common():
        print(f"    {count:>6}  {name}")
    if not problems:
        print("    every one resolves to a table that exists and a cell inside it")
    if problems:
        failures.append(f"UNRESOLVABLE_SOURCE_CELL = {dict(problems)}")
    report["resolvability"] = {"checked": sampled, "problems": dict(problems)}
    print()

    # --- 3. round-trip, on the four cases this line verified ---------------------------
    print("=== 3. round-trip: producer -> AtomicFact -> record -> reloaded ===")
    round_trips = {}
    for document_id, method, scope, why in ORACLES:
        candidates = [r for r in records
                      if str(r.get("source", {}).get("document_id")) == document_id
                      and r.get("period_binding_method") == method]
        if not candidates:
            print(f"    MISSING {document_id} {method}")
            failures.append(f"ORACLE_NOT_FOUND {document_id}/{method}")
            continue
        record = candidates[0]
        cell_id = str(record["cell_id"])
        parsed = parsed_by_doc[document_id]

        # 1. what the producer decided, read back off the parsed cell
        parsed_cell = next((c for table in parsed["tables"]
                            for c in table.get("cells") or ()
                            if str(c.get("cell_id")) == cell_id), None)
        if parsed_cell is None:
            failures.append(f"ORACLE_CELL_NOT_IN_REPARSE {document_id}/{cell_id}")
            continue
        step1 = from_payload(parsed_cell.get("period_binding_v2"))

        # 2. what the fact carries
        from src.pdf_retrieval_v4.html_semantic_adapter import build_semantic_corpus
        corpus = build_semantic_corpus([{**parsed, **parsed["document"]}])
        atomic = next((f for f in corpus["atomic_facts"]
                       if str(f.cell_id) == cell_id), None)
        step2 = canonical(atomic.to_dict()) if atomic is not None else {}

        # 3. what the store record carries, and 4. what it carries after a reload
        step3 = canonical(record)
        step4 = canonical(json.loads(json.dumps(record)))

        views = {"producer": step1, "fact": step2, "record": step3, "reloaded": step4}
        agree = all(view == step1 for view in views.values())
        differs = [name for name, view in views.items() if view != step1]
        mark = "ok " if agree else "?? "
        print(f"    {mark} {document_id:12} {method:28} {scope:10} "
              f"{len(step1.get('source_cells') or ())} source cells")
        print(f"         {why}")
        if not agree:
            print(f"         views that differ from the producer: {differs}")
            for name in differs:
                print(f"           {name}: {json.dumps(views[name], sort_keys=True)[:200]}")
            failures.append(f"ROUND_TRIP_DIFFERS {document_id}/{method} {differs}")
        round_trips[document_id] = {"method": method, "target_scope": step1["target_scope"],
                                    "source_cells": len(step1.get("source_cells") or ()),
                                    "temporal_kind": step1.get("temporal_kind"),
                                    "views_agree": agree}
        if step1.get("target_scope") != scope:
            failures.append(f"ORACLE_SCOPE {document_id}: {step1.get('target_scope')} != {scope}")
    report["round_trips"] = round_trips
    print()

    # --- 4. behaviour invariance --------------------------------------------------------
    print("=== 4. the store is otherwise untouched ===")
    print(f"    records {len(records)}   previous {len(previous)}")
    if len(records) != len(previous):
        failures.append(f"COUNT_CHANGED {len(previous)} -> {len(records)}")
    if set(by_cell) != set(previous):
        failures.append("THE SET OF CELLS CHANGED")
    drifted = collections.Counter()
    for cell_id in set(by_cell) & set(previous):
        before, after = previous[cell_id], by_cell[cell_id]
        for field in sorted(set(before) | set(after)):
            if before.get(field) != after.get(field):
                drifted[field] += 1
    unexpected = {f: c for f, c in drifted.items() if f not in W5_FIELDS}
    print(f"    fields that differ from the W4-B2 store: {dict(drifted) or 'none'}")
    if unexpected:
        print(f"    outside the fields W5 adds: {unexpected}")
        failures.append(f"UNEXPECTED_FIELD_DRIFT = {unexpected}")
    else:
        print("    every difference is a W5 addition; nothing else moved")
    report["invariance"] = {"records": len(records), "drifted": dict(drifted),
                            "unexpected": unexpected}
    print()

    print(f"  failures: {failures or 'none'}")
    report["failures"] = failures
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "provenance-acceptance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'provenance-acceptance.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
