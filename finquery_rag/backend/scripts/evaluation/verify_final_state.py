"""P1.6-A3-W6-B: the whole system, verified end to end against frozen expectations.

W1 to W5 each verified their own layer.  This asks whether the assembled system still says
what those layers said, and it holds the expectations as **frozen constants** rather than
reading them back out of the artifacts it is checking -- a validator that derives its
expected values from the thing under test cannot fail.

Four groups:

  1. the nine tables   the nine verified primary statements that produced no store records
                       at A3-0, each of which must now have a stated outcome.  A table may
                       be recovered or it may be withheld for a reason; what it may not be
                       is unexplained
  2. store final state 26,311 records and the addition/removal classes, still kept apart
  3. provenance        26,311 / 26,311 complete, every source cell resolvable by re-parsing
  4. the 47 slots      recomputed here rather than read from a summary

  python verify_final_state.py --store <dir>/store-v2.jsonl --accounting <json> \
      --funnel <json> --baseline-slots <json> --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import re
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

EVAL = _BACKEND_DIR / "scripts/evaluation"
BUILDER = EVAL / "build_store_v2.py"
RESOLVER = EVAL / "resolve_store_v2_slot.py"
CROSS = EVAL / "resolve_cross_entity_v2.py"

#: Frozen.  These are the numbers W4 and W5 established, written down here so this file
#: fails when they move rather than learning the new value and passing.
EXPECTED_RECORDS = 26311
EXPECTED_ADDITIONS = {("PARTIAL", "YEAR"): 3368, ("RESOLVED", "DAY"): 3018}
EXPECTED_REMOVALS = {"LEGACY_FALSE_POSITIVE": 691, "SOURCE_AMBIGUOUS": 4188,
                     "VALID_BUT_OUT_OF_SCOPE_GEOMETRY": 2173}
EXPECTED_SLOTS = {"RESOLVED_COMPANY_LEVEL": 37, "NO_COMPANY_LEVEL_FACT": 5, "NO_FACT": 5}
SLOT_GATES = ("wrong_scope", "wrong_metric", "value_mismatch", "dangerous_authority",
              "scaffold_authority")

RESOLVED = "RESOLVED_COMPANY_LEVEL"

#: The nine zero-record primary statements A3-0 found, by the funnel key that names them.
NINE_TABLES = ("jpm_fy2025#63193", "jpm_fy2025#64546", "ko_fy2025#11388",
               "ko_fy2025#11899", "ko_fy2025#12533", "nvda_fy2025#8766",
               "pfe_fy2024#24395", "tsla_fy2025#10407", "v_fy2025#9951")

#: Why the production decision withheld a cell -> the outcome that names it.
#:
#: The removal audit cannot answer this.  These tables produced no records under the
#: *legacy* path either, so their cells were never in the old store, never became
#: "removals", and were never adjudicated -- the join finds nothing for exactly the cells
#: whose explanation matters.  So the explanation is taken from `decide_emission_admission`
#: itself, which is the authority the outcome is supposed to be about.
#:
#: Note what is **not** here: `DISAGGREGATION_AXIS` is deliberately absent.  It is a claim
#: that the column is a breakdown rather than a period, and that claim can be false -- the
#: four cash flow statements below are withheld on it because the legacy classifier reads
#: `Year ended December 31, / 2025` as a *bucket*.  Mapping it to "out of scope geometry"
#: would turn a misclassification into an explanation, so it is resolved against the
#: column's own text instead.
REASON_OUTCOME = {
    "NON_PERIOD_AXIS": "VALID_WITHHELD_OUT_OF_SCOPE_GEOMETRY",
    "CONFLICTED_PERIOD": "VALID_WITHHELD_SOURCE_AMBIGUOUS",
    "UNRESOLVED_PERIOD": "VALID_WITHHELD_SOURCE_AMBIGUOUS",
    "NO_BINDING": "VALID_WITHHELD_SOURCE_AMBIGUOUS",
    "INCOMPLETE_PROVENANCE": "VALID_WITHHELD_INCOMPLETE_PROVENANCE",
}

#: A column header that names a year is a column that states a period, whatever kind the
#: legacy classifier gave it.  This is the test that separates "the column is genuinely a
#: breakdown" from "the column is a period the classifier mislabelled".
_STATES_A_PERIOD = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_store(path: Path) -> list[dict]:
    return [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--funnel", type=Path, required=True)
    parser.add_argument("--baseline-slots", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    builder = _module(BUILDER, "builder")
    resolver = _module(RESOLVER, "resolver")
    cross = _module(CROSS, "cross")
    from src.pdf_retrieval_v4.html_semantic_adapter import build_semantic_corpus

    records = read_store(args.store)
    by_cell = {str(r["cell_id"]): r for r in records}
    accounting = json.loads(args.accounting.read_text(encoding="utf-8"))
    funnel = json.loads(args.funnel.read_text(encoding="utf-8"))

    failures: list[str] = []
    report: dict = {"phase": "P1.6-A3-W6-B"}

    # Which parsed tables exist, and which cells each holds, so a table can be joined to
    # the cells it is made of.  Parsed once per document and reused by groups 1 and 3.
    parsed_by_doc: dict[str, dict] = {}
    corpora: dict[str, dict] = {}
    grids: dict[tuple[str, str], tuple[int, int]] = {}
    for document_id in sorted(builder.DOCUMENTS):
        ticker, accession = builder.DOCUMENTS[document_id]
        parsed = builder.parse_filing(ticker, accession, document_id)
        parsed_by_doc[document_id] = parsed
        # The same corpus the store build consumes, so the admission reasons group 1
        # tallies are the ones the production path produced.
        corpora[document_id] = build_semantic_corpus(
            [{**parsed, **parsed["document"]}])
        for table in parsed["tables"]:
            table_id = str(table["table_id"])
            extent = [0, 0]
            for cell in table.get("cells") or ():
                extent[0] = max(extent[0], int(cell.get("row_index") or 0))
                extent[1] = max(extent[1], int(cell.get("column_index") or 0))
            grids[(document_id, table_id)] = (extent[0], extent[1])

    # --- 1. the nine tables --------------------------------------------------------------
    print("=== 1. the nine zero-record primary statements, each with an outcome ===")
    # oracle_key -> table_fragment_id, from the funnel that named them.
    key_to_table: dict[str, str] = {}
    for document_id, tables in funnel["documents"].items():
        for table in tables:
            key_to_table[str(table["oracle_key"])] = str(table["table_fragment_id"])

    from src.pdf_retrieval_v4.period_binding import (
        AdmissionRequest, binding_from_payload, decide_emission_admission, temporal_kind_of)
    from src.pdf_retrieval_v4.typed_evidence_emitters import _get_numeric_value

    # Replay the emitter's own filter, in its own order, so the reasons tallied here are
    # the reasons that actually withheld the cell and not a re-derivation of them.
    def reasons_for(table_id: str) -> tuple[collections.Counter, int, int]:
        """(withheld reasons, withheld cells, withheld cells whose column states a year)."""
        tally: collections.Counter = collections.Counter()
        withheld = 0
        period_columns = 0
        for corpus in corpora.values():
            # `table_meta` is keyed by (document_id, table_fragment_id); the store and the
            # funnel name the table by the fragment id alone, and it is unique here.
            table = next((meta for (doc, tid), meta in corpus["table_meta"].items()
                          if tid == table_id), None)
            if table is None:
                continue
            sr_by_id = {sr.row_id: sr for sr in corpus["semantic_rows"]
                        if sr.table_fragment_id == table_id}
            mp_by_row = {mp.row_id: mp for mp in corpus["metric_paths"]}
            axis_by_cell = {a.cell_id: a for a in corpus["axis_bindings"]}
            for cell in table["cells"]:
                if int(cell.get("column_index") or 0) == 0:
                    continue
                row_id = str(cell.get("row_id") or "")
                sr = sr_by_id.get(row_id)
                if not sr or not sr.is_financial_data_row:
                    continue
                raw_val, norm_val = _get_numeric_value(cell)
                if not raw_val or norm_val is None:
                    continue
                axis = axis_by_cell.get(str(cell.get("cell_id") or ""))
                if not axis:
                    continue
                mp = mp_by_row.get(row_id)
                admission = decide_emission_admission(AdmissionRequest(
                    binding=binding_from_payload(cell.get("period_binding_v2")),
                    temporal_kind=temporal_kind_of(axis.temporal_kind),
                    metric_path=(mp.metric_path if mp else None),
                    metric_status=(mp.metric_status if mp else None),
                    value_normalized=norm_val,
                    cell_id=str(cell.get("cell_id") or "")))
                if not admission.admitted:
                    tally[admission.reason.value if admission.reason else "NO_REASON"] += 1
                    withheld += 1
                    if _STATES_A_PERIOD.search(str(cell.get("column_header") or "")):
                        period_columns += 1
        return tally, withheld, period_columns

    funnel_rows = {}
    unexplained = 0
    disputed = []
    for key in NINE_TABLES:
        table_id = key_to_table.get(key)
        if table_id is None:
            failures.append(f"THE FUNNEL NO LONGER NAMES {key}")
            continue
        in_store = sum(1 for r in records if str(r.get("table_fragment_id")) == table_id)
        reasons, withheld, period_columns = reasons_for(table_id)
        outcomes = {REASON_OUTCOME.get(r, f"UNMAPPED_REASON {r!r}") for r in reasons}
        # A routing refusal is only an explanation if the column really is a breakdown.
        # When every withheld cell sits in a column that states a year, the kind is wrong
        # and the withdrawal is a classifier defect -- explained, but not a scope decision.
        misclassified = reasons.get("DISAGGREGATION_AXIS", 0) and period_columns == withheld
        if in_store:
            outcome = "RECOVERED"
        elif not reasons:
            outcome = "UNEXPLAINED_ZERO_RECORD"
            unexplained += 1
        elif misclassified:
            outcome = "WITHHELD_ON_LEGACY_KIND"
            disputed.append(key)
            outcomes.discard("UNMAPPED_REASON 'DISAGGREGATION_AXIS'")
        elif len(outcomes) == 1:
            outcome = outcomes.pop()
        else:
            outcome = "MIXED:" + "+".join(sorted(outcomes))
        print(f"    {key:20} records {in_store:>5}   withheld {withheld:>5}   {outcome}")
        if misclassified:
            print(f"        every withheld cell sits in a column stating a year, and the "
                  f"legacy kind called it a bucket:")
            for reason, count in reasons.most_common():
                print(f"            {count:>5}  {reason}")
        else:
            for reason, count in reasons.most_common():
                print(f"            {count:>5}  {reason}")
        funnel_rows[key] = {"records": in_store, "outcome": outcome,
                            "withheld": dict(reasons),
                            "withheld_by_legacy_kind": misclassified}
    print(f"    tables with no outcome: {unexplained}")
    if unexplained:
        failures.append(f"UNEXPLAINED_ZERO_RECORD = {unexplained}")
    if disputed:
        print()
        print(f"    {len(disputed)} table(s) are withheld because the legacy classifier "
              f"mislabelled a period column;")
        print(f"    the mechanism is known and the outcome is stated, but the outcome is a")
        print(f"    defect rather than a scope decision:")
        for key in disputed:
            print(f"        {key}")
    report["nine_tables"] = funnel_rows
    report["withheld_on_legacy_kind"] = disputed
    print()

    # --- 2. store final state ------------------------------------------------------------
    print("=== 2. store final state ===")
    print(f"    records {len(records)}   (frozen expected {EXPECTED_RECORDS})")
    if len(records) != EXPECTED_RECORDS:
        failures.append(f"RECORDS {len(records)} != {EXPECTED_RECORDS}")
    by_identity = accounting["added_by_identity"]
    for (status, granularity), expected in sorted(EXPECTED_ADDITIONS.items()):
        got = by_identity.get(f"{status} / {granularity}", 0)
        print(f"    added {status} / {granularity:<6} {got:>5}  (frozen {expected})")
        if got != expected:
            failures.append(f"ADDED {status}/{granularity} {got} != {expected}")
    for name, expected in EXPECTED_REMOVALS.items():
        got = accounting["removed_by_class"].get(name, 0)
        print(f"    removed {name:<34} {got:>5}  (frozen {expected})")
        if got != expected:
            failures.append(f"REMOVED {name} {got} != {expected}")
    for name in ("unclassified_added", "unclassified_removed"):
        print(f"    {name} {accounting.get(name)}")
        if accounting.get(name):
            failures.append(f"{name.upper()} = {accounting[name]}")
    report["store"] = {"records": len(records),
                       "added_by_identity": by_identity,
                       "removed_by_class": accounting["removed_by_class"]}
    print()

    # --- 3. provenance --------------------------------------------------------------------
    print("=== 3. provenance integrity ===")
    bad = collections.Counter()
    complete = 0
    for record in records:
        status = str(record.get("period_binding_status") or "")
        if status not in ("RESOLVED", "PARTIAL"):
            bad[f"status {status!r}"] += 1
            continue
        if not all(record.get(f) for f in ("period_binding_method", "period_target_scope",
                                           "period_granularity", "normalized_period")):
            bad["incomplete period identity or binding provenance"] += 1
            continue
        if record["period_granularity"] == "UNKNOWN":
            bad["granularity UNKNOWN"] += 1
            continue
        if not record.get("period_source_cells"):
            bad["no source cells"] += 1
            continue
        complete += 1
    print(f"    period identity + binding provenance complete   {complete} / {len(records)}")
    if complete != len(records):
        failures.append(f"PROVENANCE_INCOMPLETE = {dict(bad)}")

    kinds = [r for r in records if str(r.get("temporal_kind") or "UNKNOWN") != "UNKNOWN"]
    kind_ok = [r for r in kinds
               if r.get("temporal_kind_method") and r.get("temporal_kind_source_cells")]
    print(f"    temporal provenance where kind != UNKNOWN      {len(kind_ok)} / {len(kinds)}")
    if len(kind_ok) != len(kinds):
        failures.append(f"TEMPORAL_PROVENANCE_INCOMPLETE = {len(kinds) - len(kind_ok)}")

    unresolvable = collections.Counter()
    checked = 0
    for record in records:
        for cell in record.get("period_source_cells") or ():
            checked += 1
            document_id = str(cell.get("document_id") or "")
            table_id = str(cell.get("table_id") or "")
            extent = grids.get((document_id, table_id))
            if extent is None:
                unresolvable["names no table in that filing"] += 1
                continue
            row, column = int(cell.get("row") or 0), int(cell.get("column") or 0)
            if not (0 <= row <= extent[0] and 0 <= column <= extent[1]):
                unresolvable["outside the grid"] += 1
    print(f"    source cells resolvable by re-parsing          "
          f"{checked - sum(unresolvable.values())} / {checked}")
    if unresolvable:
        failures.append(f"UNRESOLVABLE_SOURCE_CELL = {dict(unresolvable)}")
    report["provenance"] = {"complete": complete, "records": len(records),
                            "temporal_kinds": len(kinds), "temporal_ok": len(kind_ok),
                            "source_cells": checked,
                            "unresolvable": dict(unresolvable)}
    print()

    # --- 4. the 47 slots, recomputed -------------------------------------------------------
    print("=== 4. the 47 slots, recomputed ===")
    baseline = json.loads(args.baseline_slots.read_text(encoding="utf-8"))
    base_by = {(r["case_id"], r["entity"]): r for r in baseline["slots"]}
    statuses = collections.Counter()
    regressions, moves = [], []
    for (case_id, entity), before in sorted(base_by.items()):
        metric = before["metric"]
        period = cross.PERIOD.get(entity, "FY2025")
        out = resolver.resolve(records, entity, metric, period,
                               resolver.AUTHORITY_TABLE_ROLE)
        statuses[out["status"]] += 1
        if before["new_status"] == RESOLVED and out["status"] != RESOLVED:
            regressions.append(f"{case_id}/{entity}")
        if before["new_status"] == RESOLVED == out["status"] and \
                str(before.get("new_value")) != str(out.get("value_raw")):
            moves.append(f"{case_id}/{entity}: {before.get('new_value')} -> {out.get('value_raw')}")
    for name, expected in sorted(EXPECTED_SLOTS.items()):
        got = statuses.get(name, 0)
        mark = "ok " if got == expected else "MOVED"
        print(f"    {mark} {name:<26} {got:>3}  (frozen {expected})")
        if got != expected:
            failures.append(f"SLOTS {name} {got} != {expected}")
    print(f"    any other status             "
          f"{sum(v for k, v in statuses.items() if k not in EXPECTED_SLOTS)}  (frozen 0)")
    print(f"    existing resolved regression {len(regressions)}")
    print(f"    resolved value changed       {len(moves)}")
    if regressions:
        failures.append(f"existing resolved regression = {len(regressions)}")
    if moves:
        failures.append(f"resolved value changed = {len(moves)}")

    # The gates, recomputed from the store rather than read from the evaluator's report.
    gate = collections.Counter()
    roles = json.loads((args.store.parent / "table-roles.json").read_text(encoding="utf-8"))
    for (case_id, entity), _before in sorted(base_by.items()):
        metric = _before["metric"]
        period = cross.PERIOD.get(entity, "FY2025")
        out = resolver.resolve(records, entity, metric, period,
                               resolver.AUTHORITY_TABLE_ROLE)
        if out["status"] != RESOLVED:
            continue
        record = by_cell.get(str(out.get("cell_id")))
        role = roles.get(str(out.get("table_fragment_id")), {})
        if str(out.get("table_role")) != "PRIMARY_FINANCIAL_STATEMENT":
            gate["wrong_scope"] += 1
        if not _same_metric(record or {}, metric):
            gate["wrong_metric"] += 1
        expected_value = cross.EXPECTED.get((case_id, entity))
        if expected_value is not None and str(out.get("value_raw")) != expected_value:
            gate["value_mismatch"] += 1
        if role.get("oracle_role") == "DANGEROUS_NEGATIVE":
            gate["dangerous_authority"] += 1
        if role.get("table_eligibility") == "LAYOUT_SCAFFOLD":
            gate["scaffold_authority"] += 1
    for name in SLOT_GATES:
        value = gate.get(name, 0)
        print(f"    {mark_if(value)} {name:<22} {value}")
        if value:
            failures.append(f"{name} = {value}")
    report["slots"] = {"statuses": dict(statuses), "gate": {n: gate.get(n, 0)
                                                            for n in SLOT_GATES},
                       "regressions": regressions, "value_moves": moves}
    print()

    print(f"  failures: {failures or 'none'}")
    report["failures"] = failures
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "final-state-verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'final-state-verification.json'}")
    return 1 if failures else 0


def mark_if(value: int) -> str:
    return "ok " if value == 0 else "FAIL"


def _same_metric(record: dict, metric: str) -> bool:
    """The resolver's own metric test, restated so the gate can use it."""
    import re

    caption = re.compile(r"\s*\((?:in|except|dollars in|amounts in)[^)]*\)\s*$", re.I)
    folded = caption.sub("", " ".join(str(record.get("row_label") or "").split()).casefold())
    return folded.strip() == " ".join(str(metric or "").split()).casefold()


if __name__ == "__main__":
    raise SystemExit(main())
