"""P1.6-A3-W4-A: the new admission decision, measured beside the authoritative one.

Measurement only.  **The legacy rule in `typed_evidence_emitters` still decides** -- the
emission is byte-identical after this runs.  What this produces is the delta W4-B would
cause, cell by cell and reason by reason, so that switching authority is a decision rather
than a discovery.

Switching admission authority moves **two** things at once, and a single old-vs-new count
would blend them:

    V1  the decision rule      the kind whitelist becomes a negative form; the period is
                               still the legacy axis's
    V2  V1 + the new period source   what W4-B actually does -- `period_binding_shadow`'s
                               binding replaces the legacy axis as WHERE the period is

    legacy -> V1   the rule change         one variable, and the one W4 is about
    V1     -> V2   the producer change     a different variable, and not W4's to claim

V1 is deliberately expressible without a binding at all: `kind in {point, duration,
comparison}` becomes `kind not in {segment, bucket, category, non_temporal}`.  That is the
entire rule change, it needs nothing invented, and it differs from the old expression on
exactly one kind -- `test_the_new_rule_differs_from_the_old_one_on_exactly_unknown` pins
that.  So `legacy -> V1` has no loss side at all by construction, and any loss the run
reports below would be a defect in this measurement rather than a property of W4.

(`comparison` was going to be a second removal.  Checking instead of assuming showed the
Store V2 path emits only atomic facts, so withholding a comparison cell would delete it
rather than reclassify it.  That is not a decoupling, it is a coverage removal, and it is
not W4's to make.)

The first smoke run is why V2 is reported apart.  On Apple alone it showed 386 cells the
legacy admitted and V2 withheld, and reading them named tables of contents and exhibit
indexes -- `Page / years ended September 27`, `Exhibit Number / as of August 20`.  The
legacy's axis classifier had read a date-shaped token out of their prose and minted a
period fact; the W2 binder refuses to bind a period to a table of contents because no
period header is there.  That is the *producer* being right, and attributing it to the
admission rule would have been a false account of both.

Both directions are reported throughout, and they are not symmetric:

    *_gain   the old rule withheld, the new one admits
    *_loss   the old rule admitted, the new one withholds   the risk direction

and the two deltas are computed **independently rather than as a chain**, because a cell
can belong to both: `kind:unknown` is gained by the rule and then withheld again by the
producer when its column has no binding.  A chain would file that cell under `rule_gain`
alone and credit the rule with a fact that never arrives; the overlap is counted on its own
as `rule_gain_then_producer_loss`.

W4's mandate is that admission stop reading temporal *shape*; it is not a licence to start
withholding facts the old rule got right, so losses are never folded into a net count.

  python emission_admission_shadow.py --out <dir> [--documents a,b]
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

from src.pdf_retrieval_v4.html_semantic_adapter import _adapt_table  # noqa: E402
from src.pdf_retrieval_v4.metric_path_builder import build_metric_paths  # noqa: E402
from src.pdf_retrieval_v4.period_binding import (  # noqa: E402
    DISAGGREGATION_KINDS,
    AdmissionRequest,
    PeriodBindingV2,
    TemporalKind,
    decide_emission_admission,
    resolve_period_evidence,
    temporal_kind_of,
)
from src.pdf_retrieval_v4.semantic_row_classifier import classify_table_rows  # noqa: E402
from src.pdf_retrieval_v4.temporal_axis_graph import build_axis_bindings  # noqa: E402
from src.pdf_retrieval_v4.typed_evidence_emitters import (  # noqa: E402
    ATOMIC_ELIGIBLE_KINDS,
    _get_numeric_value,
)

PARSER = _BACKEND_DIR / "scripts/evaluation/run_nf_v2_17a4_parse.py"
BUILDER = _BACKEND_DIR / "scripts/evaluation/build_store_v2.py"
PB_SHADOW = _BACKEND_DIR / "scripts/evaluation/period_binding_shadow.py"
TK_SHADOW = _BACKEND_DIR / "scripts/evaluation/temporal_kind_shadow.py"

CORPUS = Path("/disk/qh/nano-finrag/data/financial_corpus_v2/raw/SEC")

#: Read straight out of the authoritative modules rather than restated, so this file
#: cannot drift from the rules it is measuring against.  Both sides are held as the
#: contract's enum: the legacy classifier's vocabulary is lower case and the enum's
#: `UNKNOWN`/`YEAR` are not, and comparing the raw strings would silently fail to match.
LEGACY_ELIGIBLE = frozenset(temporal_kind_of(k) for k in ATOMIC_ELIGIBLE_KINDS)
NEW_INELIGIBLE = (frozenset(DISAGGREGATION_KINDS)
                  | {TemporalKind.NON_TEMPORAL})


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _label(key) -> str:
    """A reason key as one line.  Reasons are None on an admitted cell, and a bare `join`
    over the tuple would raise rather than print the None."""
    return " | ".join(str(part) for part in key)


def legacy_verdict(mp, axis) -> tuple[bool, str]:
    """The old rule, in the old order: `emit_atomic_facts`' kind whitelist.

    The order -- kind before metric -- is kept because the blocker is what a reader
    compares across columns, and reordering it would make two reports incomparable.
    """
    if not axis:
        return False, "no_axis_binding"
    kind = temporal_kind_of(axis.temporal_kind)
    if kind is None:
        return False, f"unmappable_kind:{axis.temporal_kind}"
    if kind not in LEGACY_ELIGIBLE:
        return False, f"kind:{axis.temporal_kind}"
    if not mp:
        return False, "no_metric_path"
    if mp.metric_status == "missing":
        return False, "metric_status_missing"
    return True, ""


def v1_verdict(mp, axis) -> tuple[bool, str]:
    """The rule change alone: the same chain with the whitelist turned into its negative.

    Nothing else moves.  The period is still whatever the legacy axis says, so a
    difference between this and `legacy_verdict` is the admission rule and cannot be
    anything else.
    """
    if not axis:
        return False, "no_axis_binding"
    kind = temporal_kind_of(axis.temporal_kind)
    if kind is None:
        # Fail closed.  An unrecognised kind is a gap in this bridge, not a licence to
        # admit: treating "I cannot read this" as "not a disaggregation" is how a
        # vocabulary mismatch turns into an admission.
        return False, f"unmappable_kind:{axis.temporal_kind}"
    if kind in NEW_INELIGIBLE:
        return False, f"kind:{axis.temporal_kind}"
    if not mp:
        return False, "no_metric_path"
    if mp.metric_status == "missing":
        return False, "metric_status_missing"
    return True, ""


def period_for(cell, bound):
    """The period the new producer states for this cell's coordinate.

    A row that states its own period shadows the column context above it -- scope
    semantics, not precedence -- which is `resolve_period_evidence`'s job and why the row
    binding is passed as the declaration and the column binding as inherited.
    """
    col = int(cell.get("column_index") or 0)
    row_index = int(cell.get("row_index") or 0)
    column_binding = bound["columns"].get(col)
    row_binding = bound["rows"].get(row_index)
    inherited = [column_binding] if isinstance(column_binding, PeriodBindingV2) else []
    if isinstance(row_binding, PeriodBindingV2):
        return resolve_period_evidence(row_binding, inherited)
    return inherited[0] if inherited else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--documents", default="")
    args = parser.parse_args(argv)

    nf = load("nf17a4", PARSER)
    builder = load("store_builder", BUILDER)
    pb = load("pb_shadow", PB_SHADOW)
    tk = load("tk_shadow", TK_SHADOW)

    from src.pdf_retrieval_v4 import temporal_axis_graph as tag
    from lxml import etree, html

    wanted = [d for d in args.documents.split(",") if d] or sorted(builder.DOCUMENTS)

    report = {"phase": "P1.6-A3-W4-A", "mutation": "none", "shadow_only": True,
              "note": "the legacy rule in typed_evidence_emitters still decides",
              "documents": {}, "totals": {}}
    tally = collections.Counter()
    rule_gain: list[dict] = []
    rule_loss: list[dict] = []
    producer_gain: list[dict] = []
    producer_loss: list[dict] = []

    print("=== old vs new admission decision (shadow) ===")
    print()

    for document_id in wanted:
        ticker, accession = builder.DOCUMENTS[document_id]
        parsed = builder.parse_filing(ticker, accession, document_id)
        roles = builder.table_roles(document_id, parsed["blocks"])
        root = html.parse(str(CORPUS / ticker / accession / "primary.html"),
                          etree.HTMLParser(recover=True, no_network=True, huge_tree=True,
                                           remove_comments=True)).getroot()
        _blocks, lookup, _prior = nf.make_blocks(
            root, {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"})
        blocks_by_order = {b["source_order"]: b for b in parsed["blocks"]
                           if b["block_type"] == "TABLE"}

        doc_tally = collections.Counter()
        for table in parsed["tables"]:
            order = table.get("source_order")
            block = blocks_by_order.get(order)
            if block is None or block["table_id"] not in lookup:
                continue
            grid = nf.grid_rows(nf.direct_rows(lookup[block["table_id"]]))
            bound = pb.bind_table(nf, grid, document_id, block["table_id"])
            union, _added = pb.extended_header_idx(nf, grid)
            duration_phrase = tk.header_duration_phrase(nf, grid, union)
            column_evidence = tk.column_evidence(nf, grid, union)

            adapted = _adapt_table(table, parsed)
            tid = adapted["table_fragment_id"]
            rows = classify_table_rows(adapted, tid, adapted["document_id"], 0)
            paths = build_metric_paths(rows)
            axes = build_axis_bindings(adapted["cells"], tid)

            sr_by_row = {sr.row_id: sr for sr in rows}
            mp_by_row = {mp.row_id: mp for mp in paths}
            axis_by_cell = {a.cell_id: a for a in axes}
            role = roles.get(block["table_id"], {}).get("oracle_role")

            for cell in adapted["cells"]:
                col = int(cell.get("column_index") or 0)
                if col == 0:
                    continue
                row_id = str(cell.get("row_id") or "")
                sr = sr_by_row.get(row_id)
                if not sr or not sr.is_financial_data_row:
                    continue
                raw_val, norm_val = _get_numeric_value(cell)
                if not raw_val or norm_val is None:
                    continue

                axis = axis_by_cell.get(str(cell.get("cell_id") or ""))
                mp = mp_by_row.get(row_id)
                legacy_kind = temporal_kind_of(axis.temporal_kind) if axis else None
                new_kind = tk.classify(column_evidence.get(col, ("", ()))[0], tag,
                                       bound["columns"].get(col), duration_phrase).kind
                binding = period_for(cell, bound)
                admission = decide_emission_admission(AdmissionRequest(
                    binding=binding, temporal_kind=legacy_kind,
                    metric_path=(mp.metric_path if mp else None),
                    metric_status=(mp.metric_status if mp else None),
                    value_normalized=norm_val,
                    cell_id=str(cell.get("cell_id") or "")))

                legacy_admits, legacy_blocker = legacy_verdict(mp, axis)
                v1_admits, v1_blocker = v1_verdict(mp, axis)

                entry = {
                    "document_id": document_id, "table_id": tid, "table_role": role,
                    "source_order": order, "cell_id": cell.get("cell_id"),
                    "row_index": cell.get("row_index"), "column_index": col,
                    "row_label": cell.get("row_label"),
                    "column_header": cell.get("column_header"),
                    "value_normalized": norm_val,
                    "legacy_kind": legacy_kind.value if legacy_kind else None,
                    "new_kind": new_kind.value,
                    "legacy_blocker": legacy_blocker, "v1_blocker": v1_blocker,
                    "v1_admits": v1_admits,
                    "v2_outcome": admission.outcome.value,
                    "v2_reason": admission.reason.value if admission.reason else None,
                    "normalized_period": admission.normalized_period,
                    "binding_method": (admission.binding.method.value
                                       if admission.binding and admission.binding.method
                                       else None),
                    "binding_status": (admission.binding.status.value
                                       if admission.binding else None),
                    "binding_granularity": (admission.binding.granularity.value
                                            if admission.binding else None),
                    "source_cells": [c.to_dict() for c in admission.binding.source_cells]
                    if admission.binding else [],
                }

                doc_tally["eligible_cells"] += 1
                doc_tally[f"legacy_{'admits' if legacy_admits else 'withholds'}"] += 1
                doc_tally[f"v2_{admission.outcome.value}"] += 1

                # The two deltas are computed **independently, not as a chain**.  A cell
                # can belong to both: `kind:unknown` is gained by the rule and then
                # withheld again by the producer when no binding exists for its column.
                # An `elif` chain would file that cell under `rule_gain` alone and report
                # the rule as having delivered a fact that never arrives.
                if not legacy_admits and v1_admits:
                    tally["rule_gain"] += 1
                    rule_gain.append(entry)
                if legacy_admits and not v1_admits:
                    tally["rule_loss"] += 1
                    rule_loss.append(entry)
                if not v1_admits and admission.admitted:
                    tally["producer_gain"] += 1
                    producer_gain.append(entry)
                if v1_admits and not admission.admitted:
                    tally["producer_loss"] += 1
                    producer_loss.append(entry)
                if not legacy_admits and v1_admits and not admission.admitted:
                    tally["rule_gain_then_producer_loss"] += 1

                if admission.admitted and legacy_admits:
                    tally["unchanged_admitted"] += 1
                elif not admission.admitted and not legacy_admits:
                    tally["unchanged_withheld"] += 1
                elif admission.admitted:
                    tally["net_gain"] += 1
                else:
                    tally["net_loss"] += 1

        report["documents"][document_id] = dict(doc_tally)
        print(f"--- {document_id}")
        for key in sorted(doc_tally):
            print(f"    {key:<26} {doc_tally[key]}")
        print()

    def grouped(entries, keys):
        return collections.Counter(tuple(e[k] for k in keys) for e in entries)

    report["totals"] = dict(tally)
    report["rule_gain"] = rule_gain
    report["rule_loss"] = rule_loss
    report["producer_gain"] = producer_gain
    report["producer_loss"] = producer_loss
    report["rule_gain_by_reason"] = {
        _label(k): v for k, v in grouped(
            rule_gain, ("legacy_blocker", "v2_outcome")).items()}
    report["rule_loss_by_reason"] = {
        _label(k): v for k, v in grouped(
            rule_loss, ("legacy_kind", "v1_blocker")).items()}
    report["producer_loss_by_reason"] = {
        _label(k): v for k, v in grouped(
            producer_loss, ("v2_reason", "table_role", "legacy_kind")).items()}
    report["producer_gain_by_reason"] = {
        _label(k): v for k, v in grouped(
            producer_gain, ("v2_outcome", "v2_reason", "table_role")).items()}
    report["shape_moves"] = {
        _label(k): v for k, v in collections.Counter(
            (e["legacy_kind"], e["new_kind"]) for e in
            rule_gain + rule_loss + producer_gain + producer_loss
            if e["legacy_kind"] != e["new_kind"]).items()}

    print("=== totals ===")
    for key in sorted(tally):
        print(f"  {key:<30} {tally[key]}")
    print()
    print("=== the rule change: legacy -> V1 (this is W4) ===")
    print(f"  gained {len(rule_gain)}   lost {len(rule_loss)}")
    overlap = tally["rule_gain_then_producer_loss"]
    if overlap:
        print(f"  of the gained, {overlap} are withheld again by the producer: the rule "
              f"would deliver them and V2 would not")
    for key, count in grouped(rule_gain, ("legacy_blocker", "v2_outcome")).most_common():
        print(f"    + {str(key[0]):<24} -> {key[1]:<20} {count}")
    for key, count in grouped(rule_loss, ("legacy_kind", "v1_blocker")).most_common():
        print(f"    - legacy {str(key[0]):<12} -> {str(key[1]):<22} {count}")
    print()
    print("=== the producer change: V1 -> V2 (NOT W4's to claim) ===")
    print(f"  gained {len(producer_gain)}   lost {len(producer_loss)}")
    for key, count in grouped(
            producer_loss, ("v2_reason", "table_role", "legacy_kind")).most_common(10):
        print(f"    - withheld for {str(key[0]):<22} role={str(key[1]):<18} "
              f"legacy kind {str(key[2]):<10} {count}")
    for key, count in grouped(
            producer_gain, ("v2_outcome", "v2_reason", "table_role")).most_common(6):
        print(f"    + {str(key[0]):<20} {str(key[1]):<22} role={str(key[2]):<18} {count}")
    print()
    print("=== where the shape itself moved (legacy kind -> W3 kind) ===")
    for (before, after), count in collections.Counter(
            (e["legacy_kind"], e["new_kind"]) for e in
            rule_gain + rule_loss + producer_gain + producer_loss
            if e["legacy_kind"] != e["new_kind"]).most_common(10):
        print(f"  {str(before):<14} -> {str(after):<14} {count}")
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "emission-admission-shadow.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'emission-admission-shadow.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
