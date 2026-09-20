"""P1.6-A3-0: where do a table's valued cells stop becoming facts?

Measurement only.  **No emitter change, no row-classifier change, no schema change.**
Nine oracle-verified primary statements produce no store records while their tables
hold hundreds of valued cells, and the point of this pass is to find the first place
that happens, not to guess at it.

The funnel, and the guard that ends each stage:

    cells                                   every cell parse_table produced
      -> column_index > 0                   the label column is skipped
      -> has a SemanticRow                  classify_table_rows produced one
      -> is_financial_data_row              the row classifier admitted it
      -> has a numeric value                raw_value present and normalizable
      -> has an axis binding                build_axis_bindings bound the cell
      -> temporal_kind in point/duration/   the cell's axis is a period, not a
         comparison                         bucket, segment or category
      -> has a metric path                  build_metric_paths produced one
      -> metric_status != "missing"
      -> AtomicFact emitted
      -> Store V2 record

Each stage's count is reported beside the same stage for a control table that works,
because a bare number cannot say whether it is the table or the stage that is unusual.

  python audit_emission_funnel.py --out <dir> [--documents a,b]
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
from src.pdf_retrieval_v4.semantic_currency_resolver import resolve_table_currency  # noqa: E402
from src.pdf_retrieval_v4.semantic_row_classifier import classify_table_rows  # noqa: E402
from src.pdf_retrieval_v4.semantic_scale_resolver import resolve_table_scale  # noqa: E402
from src.pdf_retrieval_v4.temporal_axis_graph import build_axis_bindings  # noqa: E402
from src.pdf_retrieval_v4.typed_evidence_emitters import (  # noqa: E402
    _get_numeric_value,
    emit_atomic_facts,
)

STORE = Path(
    "/disk/qh/nano-finrag/artifacts/evaluation/p1-6-a2b20-store-v2-role/store-v2.jsonl"
)
ORACLE = Path(
    "/disk/qh/nano-finrag/artifacts/evaluation/p1-6-a2b19a-oracle/table-authority-oracle.json"
)
BUILDER = _BACKEND_DIR / "scripts/evaluation/build_store_v2.py"

#: The stages, in the order the emitter applies them.
STAGES = (
    "cells",
    "not_label_column",
    "has_semantic_row",
    "is_financial_data_row",
    "has_numeric_value",
    "has_axis_binding",
    "temporal_kind",
    "has_metric_path",
    "metric_status",
)


def funnel_for(table: dict, document: dict) -> dict:
    """Count the cells surviving each guard, and name the first one that bites.

    This repeats `adapt_document_tables`' per-table body so that each guard can be
    counted separately; the emitted total is checked against the real pipeline's for
    the same table, so the repetition cannot silently diverge from it.
    """
    adapted = _adapt_table(table, document)
    tid = adapted["table_fragment_id"]
    cells = adapted["cells"]

    rows = classify_table_rows(adapted, tid, adapted["document_id"], 0)
    paths = build_metric_paths(rows)
    axes = build_axis_bindings(cells, tid)
    facts = emit_atomic_facts(
        rows, paths, axes, cells,
        resolve_table_scale(adapted, tid), resolve_table_currency(adapted, tid), {},
    )

    sr_by_row = {sr.row_id: sr for sr in rows}
    mp_by_row = {mp.row_id: mp for mp in paths}
    axis_by_cell = {a.cell_id: a for a in axes}

    counts = collections.Counter()
    drops = collections.Counter()
    counts["cells"] = len(cells)
    for cell in cells:
        if int(cell.get("column_index") or 0) == 0:
            drops["label_column"] += 1
            continue
        counts["not_label_column"] += 1
        sr = sr_by_row.get(str(cell.get("row_id") or ""))
        if not sr:
            drops["no_semantic_row"] += 1
            continue
        counts["has_semantic_row"] += 1
        if not sr.is_financial_data_row:
            drops["row_not_financial_data"] += 1
            continue
        counts["is_financial_data_row"] += 1
        raw_val, norm_val = _get_numeric_value(cell)
        if not raw_val or norm_val is None:
            drops["no_numeric_value"] += 1
            continue
        counts["has_numeric_value"] += 1
        axis = axis_by_cell.get(str(cell.get("cell_id") or ""))
        if not axis:
            drops["no_axis_binding"] += 1
            continue
        counts["has_axis_binding"] += 1
        if axis.temporal_kind not in ("point", "duration", "comparison"):
            drops[f"temporal_kind_{axis.temporal_kind}"] += 1
            continue
        counts["temporal_kind"] += 1
        mp = mp_by_row.get(str(cell.get("row_id") or ""))
        if not mp:
            drops["no_metric_path"] += 1
            continue
        counts["has_metric_path"] += 1
        if mp.metric_status == "missing":
            drops["metric_status_missing"] += 1
            continue
        counts["metric_status"] += 1

    return {
        "table_fragment_id": tid,
        "counts": {stage: counts.get(stage, 0) for stage in STAGES},
        "drops": dict(drops),
        "emitted_by_audit": counts["metric_status"],
        "emitted_by_pipeline": len(facts),
        "semantic_rows": len(rows),
        "financial_data_rows": sum(1 for r in rows if r.is_financial_data_row),
        "metric_paths": len(paths),
        "axis_bindings": len(axes),
        "temporal_kinds": dict(collections.Counter(a.temporal_kind for a in axes)),
        "row_types": dict(collections.Counter(
            str(getattr(r, "row_type", None)) for r in rows)),
        "first_bite": drops.most_common(1)[0] if drops else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--documents", default="")
    args = parser.parse_args(argv)

    spec = importlib.util.spec_from_file_location("store_builder", BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    store_records = collections.Counter()
    if STORE.is_file():
        for line in STORE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                store_records[json.loads(line)["table_fragment_id"]] += 1

    wanted = [d for d in args.documents.split(",") if d] or sorted(builder.DOCUMENTS)
    report = {"phase": "P1.6-A3-0", "mutation": "none", "documents": {}}

    print("=== emission funnel ===")
    print()
    for document_id in wanted:
        ticker, accession = builder.DOCUMENTS[document_id]
        parsed = builder.parse_filing(ticker, accession, document_id)
        by_order = {b["source_order"]: b for b in parsed["blocks"]
                    if b["block_type"] == "TABLE"}
        roles = builder.table_roles(document_id, parsed["blocks"])
        tables_by_id = {t["table_id"]: t for t in parsed["tables"]}

        audited = []
        for block in parsed["blocks"]:
            if block["block_type"] != "TABLE":
                continue
            table = tables_by_id.get(block["table_id"])
            if table is None:
                continue
            result = funnel_for(table, parsed)
            result["document_id"] = document_id
            result["oracle_key"] = f"{document_id}#{block['source_order']}"
            result["table_ordinal"] = next(
                (t["table_ordinal"] for t in oracle["documents"][document_id]["tables"]
                 if t["source_order"] == block["source_order"]), None)
            result["oracle_role"] = roles.get(block["table_id"], {}).get("oracle_role")
            result["store_records"] = store_records.get(block["table_id"], 0)
            audited.append(result)

        report["documents"][document_id] = audited

        # The tables this pass exists for: an oracle-verified primary statement that the
        # store received nothing from.
        missing = [a for a in audited
                   if a["oracle_role"] == "PRIMARY" and a["store_records"] == 0]
        present = [a for a in audited
                   if a["oracle_role"] == "PRIMARY" and a["store_records"] > 0]
        print(f"--- {document_id}  tables {len(audited)}  "
              f"oracle-primary with records {len(present)}, with none {len(missing)}")
        for a in missing + present:
            mark = "ZERO" if a["store_records"] == 0 else " ok "
            print(f"    {mark} {a['oracle_key']:<20} ord={str(a['table_ordinal']):<4} "
                  f"cells {a['counts']['cells']:>4} -> fdrows {a['financial_data_rows']:>3} "
                  f"-> axes {a['axis_bindings']:>4} -> facts {a['emitted_by_pipeline']:>4} "
                  f"-> store {a['store_records']:>4}   first bite {a['first_bite']}")
            if a["store_records"] == 0 or a["emitted_by_pipeline"] == 0:
                print(f"         drops {a['drops']}")
                print(f"         temporal {a['temporal_kinds']}")
                print(f"         row_types {a['row_types']}")
        print()

        # Any table where the audit and the real pipeline disagree would mean the
        # repetition above is not faithful; that must be visible, not assumed.
        for a in audited:
            if a["emitted_by_audit"] != a["emitted_by_pipeline"]:
                print(f"    !! audit/pipeline disagree on {a['oracle_key']}: "
                      f"{a['emitted_by_audit']} vs {a['emitted_by_pipeline']}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "emission-funnel.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    # A global view: where does the first bite land, split by what the table is.
    tally = collections.Counter()
    for document in report["documents"].values():
        for a in document:
            if a["first_bite"]:
                tally[(a["oracle_role"], a["first_bite"][0])] += 1
            else:
                tally[(a["oracle_role"], "emitted_cleanly")] += 1
    print("=== first bite, by table role ===")
    for (role, bite), count in sorted(tally.items(), key=lambda kv: (str(kv[0][0]), -kv[1])):
        print(f"  {str(role):<20} {bite:<28} {count}")
    print()
    print(f"  written to {args.out / 'emission-funnel.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
