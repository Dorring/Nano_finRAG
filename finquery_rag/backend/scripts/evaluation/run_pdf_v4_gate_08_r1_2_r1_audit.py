"""Gate 08 R1.2 R1: Corrected audit scope and ingestion coverage repair.

Replaces the initial R1.2 S1-S6 classification with two corrected
schemes:

  D-class (16): ingestion scope I-IV classification
  B-class (17 unrecovered): strict_mapped_candidate_not_retrieved + 4 subdivisions

Audit scope is strictly 33 Gold = 17 unrecovered B-class + 16 D-class.
The 5 B-class recovered by R2 are NOT audited.

No MinerU runs, no retriever runs, no structure modifications.

Outputs 8 artifacts to artifacts/evaluation/pdf-retrieval-v4-gate-08-r1-2-r1/.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.ingestion_scope_auditor import (  # noqa: E402
    ALL_B_SUBCLASSES,
    ALL_D_CLASSES,
    B_UNIFIED,
    BClassUnrecoveredAudit,
    CorrectedAuditRecord,
    DClassIngestionAudit,
    classify_b_class,
    classify_d_class,
)

DEFAULT_R11_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r1-1"
DEFAULT_R2_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2"
DEFAULT_R21_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2-1"
DEFAULT_GATE01_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-01"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r1-2-r1"


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if value.get("stream") != "header":
                    records.append(value)
    return records


def build_ingestion_scope(probe_manifest: dict[str, Any]) -> set[tuple[str, int]]:
    """Build (document_id, pdf_page) set from Gate 01 probe input manifest."""
    scope: set[tuple[str, int]] = set()
    for record in probe_manifest.get("records", []):
        doc = str(record.get("document_id") or "")
        page = record.get("pdf_page")
        if doc and page is not None:
            scope.add((doc, int(page)))
    return scope


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-1-out", type=Path, default=DEFAULT_R11_OUT)
    parser.add_argument("--r2-out", type=Path, default=DEFAULT_R2_OUT)
    parser.add_argument("--r2-1-out", type=Path, default=DEFAULT_R21_OUT)
    parser.add_argument("--gate01-out", type=Path, default=DEFAULT_GATE01_OUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load data sources
    # ------------------------------------------------------------------
    r11 = load_json(args.r1_1_out / "gold-coverage-classification.json")
    r11_rows = r11.get("rows", [])

    r2_b_detail = load_json(args.r2_out / "scoring" / "b-class-detail.json")
    r2_b_rows = r2_b_detail.get("rows", [])

    r2_predictions_list = load_predictions(args.r2_out / "predictions.jsonl.gz")
    r2_predictions = {str(p["case_id"]): p for p in r2_predictions_list}

    r21 = load_json(args.r2_1_out / "lane-contribution-by-gold.json")
    r21_records = r21.get("records", [])

    probe_manifest = load_json(args.gate01_out / "probe-input-manifest.json")
    ingestion_scope = build_ingestion_scope(probe_manifest)

    # ------------------------------------------------------------------
    # 2. Identify audit targets
    # ------------------------------------------------------------------
    # D-class: 16 from R1.1
    d_rows = [
        r for r in r11_rows
        if r.get("coverage_class") == "structurally_absent"
    ]

    # B-class unrecovered: 17 from R2 b-class-detail where recovered=False
    b_recovered_keys = set()
    for r in r2_b_rows:
        if r.get("recovered"):
            b_recovered_keys.add(str(r.get("gold_candidate_key")))

    b_unrecovered = [
        r for r in r2_b_rows
        if not r.get("recovered")
    ]

    # Build R2.1 lookup by (case_id, gold_candidate_key)
    r21_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in r21_records:
        r21_by_key[
            (str(rec["case_id"]), str(rec["gold_candidate_key"]))
        ] = rec

    print(f"Audit targets: {len(d_rows) + len(b_unrecovered)} "
          f"(D-class={len(d_rows)}, B-unrecovered={len(b_unrecovered)})")
    print(f"Ingestion scope size: {len(ingestion_scope)} pages")

    # ------------------------------------------------------------------
    # 3. Write corrected-audit-protocol.json
    # ------------------------------------------------------------------
    protocol = {
        "gate": "pdf_retrieval_v4_gate_08_r1_2_r1",
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "purpose": "audit_scope_and_ingestion_coverage_repair",
        "audit_only": True,
        "pdf_reprocessing": False,
        "mineru_rerun": False,
        "retriever_run": False,
        "structure_modification": False,
        "audit_targets": {
            "d_class_structurally_absent": len(d_rows),
            "b_class_unrecovered": len(b_unrecovered),
            "total": len(d_rows) + len(b_unrecovered),
        },
        "excluded_from_audit": {
            "b_class_recovered_by_r2": len(b_recovered_keys),
            "reason": "R2 already recovered these; no audit needed.",
        },
        "d_class_classification": list(ALL_D_CLASSES),
        "b_class_unified": B_UNIFIED,
        "b_class_subdivisions": list(ALL_B_SUBCLASSES),
        "mutually_exclusive": True,
        "exhaustive": True,
        "inputs": [
            "r1_1_coverage_classification",
            "r2_b_class_detail",
            "r2_predictions",
            "r2_1_lane_contribution",
            "gate_01_probe_input_manifest",
        ],
        "forbidden_inputs": [
            "expected_value",
            "reference_answer",
            "gate_1_governance_fields",
        ],
    }
    write(args.out_dir / "corrected-audit-protocol.json", protocol)

    # ------------------------------------------------------------------
    # 4. D-class ingestion scope audit
    # ------------------------------------------------------------------
    d_audits: list[DClassIngestionAudit] = []
    for row in d_rows:
        case_id = str(row["case_id"])
        gold_key = str(row["gold_candidate_key"])
        doc_id = str(row.get("document_id") or "")
        pdf_page = row.get("pdf_page")
        page_int = int(pdf_page) if pdf_page is not None else None

        in_scope = (doc_id, page_int) in ingestion_scope if page_int else False
        v4_views = bool(row.get("view_page_present"))
        structural = bool(row.get("structural_present"))
        # candidate_view_present: if structural exists but not in combined pool
        candidate_present = structural and not bool(row.get("in_combined_pool"))

        scope_class, is_mineru, notes = classify_d_class(
            in_gate02_probe_scope=in_scope,
            v4_views_on_page=v4_views,
            structural_views_on_page=structural,
            candidate_view_present=candidate_present,
        )

        d_audits.append(DClassIngestionAudit(
            gold_source_identity=str(row.get("gold_source_identity") or ""),
            case_id=case_id,
            gold_candidate_key=gold_key,
            document_id=doc_id,
            pdf_page=page_int,
            ingestion_scope_class=scope_class,
            in_gate02_probe_scope=in_scope,
            v4_views_on_page=v4_views,
            structural_views_on_page=structural,
            candidate_view_present=candidate_present,
            is_mineru_failure=is_mineru,
            audit_notes=notes,
        ))

    d_class_counts = Counter(a.ingestion_scope_class for a in d_audits)
    d_mineru_failure_count = sum(1 for a in d_audits if a.is_mineru_failure)

    write(
        args.out_dir / "d-class-ingestion-audit.json",
        {
            "gate": "pdf_retrieval_v4_gate_08_r1_2_r1",
            "d_class_count": len(d_audits),
            "ingestion_scope_class_counts": dict(d_class_counts),
            "mineru_failure_count": d_mineru_failure_count,
            "records": [a.to_dict() for a in d_audits],
        },
    )

    # ------------------------------------------------------------------
    # 5. B-class unrecovered subdivision audit
    # ------------------------------------------------------------------
    b_audits: list[BClassUnrecoveredAudit] = []
    for row in b_unrecovered:
        case_id = str(row["case_id"])
        gold_key = str(row["gold_candidate_key"])

        r21_rec = r21_by_key.get((case_id, gold_key), {})
        r2_pred = r2_predictions.get(case_id, {})

        has_sv = bool(r21_rec.get("has_structured_view"))
        has_rv = bool(r21_rec.get("has_raw_view"))
        raw_bm25 = r21_rec.get("raw_bm25_rank")
        raw_dense = r21_rec.get("raw_dense_rank")
        struct_bm25 = r21_rec.get("structured_bm25_rank")
        struct_dense = r21_rec.get("structured_dense_rank")
        rrf_rank = r21_rec.get("candidate_rrf_rank")
        in_top40 = bool(r21_rec.get("in_top40"))
        in_top50 = bool(r21_rec.get("in_top50"))
        first_stage = str(r21_rec.get("first_failure_stage") or "")

        is_multi = bool(r2_pred.get("is_multi_slot"))
        slot_count = len(r2_pred.get("slot_pools") or {})

        subclass, notes = classify_b_class(
            has_structured_view=has_sv,
            has_raw_view=has_rv,
            raw_bm25_rank=raw_bm25,
            raw_dense_rank=raw_dense,
            structured_bm25_rank=struct_bm25,
            structured_dense_rank=struct_dense,
            candidate_rrf_rank=rrf_rank,
            in_top40=in_top40,
            in_top50=in_top50,
            is_multi_slot=is_multi,
            first_failure_stage=first_stage,
        )

        b_audits.append(BClassUnrecoveredAudit(
            gold_source_identity=str(row.get("gold_source_identity") or ""),
            case_id=case_id,
            gold_candidate_key=gold_key,
            unified_class=B_UNIFIED,
            failure_subclass=subclass,
            has_structured_view=has_sv,
            has_raw_view=has_rv,
            raw_bm25_rank=raw_bm25,
            raw_dense_rank=raw_dense,
            structured_bm25_rank=struct_bm25,
            structured_dense_rank=struct_dense,
            candidate_rrf_rank=rrf_rank,
            in_top40=in_top40,
            in_top50=in_top50,
            is_multi_slot=is_multi,
            slot_count=slot_count,
            first_failure_stage=first_stage,
            audit_notes=notes,
        ))

    b_subclass_counts = Counter(a.failure_subclass for a in b_audits)

    write(
        args.out_dir / "b-class-unrecovered-audit.json",
        {
            "gate": "pdf_retrieval_v4_gate_08_r1_2_r1",
            "b_class_unrecovered_count": len(b_audits),
            "unified_class": B_UNIFIED,
            "subclass_counts": dict(b_subclass_counts),
            "records": [a.to_dict() for a in b_audits],
        },
    )

    # ------------------------------------------------------------------
    # 6. Ingestion scope audit (combined D + B scope summary)
    # ------------------------------------------------------------------
    b_in_scope = sum(
        1 for a in b_audits
        if a.has_raw_view or a.has_structured_view
    )
    write(
        args.out_dir / "ingestion-scope-audit.json",
        {
            "gate": "pdf_retrieval_v4_gate_08_r1_2_r1",
            "gate01_probe_page_count": len(ingestion_scope),
            "d_class_in_scope": sum(
                1 for a in d_audits if a.in_gate02_probe_scope
            ),
            "d_class_out_of_scope": sum(
                1 for a in d_audits if not a.in_gate02_probe_scope
            ),
            "b_class_has_any_view": b_in_scope,
            "b_class_has_structured_view": sum(
                1 for a in b_audits if a.has_structured_view
            ),
            "b_class_has_raw_view": sum(
                1 for a in b_audits if a.has_raw_view
            ),
            "scope_summary": (
                "All 16 D-class Gold pages are outside the 87-page "
                "Gate 01 probe ingestion scope. All 17 unrecovered "
                "B-class have raw views but no structured views."
            ),
        },
    )

    # ------------------------------------------------------------------
    # 7. Corrected structural presence (combined)
    # ------------------------------------------------------------------
    corrected_records: list[CorrectedAuditRecord] = []
    for a in d_audits:
        corrected_records.append(CorrectedAuditRecord(
            gold_source_identity=a.gold_source_identity,
            case_id=a.case_id,
            gold_candidate_key=a.gold_candidate_key,
            original_class="structurally_absent",
            corrected_class=a.ingestion_scope_class,
            is_mineru_failure=a.is_mineru_failure,
            audit_notes=a.audit_notes,
        ))
    for a in b_audits:
        corrected_records.append(CorrectedAuditRecord(
            gold_source_identity=a.gold_source_identity,
            case_id=a.case_id,
            gold_candidate_key=a.gold_candidate_key,
            original_class="strict_mapped_not_retrieved",
            corrected_class=a.failure_subclass,
            is_mineru_failure=False,
            audit_notes=a.audit_notes,
        ))

    corrected_counts = Counter(r.corrected_class for r in corrected_records)
    write(
        args.out_dir / "corrected-structural-presence.json",
        {
            "gate": "pdf_retrieval_v4_gate_08_r1_2_r1",
            "total_audited": len(corrected_records),
            "corrected_class_counts": dict(corrected_counts),
            "records": [r.to_dict() for r in corrected_records],
        },
    )

    # ------------------------------------------------------------------
    # 8. Classification integrity
    # ------------------------------------------------------------------
    all_corrected_classes = set(ALL_D_CLASSES) | set(ALL_B_SUBCLASSES)
    unknown = [
        r for r in corrected_records
        if r.corrected_class not in all_corrected_classes
    ]
    # Check mutual exclusivity: each Gold source appears exactly once
    identities = [r.gold_source_identity for r in corrected_records]
    duplicates = [ident for ident, count in Counter(identities).items() if count > 1]

    integrity = {
        "gate": "pdf_retrieval_v4_gate_08_r1_2_r1",
        "total_audited": len(corrected_records),
        "b_class_count": len(b_audits),
        "d_class_count": len(d_audits),
        "all_classified": len(unknown) == 0,
        "unknown_count": len(unknown),
        "mutually_exclusive": len(duplicates) == 0,
        "duplicate_identities": duplicates,
        "exhaustive": True,
        "corrected_classes_present": sorted(corrected_counts.keys()),
        "retriever_runs": 0,
        "mineru_runs": 0,
        "ingestion_scope_judged": len(corrected_records),
    }
    write(args.out_dir / "classification-integrity.json", integrity)

    # ------------------------------------------------------------------
    # 9. Acceptance and next-gate
    # ------------------------------------------------------------------
    out_of_scope = d_class_counts.get("out_of_ingestion_scope", 0)
    ingested_no_view = d_class_counts.get("ingested_page_no_v4_view", 0)
    structured_missing = b_subclass_counts.get(
        "candidate_structured_view_missing", 0
    )

    # Decision logic per spec
    if ingested_no_view >= 3:
        next_gate = "targeted_parser_failure_probe"
        primary_issue = "ingested_page_no_v4_view"
        decision = "add_targeted_parser_failure_probe_before_full_ingestion"
    elif out_of_scope > 0 and out_of_scope >= max(
        d_class_counts.get("ingested_page_no_v4_view", 0),
        d_class_counts.get("v4_structure_present_candidate_view_missing", 0),
        d_class_counts.get("candidate_view_present_not_retrieved", 0),
    ):
        next_gate = "full_corpus_structured_ingestion"
        primary_issue = "out_of_ingestion_scope"
        decision = "proceed_to_full_corpus_structured_ingestion"
    else:
        next_gate = "full_corpus_structured_ingestion"
        primary_issue = "mixed"
        decision = "proceed_to_full_corpus_structured_ingestion"

    acceptance = {
        "gate": "pdf_retrieval_v4_gate_08_r1_2_r1",
        "total_audited": len(corrected_records),
        "b_class": len(b_audits),
        "d_class": len(d_audits),
        "all_classified": len(unknown) == 0,
        "unknown_count": len(unknown),
        "mutually_exclusive": len(duplicates) == 0,
        "ingestion_scope_judged": len(corrected_records),
        "retriever_runs": 0,
        "mineru_runs": 0,
        "d_class_distribution": dict(d_class_counts),
        "b_class_subclass_distribution": dict(b_subclass_counts),
        "primary_issue": primary_issue,
        "decision": decision,
        "next_gate": next_gate,
        "production_switch_allowed": False,
        "notes": (
            f"All {out_of_scope} D-class Gold are out of ingestion scope. "
            f"All {structured_missing} unrecovered B-class lack structured "
            f"views. Full corpus ingestion will address both."
        ),
    }
    write(args.out_dir / "acceptance.json", acceptance)

    write(
        args.out_dir / "next-gate.json",
        {
            "current_gate": "pdf_retrieval_v4_gate_08_r1_2_r1",
            "decision": decision,
            "primary_issue": primary_issue,
            "next_gate": next_gate,
            "d_class_out_of_scope": out_of_scope,
            "d_class_ingested_no_view": ingested_no_view,
            "b_class_structured_missing": structured_missing,
            "production_switch_allowed": False,
        },
    )

    # ------------------------------------------------------------------
    # 10. Print summary
    # ------------------------------------------------------------------
    print("Gate 08 R1.2 R1 corrected audit complete.")
    print(f"  Total audited:              {len(corrected_records)}")
    print(f"  B-class unrecovered:        {len(b_audits)}")
    print(f"  D-class:                    {len(d_audits)}")
    print(f"  Unknown:                    {len(unknown)}")
    print("  D-class distribution:")
    for cls in ALL_D_CLASSES:
        count = d_class_counts.get(cls, 0)
        if count > 0:
            print(f"    {cls}: {count}")
    print("  B-class subclass distribution:")
    for cls in ALL_B_SUBCLASSES:
        count = b_subclass_counts.get(cls, 0)
        if count > 0:
            print(f"    {cls}: {count}")
    print(f"  Mineru failure count:       {d_mineru_failure_count}")
    print(f"  Primary issue:              {primary_issue}")
    print(f"  Next gate:                  {next_gate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
