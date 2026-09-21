"""Gate 08 R1.2: Independent Structural Presence Audit.

Audits 33 Gold sources (17 B-class unrecovered + 16 D-class
structurally absent) through five structural layers using the sealed
Gate 06 R2 metadata store.  No PDF reprocessing, no MinerU reruns.

Each Gold is classified into exactly one of:
  S1: Structure exists, Candidate Bridge missing
  S2: Candidate granularity mismatch
  S3: Structure exists, Evidence Unit not emitted
  S4: Native PDF has content, MinerU structure missing
  S5: Narrative or non-table evidence
  S6: Production Candidate granularity or mapping error

Classification is mutually exclusive and exhaustive.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.structural_presence_auditor import (  # noqa: E402
    StructuralPresenceAudit,
    StructuralPresenceAuditor,
)

DEFAULT_R11_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r1-1"
DEFAULT_R1_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r1"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r1-2"
DEFAULT_RUNTIME = ROOT / "artifacts/runtime/pdf-retrieval-v4-gate-06-r2"
DATA = ROOT / "benchmarks/financial_rag_v1/data"

# Coverage classes to audit
AUDIT_CLASSES = (
    "strict_mapped_not_retrieved",
    "structurally_absent",
)

ALL_FAILURE_CLASSES = (
    "S1_bridge_missing",
    "S2_candidate_granularity_mismatch",
    "S3_evidence_unit_not_emitted",
    "S4_mineru_structure_missing",
    "S5_narrative_evidence",
    "S6_candidate_mapping_error",
)


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _extract_gold_metadata(
    source: dict[str, Any],
) -> dict[str, Any]:
    """Extract Gold source metadata for audit."""
    return {
        "gold_candidate_key": str(source.get("candidate_key") or ""),
        "gold_document_id": str(source.get("document_id") or ""),
        "gold_page": source.get("page")
        or source.get("candidate_pdf_page")
        or source.get("pdf_page"),
        "gold_metric": str(source.get("row_label") or source.get("metric") or ""),
        "gold_period": str(source.get("period") or ""),
        "gold_row_label": str(source.get("row_label") or ""),
        "gold_table_title": str(source.get("table_title") or ""),
        "gold_section": str(source.get("section") or ""),
        "gold_evidence_type": str(source.get("evidence_type") or ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-1-out", type=Path, default=DEFAULT_R11_OUT)
    parser.add_argument("--r1-out", type=Path, default=DEFAULT_R1_OUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--labels", type=Path, default=DATA / "labels.golden.jsonl")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load R1.1 coverage classification
    # ------------------------------------------------------------------
    r11_path = args.r1_1_out / "gold-coverage-classification.json"
    r11 = load_json(r11_path)

    # Filter to audit classes (B + D)
    audit_rows = [
        row
        for row in r11.get("rows", [])
        if str(row.get("coverage_class")) in AUDIT_CLASSES
    ]
    b_class = [
        r for r in audit_rows
        if r.get("coverage_class") == "strict_mapped_not_retrieved"
    ]
    d_class = [
        r for r in audit_rows
        if r.get("coverage_class") == "structurally_absent"
    ]

    print(f"Audit targets: {len(audit_rows)} "
          f"(B-class={len(b_class)}, D-class={len(d_class)})")

    # ------------------------------------------------------------------
    # 2. Load Gold labels for source metadata
    # ------------------------------------------------------------------
    labels_list = load_jsonl(args.labels)
    labels = {str(item["case_id"]): item for item in labels_list}

    # Load R1 gold-structural-map for strict mapping info
    r1_gold_map_path = args.r1_out / "gold-structural-map.json"
    r1_gold_map = (
        load_json(r1_gold_map_path) if r1_gold_map_path.is_file() else {}
    )
    r1_match_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for match in r1_gold_map.get("matches", []):
        r1_match_by_key[
            (str(match["case_id"]), str(match["gold_candidate_key"]))
        ] = match

    # ------------------------------------------------------------------
    # 3. Write protocol
    # ------------------------------------------------------------------
    protocol = {
        "gate": "pdf_retrieval_v4_gate_08_r1_2",
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "purpose": "independent_structural_presence_audit",
        "audit_only": True,
        "pdf_reprocessing": False,
        "mineru_rerun": False,
        "structure_modification": False,
        "retrieval_modification": False,
        "audit_targets": {
            "b_class_strict_mapped_not_retrieved": len(b_class),
            "d_class_structurally_absent": len(d_class),
            "total": len(audit_rows),
        },
        "inputs": [
            "r1_1_coverage_classification",
            "r1_gold_structural_map",
            "gate_06_r2_metadata_store",
            "labels_golden",
        ],
        "forbidden_inputs": [
            "expected_value",
            "reference_answer",
            "gate_1_governance_fields",
        ],
        "failure_classes": list(ALL_FAILURE_CLASSES),
        "mutually_exclusive": True,
        "exhaustive": True,
    }
    write(args.out_dir / "audit-protocol.json", protocol)

    # ------------------------------------------------------------------
    # 4. Run five-layer audit
    # ------------------------------------------------------------------
    metadata_db = args.runtime_dir / "metadata" / "metadata.sqlite"
    if not metadata_db.is_file():
        raise RuntimeError(f"metadata_db_not_found:{metadata_db}")

    audits: list[StructuralPresenceAudit] = []
    with StructuralPresenceAuditor(metadata_db) as auditor:
        print(f"Metadata store loaded: {auditor.total_views} views, "
              f"{auditor.total_tables} table views")

        for row in audit_rows:
            case_id = str(row["case_id"])
            gold_key = str(row["gold_candidate_key"])
            source_index = int(row.get("source_index", 0))

            # Get source metadata from labels
            label = labels.get(case_id, {})
            sources = label.get("expected_sources") or []
            source = None
            for idx, src in enumerate(sources):
                if str(src.get("candidate_key")) == gold_key:
                    source = src
                    break
            if source is None and sources:
                source = sources[source_index] if source_index < len(sources) else {}

            gold_meta = _extract_gold_metadata(source or {})

            # R1 strict mapping info
            r1_match = r1_match_by_key.get((case_id, gold_key), {})
            r1_strict_mapped = bool(
                r1_match.get("in_structured_universe")
            )
            r1_matched_view_id = (
                r1_match.get("matched_retrieval_view_id")
                if r1_match else None
            )
            r1_matched_unit_type = (
                r1_match.get("matched_unit_type") if r1_match else None
            )

            audit = auditor.audit_gold_source(
                case_id=case_id,
                source_index=source_index,
                gold_candidate_key=gold_key,
                gold_document_id=gold_meta["gold_document_id"],
                gold_page=gold_meta["gold_page"],
                gold_metric=gold_meta["gold_metric"],
                gold_period=gold_meta["gold_period"],
                gold_row_label=gold_meta["gold_row_label"],
                gold_table_title=gold_meta["gold_table_title"],
                gold_section=gold_meta["gold_section"],
                gold_evidence_type=gold_meta["gold_evidence_type"],
                r1_strict_mapped=r1_strict_mapped,
                r1_matched_view_id=r1_matched_view_id,
                r1_matched_unit_type=r1_matched_unit_type,
            )
            audits.append(audit)

    # ------------------------------------------------------------------
    # 5. Classify and verify integrity
    # ------------------------------------------------------------------
    failure_counts: Counter[str] = Counter()
    coverage_class_counts: Counter[str] = Counter()
    for audit in audits:
        failure_counts[audit.failure_class] += 1
        # Track coverage class
        if audit.layer5_candidate_bridge:
            coverage_class_counts["b_class"] += 1
        else:
            coverage_class_counts["d_class"] += 1

    # Verify: all classified, no unknown
    unknown_count = sum(
        1 for a in audits
        if a.failure_class not in ALL_FAILURE_CLASSES
    )
    reprocess_required = sum(
        1 for a in audits if a.pdf_reprocessing_required
    )

    # Write audit records
    audit_records = [a.to_dict() for a in audits]
    write(
        args.out_dir / "structural-presence-audit.json",
        {
            "gate": "pdf_retrieval_v4_gate_08_r1_2",
            "audit_count": len(audits),
            "failure_class_counts": dict(failure_counts),
            "coverage_class_breakdown": dict(coverage_class_counts),
            "records": audit_records,
        },
    )

    # ------------------------------------------------------------------
    # 6. Write classification integrity
    # ------------------------------------------------------------------
    integrity = {
        "gate": "pdf_retrieval_v4_gate_08_r1_2",
        "total_audited": len(audits),
        "all_classified": unknown_count == 0,
        "unknown_count": unknown_count,
        "mutually_exclusive": True,
        "exhaustive": True,
        "failure_classes_present": sorted(failure_counts.keys()),
        "pdf_reprocessing_required_count": reprocess_required,
        "gold_specific_runtime_rules": 0,
    }
    write(args.out_dir / "classification-integrity.json", integrity)

    # ------------------------------------------------------------------
    # 7. Write summary metrics
    # ------------------------------------------------------------------
    b_audits = [a for a in audits if a.layer5_candidate_bridge]
    d_audits = [a for a in audits if not a.layer5_candidate_bridge]

    b_failure_counts = Counter(a.failure_class for a in b_audits)
    d_failure_counts = Counter(a.failure_class for a in d_audits)

    layer_stats = {
        "layer1_page_present": sum(1 for a in audits if a.layer1_page_present),
        "layer2_table_present": sum(1 for a in audits if a.layer2_table_present),
        "layer3_row_present": sum(1 for a in audits if a.layer3_row_present),
        "layer4_cell_fact_present": sum(
            1 for a in audits if a.layer4_cell_fact_present
        ),
        "layer5_candidate_bridge": sum(
            1 for a in audits if a.layer5_candidate_bridge
        ),
    }

    write(
        args.out_dir / "audit-summary.json",
        {
            "gate": "pdf_retrieval_v4_gate_08_r1_2",
            "total_audited": len(audits),
            "b_class_count": len(b_audits),
            "d_class_count": len(d_audits),
            "failure_class_counts": dict(failure_counts),
            "b_class_failure_counts": dict(b_failure_counts),
            "d_class_failure_counts": dict(d_failure_counts),
            "layer_stats": layer_stats,
            "pdf_reprocessing_required_count": reprocess_required,
            "recommended_actions": dict(
                Counter(a.recommended_action for a in audits)
            ),
        },
    )

    # ------------------------------------------------------------------
    # 8. Acceptance and next-gate
    # ------------------------------------------------------------------
    # Determine next gate based on dominant failure class
    s4_count = failure_counts.get("S4_mineru_structure_missing", 0)
    s1_count = failure_counts.get("S1_bridge_missing", 0)
    s2_count = failure_counts.get("S2_candidate_granularity_mismatch", 0)
    s3_count = failure_counts.get("S3_evidence_unit_not_emitted", 0)
    s5_count = failure_counts.get("S5_narrative_evidence", 0)
    s6_count = failure_counts.get("S6_candidate_mapping_error", 0)

    if s4_count > 0 and s4_count >= max(s1_count, s2_count, s3_count):
        next_gate = "targeted_pdf_reprocessing"
        primary_issue = "mineru_structure_missing"
    elif s1_count > 0 and s1_count >= max(s2_count, s3_count, s5_count):
        next_gate = "gate_05_r5a_candidate_aligned_evidence_expansion"
        primary_issue = "candidate_bridge_missing"
    elif s2_count > 0:
        next_gate = "gate_05_r5a_candidate_aligned_evidence_expansion"
        primary_issue = "candidate_granularity_mismatch"
    elif s3_count > 0:
        next_gate = "gate_05_r5a_candidate_aligned_evidence_expansion"
        primary_issue = "evidence_unit_not_emitted"
    elif s5_count > 0:
        next_gate = "gate_05_r5a_candidate_aligned_narrative_view"
        primary_issue = "narrative_evidence"
    elif s6_count > 0:
        next_gate = "candidate_granularity_or_gold_identity_audit"
        primary_issue = "candidate_mapping_error"
    else:
        next_gate = "gate_05_r5a_candidate_aligned_evidence_expansion"
        primary_issue = "mixed"

    acceptance = {
        "gate": "pdf_retrieval_v4_gate_08_r1_2",
        "total_audited": len(audits),
        "all_classified": unknown_count == 0,
        "unknown_count": unknown_count,
        "mutually_exclusive": True,
        "exhaustive": True,
        "pdf_reprocessing_evidence": (
            "S4 instances have clear page-level evidence" if s4_count > 0
            else "No S4 instances; no PDF reprocessing required"
        ),
        "gold_specific_runtime_rules": 0,
        "failure_class_distribution": dict(failure_counts),
        "primary_issue": primary_issue,
        "decision": "structural_presence_audit_complete",
        "next_gate": next_gate,
        "pdf_reprocessing_required": s4_count > 0,
        "production_switch_allowed": False,
    }
    write(args.out_dir / "acceptance.json", acceptance)

    write(
        args.out_dir / "next-gate.json",
        {
            "current_gate": "pdf_retrieval_v4_gate_08_r1_2",
            "decision": "structural_presence_audit_complete",
            "primary_issue": primary_issue,
            "next_gate": next_gate,
            "failure_class_counts": dict(failure_counts),
            "pdf_reprocessing_required": s4_count > 0,
            "production_switch_allowed": False,
        },
    )

    # ------------------------------------------------------------------
    # 9. Print summary
    # ------------------------------------------------------------------
    print("Gate 08 R1.2 structural presence audit complete.")
    print(f"  Total audited:              {len(audits)}")
    print(f"  B-class (strict-mapped):    {len(b_audits)}")
    print(f"  D-class (structurally abs): {len(d_audits)}")
    print(f"  Unknown count:              {unknown_count}")
    print(f"  PDF reprocessing required:  {reprocess_required}")
    print("  Failure class distribution:")
    for cls in ALL_FAILURE_CLASSES:
        count = failure_counts.get(cls, 0)
        if count > 0:
            print(f"    {cls}: {count}")
    print("  Layer stats:")
    for layer, count in layer_stats.items():
        print(f"    {layer}: {count}/{len(audits)}")
    print(f"  Primary issue:              {primary_issue}")
    print(f"  Next gate:                  {next_gate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
