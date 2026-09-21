#!/usr/bin/env python3
"""Zero-Gold Gate09 R5.1 deterministic operand binding V2 prediction."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.joint_operand_binder import bind_joint_operands  # noqa: E402
from src.pdf_retrieval_v4.metric_binding_contract_v2 import (  # noqa: E402
    bind_metric,
    infer_measurement_kind,
    normalize,
)
from src.pdf_retrieval_v4.semantic_evidence_set import (  # noqa: E402
    MAX_EVIDENCE_ITEMS,
    minimum_candidate_cover,
)
from src.pdf_retrieval_v4.unit_context_resolver import resolve_unit_context  # noqa: E402

EVAL = ROOT / "artifacts/evaluation"
R5 = EVAL / "pdf-retrieval-v4-gate-09-r5"
G03 = EVAL / "pdf-retrieval-v4-gate-03-r2"
QUERY_PLAN = EVAL / "pdf-retrieval-v4-gate-07/query-plan-predictions.json"
OUT = EVAL / "pdf-retrieval-v4-gate-09-r5-1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def exact_slot_candidate(
    plan: dict[str, Any], slot: dict[str, Any], semantic_class: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    metric_binding = bind_metric(slot, semantic_class)
    base_trace = {
        "semantic_fact_id": semantic_class["semantic_fact_id"],
        **metric_binding,
    }
    if not metric_binding["deterministic_compatible"]:
        return None, base_trace
    document_scope = {normalize(value) for value in plan.get("document_scope") or []}
    if document_scope and normalize(semantic_class.get("document_id")) not in document_scope:
        return None, {**base_trace, "rejected_reason": "document_conflict"}
    if normalize(slot.get("period")) != normalize(semantic_class.get("period")):
        return None, {**base_trace, "rejected_reason": "period_conflict"}
    if slot.get("segment_label") and normalize(slot.get("segment_label")) != normalize(semantic_class.get("segment")):
        return None, {**base_trace, "rejected_reason": "segment_conflict"}
    if slot.get("bucket_label") and normalize(slot.get("bucket_label")) != normalize(semantic_class.get("bucket")):
        return None, {**base_trace, "rejected_reason": "bucket_conflict"}
    return semantic_class, {**base_trace, "rejected_reason": None}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    r5_seal = json.loads((R5 / "prediction-seal.json").read_text(encoding="utf-8"))
    if not r5_seal.get("sealed") or r5_seal["gold_reads_before_seal"] != 0:
        raise RuntimeError("gate09_r5_input_seal_invalid")
    access_path = R5 / "evidence-access-universe.jsonl.gz"
    classes_path = R5 / "semantic-evidence-classes.jsonl.gz"
    if r5_seal["output_sha256"]["access"] != sha256(access_path) or r5_seal["output_sha256"]["classes"] != sha256(classes_path):
        raise RuntimeError("gate09_r5_input_mutation")
    access_by_case = {str(row["case_id"]): row for row in read_jsonl(access_path)}
    classes_by_case = {str(row["case_id"]): row for row in read_jsonl(classes_path)}
    plans_payload = json.loads(QUERY_PLAN.read_text(encoding="utf-8"))
    plans = {str(row["case_id"]): row["plan"] for row in plans_payload["plans"]}
    scale_path = G03 / "scale-resolutions.jsonl"
    currency_path = G03 / "currency-resolutions.jsonl"
    evidence_paths = [
        G03 / name
        for name in (
            "atomic-facts.jsonl",
            "comparison-facts.jsonl",
            "bucket-facts.jsonl",
            "row-matrices.jsonl",
            "narrative-evidence.jsonl",
        )
    ]
    scale_by_table = {str(row["table_fragment_id"]): row for row in read_jsonl(scale_path)}
    currency_by_table = {str(row["table_fragment_id"]): row for row in read_jsonl(currency_path)}
    evidence_by_id = {
        str(row["semantic_fact_id"]): row
        for path in evidence_paths
        for row in read_jsonl(path)
        if row.get("semantic_fact_id")
    }

    enhanced_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    set_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    unit_status_counts: Counter[str] = Counter()
    concept_only_count = 0
    max_set_size = 0
    for case_id in sorted(plans):
        plan = plans[case_id]
        access = access_by_case[case_id]["candidates"]
        enhanced_classes: list[dict[str, Any]] = []
        for semantic_class in classes_by_case[case_id]["semantic_classes"]:
            unit_context = resolve_unit_context(
                semantic_class,
                scale_by_table,
                currency_by_table,
                evidence_by_id,
            )
            enhanced = {
                **semantic_class,
                "measurement_kind": infer_measurement_kind(semantic_class.get("metric")),
                "unit_context": unit_context,
            }
            enhanced_classes.append(enhanced)
            unit_status_counts[unit_context["resolution_status"]] += 1
        class_by_id = {item["semantic_fact_id"]: item for item in enhanced_classes}
        slot_options: list[dict[str, Any]] = []
        metric_trace: list[dict[str, Any]] = []
        for slot in plan.get("operand_slots") or []:
            compatible: list[dict[str, Any]] = []
            traces: list[dict[str, Any]] = []
            for semantic_class in enhanced_classes:
                candidate, trace = exact_slot_candidate(plan, slot, semantic_class)
                traces.append(trace)
                concept_only_count += bool(trace["concept_hint_only"] and not trace["metric_tier"])
                if candidate:
                    compatible.append(candidate)
            compatible.sort(key=lambda item: item["semantic_fact_id"])
            slot_options.append({"slot": slot, "compatible_classes": compatible})
            metric_trace.append(
                {
                    "slot_id": slot["slot_id"],
                    "deterministic_compatible_fact_ids": [item["semantic_fact_id"] for item in compatible],
                    "concept_hint_only_fact_ids": sorted(
                        trace["semantic_fact_id"]
                        for trace in traces
                        if trace["concept_hint_only"] and not trace["metric_tier"]
                    ),
                    "traces": traces,
                }
            )
        joint = bind_joint_operands(plan, slot_options)
        status_counts[joint["binding_status"]] += 1
        selected_fact_ids = (
            joint["selected_assignment"]["semantic_fact_ids"] if joint.get("selected_assignment") else []
        )
        if selected_fact_ids and len(selected_fact_ids) != len(plan.get("operand_slots") or []):
            raise RuntimeError(
                f"joint_assignment_slot_count_mismatch:{case_id}:"
                f"{len(plan.get('operand_slots') or [])}:{len(selected_fact_ids)}"
            )
        selected_classes = [class_by_id[fact_id] for fact_id in selected_fact_ids]
        faux_slot_matches = (
            [
                {
                    "slot_id": slot["slot_id"],
                    "slot_status": "deterministic",
                    "compatible_semantic_fact_ids": [fact_id],
                }
                for slot, fact_id in zip(plan.get("operand_slots") or [], selected_fact_ids, strict=True)
            ]
            if selected_fact_ids
            else []
        )
        cover = minimum_candidate_cover(faux_slot_matches, enhanced_classes, access) if selected_fact_ids else {
            "selected_candidate_keys": [],
            "covered_semantic_fact_ids": [],
            "complete": False,
            "evidence_item_count": 0,
        }
        max_set_size = max(max_set_size, cover["evidence_item_count"])
        unit_contract = (joint.get("selected_assignment") or {}).get("unit_contract") or {}
        normalized_values = unit_contract.get("normalized_values") or []
        operands: dict[str, Any] = {}
        selected_slot_classes = (
            zip(plan.get("operand_slots") or [], selected_classes, strict=True)
            if selected_classes
            else []
        )
        for index, (slot, semantic_class) in enumerate(selected_slot_classes):
            operands[str(slot["slot_id"])] = {
                "role": slot.get("role"),
                "semantic_fact_id": semantic_class["semantic_fact_id"],
                "value": semantic_class.get("value"),
                "normalized_value": normalized_values[index] if index < len(normalized_values) else None,
                "measurement_kind": semantic_class["measurement_kind"],
                "unit_context": semantic_class["unit_context"],
                "supporting_candidate_keys": semantic_class["supporting_candidate_keys"],
                "deterministic": True,
            }
        calculation_ready = bool(
            plan.get("task_type") == "calculation_multi_operand"
            and joint["binding_status"] == "deterministic_ready"
            and cover["complete"]
        )
        enhanced_rows.append({"case_id": case_id, "semantic_classes": enhanced_classes})
        metric_rows.append({"case_id": case_id, "slot_metric_bindings": metric_trace})
        joint_rows.append({"case_id": case_id, **joint})
        projection_rows.append(
            {
                "case_id": case_id,
                "operation": plan.get("operation"),
                "binding_status": joint["binding_status"],
                "operands": operands,
                "calculation_runtime_ready": calculation_ready,
                "blocked_reason": None if calculation_ready else unit_contract.get("reason") or joint["binding_status"],
            }
        )
        set_rows.append(
            {
                "case_id": case_id,
                "binding_status": joint["binding_status"],
                "selected_semantic_fact_ids": selected_fact_ids,
                "selected_candidate_keys": cover["selected_candidate_keys"],
                "evidence_item_count": cover["evidence_item_count"],
                "evidence_set_complete": bool(selected_fact_ids and cover["complete"]),
                "calculation_runtime_ready": calculation_ready,
            }
        )

    outputs = {
        "classes_v2": OUT / "semantic-evidence-classes-v2.jsonl.gz",
        "metric_bindings": OUT / "metric-binding-candidates.jsonl.gz",
        "joint_bindings": OUT / "joint-operand-bindings.jsonl.gz",
        "projections_v2": OUT / "operand-projections-v2.jsonl.gz",
        "sets_v2": OUT / "evidence-set-predictions-v2.jsonl.gz",
    }
    for path, rows in (
        (outputs["classes_v2"], enhanced_rows),
        (outputs["metric_bindings"], metric_rows),
        (outputs["joint_bindings"], joint_rows),
        (outputs["projections_v2"], projection_rows),
        (outputs["sets_v2"], set_rows),
    ):
        write_jsonl_gz(path, rows)
    protocol = {
        "gate": "pdf_retrieval_v4_gate_09_r5_1",
        "metric_contract": "M0_exact_path_or_M1_exact_leaf",
        "concept_candidates": "support_hint_only",
        "measurement_kind_hard_conflict": True,
        "joint_operand_binding": True,
        "rank_resolves_ambiguity": False,
        "operation_aware_units": True,
        "max_evidence_items": MAX_EVIDENCE_ITEMS,
        "gold_reads_before_seal": 0,
        "strict_binding_reads_before_seal": 0,
        "reference_answer_reads_before_seal": 0,
        "expected_value_reads_before_seal": 0,
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "embedding_calls": 0,
        "bridge_runs": 0,
        "candidate_mutation": 0,
        "semantic_registry_mutation": 0,
        "rule_scan": False,
        "production_writes": 0,
    }
    integrity = {
        "case_count": len(set_rows),
        "binding_status_counts": dict(sorted(status_counts.items())),
        "unit_resolution_status_counts": dict(sorted(unit_status_counts.items())),
        "concept_hint_only_occurrences": concept_only_count,
        "max_evidence_set_size": max_set_size,
        "candidate_outside_r5_access": 0,
        "candidate_mutation": 0,
        "semantic_registry_mutation": 0,
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "input-integrity.json", integrity)
    write_json(OUT / "unit-context-audit.json", {"unit_resolution_status_counts": dict(sorted(unit_status_counts.items()))})
    output_hashes = {name: sha256(path) for name, path in outputs.items()}
    source_files = [
        "metric_binding_contract_v2.py", "joint_operand_binder.py", "unit_context_resolver.py", "operation_unit_contract.py"
    ]
    manifest = {
        "r5_prediction_seal_sha256": sha256(R5 / "prediction-seal.json"),
        "r5_access_universe_sha256": sha256(access_path),
        "r5_semantic_classes_sha256": sha256(classes_path),
        "query_plan_sha256": sha256(QUERY_PLAN),
        "scale_resolutions_sha256": sha256(scale_path),
        "currency_resolutions_sha256": sha256(currency_path),
        "authoritative_evidence_catalog_sha256": {
            path.name: sha256(path) for path in evidence_paths
        },
        "source_sha256": {
            name: sha256(ROOT / "src/pdf_retrieval_v4" / name) for name in source_files
        },
        "prediction_source_sha256": sha256(Path(__file__)),
        "output_sha256": output_hashes,
    }
    write_json(OUT / "prediction-manifest.json", manifest)
    write_json(
        OUT / "prediction-seal.json",
        {**protocol, **integrity, **manifest, "prediction_count": len(set_rows), "sealed": True, "production_switch_allowed": False},
    )
    print(json.dumps({**integrity, **protocol}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
