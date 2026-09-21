#!/usr/bin/env python3
"""Post-seal calculation binding failure audit for Gate09 R5.2-R0."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.canonical_metric_identity import (  # noqa: E402
    canonical_metric_id,
    canonical_metric_tokens,
)
from src.pdf_retrieval_v4.metric_binding_contract_v2 import (  # noqa: E402
    bind_metric,
    infer_measurement_kind,
    normalize,
)

EVAL = ROOT / "artifacts/evaluation"
R51 = EVAL / "pdf-retrieval-v4-gate-09-r5-1"
G03 = EVAL / "pdf-retrieval-v4-gate-03-r2"
G04 = EVAL / "pdf-retrieval-v4-gate-04"
QUERY_PLAN = EVAL / "pdf-retrieval-v4-gate-07/query-plan-predictions.json"
OUT = EVAL / "pdf-retrieval-v4-gate-09-r5-2-r0"


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


def _lineage(item: dict[str, Any], field: str) -> set[str]:
    return {
        str(value[field])
        for value in item.get("physical_provenance") or []
        if value.get(field)
    }


def _coherent(assignment: tuple[dict[str, Any], ...], field: str) -> bool:
    values = [_lineage(item, field) for item in assignment]
    return bool(values) and all(values) and bool(set.intersection(*values))


def _logical_ids(item: dict[str, Any], fragment_to_logical: dict[str, str]) -> set[str]:
    return {
        fragment_to_logical[fragment]
        for fragment in _lineage(item, "table_fragment_id")
        if fragment in fragment_to_logical
    }


def _logical_coherent(
    assignment: tuple[dict[str, Any], ...], fragment_to_logical: dict[str, str]
) -> bool:
    values = [_logical_ids(item, fragment_to_logical) for item in assignment]
    return bool(values) and all(values) and bool(set.intersection(*values))


def _dimension_status(plan: dict[str, Any], slot: dict[str, Any], item: dict[str, Any]) -> str | None:
    scope = {normalize(value) for value in plan.get("document_scope") or []}
    if scope and normalize(item.get("document_id")) not in scope:
        return "document_conflict"
    if normalize(slot.get("period")) != normalize(item.get("period")):
        return "period_conflict"
    if slot.get("segment_label") and normalize(slot.get("segment_label")) != normalize(item.get("segment")):
        return "segment_conflict"
    if slot.get("bucket_label") and normalize(slot.get("bucket_label")) != normalize(item.get("bucket")):
        return "bucket_conflict"
    return None


def _shape_status(item: dict[str, Any]) -> tuple[bool, str]:
    provenance = item.get("physical_provenance") or []
    if any(value.get("authoritative_evidence_type") == "atomic" for value in provenance):
        return True, "atomic_fact"
    if any(
        value.get("authoritative_evidence_type") == "matrix"
        and value.get("matrix_dimension_index") is not None
        and value.get("cell_id")
        for value in provenance
    ):
        return True, "row_matrix_exact_dimension"
    if provenance:
        return False, "non_numeric_or_unprojected_evidence"
    return False, "no_typed_semantic_provenance"


def _operand_key(item: dict[str, Any], canonical: bool) -> tuple[str, ...]:
    unit = item.get("unit_context") or {}
    metric = canonical_metric_id(item.get("metric")) if canonical else normalize(item.get("metric"))
    return (
        normalize(item.get("document_id")),
        str(metric or ""),
        normalize(item.get("period")),
        normalize(item.get("segment")),
        normalize(item.get("bucket")),
        normalize(item.get("value")),
        normalize(item.get("measurement_kind")),
        normalize(unit.get("scale")),
        normalize(unit.get("currency")),
    )


def _assignment_audit(
    slot_options: list[list[dict[str, Any]]], fragment_to_logical: dict[str, str]
) -> dict[str, Any]:
    if any(not values for values in slot_options):
        return {
            "assignment_count": 0,
            "operand_tuple_count": 0,
            "same_row_assignment_count": 0,
            "same_fragment_assignment_count": 0,
            "same_logical_table_assignment_count": 0,
            "same_row_operand_tuple_count": 0,
            "same_fragment_operand_tuple_count": 0,
            "same_logical_table_operand_tuple_count": 0,
        }
    assignments = [
        value
        for value in product(*slot_options)
        if len({item["document_id"] for item in value}) == 1
    ]
    same_row = [value for value in assignments if _coherent(value, "row_id")]
    same_fragment = [value for value in assignments if _coherent(value, "table_fragment_id")]
    same_logical = [value for value in assignments if _logical_coherent(value, fragment_to_logical)]

    def tuple_count(values: list[tuple[dict[str, Any], ...]]) -> int:
        return len({tuple(_operand_key(item, canonical=True) for item in value) for value in values})

    return {
        "assignment_count": len(assignments),
        "operand_tuple_count": tuple_count(assignments),
        "same_row_assignment_count": len(same_row),
        "same_fragment_assignment_count": len(same_fragment),
        "same_logical_table_assignment_count": len(same_logical),
        "same_row_operand_tuple_count": tuple_count(same_row),
        "same_fragment_operand_tuple_count": tuple_count(same_fragment),
        "same_logical_table_operand_tuple_count": tuple_count(same_logical),
        "logical_table_identity_is_distinct_from_fragment_identity": True,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    seal_path = R51 / "prediction-seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not seal.get("sealed") or seal["prediction_count"] != 72:
        raise RuntimeError("r5_1_prediction_seal_invalid")
    frozen_files = {
        "classes_v2": R51 / "semantic-evidence-classes-v2.jsonl.gz",
        "metric_bindings": R51 / "metric-binding-candidates.jsonl.gz",
        "joint_bindings": R51 / "joint-operand-bindings.jsonl.gz",
        "projections_v2": R51 / "operand-projections-v2.jsonl.gz",
        "sets_v2": R51 / "evidence-set-predictions-v2.jsonl.gz",
    }
    for name, path in frozen_files.items():
        if seal["output_sha256"][name] != sha256(path):
            raise RuntimeError(f"r5_1_prediction_mutation:{name}")

    plans_payload = json.loads(QUERY_PLAN.read_text(encoding="utf-8"))
    plans = {
        str(row["case_id"]): row["plan"]
        for row in plans_payload["plans"]
        if row["plan"].get("task_type") == "calculation_multi_operand"
    }
    if len(plans) != 11:
        raise RuntimeError(f"calculation_case_count_mismatch:{len(plans)}")
    classes = {
        str(row["case_id"]): row["semantic_classes"]
        for row in read_jsonl(frozen_files["classes_v2"])
        if str(row["case_id"]) in plans
    }
    bindings = {
        str(row["case_id"]): row
        for row in read_jsonl(frozen_files["joint_bindings"])
        if str(row["case_id"]) in plans
    }
    logical_payload = json.loads((G04 / "logical-tables.json").read_text(encoding="utf-8"))
    fragment_to_logical = {
        str(fragment): str(table["logical_table_id"])
        for table in logical_payload["logical_tables"]
        for fragment in table.get("fragment_ids") or []
    }

    slot_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    candidate_fact_tier_counts: Counter[str] = Counter()
    slot_tier_counts: Counter[str] = Counter()
    for case_id, plan in sorted(plans.items()):
        case_options: list[list[dict[str, Any]]] = []
        case_slot_rows: list[dict[str, Any]] = []
        for slot in plan.get("operand_slots") or []:
            slot_metric_id = canonical_metric_id(slot.get("raw_metric_phrase"))
            exact_metric: list[dict[str, Any]] = []
            canonical_metric: list[dict[str, Any]] = []
            exact_dimension: list[dict[str, Any]] = []
            canonical_dimension: list[dict[str, Any]] = []
            exact_shape: list[dict[str, Any]] = []
            canonical_shape: list[dict[str, Any]] = []
            exact_shape_tiers: set[str] = set()
            traces: list[dict[str, Any]] = []
            for item in classes[case_id]:
                current = bind_metric(slot, item)
                evidence_metric_id = canonical_metric_id(item.get("metric"))
                canonical_exact = bool(slot_metric_id and slot_metric_id == evidence_metric_id)
                query_kind = infer_measurement_kind(slot.get("raw_metric_phrase"))
                evidence_kind = infer_measurement_kind(item.get("metric"), item.get("unit_kind"))
                kind_conflict = (
                    query_kind != "unknown"
                    and evidence_kind != "unknown"
                    and query_kind != evidence_kind
                )
                dimension_failure = _dimension_status(plan, slot, item)
                shape_ok, shape_kind = _shape_status(item)
                if current["deterministic_compatible"]:
                    exact_metric.append(item)
                    candidate_fact_tier_counts[str(current["metric_tier"])] += 1
                    if not dimension_failure:
                        exact_dimension.append(item)
                        if shape_ok:
                            exact_shape.append(item)
                            exact_shape_tiers.add(str(current["metric_tier"]))
                if canonical_exact and not kind_conflict:
                    canonical_metric.append(item)
                    if not dimension_failure:
                        canonical_dimension.append(item)
                        if shape_ok:
                            canonical_shape.append(item)
                traces.append(
                    {
                        "semantic_fact_id": item["semantic_fact_id"],
                        "metric": item.get("metric"),
                        "metric_tier": current["metric_tier"],
                        "m0_m1_compatible": current["deterministic_compatible"],
                        "canonical_metric_id": evidence_metric_id,
                        "canonical_metric_exact": canonical_exact,
                        "measurement_kind_conflict": kind_conflict,
                        "dimension_failure": dimension_failure,
                        "shape_compatible": shape_ok,
                        "shape_kind": shape_kind,
                    }
                )

            exact_tuples = {_operand_key(item, canonical=False) for item in exact_shape}
            canonical_tuples = {_operand_key(item, canonical=True) for item in canonical_shape}
            if not classes[case_id]:
                category = "true_metric_absence"
            elif not exact_metric:
                if canonical_shape:
                    category = "metric_representation_mismatch"
                elif canonical_metric and not canonical_dimension:
                    category = "period_or_dimension_conflict"
                elif canonical_dimension and not canonical_shape:
                    category = "shape_conflict"
                elif canonical_metric:
                    category = "other"
                else:
                    category = "true_metric_absence"
            elif not exact_dimension:
                category = "period_or_dimension_conflict"
            elif not exact_shape:
                category = "shape_conflict"
            elif len(exact_tuples) > 1:
                category = "multiple_operand_tuples"
            else:
                category = "other"
            category_counts[category] += 1
            if "M0_exact_path" in exact_shape_tiers:
                slot_metric_tier = "M0_exact_path"
            elif "M1_exact_leaf" in exact_shape_tiers:
                slot_metric_tier = "M1_exact_leaf"
            elif canonical_shape:
                slot_metric_tier = "M2_diagnostic_exact"
            else:
                slot_metric_tier = "no_match"
            slot_tier_counts[slot_metric_tier] += 1
            row = {
                "case_id": case_id,
                "slot_id": slot["slot_id"],
                "role": slot.get("role"),
                "raw_metric_phrase": slot.get("raw_metric_phrase"),
                "canonical_metric_tokens": list(canonical_metric_tokens(slot.get("raw_metric_phrase"))),
                "canonical_metric_id": slot_metric_id,
                "first_failure": category,
                "slot_metric_tier": slot_metric_tier,
                "m0_m1_metric_match_count": len(exact_metric),
                "m0_m1_dimension_match_count": len(exact_dimension),
                "m0_m1_shape_match_count": len(exact_shape),
                "m0_m1_operand_tuple_count": len(exact_tuples),
                "m2_diagnostic_metric_match_count": len(canonical_metric),
                "m2_diagnostic_dimension_match_count": len(canonical_dimension),
                "m2_diagnostic_shape_match_count": len(canonical_shape),
                "m2_diagnostic_operand_tuple_count": len(canonical_tuples),
                "m2_diagnostic_only": True,
                "traces": traces,
            }
            slot_rows.append(row)
            case_slot_rows.append({key: value for key, value in row.items() if key != "traces"})
            case_options.append(exact_shape)

        assignment_audit = _assignment_audit(case_options, fragment_to_logical)
        case_rows.append(
            {
                "case_id": case_id,
                "operation": plan.get("operation"),
                "r5_1_binding_status": bindings[case_id]["binding_status"],
                "slot_blockers": [
                    {"slot_id": item["slot_id"], "first_failure": item["first_failure"]}
                    for item in case_slot_rows
                ],
                "slot_metrics": case_slot_rows,
                "joint_assignment_diagnostic": assignment_audit,
            }
        )

    blocked_slot_count = sum(value for key, value in category_counts.items() if key != "other")
    significance_threshold = max(2, (blocked_slot_count + 4) // 5)
    metric_count = category_counts["metric_representation_mismatch"]
    tuple_count = category_counts["multiple_operand_tuples"]
    metric_significant = metric_count >= significance_threshold
    tuple_significant = tuple_count >= significance_threshold
    if metric_significant and tuple_significant:
        decision = "metric_representation_and_multiple_tuple_failures_significant"
        next_gate = "canonical_metric_and_structural_joint_binding"
        recommended_ablation = "B3"
    elif metric_significant:
        decision = "metric_representation_failure_dominant"
        next_gate = "canonical_metric_deterministic_binding"
        recommended_ablation = "B1"
    elif tuple_significant:
        decision = "multiple_operand_tuple_failure_dominant"
        next_gate = "structural_joint_binder_repair"
        recommended_ablation = "B2"
    else:
        decision = "calculation_binding_failure_not_concentrated"
        next_gate = "binder_contract_diagnosis"
        recommended_ablation = "B0"

    aggregate = {
        "calculation_case_count": len(case_rows),
        "required_slot_count": len(slot_rows),
        "first_failure_counts": dict(sorted(category_counts.items())),
        "slot_metric_tier_counts": dict(sorted(slot_tier_counts.items())),
        "candidate_fact_metric_tier_counts": dict(
            sorted(candidate_fact_tier_counts.items())
        ),
        "blocked_slot_count": blocked_slot_count,
        "significance_threshold": significance_threshold,
        "metric_representation_significant": metric_significant,
        "multiple_operand_tuples_significant": tuple_significant,
        "r5_1_case_status_counts": dict(
            sorted(Counter(item["r5_1_binding_status"] for item in case_rows).items())
        ),
    }
    protocol = {
        "gate": "pdf_retrieval_v4_gate_09_r5_2_r0",
        "stage": "post_seal_diagnostic_only",
        "prediction_rerun": 0,
        "binder_mutation": 0,
        "query_plan_mutation": 0,
        "unit_contract_mutation": 0,
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "embedding_calls": 0,
        "calculator_calls": 0,
        "canonical_metric_used_for_binding": False,
        "canonical_metric_allowed_operations": [
            "casefold",
            "punctuation_and_path_separator_normalization",
            "whitespace_normalization",
            "closed_english_morphology_map",
            "remove_the_of_for",
        ],
        "canonical_metric_forbidden": [
            "synonym",
            "concept_candidate",
            "embedding",
            "fuzzy",
            "subset",
            "llm",
        ],
        "rank_resolves_ambiguity": False,
    }
    input_integrity = {
        "r5_1_prediction_seal_sha256": sha256(seal_path),
        "r5_1_output_sha256": {name: sha256(path) for name, path in frozen_files.items()},
        "r5_1_output_hashes_exact": True,
        "query_plan_sha256": sha256(QUERY_PLAN),
        "logical_tables_sha256": sha256(G04 / "logical-tables.json"),
        "candidate_mutation": 0,
        "semantic_registry_mutation": 0,
    }
    ablation = {
        "diagnostic_only": True,
        "groups": {
            "B0": "R5.1 exact metric plus current structural binder",
            "B1": "M2 canonical metric only",
            "B2": "logical-table and canonical-row structural binder only",
            "B3": "M2 canonical metric plus structural binder",
        },
        "recommended_next_group": recommended_ablation,
        "no_ablation_executed_in_r0": True,
    }
    slot_path = OUT / "calculation-slot-audit.jsonl.gz"
    write_jsonl_gz(slot_path, slot_rows)
    write_json(OUT / "calculation-case-blocker-matrix.json", {"cases": case_rows})
    write_json(OUT / "aggregate-counts.json", aggregate)
    write_json(OUT / "ablation-preregistration.json", ablation)
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "input-integrity.json", input_integrity)
    manifest = {
        "input_integrity": input_integrity,
        "output_sha256": {
            "slot_audit": sha256(slot_path),
            "case_matrix": sha256(OUT / "calculation-case-blocker-matrix.json"),
            "aggregate": sha256(OUT / "aggregate-counts.json"),
            "ablation": sha256(OUT / "ablation-preregistration.json"),
        },
        "audit_source_sha256": sha256(Path(__file__)),
        "canonical_metric_source_sha256": sha256(
            ROOT / "src/pdf_retrieval_v4/canonical_metric_identity.py"
        ),
    }
    write_json(OUT / "audit-manifest.json", manifest)
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_09_r5_2_r0",
        "decision": decision,
        "next_gate": next_gate,
        "recommended_ablation": recommended_ablation,
        **aggregate,
        "prediction_mutation": 0,
        "binder_mutation": 0,
        "query_plan_mutation": 0,
        "unit_contract_mutation": 0,
        "production_switch_allowed": False,
    }
    write_json(OUT / "acceptance.json", acceptance)
    write_json(
        OUT / "next-gate.json",
        {
            "decision": decision,
            "next_gate": next_gate,
            "recommended_ablation": recommended_ablation,
            "production_switch_allowed": False,
        },
    )
    print(json.dumps(acceptance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
