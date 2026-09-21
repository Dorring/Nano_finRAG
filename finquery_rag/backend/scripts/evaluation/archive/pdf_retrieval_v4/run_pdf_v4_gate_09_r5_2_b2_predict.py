#!/usr/bin/env python3
"""Zero-Gold Gate09 R5.2 B2 structural joint binding prediction."""

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

from src.pdf_retrieval_v4.semantic_evidence_set import (  # noqa: E402
    MAX_EVIDENCE_ITEMS,
    minimum_candidate_cover,
)
from src.pdf_retrieval_v4.structural_joint_binder_v2 import (  # noqa: E402
    bind_structural_operands_b2,
    canonical_row_label,
    hydrate_structural_provenance,
)

EVAL = ROOT / "artifacts/evaluation"
R51 = EVAL / "pdf-retrieval-v4-gate-09-r5-1"
R0 = EVAL / "pdf-retrieval-v4-gate-09-r5-2-r0"
G03 = EVAL / "pdf-retrieval-v4-gate-03-r2"
G04 = EVAL / "pdf-retrieval-v4-gate-04"
QUERY_PLAN = EVAL / "pdf-retrieval-v4-gate-07/query-plan-predictions.json"
OUT = EVAL / "pdf-retrieval-v4-gate-09-r5-2-b2"


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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    r51_seal_path = R51 / "prediction-seal.json"
    r51_seal = json.loads(r51_seal_path.read_text(encoding="utf-8"))
    if not r51_seal.get("sealed") or r51_seal["gold_reads_before_seal"] != 0:
        raise RuntimeError("r5_1_prediction_seal_invalid")
    r51_files = {
        "classes_v2": R51 / "semantic-evidence-classes-v2.jsonl.gz",
        "metric_bindings": R51 / "metric-binding-candidates.jsonl.gz",
        "joint_bindings": R51 / "joint-operand-bindings.jsonl.gz",
        "projections_v2": R51 / "operand-projections-v2.jsonl.gz",
        "sets_v2": R51 / "evidence-set-predictions-v2.jsonl.gz",
    }
    for name, path in r51_files.items():
        if r51_seal["output_sha256"][name] != sha256(path):
            raise RuntimeError(f"r5_1_prediction_mutation:{name}")
    r0_integrity = json.loads((R0 / "input-integrity.json").read_text(encoding="utf-8"))
    if r0_integrity["r5_1_prediction_seal_sha256"] != sha256(r51_seal_path):
        raise RuntimeError("r5_2_r0_input_contract_mismatch")

    plans_payload = json.loads(QUERY_PLAN.read_text(encoding="utf-8"))
    plans = {str(row["case_id"]): row["plan"] for row in plans_payload["plans"]}
    classes_by_case = {
        str(row["case_id"]): row["semantic_classes"]
        for row in read_jsonl(r51_files["classes_v2"])
    }
    metric_by_case = {
        str(row["case_id"]): row["slot_metric_bindings"]
        for row in read_jsonl(r51_files["metric_bindings"])
    }
    old_bindings = {
        str(row["case_id"]): row for row in read_jsonl(r51_files["joint_bindings"])
    }
    r5_access = {
        str(row["case_id"]): row["candidates"]
        for row in read_jsonl(EVAL / "pdf-retrieval-v4-gate-09-r5/evidence-access-universe.jsonl.gz")
    }
    logical_path = G04 / "logical-tables.json"
    logical_payload = json.loads(logical_path.read_text(encoding="utf-8"))
    fragment_to_logical = {
        str(fragment): str(table["logical_table_id"])
        for table in logical_payload["logical_tables"]
        for fragment in table.get("fragment_ids") or []
    }
    semantic_rows_path = G03 / "semantic-rows.jsonl"
    semantic_rows = list(read_jsonl(semantic_rows_path))
    row_labels = {str(row["row_id"]): str(row.get("raw_label") or "") for row in semantic_rows}

    binding_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    set_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    hydrated_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    calculation_status_counts: Counter[str] = Counter()
    recovered_by: Counter[str] = Counter()
    resolved_logical = unresolved_logical = resolved_rows = unresolved_rows = 0
    max_set_size = 0
    for case_id in sorted(plans):
        plan = plans[case_id]
        hydrated = [
            hydrate_structural_provenance(item, fragment_to_logical, row_labels)
            for item in classes_by_case[case_id]
        ]
        for item in hydrated:
            for source in item.get("physical_provenance") or []:
                resolved_logical += bool(source.get("logical_table_id"))
                unresolved_logical += not bool(source.get("logical_table_id"))
                resolved_rows += bool(source.get("canonical_row_identity"))
                unresolved_rows += not bool(source.get("canonical_row_identity"))
        class_by_id = {str(item["semantic_fact_id"]): item for item in hydrated}
        trace_by_slot = {str(item["slot_id"]): item for item in metric_by_case[case_id]}
        slot_options: list[dict[str, Any]] = []
        for slot in plan.get("operand_slots") or []:
            fact_ids = trace_by_slot[str(slot["slot_id"])]["deterministic_compatible_fact_ids"]
            slot_options.append(
                {
                    "slot": slot,
                    "compatible_classes": [class_by_id[fact_id] for fact_id in fact_ids],
                }
            )
        binding = bind_structural_operands_b2(plan, slot_options)
        status_counts[binding["binding_status"]] += 1
        is_calculation = plan.get("task_type") == "calculation_multi_operand"
        if is_calculation:
            calculation_status_counts[binding["binding_status"]] += 1
        selected_assignment = binding.get("selected_assignment") or {}
        selected_ids = selected_assignment.get("semantic_fact_ids") or []
        faux_matches = (
            [
                {
                    "slot_id": slot["slot_id"],
                    "slot_status": "deterministic",
                    "compatible_semantic_fact_ids": [fact_id],
                }
                for slot, fact_id in zip(plan.get("operand_slots") or [], selected_ids, strict=True)
            ]
            if selected_ids
            else []
        )
        cover = (
            minimum_candidate_cover(faux_matches, hydrated, r5_access[case_id])
            if selected_ids
            else {
                "selected_candidate_keys": [],
                "covered_semantic_fact_ids": [],
                "complete": False,
                "evidence_item_count": 0,
            }
        )
        max_set_size = max(max_set_size, cover["evidence_item_count"])
        calculation_ready = bool(
            is_calculation
            and binding["binding_status"] == "deterministic_ready"
            and cover["complete"]
        )
        unit_contract = selected_assignment.get("unit_contract") or {}
        normalized_values = unit_contract.get("normalized_values") or []
        operands: dict[str, Any] = {}
        if selected_ids:
            for index, (slot, fact_id) in enumerate(
                zip(plan.get("operand_slots") or [], selected_ids, strict=True)
            ):
                semantic_class = class_by_id[fact_id]
                operands[str(slot["slot_id"])] = {
                    "role": slot.get("role"),
                    "semantic_fact_id": fact_id,
                    "equivalent_semantic_fact_ids": selected_assignment[
                        "equivalent_semantic_fact_ids"
                    ][index],
                    "value": semantic_class.get("value"),
                    "normalized_value": normalized_values[index]
                    if index < len(normalized_values)
                    else None,
                    "measurement_kind": semantic_class.get("measurement_kind"),
                    "unit_context": semantic_class.get("unit_context"),
                    "deterministic": True,
                }
        old_status = old_bindings[case_id]["binding_status"]
        if (
            is_calculation
            and old_status == "runtime_operand_ambiguity"
            and binding["binding_status"] == "deterministic_ready"
        ):
            lineage = binding["assignment_lineage"]
            if (
                lineage["before"]["operand_tuples"] > 1
                and lineage["after_same_canonical_row"]["operand_tuples"] == 1
            ):
                recovered_by["canonical_row"] += 1
            elif (
                lineage["after_same_canonical_row"]["operand_tuples"] > 1
                and lineage["after_same_logical_table"]["operand_tuples"] == 1
            ):
                recovered_by["logical_table"] += 1
            elif lineage["after_semantic_tuple_collapse"]["operand_tuples"] == 1:
                recovered_by["semantic_tuple_collapse"] += 1
        record = {"case_id": case_id, **binding}
        binding_rows.append(record)
        projection_rows.append(
            {
                "case_id": case_id,
                "operation": plan.get("operation"),
                "binding_status": binding["binding_status"],
                "operands": operands,
                "calculation_runtime_ready": calculation_ready,
                "blocked_reason": None
                if calculation_ready
                else unit_contract.get("reason") or binding["binding_status"],
            }
        )
        set_rows.append(
            {
                "case_id": case_id,
                "binding_status": binding["binding_status"],
                "selected_semantic_fact_ids": selected_ids,
                "selected_candidate_keys": cover["selected_candidate_keys"],
                "evidence_item_count": cover["evidence_item_count"],
                "evidence_set_complete": bool(selected_ids and cover["complete"]),
                "calculation_runtime_ready": calculation_ready,
            }
        )
        hydrated_rows.append({"case_id": case_id, "semantic_classes": hydrated})
        if is_calculation and old_status == "runtime_operand_ambiguity":
            lineage_rows.append(
                {
                    "case_id": case_id,
                    **binding["assignment_lineage"],
                    "canonical_row_filter_applied": binding["canonical_row_filter_applied"],
                    "logical_table_filter_applied": binding["logical_table_filter_applied"],
                    "final_status": binding["binding_status"],
                }
            )

    outputs = {
        "hydrated_classes": OUT / "semantic-classes-structural-b2.jsonl.gz",
        "joint_bindings_b2": OUT / "joint-bindings-b2.jsonl.gz",
        "projections_b2": OUT / "operand-projections-b2.jsonl.gz",
        "sets_b2": OUT / "evidence-sets-b2.jsonl.gz",
    }
    for path, rows in (
        (outputs["hydrated_classes"], hydrated_rows),
        (outputs["joint_bindings_b2"], binding_rows),
        (outputs["projections_b2"], projection_rows),
        (outputs["sets_b2"], set_rows),
    ):
        write_jsonl_gz(path, rows)
    write_json(OUT / "calculation-assignment-lineage.json", {"cases": lineage_rows})
    write_json(
        OUT / "logical-table-hydration.json",
        {
            "fragment_to_logical_count": len(fragment_to_logical),
            "resolved_provenance": resolved_logical,
            "unresolved_provenance": unresolved_logical,
            "fragment_used_as_logical_fallback": False,
        },
    )
    write_json(
        OUT / "canonical-row-registry.json",
        {
            "row_count": len(row_labels),
            "resolved_provenance": resolved_rows,
            "unresolved_provenance": unresolved_rows,
            "entries": [
                {
                    "row_id": row_id,
                    "raw_row_label": label,
                    "canonical_row_label": canonical_row_label(label),
                }
                for row_id, label in sorted(row_labels.items())
            ],
        },
    )
    write_json(
        OUT / "structural-filter-attribution.json",
        {
            "recovered_by": dict(sorted(recovered_by.items())),
            "r5_1_calculation_status_counts": {
                "deterministic_ready": 2,
                "runtime_operand_ambiguity": 6,
                "undercovered": 3,
            },
            "b2_calculation_status_counts": dict(sorted(calculation_status_counts.items())),
        },
    )
    outputs.update(
        {
            "assignment_lineage": OUT / "calculation-assignment-lineage.json",
            "logical_table_hydration": OUT / "logical-table-hydration.json",
            "canonical_row_registry": OUT / "canonical-row-registry.json",
            "structural_filter_attribution": OUT / "structural-filter-attribution.json",
        }
    )
    protocol = {
        "gate": "pdf_retrieval_v4_gate_09_r5_2_b2",
        "only_variable": "logical_table_and_canonical_row_structural_binding",
        "metric_contract": "R5.1_M0_M1_exact",
        "m2_canonical_metric": False,
        "concept_candidate_deterministic": False,
        "query_plan_mutation": 0,
        "unit_contract_mutation": 0,
        "evidence_access_mutation": 0,
        "semantic_registry_mutation": 0,
        "gold_reads_before_seal": 0,
        "strict_binding_reads_before_seal": 0,
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "embedding_calls": 0,
        "bridge_runs": 0,
        "calculator_calls": 0,
        "rank_resolves_ambiguity": False,
        "max_evidence_items": MAX_EVIDENCE_ITEMS,
    }
    integrity = {
        "case_count": len(binding_rows),
        "calculation_case_count": sum(
            plan.get("task_type") == "calculation_multi_operand" for plan in plans.values()
        ),
        "binding_status_counts": dict(sorted(status_counts.items())),
        "calculation_status_counts": dict(sorted(calculation_status_counts.items())),
        "max_evidence_set_size": max_set_size,
        "candidate_mutation": 0,
        "semantic_registry_mutation": 0,
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "input-integrity.json", integrity)
    manifest = {
        "r5_1_prediction_seal_sha256": sha256(r51_seal_path),
        "r5_1_output_sha256": {name: sha256(path) for name, path in r51_files.items()},
        "r5_2_r0_input_integrity_sha256": sha256(R0 / "input-integrity.json"),
        "query_plan_sha256": sha256(QUERY_PLAN),
        "logical_tables_sha256": sha256(logical_path),
        "semantic_rows_sha256": sha256(semantic_rows_path),
        "metric_contract_source_sha256": sha256(
            ROOT / "src/pdf_retrieval_v4/metric_binding_contract_v2.py"
        ),
        "unit_contract_source_sha256": sha256(
            ROOT / "src/pdf_retrieval_v4/operation_unit_contract.py"
        ),
        "binder_b2_source_sha256": sha256(
            ROOT / "src/pdf_retrieval_v4/structural_joint_binder_v2.py"
        ),
        "prediction_source_sha256": sha256(Path(__file__)),
        "output_sha256": {name: sha256(path) for name, path in outputs.items()},
    }
    write_json(OUT / "prediction-manifest.json", manifest)
    write_json(
        OUT / "prediction-seal.json",
        {
            **protocol,
            **integrity,
            **manifest,
            "prediction_count": len(binding_rows),
            "sealed": True,
            "production_switch_allowed": False,
        },
    )
    print(json.dumps({**protocol, **integrity}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
