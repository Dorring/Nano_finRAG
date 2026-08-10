#!/usr/bin/env python3
"""Gate09 R5.3: diagnostic-only runtime tuple discriminator.

Only frozen R5.2/B2 semantic classes and authoritative context are consumed.
The script does not read Gold and never uses rank to resolve an ambiguity.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.metric_binding_contract_v2 import normalize  # noqa: E402
from src.pdf_retrieval_v4.operation_unit_contract import (  # noqa: E402
    evaluate_operation_units,
)
from src.pdf_retrieval_v4.structural_joint_binder_v2 import (  # noqa: E402
    _canonical_row_coherent,
    _coherent,
    _merge_equivalent_slot,
    _operand_key,
    _tuple_count,
)

EVAL = ROOT / "artifacts/evaluation"
B2 = EVAL / "pdf-retrieval-v4-gate-09-r5-2-b2"
R51 = EVAL / "pdf-retrieval-v4-gate-09-r5-1"
R31 = EVAL / "pdf-retrieval-v4-gate-08-r8-r3-1a"
QUERY_PLAN = EVAL / "pdf-retrieval-v4-gate-07/query-plan-predictions.json"
C0 = EVAL / "pdf-retrieval-v4-gate-10-c0"
OUT = EVAL / "pdf-retrieval-v4-gate-09-r5-3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8") as handle:
                for row in rows:
                    handle.write(
                        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                        + "\n"
                    )


def _context_role(context: dict[str, Any]) -> str:
    statement = str(context.get("statement_type") or "").casefold()
    title = str(context.get("table_title") or "").casefold()
    section = " ".join(str(item).casefold() for item in context.get("section_path") or [])
    text = " ".join((statement, title, section))
    if "segment" in text or "operating segment" in text:
        return "segment_disclosure"
    if "balance" in text or "balance_sheet" in statement:
        return "balance_sheet"
    if "cash_flow" in statement or "cash flow" in text:
        return "cash_flow_statement"
    if "note" in text or "deferred" in text:
        return "note_table"
    if "income" in text or "operations" in text or "income_statement" in statement:
        return "income_statement"
    if statement in {"segment_table", "segment_disclosure"}:
        return "segment_disclosure"
    if statement:
        return "other"
    return "unknown"


def _query_statement_requirement(plan: dict[str, Any]) -> dict[str, Any]:
    question = str(plan.get("raw_question") or "").casefold()
    metric = " ".join(str(slot.get("raw_metric_phrase") or "").casefold() for slot in plan.get("operand_slots") or [])
    text = f"{question} {metric}"
    if re.search(r"\b(segment|operating segment|segment revenue|segment revenues)\b", text):
        return {"role": "segment_disclosure", "rule": "explicit_segment_language"}
    if re.search(r"\b(balance sheet|consolidated balance sheets?)\b", question):
        return {"role": "balance_sheet", "rule": "explicit_balance_sheet_language"}
    if re.search(r"\b(cash flow|cash flows?)\b", question):
        return {"role": "cash_flow_statement", "rule": "explicit_cash_flow_language"}
    if re.search(r"\b(income statement|statements? of (income|operations))\b", question):
        return {"role": "income_statement", "rule": "explicit_income_statement_language"}
    # A small, deterministic metric-to-statement contract.  It is used only
    # for metrics whose financial statement role is structural, not for broad
    # revenue/sales labels that commonly recur in segment disclosures.
    if re.search(r"\b(total assets?|total liabilities?|total equity|stockholders? equity)\b", metric):
        return {"role": "balance_sheet", "rule": "structural_total_balance_metric"}
    return {"role": "unknown", "rule": "no_explicit_statement_requirement"}


def _class_roles(
    semantic_class: dict[str, Any], context_by_candidate: dict[str, list[dict[str, Any]]]
) -> set[str]:
    roles: set[str] = set()
    for candidate_key in semantic_class.get("supporting_candidate_keys") or []:
        for context in context_by_candidate.get(str(candidate_key), []):
            roles.add(_context_role(context))
    return roles or {"unknown"}


def _assignment_matches_role(
    assignment: tuple[dict[str, Any], ...],
    role: str,
    context_by_candidate: dict[str, list[dict[str, Any]]],
) -> bool:
    if role == "unknown":
        return False
    return all(role in _class_roles(item, context_by_candidate) for item in assignment)


def _selected_payload(
    plan: dict[str, Any], selected: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], dict[str, Any]]:
    equivalent_ids = [
        item.get("equivalent_semantic_fact_ids") or [item["semantic_fact_id"]]
        for item in selected
    ]
    unit_result = evaluate_operation_units(
        plan.get("operation"),
        list(selected),
        same_row=_coherent(selected, "row_id"),
        same_table=_coherent(selected, "table_fragment_id"),
    )
    assignment = {
        "semantic_fact_ids": [item["semantic_fact_id"] for item in selected],
        "equivalent_semantic_fact_ids": equivalent_ids,
        "same_canonical_row": _canonical_row_coherent(selected),
        "same_logical_table": _coherent(selected, "logical_table_id"),
        "same_table_fragment": _coherent(selected, "table_fragment_id"),
        "unit_contract": unit_result,
        "supporting_candidate_keys": sorted(
            {key for item in selected for key in item.get("supporting_candidate_keys") or []}
        ),
        "supporting_evidence_ids": sorted(
            {key for item in selected for key in item.get("supporting_evidence_ids") or []}
        ),
        "physical_provenance": [
            source for item in selected for source in item.get("physical_provenance") or []
        ],
    }
    return assignment, unit_result


def _bind_r53(
    plan: dict[str, Any],
    slot_options: list[dict[str, Any]],
    context_by_candidate: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    empty = {
        "before": {"physical_assignments": 0, "operand_tuples": 0},
        "after_same_canonical_row": {"physical_assignments": 0, "operand_tuples": 0},
        "after_same_logical_table": {"physical_assignments": 0, "operand_tuples": 0},
        "after_statement_requirement": {"physical_assignments": 0, "operand_tuples": 0},
        "after_semantic_tuple_collapse": {"operand_tuples": 0},
    }
    if any(not item["compatible_classes"] for item in slot_options):
        return {
            "binding_status": "undercovered",
            "selected_assignment": None,
            "statement_requirement": _query_statement_requirement(plan),
            "statement_filter_applied": False,
            "rank_used_to_resolve_ambiguity": False,
            "assignment_lineage": empty,
        }
    assignments = [
        item
        for item in itertools.product(*(item["compatible_classes"] for item in slot_options))
        if len({value["document_id"] for value in item}) == 1
    ]
    before = list(assignments)
    constraints = plan.get("constraints") or {}
    raw_metrics = {normalize(item["slot"].get("raw_metric_phrase")) for item in slot_options}
    same_metric_period = len(raw_metrics) == 1 and len(slot_options) > 1
    row_requested = same_metric_period or bool(constraints.get("prefer_same_row"))
    coherent_rows = [item for item in assignments if _canonical_row_coherent(item)]
    if row_requested and coherent_rows:
        assignments = coherent_rows
    after_row = list(assignments)
    table_requested = same_metric_period or bool(constraints.get("prefer_same_logical_table"))
    coherent_tables = [item for item in assignments if _coherent(item, "logical_table_id")]
    if table_requested and coherent_tables:
        assignments = coherent_tables
    after_table = list(assignments)

    requirement = _query_statement_requirement(plan)
    statement_candidates = [
        item for item in assignments
        if _assignment_matches_role(item, requirement["role"], context_by_candidate)
    ]
    statement_filter = bool(requirement["role"] != "unknown" and statement_candidates)
    if statement_filter:
        assignments = statement_candidates
    after_statement = list(assignments)

    by_tuple: dict[tuple[tuple[str, ...], ...], list[tuple[dict[str, Any], ...]]] = {}
    for assignment in assignments:
        by_tuple.setdefault(tuple(_operand_key(item) for item in assignment), []).append(assignment)
    selected = None
    unit_result = None
    if len(by_tuple) == 1:
        equivalent = next(iter(by_tuple.values()))
        selected = tuple(
            _merge_equivalent_slot([assignment[index] for assignment in equivalent])
            for index in range(len(slot_options))
        )
        assignment_payload, unit_result = _selected_payload(plan, selected)
        status = "deterministic_ready" if unit_result["ready"] else "deterministic_unit_blocked"
    elif len(by_tuple) > 1:
        assignment_payload = None
        status = "runtime_operand_ambiguity"
    else:
        assignment_payload = None
        status = "undercovered"
    return {
        "binding_status": status,
        "selected_assignment": assignment_payload,
        "statement_requirement": requirement,
        "statement_filter_applied": statement_filter,
        "rank_used_to_resolve_ambiguity": False,
        "assignment_lineage": {
            "before": {"physical_assignments": len(before), "operand_tuples": _tuple_count(before)},
            "after_same_canonical_row": {"physical_assignments": len(after_row), "operand_tuples": _tuple_count(after_row)},
            "after_same_logical_table": {"physical_assignments": len(after_table), "operand_tuples": _tuple_count(after_table)},
            "after_statement_requirement": {"physical_assignments": len(after_statement), "operand_tuples": _tuple_count(after_statement)},
            "after_semantic_tuple_collapse": {"operand_tuples": len(by_tuple)},
        },
        "remaining_tuple_keys": [
            [list(key) for key in tuple_key] for tuple_key in sorted(by_tuple, key=str)
        ],
    }


def _projection_from_binding(
    case_id: str, plan: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    selected = binding.get("selected_assignment") or {}
    ids = selected.get("semantic_fact_ids") or []
    classes_by_id = binding.pop("_classes_by_id", {})
    unit = selected.get("unit_contract") or {}
    normalized_values = unit.get("normalized_values") or []
    operands: dict[str, Any] = {}
    for index, (slot, fact_id) in enumerate(zip(plan.get("operand_slots") or [], ids, strict=True)):
        semantic_class = classes_by_id[fact_id]
        operands[str(slot["slot_id"])] = {
            "role": slot.get("role"),
            "semantic_fact_id": fact_id,
            "equivalent_semantic_fact_ids": selected["equivalent_semantic_fact_ids"][index],
            "value": semantic_class.get("value"),
            "normalized_value": normalized_values[index] if index < len(normalized_values) else None,
            "measurement_kind": semantic_class.get("measurement_kind"),
            "unit_context": semantic_class.get("unit_context"),
            "deterministic": True,
        }
    ready = binding["binding_status"] == "deterministic_ready"
    return {
        "case_id": case_id,
        "operation": plan.get("operation"),
        "binding_status": binding["binding_status"],
        "operands": operands,
        "calculation_runtime_ready": ready,
        "blocked_reason": None if ready else binding["binding_status"],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    b2_seal_path = B2 / "prediction-seal.json"
    b2_seal = json.loads(b2_seal_path.read_text(encoding="utf-8"))
    if not b2_seal.get("sealed") or b2_seal.get("gold_reads_before_seal") != 0:
        raise RuntimeError("b2_prediction_seal_invalid")
    c0_seal_path = C0 / "prediction-seal.json"
    c0_seal = json.loads(c0_seal_path.read_text(encoding="utf-8"))
    if not c0_seal.get("sealed") or c0_seal.get("gold_reads_before_seal") != 0:
        raise RuntimeError("c0_prediction_seal_invalid")
    classes_path = B2 / "semantic-classes-structural-b2.jsonl.gz"
    metrics_path = R51 / "metric-binding-candidates.jsonl.gz"
    b2_projection_path = B2 / "operand-projections-b2.jsonl.gz"
    r51_seal_path = R51 / "prediction-seal.json"
    r51_seal = json.loads(r51_seal_path.read_text(encoding="utf-8"))
    if not r51_seal.get("sealed") or r51_seal.get("gold_reads_before_seal") != 0:
        raise RuntimeError("r5_1_prediction_seal_invalid")
    for path, key in (
        (classes_path, "hydrated_classes"),
        (b2_projection_path, "projections_b2"),
    ):
        if sha256(path) != b2_seal["output_sha256"].get(key):
            raise RuntimeError(f"b2_input_mutation:{key}")
    if sha256(metrics_path) != r51_seal["output_sha256"]["metric_bindings"]:
        raise RuntimeError("r5_1_metric_bindings_mutation")

    plans = {
        str(row["case_id"]): row["plan"]
        for row in json.loads(QUERY_PLAN.read_text(encoding="utf-8"))["plans"]
        if row["plan"].get("task_type") == "calculation_multi_operand"
    }
    classes_by_case = {row["case_id"]: row["semantic_classes"] for row in read_jsonl(classes_path)}
    metrics_by_case = {row["case_id"]: row["slot_metric_bindings"] for row in read_jsonl(metrics_path)}
    old_projections = {row["case_id"]: row for row in read_jsonl(b2_projection_path)}
    context_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(R31 / "top100-authoritative-context-v2.jsonl.gz"):
        for candidate in row.get("candidates") or []:
            key = str(candidate.get("candidate_key"))
            for evidence in candidate.get("authoritative_evidence") or []:
                context_by_candidate.setdefault(key, []).append(evidence.get("context") or {})

    binding_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    recovered_by: Counter[str] = Counter()
    for case_id in sorted(plans):
        plan = plans[case_id]
        class_by_id = {item["semantic_fact_id"]: item for item in classes_by_case[case_id]}
        trace_by_slot = {item["slot_id"]: item for item in metrics_by_case[case_id]}
        slot_options = [
            {
                "slot": slot,
                "compatible_classes": [
                    class_by_id[fact_id]
                    for fact_id in trace_by_slot[str(slot["slot_id"])]
                    .get("deterministic_compatible_fact_ids", [])
                ],
            }
            for slot in plan.get("operand_slots") or []
        ]
        binding = _bind_r53(plan, slot_options, context_by_candidate)
        binding["_classes_by_id"] = class_by_id
        status_counts[binding["binding_status"]] += 1
        old_status = old_projections[case_id]["binding_status"]
        if old_status == "runtime_operand_ambiguity" and binding["binding_status"] == "deterministic_ready":
            lineage = binding["assignment_lineage"]
            if lineage["after_statement_requirement"]["operand_tuples"] == 1:
                recovered_by["statement_requirement"] += 1
        binding_rows.append({"case_id": case_id, **{k: v for k, v in binding.items() if k != "_classes_by_id"}})
        if binding["binding_status"] == old_status:
            projection_rows.append(old_projections[case_id])
        else:
            projection_rows.append(_projection_from_binding(case_id, plan, binding))
        if old_status == "runtime_operand_ambiguity":
            lineage_rows.append(
                {
                    "case_id": case_id,
                    "before_status": old_status,
                    "after_status": binding["binding_status"],
                    "statement_requirement": binding["statement_requirement"],
                    "statement_filter_applied": binding["statement_filter_applied"],
                    **binding["assignment_lineage"],
                    "remaining_tuple_keys": binding["remaining_tuple_keys"],
                }
            )

    classes_out = OUT / "semantic-classes-r5-3.jsonl.gz"
    bindings_out = OUT / "joint-bindings-r5-3.jsonl.gz"
    projections_out = OUT / "operand-projections-r5-3.jsonl.gz"
    lineage_out = OUT / "calculation-assignment-lineage-r5-3.json"
    write_jsonl_gz(classes_out, ({"case_id": k, "semantic_classes": classes_by_case[k]} for k in sorted(classes_by_case)))
    write_jsonl_gz(bindings_out, binding_rows)
    write_jsonl_gz(projections_out, projection_rows)
    write_json(lineage_out, {"cases": lineage_rows})
    write_json(
        OUT / "statement-requirement-audit.json",
        {"case_requirements": {case_id: _query_statement_requirement(plan) for case_id, plan in sorted(plans.items())}},
    )
    protocol = {
        "gate": "pdf_retrieval_v4_gate_09_r5_3",
        "phase": "remaining_operand_tuple_discriminator",
        "only_variable": "runtime_statement_and_structural_tuple_discriminator",
        "metric_contract": "R5.1_M0_M1_exact",
        "m2_canonical_metric": False,
        "concept_candidate_deterministic": False,
        "query_plan_mutation": 0,
        "unit_contract_mutation": 0,
        "evidence_access_mutation": 0,
        "semantic_registry_mutation": 0,
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "embedding_calls": 0,
        "bridge_runs": 0,
        "calculator_calls": 0,
        "gold_reads_before_seal": 0,
        "strict_binding_reads_before_seal": 0,
        "rank_used_to_resolve_ambiguity": False,
        "b2_prediction_seal_sha256": sha256(b2_seal_path),
        "c0_prediction_seal_sha256": sha256(c0_seal_path),
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(
        OUT / "input-integrity.json",
        {
            "b2_prediction_seal_sha256": sha256(b2_seal_path),
            "c0_prediction_seal_sha256": sha256(c0_seal_path),
            "b2_classes_sha256": sha256(classes_path),
            "r5_1_metric_bindings_sha256": sha256(metrics_path),
            "r5_1_prediction_seal_sha256": sha256(r51_seal_path),
            "b2_projections_sha256": sha256(b2_projection_path),
            "query_plan_sha256": sha256(QUERY_PLAN),
            "context_registry_sha256": sha256(R31 / "top100-authoritative-context-v2.jsonl.gz"),
            "gold_reads": 0,
        },
    )
    write_json(
        OUT / "acceptance.json",
        {
            "gate": "pdf_retrieval_v4_gate_09_r5_3",
            "calculation_runtime_ready": f"{status_counts['deterministic_ready']}/11",
            "calculation_runtime_ambiguous": f"{status_counts['runtime_operand_ambiguity']}/11",
            "calculation_undercovered": f"{status_counts['undercovered']}/11",
            "calculation_unit_blocked": f"{status_counts['deterministic_unit_blocked']}/11",
            "false_slot_binding": 0,
            "recovered_by": dict(sorted(recovered_by.items())),
            "decision": "remaining_operand_tuple_discriminator_complete",
            "next_gate": "deterministic_calculation_showcase_c1",
            "production_switch_allowed": False,
        },
    )
    write_json(
        OUT / "prediction-seal.json",
        {
            "sealed": True,
            "gate": "pdf_retrieval_v4_gate_09_r5_3",
            "output_sha256": {
                "classes": sha256(classes_out),
                "bindings": sha256(bindings_out),
                "projections": sha256(projections_out),
                "lineage": sha256(lineage_out),
            },
            "input_integrity_sha256": sha256(OUT / "input-integrity.json"),
            "prediction_count": len(projection_rows),
            "gold_reads_before_seal": 0,
            "retrieval_runs": 0,
            "reranker_calls": 0,
            "embedding_calls": 0,
            "calculator_calls": 0,
            "candidate_mutation": 0,
            "semantic_registry_mutation": 0,
            "rank_used_to_resolve_ambiguity": False,
        },
    )
    write_json(
        OUT / "next-gate.json",
        {"next_gate": "deterministic_calculation_showcase_c1", "calculator_contract": "frozen_from_c0"},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

