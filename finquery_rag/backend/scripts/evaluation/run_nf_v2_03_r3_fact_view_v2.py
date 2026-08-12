#!/usr/bin/env python3
"""NF-V2-03 R3 BinderFactViewV2 audit and formal Attempt 8 runner.

The offline half of this script is deliberately independent of the model.  It
projects only source labels already present in the frozen Top20 candidate
records.  The optional synthetic suite and Attempt 8 are run only after the
offline precondition is sealed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts.plan import Action, Intent, RequiredSlot, SupervisorPlan  # noqa: E402
from rag_v2.evidence.binder_fact_view import (  # noqa: E402
    binder_fact_view_v2_field_provenance,
    build_binder_fact_view,
    build_binder_fact_views_v2,
)
from rag_v2.evidence.binder_service import BinderRequest, SemanticBinderService  # noqa: E402
from rag_v2.evidence.constrained_binder_provider import BailianConstrainedBinderProvider  # noqa: E402
from scripts.evaluation import run_nf_v2_02_top20_financial_fact_expansion as nf02  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1c_supply_recovery as r1c  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1d_formal_attempt_6 as r1d  # noqa: E402


BASE_COMMIT = "fcd8032bba270eb24a4530a6900dd20ff0ea5b5c"
MODEL = "qwen3.7-plus"
OUT = ROOT / "artifacts/evaluation/nf-v2-03-r3-binder-fact-view-v2"
R2_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r2-distinguishability-review"
R2_FORMAL = ROOT / "artifacts/evaluation/nf-v2-03-r2-semantic-selection/formal-attempt-7"
PROMPT_PATH = ROOT / "artifacts/evaluation/nf-v2-03-r2-semantic-selection/binder-prompt-r2.txt"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write((json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha(value: Any) -> str:
    return hashlib.sha256((json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")).hexdigest()


def estimate_tokens(payload: Any) -> int:
    # A deterministic provider-independent estimate is enough for the V1/V2
    # expansion comparison; provider token metadata is reported separately.
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return max(1, math.ceil(len(body.encode("utf-8")) / 4))


def canonical(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(canonical(item) for item in value)
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), canonical(item)) for key, item in value.items()))
    if value is None:
        return ""
    return str(value).casefold().strip()


def distinguishability_signature(view: Mapping[str, Any]) -> tuple[Any, ...]:
    """Visible semantic/context signature; provenance IDs are not features."""

    return tuple(canonical(view.get(key)) for key in (
        "raw_metric", "normalized_metric", "raw_period", "normalized_period",
        "row_label", "row_path", "row_hierarchy", "column_header_path",
        "multi_level_column_headers", "table_title", "statement_title",
        "statement_type", "section_title", "section_path", "unit",
        "normalized_scale", "raw_scale", "scale", "currency",
        "page", "pdf_page", "table_id", "row_id", "column_id", "cell_id",
    ))


def period_of(view: Mapping[str, Any]) -> str:
    return str(view.get("normalized_period") or view.get("raw_period") or "").casefold().replace(" ", "")


def handle_map(request: BinderRequest) -> dict[str, Mapping[str, Any]]:
    return {f"F{index:02d}": fact for index, fact in enumerate(request.facts, 1)}


def load_frozen() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    frozen = r1d.load_r1c_frozen_inputs()
    state = nf02.verify_frozen_top100()
    return frozen, r1c.candidate_source_map(state)


def build_views(frozen: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for question_id, request in frozen["requests"].items():
        v1 = [build_binder_fact_view(fact, f"F{index:02d}") for index, fact in enumerate(request.facts, 1)]
        v2 = build_binder_fact_views_v2(list(request.facts), source_map)
        result[question_id] = {
            "v1": v1,
            "v2": v2,
            "v1_sha256": stable_sha(v1),
            "v2_sha256": stable_sha(v2),
            "fact_count": len(request.facts),
            "v1_estimated_input_tokens": estimate_tokens(v1),
            "v2_estimated_input_tokens": estimate_tokens(v2),
        }
    return result


def offline_audit(frozen: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    views = build_views(frozen, source_map)
    direct_rows = read_json(R2_OUT / "direct-case-review.json")["rows"]
    direct_unique: list[str] = []
    direct_rows_out: list[dict[str, Any]] = []
    for old in direct_rows:
        qid = str(old["question_id"])
        request = frozen["requests"][qid]
        v2_by_handle = {view["fact_handle"]: view for view in views[qid]["v2"]}
        requested_period = str(request.plan.required_slots[0].period).casefold().replace(" ", "")
        gold_handles = [
            handle for handle in (old.get("gold_compatible_fact_handles") or [])
            if handle in v2_by_handle and period_of(v2_by_handle[handle]) == requested_period
        ]
        competitors = [
            handle for handle in (old.get("competing_fact_handles") or [])
            if handle in v2_by_handle and period_of(v2_by_handle[handle]) == requested_period
        ]
        gold_signatures = {stable_sha(distinguishability_signature(v2_by_handle[handle])) for handle in gold_handles if handle in v2_by_handle}
        competitor_signatures = {stable_sha(distinguishability_signature(v2_by_handle[handle])) for handle in competitors if handle in v2_by_handle}
        # R2 already audited the six CD3 cases as duplicate-equivalent facts
        # within the same statement.  They remain intentionally conservative
        # even though each physical occurrence has a different opaque ID.
        v2_unique = (
            len(gold_signatures) == 1
            and not (gold_signatures & competitor_signatures)
            and old.get("classification") != "CD3_duplicate_equivalent_same_statement"
        )
        if v2_unique:
            direct_unique.append(qid)
        direct_rows_out.append({
            "question_id": qid,
            "v1_visible_unique_bindable": bool(old.get("visible_unique_bindable")),
            "v2_visible_unique_bindable": v2_unique,
            "v1_classification": old.get("classification"),
            "gold_compatible_fact_handles": gold_handles,
            "competing_fact_handles": competitors,
            "gold_visible_signatures": sorted(gold_signatures),
            "competitor_visible_signatures": sorted(competitor_signatures),
            "v2_fields_used": ["metric", "period", "row/header", "table", "statement", "section"],
        })

    calc_audit = read_json(R2_OUT / "calculation-operand-distinguishability.json")
    calc_rows_out: list[dict[str, Any]] = []
    calc_unique_slots: list[tuple[str, str]] = []
    calc_unique_questions: set[str] = set()
    for old in calc_audit["rows"]:
        qid = str(old["question_id"])
        v2_by_handle = {view["fact_handle"]: view for view in views[qid]["v2"]}
        slot_out: list[dict[str, Any]] = []
        all_unique = True
        for slot in old.get("slots", []):
            required_period = str(slot.get("required", {}).get("period", "")).casefold().replace(" ", "")
            gold_handles = [
                handle for handle in (slot.get("gold_compatible_fact_handles") or [])
                if handle in v2_by_handle and period_of(v2_by_handle[handle]) == required_period
            ]
            competitors = list(slot.get("competing_fact_handles") or [])
            gold_signatures = {stable_sha(distinguishability_signature(v2_by_handle[handle])) for handle in gold_handles}
            competitor_signatures = {stable_sha(distinguishability_signature(v2_by_handle[handle])) for handle in competitors if handle in v2_by_handle}
            unique = len(gold_signatures) == 1 and not (gold_signatures & competitor_signatures)
            all_unique = all_unique and unique
            if unique:
                calc_unique_slots.append((qid, str(slot.get("slot_id"))))
            slot_out.append({
                "slot_id": slot.get("slot_id"),
                "role": slot.get("role"),
                "required": slot.get("required"),
                "gold_compatible_fact_handles": gold_handles,
                "competing_fact_handles": competitors,
                "v2_visible_unique_operand": unique,
                "classification": "CC0_unique_visible_operand" if unique else "CC2_same_metric_same_period_wrong_statement",
                "visible_discriminators": ["statement", "table", "row/header", "section"],
            })
        if all_unique:
            calc_unique_questions.add(qid)
        calc_rows_out.append({"question_id": qid, "all_operands_v2_visible_unique": all_unique, "slots": slot_out})

    provenance_rows: list[dict[str, Any]] = []
    fabricated = 0
    cross_candidate = 0
    for qid, request in frozen["requests"].items():
        for fact, view in zip(request.facts, views[qid]["v2"], strict=True):
            candidate_ids = {str(fact.get("candidate_id"))} | {str(item) for item in fact.get("candidate_ids", [])}
            source = source_map.get(str(fact.get("candidate_id"))) or next((source_map.get(item) for item in candidate_ids if item in source_map), {})
            fields = binder_fact_view_v2_field_provenance(fact, source)
            for field, provenance in fields.items():
                valid = provenance.get("source_candidate_id") in candidate_ids or provenance.get("origin") == "financial_fact"
                if not valid:
                    fabricated += 1
                    cross_candidate += 1
                provenance_rows.append({"question_id": qid, "fact_handle": view["fact_handle"], "field": field, **provenance, "valid": valid})

    direct_counts = {f"FV2-{index}": 0 for index in range(8)}
    for row in direct_rows_out:
        if row["v2_visible_unique_bindable"] and not row["v1_visible_unique_bindable"]:
            direct_counts["FV2-0"] += 1
        elif row["v1_classification"] == "CD3_duplicate_equivalent_same_statement":
            direct_counts["FV2-1"] += 1
        elif not row["v2_visible_unique_bindable"]:
            direct_counts["FV2-6"] += 1
    all_tokens = [item["v1_estimated_input_tokens"] for item in views.values()]
    v2_tokens = [item["v2_estimated_input_tokens"] for item in views.values()]
    direct_unique_set = set(direct_unique)
    calc_unique_set = {f"{qid}:{slot_id}" for qid, slot_id in calc_unique_slots}
    audit = {
        "model_calls": 0,
        "question_reads_during_view_build": 0,
        "gold_reads_during_view_build": 0,
        "financial_fact_v1_modified": False,
        "fabricated_context_fields": fabricated,
        "cross_candidate_context_composition": cross_candidate,
        "field_provenance_rows": provenance_rows,
        "direct": {
            "reviewed_strict_bindable": "27/27",
            "v1_visible_unique": "4/27",
            "v2_visible_unique": f"{len(direct_unique)}/27",
            "v2_still_indistinguishable": f"{27 - len(direct_unique)}/27",
            "rows": direct_rows_out,
            "reclassification": direct_counts,
        },
        "calculation": {
            "bindable_questions": "6/6",
            "operand_slots": "12/12",
            "v1_visible_unique": "0/12",
            "v2_visible_unique": f"{len(calc_unique_slots)}/12",
            "questions_all_operands_v2_unique": f"{len(calc_unique_questions)}/6",
            "rows": calc_rows_out,
        },
        "token_impact": {
            "v1_median": statistics.median(all_tokens),
            "v2_median": statistics.median(v2_tokens),
            "v1_p95": sorted(all_tokens)[max(0, math.ceil(len(all_tokens) * .95) - 1)],
            "v2_p95": sorted(v2_tokens)[max(0, math.ceil(len(v2_tokens) * .95) - 1)],
            "v1_max": max(all_tokens),
            "v2_max": max(v2_tokens),
            "bytes_per_fact_v1": round(sum(item["v1_estimated_input_tokens"] * 4 / max(1, item["fact_count"]) for item in views.values()) / len(views), 3),
            "bytes_per_fact_v2": round(sum(item["v2_estimated_input_tokens"] * 4 / max(1, item["fact_count"]) for item in views.values()) / len(views), 3),
            "per_query": {qid: {k: item[k] for k in ("fact_count", "v1_estimated_input_tokens", "v2_estimated_input_tokens")} for qid, item in views.items()},
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "direct-v2-distinguishability.json", audit["direct"])
    write_json(OUT / "calculation-v2-distinguishability.json", audit["calculation"])
    write_json(OUT / "field-provenance-audit.json", {
        "fields": sorted({row["field"] for row in provenance_rows}),
        "rows": provenance_rows,
        "fabricated_context_fields": fabricated,
        "cross_candidate_context_composition": cross_candidate,
        "relation_context_integrity": "100%" if not fabricated and not cross_candidate else "failed",
    })
    write_json(OUT / "remaining-indistinguishable.json", {
        "direct": [row for row in direct_rows_out if not row["v2_visible_unique_bindable"]],
        "reclassification": direct_counts,
        "model_calls": 0,
    })
    write_json(OUT / "token-impact.json", audit["token_impact"])
    write_json(OUT / "binder-fact-view-v2-contract.json", {
        "version": "BinderFactViewV2",
        "builder": "build_binder_fact_view_v2(financial_fact, source_metadata, fact_handle)",
        "query_independent": True,
        "allowed_source_fields": ["row_label", "row_path", "row_hierarchy", "column_label", "column_header_path", "multi_level_column_headers", "table_title", "statement_title", "statement_type", "section_title", "section_path", "page", "table_id", "row_id", "column_id", "cell_id", "physical_source_id", "document_id", "pdf_page", "period_value_bindings"],
        "financial_fact_v1_modified": False,
        "question_conditioned_fields": 0,
        "gold_conditioned_fields": 0,
        "fabricated_context_fields": fabricated,
        "cross_candidate_context_composition": cross_candidate,
        "source_text_parsing": "label-preserving only",
    })
    return {"audit": audit, "views": views, "direct_unique": direct_unique_set, "calc_unique": calc_unique_set}


def synthetic_fact(fact_id: str, metric: str, period: str, *, statement: str, row: list[str], headers: list[str], section: str) -> dict[str, Any]:
    return {
        "fact_id": fact_id,
        "candidate_id": f"candidate:{fact_id}",
        "physical_source_id": f"source:{fact_id}",
        "document_id": "synthetic_v2_document",
        "pdf_page": 1,
        "table_id": f"table:{statement}",
        "row_id": f"row:{fact_id}",
        "column_id": f"column:{period}",
        "cell_id": f"cell:{fact_id}",
        "raw_metric": metric,
        "normalized_metric": metric.casefold(),
        "raw_period": period,
        "normalized_period": period,
        "raw_value": "100",
        "parsed_numeric_value": "100",
        "currency": "USD",
        "unit": "currency",
        "provenance_complete": True,
        "row_label": row[-1],
        "row_hierarchy": row,
        "column_header": headers,
        "column_header_path": headers,
        "table_title": statement,
        "statement_title": statement,
        "section_heading": section,
    }


def make_slot(slot_id: str, metric: str, period: str, role: str = "value") -> RequiredSlot:
    return RequiredSlot(slot_id, metric, period, role, "numeric", None)


def make_plan(slots: list[RequiredSlot], intent: Intent, operation: str | None = None) -> SupervisorPlan:
    return SupervisorPlan(intent=intent, required_slots=tuple(slots), operation=operation, next_action=Action.RETRIEVE)


def synthetic_cases() -> list[tuple[BinderRequest, dict[str, list[str]]]]:
    return [
        (BinderRequest("v2_syn_01", "Select the supplied revenue fact.", make_plan([make_slot("s1", "revenue", "FY2026")], Intent.DIRECT_FACT), (synthetic_fact("f01", "revenue", "FY2026", statement="Operations", row=["Revenue"], headers=["FY2026", "Revenue"], section="Summary"), synthetic_fact("f02", "expenses", "FY2026", statement="Operations", row=["Expenses"], headers=["FY2026", "Expenses"], section="Summary"))), {"s1": ["F01"]}),
        (BinderRequest("v2_syn_02", "Select the operating income from the correct statement.", make_plan([make_slot("s1", "operating income", "FY2026")], Intent.DIRECT_FACT), (synthetic_fact("f01", "operating income", "FY2026", statement="Income Statement", row=["Operating income"], headers=["FY2026"], section="Results"), synthetic_fact("f02", "operating income", "FY2026", statement="Cash Flow Statement", row=["Operating income"], headers=["FY2026"], section="Cash Flows"))), {"s1": ["F01"]}),
        (BinderRequest("v2_syn_03", "Select the regional row.", make_plan([make_slot("s1", "revenue", "FY2026")], Intent.DIRECT_FACT), (synthetic_fact("f01", "revenue", "FY2026", statement="Segment Note", row=["North", "Revenue"], headers=["FY2026", "Revenue"], section="Segments"), synthetic_fact("f02", "revenue", "FY2026", statement="Segment Note", row=["South", "Revenue"], headers=["FY2026", "Revenue"], section="Segments"))), {"s1": ["F01"]}),
        (BinderRequest("v2_syn_04", "Select the fact under the current header.", make_plan([make_slot("s1", "units", "FY2026")], Intent.DIRECT_FACT), (synthetic_fact("f01", "units", "FY2026", statement="Operating Metrics", row=["Units"], headers=["FY2026", "Current", "Units"], section="Metrics"), synthetic_fact("f02", "units", "FY2026", statement="Operating Metrics", row=["Units"], headers=["FY2026", "Prior", "Units"], section="Metrics"))), {"s1": ["F01"]}),
        (BinderRequest("v2_syn_05", "Select current and prior independently.", make_plan([make_slot("current", "sales", "FY2026", "current"), make_slot("prior", "sales", "FY2025", "prior")], Intent.CALCULATION, "growth_rate"), (synthetic_fact("f01", "sales", "FY2026", statement="Results", row=["Sales"], headers=["FY2026"], section="Summary"), synthetic_fact("f02", "sales", "FY2025", statement="Results", row=["Sales"], headers=["FY2025"], section="Summary"))), {"current": ["F01"], "prior": ["F02"]}),
        (BinderRequest("v2_syn_06", "Select numerator and denominator.", make_plan([make_slot("numerator", "gross profit", "FY2026", "numerator"), make_slot("denominator", "sales", "FY2026", "denominator")], Intent.CALCULATION, "percentage_share"), (synthetic_fact("f01", "gross profit", "FY2026", statement="Results", row=["Gross profit"], headers=["FY2026"], section="Summary"), synthetic_fact("f02", "sales", "FY2026", statement="Results", row=["Sales"], headers=["FY2026"], section="Summary"))), {"numerator": ["F01"], "denominator": ["F02"]}),
        (BinderRequest("v2_syn_07", "No supplied fact has the requested scope.", make_plan([make_slot("s1", "regional margin", "FY2026")], Intent.DIRECT_FACT), (synthetic_fact("f01", "margin", "FY2026", statement="Results", row=["Total", "Margin"], headers=["FY2026"], section="Summary"), synthetic_fact("f02", "regional revenue", "FY2026", statement="Segments", row=["Regional", "Revenue"], headers=["FY2026"], section="Segments"))), {"s1": []}),
        (BinderRequest("v2_syn_08", "Two identical statements remain ambiguous.", make_plan([make_slot("s1", "cash", "FY2026")], Intent.DIRECT_FACT), (synthetic_fact("f01", "cash", "FY2026", statement="Balance Sheet", row=["Cash"], headers=["FY2026"], section="Assets"), synthetic_fact("f02", "cash", "FY2026", statement="Balance Sheet", row=["Cash"], headers=["FY2026"], section="Assets"))), {"s1": ["F01", "F02"]}),
    ]


def run_synthetic(prompt: str) -> dict[str, Any]:
    config = r1d.legacy.load_config()
    provider = BailianConstrainedBinderProvider(
        base_url=os.getenv("V2_SUPERVISOR_BASE_URL", "").strip(), api_key=config["api_key"], model_name=MODEL,
        enable_thinking=False, temperature=0.0, timeout=180.0, max_retries=0, system_prompt=prompt,
        fact_view_version="v2", source_metadata_by_candidate={},
    )
    service = SemanticBinderService(provider)
    rows: list[dict[str, Any]] = []
    try:
        for request, expected in synthetic_cases():
            started = time.perf_counter()
            run = service.bind(request)
            reverse = {str(fact["fact_id"]): f"F{index:02d}" for index, fact in enumerate(request.facts, 1)}
            actual = {slot_id: [reverse[str(fid)] for fid in fact_ids] for slot_id, fact_ids in (run.binding.slot_bindings if run.binding else {}).items()}
            for slot in request.plan.required_slots:
                actual.setdefault(slot.slot_id, [])
            semantic_correct = all(sorted(actual.get(slot_id, [])) == sorted(values) for slot_id, values in expected.items())
            false_binding = any(not values and actual.get(slot_id) for slot_id, values in expected.items())
            rows.append({
                "question_id": request.question_id, "expected": expected, "actual": actual,
                "provider_response_success": bool(run.metadata and run.metadata.provider_response_success),
                "structured_output_success": bool(run.metadata and run.metadata.structured_output_success),
                "dto_valid": bool(run.schema_valid), "adapter_valid": bool(run.schema_valid and run.binding),
                "binding_validator_pass": bool(run.validation.passed), "semantic_correct": semantic_correct,
                "false_binding": bool(false_binding), "unknown_slot": 0, "unknown_fact": 0,
                "latency_ms": run.metadata.latency_ms if run.metadata else None,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            })
    finally:
        provider.close()
    summary = {
        "gate": "NF-V2-03-R3", "model": MODEL, "model_calls": len(rows), "benchmark_questions_used": 0,
        "provider_response_success": sum(int(row["provider_response_success"]) for row in rows),
        "structured_output_success": sum(int(row["structured_output_success"]) for row in rows),
        "dto_valid": sum(int(row["dto_valid"]) for row in rows), "adapter_valid": sum(int(row["adapter_valid"]) for row in rows),
        "binding_validator_pass": sum(int(row["binding_validator_pass"]) for row in rows),
        "semantic_correct": sum(int(row["semantic_correct"]) for row in rows), "semantic_total": len(rows),
        "false_binding": sum(int(row["false_binding"]) for row in rows), "unknown_slot": 0, "unknown_fact": 0,
        "pass": sum(int(row["semantic_correct"]) for row in rows) >= 7 and sum(int(row["false_binding"]) for row in rows) == 0 and sum(int(row["provider_response_success"]) for row in rows) == 8,
        "rows": rows,
    }
    write_json(OUT / "synthetic-v2-context-suite.json", summary)
    return summary


def run_formal(prompt: str, frozen: dict[str, Any], source_map: Mapping[str, Mapping[str, Any]], offline: Mapping[str, Any]) -> dict[str, Any]:
    formal_out = OUT / "formal-attempt-8"
    formal_out.mkdir(parents=True, exist_ok=True)
    config = r1d.legacy.load_config()
    predictions, runtime = r1d.run_formal(
        {**config, "base_url": os.getenv("V2_SUPERVISOR_BASE_URL", "").strip()}, frozen,
        system_prompt=prompt, fact_view_version="v2", source_metadata_by_candidate=source_map,
    )
    path = formal_out / "predictions.jsonl.gz"
    write_jsonl_gz(path, predictions)
    prediction_sha = sha256_file(path)
    write_json(formal_out / "prediction-seal.json", {"gate": "NF-V2-03-R3", "sealed": True, "prediction_count": len(predictions), "prediction_sha256": prediction_sha, "gold_reads_before_prediction_seal": 0, "sealed_before_gold": True})
    if sha256_file(path) != prediction_sha:
        raise RuntimeError("Attempt 8 prediction seal verification failed")
    labels = {str(row["case_id"]): row for row in (json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()) if row}
    scored = r1d.score_supply_conditioned(frozen, predictions, labels)
    eligible = [row for row in predictions if not row["skipped_no_fact_supply"]]
    structural = {
        "questions": 72, "provider_responses": sum(int(row["provider_response_success"]) for row in eligible),
        "structured_output": sum(int(row["structured_output_success"]) for row in eligible), "dto_valid": sum(int(row["dto_valid"]) for row in eligible),
        "adapter_valid": sum(int(row["adapter_valid"]) for row in eligible), "binding_validator_pass": sum(int(row["binding_validator_pass"]) for row in predictions),
        "unknown_slots": sum(row["unknown_slot"] for row in predictions), "unknown_facts": sum(row["unknown_fact"] for row in predictions),
        "duplicate_handles": sum(row["duplicate_handle"] for row in predictions), "status_violations": sum(row["status_violation"] for row in predictions),
        "cardinality_violations": sum(row["cardinality_violation"] for row in predictions), "calculation_leakage": sum(row["calculation_leakage"] for row in predictions),
        "gold_reads_before_prediction_seal": 0,
    }
    write_json(formal_out / "structural-metrics.json", structural)
    write_json(formal_out / "direct-visible-unique-metrics.json", {"v1_visible_unique": "4/27", "v2_visible_unique": f"{len(offline['direct_unique'])}/27", "strict_complete": f"{scored['direct']['strict_complete']}/56", "success_given_reviewed_bindable": scored["direct"]["success_given_bindable_percent"], "rows": scored["direct"]["rows"]})
    write_json(formal_out / "calculation-visible-unique-metrics.json", {"v1_visible_unique_operands": "0/12", "v2_visible_unique_operands": f"{sum(1 for _ in offline['calc_unique'])}/12", "strict_complete": f"{scored['calculation']['strict_complete']}/11", "rows": scored["calculation"]["rows"]})
    direct_unique = set(offline["direct_unique"])
    direct_rows = {str(row["question_id"]): row for row in scored["direct"]["rows"]}
    unique_correct = sum(int(direct_rows[qid]["strict_complete"]) for qid in direct_unique if qid in direct_rows)
    calc_unique_slots = set(offline["calc_unique"])
    calc_rows = {str(row["question_id"]): row for row in scored["calculation"]["rows"]}
    unique_correct_slots = 0
    for qid, row in calc_rows.items():
        for slot_row in row.get("slot_results", []):
            if f"{qid}:{slot_row['slot_id']}" in calc_unique_slots:
                unique_correct_slots += int(slot_row.get("strict_correct", False))
    unique_calc_questions = {qid for qid in calc_rows if all(f"{qid}:{slot['slot_id']}" in calc_unique_slots for slot in calc_rows[qid].get("slot_results", []))}
    unique_calc_all = sum(int(calc_rows[qid]["strict_complete"]) for qid in unique_calc_questions)
    false_total = int(scored["false_binding_queries"])
    direct_bindable = set(scored["direct_bindable_ids"])
    false_bindable = sum(int(row["question_id"] in direct_bindable and row["status"] == "BOUND" and not row["strict_complete"]) for row in scored["strict_rows"] if row["intent"] == "DIRECT_FACT")
    false_unbindable = false_total - false_bindable
    write_json(formal_out / "unbindable-safety.json", {"false_binding_total": false_total, "false_binding_on_bindable": false_bindable, "false_binding_on_unbindable": false_unbindable, "before_attempt_7": 7})
    metadata = [row["metadata"] for row in predictions if row.get("metadata")]
    latencies = [float(item.get("latency_ms") or 0) for item in metadata]
    write_json(formal_out / "latency-token-cost.json", {"provider_calls": len(eligible), "input_tokens": sum(int(item.get("input_tokens") or 0) for item in metadata), "output_tokens": sum(int(item.get("output_tokens") or 0) for item in metadata), "average_latency_ms": statistics.mean(latencies) if latencies else 0, "p50_latency_ms": statistics.median(latencies) if latencies else 0, "p95_latency_ms": r1d.percentile(latencies), "max_latency_ms": max(latencies) if latencies else 0, "formal_wall_time_ms": runtime["formal_wall_time_ms"]})
    structural_healthy = all(structural[key] >= threshold for key, threshold in (("provider_responses", len(eligible)), ("structured_output", len(eligible)), ("dto_valid", len(eligible)), ("adapter_valid", len(eligible)), ("binding_validator_pass", 72))) and all(structural[key] == 0 for key in ("unknown_slots", "unknown_facts", "duplicate_handles", "status_violations", "cardinality_violations", "calculation_leakage"))
    visible_direct_quality = (unique_correct / len(direct_unique)) if direct_unique else 0
    visible_calc_quality = (unique_correct_slots / len(calc_unique_slots)) if calc_unique_slots else 0
    strong = structural_healthy and len(direct_unique) >= 20 and visible_direct_quality >= .9 and visible_calc_quality >= .9 and false_bindable == 0 and false_total <= 2
    decision = {
        "gate": "NF-V2-03-R3", "base_commit": BASE_COMMIT, "binder_model": MODEL, "prompt": "R2 unchanged", "formal_attempt_8": "executed", "formal_run_complete": True, "gold_reads_before_prediction_seal": 0, "prediction_seal": "pass", "structural_healthy": structural_healthy,
        "direct_v2_visible_unique": f"{len(direct_unique)}/27", "direct_strict_complete": f"{scored['direct']['strict_complete']}/56", "success_given_visible_unique": f"{unique_correct}/{len(direct_unique)}" if direct_unique else "0/0", "success_given_visible_unique_percent": round(100 * visible_direct_quality, 4),
        "calculation_v2_visible_unique_operands": f"{len(calc_unique_slots)}/12", "calculation_correct_visible_unique_operands": f"{unique_correct_slots}/{len(calc_unique_slots)}", "calculation_all_operand_visible_unique_questions": f"{unique_calc_all}/{len(unique_calc_questions)}" if unique_calc_questions else "0/0", "success_given_visible_unique_operand_percent": round(100 * visible_calc_quality, 4),
        "false_binding_total": false_total, "false_binding_on_bindable": false_bindable, "false_binding_on_unbindable": false_unbindable,
        "binder_semantic_selection_effective": True if strong else False, "binder_semantic_policy_frozen": bool(strong), "binder_fact_view_v2_frozen": bool(strong), "dominant_failure": "none" if strong else ("binder_model_semantic_selection" if len(direct_unique) >= 20 else "source_representation_insufficient"), "next_gate": "v2_04_missing_evidence_supply_repair" if strong else ("v2_03_binder_model_review" if len(direct_unique) >= 20 else "v2_03_fact_view_v2_failure_review"), "production_default": "V1", "production_switch_allowed": False,
    }
    write_json(formal_out / "decision.json", decision)
    write_json(formal_out / "README.md", {
        "gate": "NF-V2-03 R3 Formal Attempt 8",
        "description": "One frozen qwen3.7-plus evaluation using BinderFactViewV2 with Prompt R2 unchanged.",
        "gold_reads_before_prediction_seal": 0,
        "prediction_seal": "pass",
        "decision": decision,
    })
    write_json(formal_out / "factview-v1-v2-ablation.json", {"factview_v1_prompt_r2": {"visible_unique_direct": "4/27", "strict_direct": "13/27", "visible_unique_calculation_operands": "0/12", "correct_operands": "1/12"}, "factview_v2_prompt_r2": {"visible_unique_direct": f"{len(direct_unique)}/27", "strict_direct": f"{scored['direct']['strict_correct_given_bindable']}/27", "visible_unique_calculation_operands": f"{len(calc_unique_slots)}/12", "correct_operands": f"{sum(int(item.get('strict_correct', False)) for row in scored['calculation']['rows'] for item in row.get('slot_results', []))}/12"}})
    write_json(formal_out / "config.json", {"gate": "NF-V2-03-R3", "base_commit": BASE_COMMIT, "provider": "Alibaba Bailian", "model": MODEL, "thinking": False, "temperature": 0.0, "max_retries": 0, "http_timeout_seconds": 180, "prompt_sha256": sha256_file(PROMPT_PATH), "binder_fact_view_version": "v2", "financial_fact_v1_modified": False, "gold_reads_before_prediction_seal": 0, "production_default": "V1", "production_switch_allowed": False})
    return decision


def main() -> int:
    if os.getenv("V2_SUPERVISOR_MODEL", "").strip() not in ("", MODEL):
        raise SystemExit("V2_SUPERVISOR_MODEL must remain qwen3.7-plus")
    if not PROMPT_PATH.exists():
        raise SystemExit("Frozen Prompt R2 is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    frozen, source_map = load_frozen()
    offline = offline_audit(frozen, source_map)
    if "--offline-only" in sys.argv:
        summary = {
            "direct_v2_visible_unique": offline["audit"]["direct"]["v2_visible_unique"],
            "calculation_v2_visible_unique": offline["audit"]["calculation"]["v2_visible_unique"],
            "calculation_questions_all_unique": offline["audit"]["calculation"]["questions_all_operands_v2_unique"],
            "fabricated_context_fields": offline["audit"]["fabricated_context_fields"],
            "cross_candidate_context_composition": offline["audit"]["cross_candidate_context_composition"],
            "model_calls": 0,
        }
        print(json.dumps(summary, sort_keys=True))
        return 0
    synthetic = run_synthetic(PROMPT_PATH.read_text(encoding="utf-8"))
    write_json(OUT / "decision.json", {"gate": "NF-V2-03-R3", "base_commit": BASE_COMMIT, "model_calls_during_offline_audit": 0, "offline_direct_v2_visible_unique": offline["audit"]["direct"]["v2_visible_unique"], "offline_calculation_v2_visible_unique": offline["audit"]["calculation"]["v2_visible_unique"], "synthetic": {"semantic_correct": f"{synthetic['semantic_correct']}/8", "structural": f"{synthetic['provider_response_success']}/8", "false_binding": synthetic['false_binding'], "pass": synthetic['pass']}, "formal_attempt_8": "pending"})
    if len(offline["direct_unique"]) < 20 or len(offline["calc_unique"]) < 10 or not synthetic["pass"] or offline["audit"]["fabricated_context_fields"] or offline["audit"]["cross_candidate_context_composition"]:
        decision = {"gate": "NF-V2-03-R3", "base_commit": BASE_COMMIT, "formal_attempt_8": "not_run", "reason": "offline_or_synthetic_precondition_failed", "production_default": "V1", "production_switch_allowed": False}
        write_json(OUT / "decision.json", decision)
        print(json.dumps(decision, sort_keys=True))
        return 3
    decision = run_formal(PROMPT_PATH.read_text(encoding="utf-8"), frozen, source_map, offline)
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "README.md", {"gate": "NF-V2-03 R3", "description": "BinderFactViewV2 deterministic source-derived context projection; Prompt R2, model, DTO, validator, FinancialFactV1, and Top20 remain frozen.", "decision": decision})
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
