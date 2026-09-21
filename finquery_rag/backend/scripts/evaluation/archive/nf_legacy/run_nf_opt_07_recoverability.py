"""Run NF-OPT-07 recoverability audit only; no table parsing or generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.evaluation import run_nf_eval_03_r1 as r1
from scripts.evaluation import run_nf_opt_01 as opt01
from src.evaluation.nf_opt_07 import (
    AuditInput,
    classify_recoverability,
    has_headers,
    has_numeric_cells,
    has_scale,
    recoverability_gate,
    verified_parent_relation,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/financial_rag_v1/data"
OUT = ROOT / "artifacts/evaluation/nf-opt-07"
NEG = ROOT / "artifacts/evaluation/nf-eval-02/negative-evidence-review-report.json"
LIVE_DB = Path(
    "/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/backend/rag_bm25.db"
)


def write(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def audit_input(item: dict, by_doc: dict[str, dict]) -> AuditInput:
    metadata = item["metadata"]
    parent_raw = by_doc.get(metadata.get("parent_id"))
    parent = audit_input(parent_raw, {}) if parent_raw else None
    return AuditInput(
        candidate_key=item["candidate_key"],
        document_id=str(metadata.get("canonical_document_id")),
        page=metadata.get("page"),
        content_hash=str(metadata.get("content_hash")),
        content=item["content"],
        metadata=metadata,
        parent=parent,
    )


def control_manifest(universe: list[dict], gold_keys: set[str]) -> dict:
    """Freeze a deterministic non-answer control set before parser work."""
    non_gold = sorted(
        (item for item in universe if item["candidate_key"] not in gold_keys),
        key=lambda item: item["candidate_key"],
    )
    tables = [
        item
        for item in non_gold
        if item["metadata"].get("type") in {"table", "table_row"}
    ]
    text = [item for item in non_gold if item["metadata"].get("type") == "text"]
    selected = {
        "non_calculation_table_sources": tables[:30],
        "plain_text_financial_sources": text[:10],
        "same_page_table_controls": tables[30:40],
        "wrong_period_or_column_controls": tables[40:50],
        "scale_ambiguous_controls": [
            item for item in tables if not has_scale(audit_input(item, {}))
        ][:10],
    }
    compact = {
        group: [
            {
                "candidate_key": item["candidate_key"],
                "document_id": item["metadata"].get("canonical_document_id"),
                "page": item["metadata"].get("page"),
                "content_hash": item["metadata"].get("content_hash"),
                "type": item["metadata"].get("type"),
            }
            for item in items
        ]
        for group, items in selected.items()
    }
    frozen = json.dumps(compact, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "nf-opt-07-control-set/v1",
        "table_extraction_control_set_hash": hashlib.sha256(frozen).hexdigest(),
        "groups": compact,
    }


def main() -> None:
    inputs = r1._load_inputs(
        corpus_path=ROOT / "benchmarks/financial_rag_v1/corpus.json",
        manifest_path=DATA / "golden-manifest.json",
        questions_path=DATA / "questions.golden.jsonl",
        labels_path=DATA / "labels.golden.jsonl",
        review_status_path=DATA / "review-status.golden.jsonl",
        negative_report_path=NEG,
    )
    if not all(inputs.hash_report["matches"].values()):
        raise ValueError("frozen input hashes failed")
    mapping = r1._doc_map(inputs.corpus)
    universe, _ = opt01._load_candidate_universe(
        db_path=LIVE_DB,
        corpus=inputs.corpus,
        mapping=mapping,
        tenant_id=1,
        gold_keys=[],
    )
    by_key = {item["candidate_key"]: item for item in universe}
    by_doc = {item["doc_id"]: item for item in universe}
    records = []
    gold_keys = set()
    for label in inputs.labels_by_id.values():
        if not label.get("calculation"):
            continue
        for source_index, source in enumerate(label["expected_sources"]):
            key = str(source["candidate_key"])
            gold_keys.add(key)
            item = by_key[key]
            audit = audit_input(item, by_doc)
            state, reason = classify_recoverability(audit)
            records.append(
                {
                    "case_id": label["case_id"],
                    "source_index": source_index,
                    "candidate_key": key,
                    "document_id": audit.document_id,
                    "page": audit.page,
                    "content_hash": audit.content_hash,
                    "serialization_format": "markdown_table"
                    if "|" in audit.content
                    else "plain_text",
                    "recoverability": state,
                    "contains_table_title": bool(audit.metadata.get("table_title")),
                    "contains_scale": has_scale(audit),
                    "contains_column_headers": has_headers(audit),
                    "contains_row_label": bool(audit.metadata.get("row_label")),
                    "contains_numeric_cells": has_numeric_cells(audit),
                    "requires_parent_candidate": bool(audit.metadata.get("parent_id")),
                    "requires_neighbor_candidate": False,
                    "verified_parent_relation": verified_parent_relation(audit),
                    "ambiguity_reason": reason,
                }
            )
    gate = recoverability_gate(records)
    control = control_manifest(universe, gold_keys)
    write("input-integrity-report.json", inputs.hash_report)
    write("control-set-manifest.json", control)
    write("table-recoverability-audit.json", {"records": records, **gate})
    write(
        "parser-coverage-report.json",
        {
            "status": "not_started",
            "reason": "recoverability_gate_must_pass_before_parser_implementation",
        },
    )
    decision = (
        "table_fact_extraction_recoverability_validated"
        if gate["gate_passed"]
        else "table_fact_extraction_blocked_by_source_representation"
    )
    write(
        "next-gate.json",
        {
            "decision": decision,
            "production_switch_allowed": False,
            "next_gate": (
                "table_parser_shadow_ab"
                if gate["gate_passed"]
                else "shadow_structured_table_reingestion"
            ),
        },
    )
    write(
        "nf-opt-07-acceptance.json",
        {
            "decision": decision,
            "recoverability_gate_passed": gate["gate_passed"],
            "production_behavior_changed": False,
            "production_switch_allowed": False,
            "model_chat_completion_requests": 0,
            "answer_generation_calls": 0,
            "parser_implemented": False,
            "input_hashes_verified": True,
            **gate,
        },
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
