"""NF-OPT-15 Gate A: build and audit shadow structured retrieval views."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.evaluation.nf_opt_15 import build_retrieval_view

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "benchmarks/financial_rag_v1/corpus.json"
DEFAULT_LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
DEFAULT_QUESTIONS = ROOT / "benchmarks/financial_rag_v1/data/questions.golden.jsonl"
DEFAULT_OUT = ROOT / "artifacts/evaluation/nf-opt-15"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rate(values: list[dict[str, Any]], predicate: Any) -> dict[str, int]:
    return {"present": sum(bool(predicate(item)) for item in values), "total": len(values)}


def run(args: argparse.Namespace) -> int:
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    documents = {str(item["filename"]): item for item in corpus["documents"]}
    labels = _jsonl(args.labels)
    questions = _jsonl(args.questions)
    answerable = [item for item in questions if item.get("answerable")]
    no_answer = [item for item in questions if not item.get("answerable")]
    if len(answerable) != 64 or len(no_answer) != 8:
        raise ValueError("expected 64 answerable and 8 no-answer questions")
    started = time.perf_counter()
    connection = sqlite3.connect(f"file:{args.candidate_db}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT doc_id, content, metadata_json, doc_name FROM chunk_store").fetchall()
    finally:
        connection.close()
    views: list[dict[str, Any]] = []
    failures: list[str] = []
    for doc_id, content, metadata_json, doc_name in rows:
        document = documents.get(str(doc_name))
        if document is None:
            continue
        try:
            metadata = json.loads(metadata_json or "{}")
            views.append(build_retrieval_view(doc_id=str(doc_id), content=str(content or ""), metadata=metadata, document=document))
        except Exception as exc:  # Fail closed: no partial view gets emitted.
            failures.append(f"{doc_id}: {type(exc).__name__}")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    view_by_identity = {
        (str(view["candidate_key"]), str(view["evidence_id"])): view for view in views
    }
    views_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for view in views:
        views_by_key[str(view["candidate_key"])].append(view)
    field_coverage = defaultdict(dict)
    for evidence_type in sorted({str(view["evidence_type"]) for view in views}):
        items = [view for view in views if view["evidence_type"] == evidence_type]
        field_coverage[evidence_type] = {
            "view_count": len(items),
            "metric": _rate(items, lambda item: item["metric_field"]["status"] == "present"),
            "period": _rate(items, lambda item: item["period_field"]["status"] == "present"),
            "statement_or_section": _rate(items, lambda item: bool(item["section_field"]["statement_title"] or item["section_field"]["section_path"])),
            "currency": _rate(items, lambda item: item["unit_field"]["currency"] is not None),
            "scale": _rate(items, lambda item: item["unit_field"]["scale"] is not None),
            "document_fiscal_year": _rate(items, lambda item: item["document_field"]["fiscal_year"] is not None),
        }
    gold_rows: list[dict[str, Any]] = []
    for label in labels:
        if label.get("expected_no_answer"):
            continue
        for source_index, source in enumerate(label.get("expected_sources") or []):
            key = str(source["candidate_key"])
            view = view_by_identity.get((key, str(source["evidence_id"])))
            if view is None and len(views_by_key.get(key, ())) == 1:
                view = views_by_key[key][0]
            gold_rows.append(
                {
                    "case_id": str(label["case_id"]),
                    "source_index": source_index,
                    "candidate_key": str(source["candidate_key"]),
                    "view_exists": view is not None,
                    "metric_present": bool(view and view["metric_field"]["status"] == "present"),
                    "period_present": bool(view and view["period_field"]["status"] == "present"),
                    "metric_period_present": bool(view and view["metric_field"]["status"] == "present" and view["period_field"]["status"] == "present"),
                    "statement_present": bool(view and (view["section_field"]["statement_title"] or view["section_field"]["section_path"])),
                    "currency_present": bool(view and view["unit_field"]["currency"] is not None),
                    "scale_present": bool(view and view["unit_field"]["scale"] is not None),
                }
            )
    if len(gold_rows) != 80:
        raise ValueError(f"expected 80 gold source instances, got {len(gold_rows)}")
    gold_summary = {
        "gold_source_count": 80,
        "view_exists": sum(item["view_exists"] for item in gold_rows),
        "metric_present": sum(item["metric_present"] for item in gold_rows),
        "period_present": sum(item["period_present"] for item in gold_rows),
        "metric_period_present": sum(item["metric_period_present"] for item in gold_rows),
        "statement_present": sum(item["statement_present"] for item in gold_rows),
        "currency_present": sum(item["currency_present"] for item in gold_rows),
        "scale_present": sum(item["scale_present"] for item in gold_rows),
    }
    passed = gold_summary["view_exists"] == 80 and gold_summary["metric_present"] >= 60 and gold_summary["period_present"] >= 55 and gold_summary["metric_period_present"] >= 45 and not failures
    view_hash = hashlib.sha256("\n".join(sorted(str(view["retrieval_view_id"]) for view in views)).encode("utf-8")).hexdigest()
    schema = {
        "view_schema": "nf-opt-15/retrieval-view/v1",
        "candidate_identity_preserved": True,
        "citation_identity": ["candidate_key", "evidence_id", "document_id", "pdf_page"],
        "view_identity_inputs_exclude": ["case_id", "expected_source", "expected_metric", "expected_period", "expected_value", "reference_answer", "benchmark_source_index"],
        "view_fields": ["document_field", "section_field", "metric_field", "period_field", "unit_field", "value_field", "field_lineage"],
    }
    manifest = {
        "view_count": len(views),
        "candidate_store_rows": len(rows),
        "scope_document_count": len(documents),
        "view_build_failure_count": len(failures),
        "candidate_identity_mapping_failure_count": 0 if not failures else len(failures),
        "candidate_key_multi_member_count": sum(len(items) > 1 for items in views_by_key.values()),
        "retrieval_view_id_set_sha256": view_hash,
        "full_view_content_committed": False,
        "view_construction_gold_fields_read": False,
    }
    acceptance = {
        "artifact_schema": "nf-opt-15/gate-a/acceptance/v1",
        "baseline_master_merge_commit": "d52c1ed02c387c551a521f5f7c5ab61180d21e05",
        "baseline_tree_sha": "e6ccdd5832a651cf53422a7273287702ec7532ca",
        "question_count": 72,
        "answerable_case_count": 64,
        "no_answer_case_count": 8,
        "gold_source_count": 80,
        "input_hashes": {"corpus_sha256": _sha(args.corpus), "questions_sha256": _sha(args.questions), "labels_sha256": _sha(args.labels)},
        "view_construction_gold_fields_read": False,
        "gold_used_only_for_posthoc_coverage": True,
        "original_candidate_identity_modified": False,
        "dense_behavior_modified": False,
        "rrf_behavior_modified": False,
        "reranker_behavior_modified": False,
        "final_top_k_modified": False,
        "model_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_code_wired": False,
        "production_switch_allowed": False,
        "decision": "structured_retrieval_view_contract_validated" if passed else "structured_retrieval_view_field_coverage_blocked",
    }
    _write(args.out_dir / "retrieval-view-schema.json", schema)
    _write(args.out_dir / "retrieval-view-manifest.json", manifest)
    _write(args.out_dir / "retrieval-view-field-coverage.json", {"by_evidence_type": field_coverage})
    _write(args.out_dir / "gold-view-coverage-posthoc.json", {"summary": gold_summary, "sources": gold_rows})
    _write(args.out_dir / "bm25f-configuration.json", {"status": "not_run_gate_a", "reason": "field coverage gate must pass before frozen BM25F evaluation"})
    _write(args.out_dir / "bm25f-stage-results.json", {"status": "not_run_gate_a"})
    _write(args.out_dir / "hybrid-transfer-results.json", {"status": "not_run_gate_a"})
    _write(args.out_dir / "strict-hit-regression-report.json", {"status": "not_run_gate_a"})
    _write(args.out_dir / "latency-and-index-report.json", {"view_build_ms": elapsed_ms, "shadow_bm25f_index_built": False, "production_index_written": False})
    _write(args.out_dir / "next-gate.json", {"decision": acceptance["decision"], "next_gate": "nf-opt-15-gate-b-frozen-bm25f-shadow" if passed else "stop_structured_retrieval_view_before_bm25f", "production_switch_allowed": False})
    _write(args.out_dir / "nf-opt-15-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-db", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
