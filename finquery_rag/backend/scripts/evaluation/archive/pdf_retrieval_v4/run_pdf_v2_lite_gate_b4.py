"""Evaluate V2-Lite with deterministic natural-query paraphrases and controls."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

from scripts.evaluation.run_pdf_retrieval_v2_lite import _bm25, _tokens, _write
from scripts.evaluation.run_pdf_v2_lite_gate_b3 import DEFAULT_OUT as B3_OUT, _run_variant
from src.retrieval.embedding_provider import ExistingMiniLMEmbeddingProvider
from src.services.retrieval_config import get_embedding_model_name

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-representation-v2-lite-gate-b4"
GENERIC_ALIASES = (
    (re.compile(r"(?i)\bnet sales\b"), "sales"),
    (re.compile(r"(?i)\bnet revenues?\b"), "revenue"),
    (re.compile(r"(?i)\btotal revenues?\b"), "revenue"),
    (re.compile(r"(?i)\boperating income\b"), "operating profit"),
    (re.compile(r"(?i)\baccounts receivable\b"), "receivables"),
    (re.compile(r"(?i)\bcash and cash equivalents\b"), "cash"),
    (re.compile(r"(?i)\bproperty,? plant and equipment\b"), "property and equipment"),
    (re.compile(r"(?i)\bstockholders[’']? equity\b"), "shareholders equity"),
    (re.compile(r"(?i)\btotal assets\b"), "assets"),
    (re.compile(r"(?i)\btotal liabilities\b"), "liabilities"),
)
NO_ANSWER_METRICS = ("lunar mining revenue", "quantum battery inventory", "Martian payroll expense", "underwater data-center assets")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _alias(metric: str) -> tuple[str, str]:
    normalized = " ".join(metric.split())
    for pattern, replacement in GENERIC_ALIASES:
        if pattern.search(normalized):
            return pattern.sub(replacement, normalized), "generic_financial_alias"
    shortened = re.sub(r"(?i)\b(total|net)\b", "", normalized)
    shortened = " ".join(shortened.split())
    return (shortened if shortened else normalized), "generic_modifier_elision" if shortened != normalized else "natural_question_wrapper"


def _issuer(document_id: str) -> str:
    return document_id.removesuffix("_pdf_dev").replace("_fy2026", "").replace("_fy2025", "").replace("_", " ").title()


def _natural_cases(source_cases: list[dict[str, object]], views: list[dict[str, object]], *, limit_per_document: int) -> list[dict[str, object]]:
    by_key = {str(view["candidate_key"]): view for view in views}
    counts: Counter[str] = Counter()
    output = []
    templates = (
        "What did {issuer} report for {metric} in fiscal {year}?",
        "How much {metric} did {issuer} have in FY{year}?",
        "For the year ended {year}, what was {issuer}'s {metric}?",
    )
    for source in source_cases:
        view = by_key[str(source["gold_candidate_key"])]
        document_id = str(view["document_id"])
        if counts[document_id] >= limit_per_document:
            continue
        periods = list(view["table_period_tokens"])
        if not periods:
            continue
        metric, method = _alias(str(view["metric"]))
        year = periods[0].removeprefix("FY")
        template_index = counts[document_id] % len(templates)
        query = templates[template_index].format(issuer=_issuer(document_id), metric=metric, year=year)
        output.append({"case_id": f"natural:{source['case_id']}", "document_id": document_id, "query": query, "gold_candidate_key": source["gold_candidate_key"], "paraphrase_method": method, "statement_omitted": True, "expected_value_used": False, "source_query_text_used": False})
        counts[document_id] += 1
    return output


def _no_answer_controls(views: list[dict[str, object]]) -> list[dict[str, object]]:
    documents = sorted({str(view["document_id"]) for view in views})
    controls = []
    for index, document_id in enumerate(documents):
        for metric in NO_ANSWER_METRICS:
            controls.append({"case_id": f"no-answer:{document_id}:{index}:{metric.replace(' ', '-')}", "document_id": document_id, "query": f"What was {_issuer(document_id)}'s {metric} in fiscal 2025?", "metric_terms": _tokens(metric), "expected_route": "no_answer"})
    return controls


def _intrusions(controls: list[dict[str, object]], views: list[dict[str, object]], field: str) -> int:
    documents = [str(view[field]) for view in views]
    count = 0
    for control in controls:
        ranked = _bm25(str(control["query"]), documents)[:5]
        metric_terms = set(control["metric_terms"])
        count += int(any(metric_terms <= set(_tokens(str(views[index]["metric"]))) for index in ranked))
    return count


def run(args: argparse.Namespace) -> int:
    b3_acceptance = B3_OUT / "pdf-v2-lite-gate-b3-acceptance.json"
    b3 = json.loads(b3_acceptance.read_text(encoding="utf-8"))
    if not b3["gate_passed"]:
        raise RuntimeError("Gate B3 must pass before B4")
    views_path = args.runtime_dir / "pdf-v2-lite-retrieval-views.json"
    source_cases_path = args.runtime_dir / "pdf-v2-lite-development-benchmark.json"
    views = json.loads(views_path.read_text(encoding="utf-8"))["views"]
    source_cases = json.loads(source_cases_path.read_text(encoding="utf-8"))["cases"]
    cases = _natural_cases(source_cases, views, limit_per_document=args.limit_per_document)
    controls = _no_answer_controls(views)
    model_name = get_embedding_model_name()
    provider = ExistingMiniLMEmbeddingProvider(model_name_or_path=model_name, device=args.device)
    fields = ("raw_row_text", "enriched_retrieval_text")
    embeddings = {field: provider.encode_documents([str(view[field]) for view in views]) for field in fields}
    query_vectors = provider.encode_queries([str(case["query"]) for case in cases])
    baseline = _run_variant(name="natural_raw", bm25_field="raw_row_text", dense_field="raw_row_text", views=views, cases=cases, embeddings=embeddings, query_vectors=query_vectors, reranker_text_field="raw_row_text")
    enriched = _run_variant(name="natural_enriched", bm25_field="enriched_retrieval_text", dense_field="enriched_retrieval_text", views=views, cases=cases, embeddings=embeddings, query_vectors=query_vectors, reranker_text_field="raw_row_text")
    baseline_final = set(baseline.pop("final_hit_keys"))
    enriched_final = set(enriched.pop("final_hit_keys"))
    baseline.pop("traces")
    enriched.pop("traces")
    bm25_gain = enriched["stage_recalls"]["bm25_5"] - baseline["stage_recalls"]["bm25_5"]
    final_gain = enriched["stage_recalls"]["final_5"] - baseline["stage_recalls"]["final_5"]
    regression_count = len(baseline_final - enriched_final)
    baseline_intrusion = _intrusions(controls, views, "raw_row_text")
    enriched_intrusion = _intrusions(controls, views, "enriched_retrieval_text")
    gate_passed = len(cases) >= 80 and bm25_gain >= 0.10 and final_gain >= 0.05 and regression_count == 0 and enriched_intrusion <= baseline_intrusion
    runtime_cases = args.runtime_dir / "pdf-v2-lite-natural-development-benchmark.json"
    _write(runtime_cases, {"cases": cases, "no_answer_controls": controls})
    acceptance = {"schema": "pdf-retrieval-representation-v2-lite/gate-b4/acceptance/v1", "b3_acceptance_sha256": _sha(b3_acceptance), "runtime_views_sha256": _sha(views_path), "runtime_natural_cases_sha256": _sha(runtime_cases), "natural_case_count": len(cases), "no_answer_control_count": len(controls), "bm25_recall_at_5_gain": bm25_gain, "hybrid_final_recall_at_5_gain": final_gain, "regressed_final_hit_count": regression_count, "baseline_no_answer_metric_intrusion_count": baseline_intrusion, "enriched_no_answer_metric_intrusion_count": enriched_intrusion, "gate_passed": gate_passed, "annotation_mode": "deterministic_natural_paraphrase_with_general_financial_aliases", "independent_human_annotation": False, "queries_copy_full_statement_title": False, "expected_value_reads": 0, "frozen_72_question_reads": 0, "model_training_calls": 0, "answer_generation_calls": 0, "production_index_writes": 0, "production_behavior_changed": False, "production_switch_allowed": False, "frozen_transfer_allowed": gate_passed, "decision": "pdf_v2_lite_natural_query_gate_passed" if gate_passed else "pdf_v2_lite_natural_query_gain_insufficient", "next_gate": "single_frozen_72_transfer" if gate_passed else "stop_pdf_v2_lite"}
    _write(args.out_dir / "natural-query-manifest.json", {"case_count": len(cases), "document_counts": dict(Counter(case["document_id"] for case in cases)), "paraphrase_method_counts": dict(Counter(case["paraphrase_method"] for case in cases)), "statement_omitted_count": sum(case["statement_omitted"] for case in cases), "runtime_query_text_committed": False})
    _write(args.out_dir / "natural-query-hybrid-results.json", {"baseline": baseline, "enriched": enriched, "new_final_hit_count": len(enriched_final - baseline_final), "regressed_final_hit_count": regression_count})
    _write(args.out_dir / "no-answer-control-report.json", {"control_count": len(controls), "metric_intrusion_definition": "all synthetic metric terms present in a top-5 candidate metric", "baseline_intrusion_count": baseline_intrusion, "enriched_intrusion_count": enriched_intrusion})
    _write(args.out_dir / "next-gate.json", {"decision": acceptance["decision"], "next_gate": acceptance["next_gate"], "production_switch_allowed": False})
    _write(args.out_dir / "pdf-v2-lite-gate-b4-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit-per-document", type=int, default=30)
    parser.add_argument("--device", default=os.getenv("PDF_V2_LITE_EMBEDDING_DEVICE", "cpu"))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
