"""Build retrieval-only row views and compare raw-row BM25 with header enrichment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

from scripts.evaluation.run_pdf_sr_v2_gate_a import SOURCES, _camelot_adapter, _candidate_pages, _download, _write
from scripts.evaluation.run_pdf_sr_v2_gate_a_r2 import DEFAULT_OUT as R2_OUT, _table_record
from scripts.evaluation.run_pdf_sr_v2_gate_a_r3 import DEFAULT_OUT as R3_OUT, _extended_record
from src.evaluation.pdf_source_representation_v2 import YEAR_RE, normalize_text, parse_number, row_label, stable_identity

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-representation-v2-lite"
TOKEN_RE = re.compile(r"[A-Za-z]+|(?:19|20)\d{2}|\d+(?:\.\d+)?")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def _bm25(query: str, documents: list[str], *, k1: float = 1.2, b: float = 0.75) -> list[int]:
    tokenized = [_tokens(document) for document in documents]
    query_tokens = _tokens(query)
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / len(lengths) if lengths else 0
    document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
    scores = []
    for index, tokens in enumerate(tokenized):
        frequencies = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            inverse_frequency = math.log(1 + (len(tokenized) - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5))
            denominator = frequency + k1 * (1 - b + b * lengths[index] / average_length) if average_length else frequency
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        scores.append((score, index))
    return [index for _, index in sorted(scores, key=lambda item: (-item[0], item[1]))]


def _issuer(document_id: str) -> str:
    return document_id.removesuffix("_pdf_dev").replace("_fy2026", "").replace("_fy2025", "").replace("_", " ")


def _view(record: dict[str, object], row_index: int, row: list[str]) -> dict[str, object] | None:
    metric = row_label(row)
    numeric_columns = [index for index, cell in enumerate(row) if parse_number(cell) is not None and not YEAR_RE.fullmatch(normalize_text(cell))]
    if not metric or not numeric_columns:
        return None
    table_periods = sorted({period["normalized_period"] for period in record["periods"] if period})
    resolved_numeric = [index for index in numeric_columns if index < len(record["periods"]) and record["periods"][index]]
    lineage = record["lineage"] or {}
    raw_row = normalize_text(" | ".join(row))
    candidate_key = stable_identity("pdf-v2-lite-row", record["table_id"], row_index, row)
    binding_status = "strict_cell_aware" if len(resolved_numeric) == len(numeric_columns) else "table_level_only"
    enriched = normalize_text(
        " ".join(
            [
                f"document {_issuer(str(record['document_id']))}",
                f"metric {metric}",
                f"section {lineage.get('lineage_title') or ''}",
                f"table periods {' '.join(table_periods)}",
                f"period binding {binding_status}",
                f"scale {record.get('scale') or ''}",
                f"row {raw_row}",
            ]
        )
    )
    return {
        "candidate_key": candidate_key,
        "document_id": record["document_id"],
        "pdf_page": record["pdf_page"],
        "source_table_id": record["table_id"],
        "row_index": row_index,
        "metric": metric,
        "statement_or_section": lineage.get("lineage_title"),
        "lineage_type": lineage.get("lineage_type"),
        "table_period_tokens": table_periods,
        "period_binding_status": binding_status,
        "scale": record.get("scale"),
        "raw_row_text": raw_row,
        "enriched_retrieval_text": enriched,
        "cell_period_claim_emitted": False,
        "calculator_eligible": binding_status == "strict_cell_aware",
    }


def _benchmark(views: list[dict[str, object]], limit_per_document: int) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    signatures = Counter()
    for view in views:
        if not view["table_period_tokens"] or not view["statement_or_section"]:
            continue
        signature = (
            view["document_id"],
            normalize_text(str(view["metric"])).casefold(),
            tuple(view["table_period_tokens"]),
            normalize_text(str(view["statement_or_section"])).casefold(),
        )
        signatures[signature] += 1
        groups[str(view["document_id"])].append({"view": view, "signature": signature})
    cases = []
    for document_id, items in sorted(groups.items()):
        unique = [item["view"] for item in sorted(items, key=lambda item: str(item["view"]["candidate_key"])) if signatures[item["signature"]] == 1]
        for view in unique[:limit_per_document]:
            periods = " ".join(view["table_period_tokens"])
            query = normalize_text(f"{_issuer(document_id)} {view['metric']} {periods} {view['statement_or_section']}")
            cases.append({"case_id": stable_identity("pdf-v2-lite-query", view["candidate_key"], query), "document_id": document_id, "query": query, "gold_candidate_key": view["candidate_key"], "annotation_mode": "deterministic_unique_signature", "expected_value_used": False})
    return cases


def _evaluate(cases: list[dict[str, object]], views: list[dict[str, object]], field: str) -> dict[str, object]:
    documents = [str(view[field]) for view in views]
    keys = [str(view["candidate_key"]) for view in views]
    ranks = []
    hits = {5: 0, 40: 0, 200: 0}
    hit_keys: dict[int, list[str]] = {5: [], 40: [], 200: []}
    for case in cases:
        ranked = _bm25(str(case["query"]), documents)
        rank = next((position + 1 for position, index in enumerate(ranked) if keys[index] == case["gold_candidate_key"]), None)
        ranks.append(rank)
        for cutoff in hits:
            if rank is not None and rank <= cutoff:
                hits[cutoff] += 1
                hit_keys[cutoff].append(str(case["gold_candidate_key"]))
    return {
        "case_count": len(cases),
        "recall_at_5": hits[5] / len(cases) if cases else 0,
        "recall_at_40": hits[40] / len(cases) if cases else 0,
        "recall_at_200": hits[200] / len(cases) if cases else 0,
        "hit_count_at_5": hits[5],
        "hit_count_at_40": hits[40],
        "hit_count_at_200": hits[200],
        "mrr": sum(1 / rank for rank in ranks if rank) / len(cases) if cases else 0,
        "hit_keys": hit_keys,
    }


def run(args: argparse.Namespace) -> int:
    import camelot
    import pymupdf

    r3_acceptance = R3_OUT / "pdf-sr-v2-gate-a-r3-acceptance.json"
    alignment_path = R2_OUT / "parser-logical-table-alignment.json"
    r3 = json.loads(r3_acceptance.read_text(encoding="utf-8"))
    if r3["decision"] != "pdf_sr_v2_period_structure_recovery_insufficient":
        raise RuntimeError("R3 must be closed before V2-Lite")
    selected_ids = {record["selected_table_id"] for record in json.loads(alignment_path.read_text(encoding="utf-8"))["records"] if record["selected_table_id"]}
    records = []
    for source in SOURCES:
        pdf = _download(source, args.runtime_dir, args.user_agent)
        with pymupdf.open(pdf) as document:
            for page_index in _candidate_pages(document):
                page = document[page_index]
                parser_tables = [("pymupdf", list(page.find_tables().tables))]
                stream = [item for item in camelot.read_pdf(str(pdf), pages=str(page_index + 1), flavor="stream", edge_tol=50, row_tol=10) if item.df.shape[0] >= 2 and item.df.shape[1] >= 2]
                parser_tables.append(("camelot_stream", [_camelot_adapter(item, page_height=float(page.rect.height)) for item in stream]))
                for parser_name, tables in parser_tables:
                    for table_index, table in enumerate(tables):
                        base = _table_record(source["document_id"], page_index + 1, page, table, parser_name, table_index)
                        if base["table_id"] in selected_ids:
                            records.append(_extended_record(source["document_id"], page_index + 1, page, table, parser_name, table_index))
    views = [view for record in records for row_index, row in enumerate(record["matrix"]) if (view := _view(record, row_index, row))]
    if len({view["candidate_key"] for view in views}) != len(views):
        raise RuntimeError("candidate identity collision")
    cases = _benchmark(views, args.limit_per_document)
    baseline = _evaluate(cases, views, "raw_row_text")
    enriched = _evaluate(cases, views, "enriched_retrieval_text")
    baseline_hits = {cutoff: set(values) for cutoff, values in baseline.pop("hit_keys").items()}
    enriched_hits = {cutoff: set(values) for cutoff, values in enriched.pop("hit_keys").items()}
    regression = {
        f"at_{cutoff}": {
            "new_hit_count": len(enriched_hits[cutoff] - baseline_hits[cutoff]),
            "regressed_hit_count": len(baseline_hits[cutoff] - enriched_hits[cutoff]),
        }
        for cutoff in (5, 40, 200)
    }
    runtime_views = args.runtime_dir / "pdf-v2-lite-retrieval-views.json"
    runtime_cases = args.runtime_dir / "pdf-v2-lite-development-benchmark.json"
    _write(runtime_views, {"views": views})
    _write(runtime_cases, {"cases": cases})
    field_coverage = {
        "view_count": len(views),
        "metric_count": sum(bool(view["metric"]) for view in views),
        "statement_or_section_count": sum(bool(view["statement_or_section"]) for view in views),
        "table_period_count": sum(bool(view["table_period_tokens"]) for view in views),
        "scale_count": sum(bool(view["scale"]) for view in views),
        "strict_cell_aware_count": sum(view["period_binding_status"] == "strict_cell_aware" for view in views),
        "retrieval_only_count": sum(view["period_binding_status"] == "table_level_only" for view in views),
        "cell_period_claim_count": sum(bool(view["cell_period_claim_emitted"]) for view in views),
        "candidate_identity_error_count": 0,
    }
    gate_passed = enriched["hit_count_at_40"] > baseline["hit_count_at_40"] and enriched["hit_count_at_200"] > baseline["hit_count_at_200"] and regression["at_40"]["regressed_hit_count"] == 0 and regression["at_200"]["regressed_hit_count"] == 0
    acceptance = {
        "schema": "pdf-retrieval-representation-v2-lite/acceptance/v1",
        "r3_acceptance_sha256": _sha(r3_acceptance),
        "r2_alignment_sha256": _sha(alignment_path),
        "view_runtime_sha256": _sha(runtime_views),
        "benchmark_runtime_sha256": _sha(runtime_cases),
        "development_case_count": len(cases),
        "development_benchmark_semantics": "deterministic_unique_signature_row_retrieval_capability",
        "benchmark_queries_derived_from_view_fields": True,
        "independent_human_annotation": False,
        "sufficient_for_frozen_benchmark_transfer": False,
        "gate_b1_passed": field_coverage["candidate_identity_error_count"] == 0 and field_coverage["cell_period_claim_count"] == 0,
        "gate_b2_passed": gate_passed,
        "frozen_72_question_reads": 0,
        "gold_source_reads": 0,
        "expected_value_reads": 0,
        "candidate_construction_scope": "retrieval_view_only",
        "calculator_calls": 0,
        "embedding_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "decision": "pdf_v2_lite_bm25_shadow_gate_passed" if gate_passed else "pdf_v2_lite_bm25_gain_insufficient",
        "next_gate": "pdf_v2_lite_hybrid_shadow" if gate_passed else "stop_pdf_v2_lite",
    }
    _write(args.out_dir / "retrieval-view-contract.json", {"allowed_fields": ["metric", "statement_or_section", "table_period_tokens", "period_binding_status", "scale", "raw_row_text", "source_table_id"], "forbidden_claims": ["unverified_cell_period", "unverified_cell_value_semantics", "calculator_eligibility_for_table_level_only"], "production_candidate_identity_modified": False})
    _write(args.out_dir / "retrieval-view-manifest.json", {**field_coverage, "runtime_view_content_committed": False})
    _write(args.out_dir / "development-benchmark-manifest.json", {"case_count": len(cases), "document_counts": dict(Counter(case["document_id"] for case in cases)), "annotation_mode": "deterministic_unique_signature", "queries_derived_from_view_fields": True, "independent_human_annotation": False, "benchmark_semantics": "row_retrieval_capability_not_open_domain_pdf_recall", "benchmark_limit_per_document": args.limit_per_document, "runtime_case_content_committed": False})
    _write(args.out_dir / "bm25-shadow-results.json", {"baseline_raw_row": baseline, "header_enriched_row": enriched})
    _write(args.out_dir / "strict-hit-regression-report.json", regression)
    _write(args.out_dir / "next-gate.json", {"decision": acceptance["decision"], "next_gate": acceptance["next_gate"], "production_switch_allowed": False})
    _write(args.out_dir / "pdf-v2-lite-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit-per-document", type=int, default=40)
    parser.add_argument("--user-agent", default="nano-finance-research contact@example.com")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
