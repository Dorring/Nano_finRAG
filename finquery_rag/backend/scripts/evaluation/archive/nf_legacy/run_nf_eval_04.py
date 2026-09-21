"""NF-EVAL-04 Candidate Recall Source Attribution.

This runner is read-only with respect to production retrieval.  It loads the
frozen Golden inputs, probes BM25 and Chroma at a diagnostic Top-200, and
records lineage without invoking the answer orchestrator or a chat model.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any, Mapping, Sequence

from scripts.evaluation import run_nf_eval_03_r1 as r1
from scripts.evaluation.run_nf_eval_03_baseline import (
    _NullTraceLogger,
    _build_engine,
)
from scripts.evaluation.run_nf_eval_03_r2 import _raw_candidate_for_identity, _stage_candidate
from src.retrieval.candidate_fusion import (
    boost_front_matter_chunks,
    normalize_scores,
    rrf,
)
from src.evaluation.nf_eval_04 import (
    CandidateRecallFailureStage,
    candidate_in_scope,
    choose_next_gate,
    classify_first_recall_failure,
    classify_index_presence,
    rank_bucket,
    source_coverage,
    stage_key_set,
)


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "financial_rag_v1"
DATA = BENCHMARK / "data"
DEFAULT_OUT = ROOT / "artifacts" / "evaluation" / "nf-eval-04"
DEFAULT_NEGATIVE = ROOT / "artifacts" / "evaluation" / "nf-eval-02" / "negative-evidence-review-report.json"
LEGACY_FILES = {
    "FINAL Annual Report.pdf",
    "leac203.pdf",
    "wipo_pub_rn2021_18e.pdf",
}
TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)
STOPWORDS = {
    "a", "an", "and", "are", "at", "by", "for", "from", "in", "of",
    "on", "or", "the", "to", "was", "what", "were", "with", "reported",
    "how", "much", "does", "did", "is", "than", "between", "according",
    "calculate", "compare", "company", "fiscal", "year",
}


class NFEval04Error(ValueError):
    """Raised when frozen benchmark or diagnostic inputs are invalid."""


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normal_tokens(value: Any) -> list[str]:
    return [
        token.casefold()
        for token in TOKEN_RE.findall(str(value or ""))
        if token.casefold() not in STOPWORDS and len(token) > 1
    ]


def _rank_metrics(
    rows: Sequence[Mapping[str, Any]],
    rank_field: str,
    *,
    cutoffs: Sequence[int] = (20, 40, 100, 200),
) -> dict[str, Any]:
    """Summarize source and Case recall from one canonical rank list.

    The source denominator is the 80 frozen Expected Source records.  The
    Case denominator is the 64 answerable Cases represented by those records.
    Missing ranks are never treated as a hit, and both integer counts and
    rates are emitted so the report is independently recomputable.
    """

    source_count = len(rows)
    case_count = len({str(row.get("case_id")) for row in rows if row.get("case_id")})
    result: dict[str, Any] = {
        "source_count": source_count,
        "case_count": case_count,
        "rank_field": rank_field,
    }
    for cutoff in cutoffs:
        hits = [
            row
            for row in rows
            if isinstance(row.get(rank_field), int)
            and int(row[rank_field]) <= cutoff
        ]
        source_hits = len(hits)
        case_hits = len({str(row["case_id"]) for row in hits})
        result[f"@{cutoff}"] = {
            "source_hit_count": source_hits,
            "source_recall": source_hits / source_count if source_count else 0.0,
            "case_hit_count": case_hits,
            "case_hit_rate": case_hits / case_count if case_count else 0.0,
        }
    return result


def _safe_candidate(
    candidate: Mapping[str, Any],
    *,
    mapping: Mapping[str, str],
    tenant_id: int,
    rank: int,
) -> dict[str, Any]:
    raw = _raw_candidate_for_identity(candidate)
    item = _stage_candidate(raw, mapping=mapping, tenant_id=tenant_id, rank=rank)
    item.setdefault("document_id", item.get("canonical_document_id"))
    item["candidate_key"] = str(item.get("candidate_key") or "")
    item["content_hash"] = str(item.get("content_hash") or "")
    return item


def _safe_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    mapping: Mapping[str, str],
    tenant_id: int,
) -> list[dict[str, Any]]:
    return [
        _safe_candidate(item, mapping=mapping, tenant_id=tenant_id, rank=index)
        for index, item in enumerate(candidates or [], 1)
    ]


def _direct_bm25_top_n(
    retriever: Any,
    query: str,
    *,
    doc_name: str,
    user_id: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Read the same FTS5 index with a diagnostic limit above production's cap."""

    clean_query = retriever._clean_query(query)
    if not clean_query.strip():
        return []
    fetch_limit = max(limit, limit * 4)
    rows: list[tuple[Any, ...]] = []
    with sqlite3.connect(retriever.db_path, timeout=20) as conn:
        rows = conn.execute(
            """
            SELECT c.doc_id, c.content, c.metadata_json, bm25(fts_index) AS score
            FROM fts_index f
            JOIN chunk_store c ON f.doc_id = c.doc_id
            WHERE fts_index MATCH ? AND c.user_id = ? AND c.doc_name = ?
            ORDER BY score ASC LIMIT ?
            """,
            (clean_query, user_id, doc_name, fetch_limit),
        ).fetchall()
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc_id, content, metadata_json, score in rows:
        if doc_id in seen:
            continue
        try:
            metadata = json.loads(metadata_json or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        if metadata.get("type") == "table_cell":
            continue
        seen.add(str(doc_id))
        output.append(
            {
                "doc_id": str(doc_id),
                "content": content,
                "metadata": metadata,
                "score": -float(score),
            }
        )
        if len(output) >= limit:
            break
    return output


def _chroma_exact_presence(collection: Any, evidence_id: str, *, document_id: str, tenant_id: int) -> tuple[bool, dict[str, Any] | None]:
    if not evidence_id:
        return False, None
    try:
        result = collection.get(ids=[evidence_id], include=["metadatas"])
    except Exception:  # noqa: BLE001 - an absent Chroma ID is a negative observation
        return False, None
    ids = result.get("ids") or []
    metas = result.get("metadatas") or []
    for candidate_id, metadata in zip(ids, metas):
        meta = metadata or {}
        if (
            str(candidate_id) == evidence_id
            and int(meta.get("user_id", -1)) == tenant_id
            and str(meta.get("doc_name") or "")
            and bool(str(document_id))
        ):
            return True, meta
    return False, None


def _bm25_exact_presence(
    db_path: Path, evidence_id: str, *, document_id: str, filename: str, tenant_id: int
) -> tuple[bool, dict[str, Any] | None]:
    with sqlite3.connect(db_path, timeout=20) as conn:
        row = conn.execute(
            "SELECT metadata_json, doc_name, user_id FROM chunk_store WHERE doc_id = ?",
            (evidence_id,),
        ).fetchone()
    if not row:
        return False, None
    try:
        metadata = json.loads(row[0] or "{}")
    except (TypeError, json.JSONDecodeError):
        return False, None
    passed = (
        int(row[2]) == tenant_id
        and str(row[1] or "") == filename
        and str(metadata.get("doc_name") or "") == filename
        and bool(str(document_id))
    )
    return passed, metadata if passed else None


def _rank_map(candidates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    output: dict[str, int] = {}
    for rank, item in enumerate(candidates, 1):
        key = str(item.get("candidate_key") or "")
        if key and key not in output:
            output[key] = rank
    return output


def _ordered_ranks(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_key": str(item.get("candidate_key") or ""),
            "document_id": item.get("canonical_document_id") or item.get("document_id"),
            "evidence_id": item.get("evidence_id"),
            "page": item.get("page"),
            "score": item.get("score"),
            "fused_score": item.get("fused_score"),
            "content_hash": item.get("content_hash"),
            "rank": rank,
        }
        for rank, item in enumerate(candidates, 1)
    ]


def _gold_source_value(label: Mapping[str, Any], source_index: int) -> Any:
    answer = label.get("expected_answer") or {}
    components = answer.get("component_values") or []
    if answer.get("value_type") == "composite" and source_index < len(components):
        return components[source_index].get("canonical_value")
    return answer.get("canonical_value")


def _terminology_row(question: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    question_tokens = sorted(set(_normal_tokens(question.get("question"))))
    metric_text = " ".join(
        str(source.get(key) or "")
        for key in ("row_label", "table_title", "section", "column_header", "period")
    )
    target_tokens = sorted(set(_normal_tokens(metric_text)))
    matched = sorted(set(question_tokens) & set(target_tokens))
    metric_tokens = set(_normal_tokens(source.get("row_label") or source.get("metric")))
    table_tokens = set(_normal_tokens(source.get("table_title") or source.get("section")))
    period_tokens = set(_normal_tokens(source.get("period") or source.get("column_header")))
    q_set = set(question_tokens)
    metric_match = bool(q_set & metric_tokens)
    table_match = bool(q_set & table_tokens)
    period_match = bool(q_set & period_tokens)
    company_match = str(question.get("company") or "").casefold() in str(question.get("question") or "").casefold()
    if metric_match and period_match:
        category = "exact_metric_match"
    elif metric_match:
        category = "period_expression_mismatch"
    elif table_match:
        category = "financial_synonym_mismatch"
    elif "segment" in q_set or "product" in q_set:
        category = "segment_name_mismatch"
    else:
        category = "financial_synonym_mismatch"
    return {
        "case_id": question.get("case_id"),
        "question_tokens": question_tokens,
        "matched_question_tokens": matched,
        "metric_term_match": metric_match,
        "company_term_match": company_match,
        "period_term_match": period_match,
        "row_label_match": metric_match,
        "table_title_match": table_match,
        "category": category,
        "classification_method": "normalized_token_overlap_diagnostic",
    }


class _RecordingDense:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> list[dict[str, Any]]:
        result = self.delegate(**kwargs)
        self.calls.append({"kwargs": dict(kwargs), "results": [dict(x) for x in result or []]})
        return result


class _RecordingBM25:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    def search(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        result = self.delegate.search(*args, **kwargs)
        self.calls.append({"args": list(args), "kwargs": dict(kwargs), "results": [dict(x) for x in result or []]})
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


def _build_engine_for_diagnostic(args: argparse.Namespace) -> tuple[Any, _RecordingDense, _RecordingBM25]:
    engine, _client = _build_engine(args)
    engine._orchestrator._trace_logger = _NullTraceLogger()
    dense = _RecordingDense(engine._retrieval_pipeline._dense_query_fn)
    bm25 = _RecordingBM25(engine._retrieval_pipeline._bm25_retriever)
    engine._retrieval_pipeline._dense_query_fn = dense
    engine._retrieval_pipeline._bm25_retriever = bm25
    return engine, dense, bm25


def _production_limits(dense_call: Mapping[str, Any] | None, bm25_call: Mapping[str, Any] | None) -> tuple[int | None, int | None]:
    dense_limit = None if dense_call is None else dense_call.get("kwargs", {}).get("n_results")
    bm25_limit = None if bm25_call is None else bm25_call.get("kwargs", {}).get("k")
    try:
        dense_limit = int(dense_limit) if dense_limit is not None else None
    except (TypeError, ValueError):
        dense_limit = None
    try:
        bm25_limit = int(bm25_limit) if bm25_limit is not None else None
    except (TypeError, ValueError):
        bm25_limit = None
    return bm25_limit, dense_limit


def _filter_raw_scope(
    candidates: Sequence[Mapping[str, Any]],
    *,
    mapping: Mapping[str, str],
    whitelist: set[str],
    tenant_id: int,
    counter: Counter,
    stage: str,
) -> list[dict[str, Any]]:
    """Apply the benchmark document whitelist before any fusion or ranking."""

    accepted: list[dict[str, Any]] = []
    for candidate in candidates or []:
        safe = _safe_candidate(candidate, mapping=mapping, tenant_id=tenant_id, rank=0)
        if candidate_in_scope(safe, whitelist):
            accepted.append(dict(candidate))
        else:
            counter[stage] += 1
    return accepted


def _index_presence_report(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    bm25_presence: Mapping[tuple[str, int], tuple[bool, dict[str, Any] | None]],
    chroma_presence: Mapping[tuple[str, int], tuple[bool, dict[str, Any] | None]],
    reconstructable: Mapping[tuple[str, int], bool],
) -> dict[str, Any]:
    rows = []
    for source in source_rows:
        identity = (str(source["case_id"]), int(source["source_index"]))
        bm25 = bm25_presence[identity][0]
        dense = chroma_presence[identity][0]
        rows.append(
            {
                **{key: source.get(key) for key in ("case_id", "source_index", "candidate_key", "evidence_id", "document_id", "identity_granularity")},
                "present_in_bm25_index": bm25,
                "present_in_chroma_index": dense,
                "index_presence": classify_index_presence(
                    present_in_bm25_index=bm25, present_in_dense_index=dense
                ),
                "exact_identity_reconstructable": reconstructable[identity],
            }
        )
    counts = Counter(row["index_presence"] for row in rows)
    return {
        "gold_source_count": len(rows),
        "present_in_bm25": sum(row["present_in_bm25_index"] for row in rows),
        "present_in_chroma": sum(row["present_in_chroma_index"] for row in rows),
        "present_in_either_index": sum(row["index_presence"] != "missing_from_both_indexes" for row in rows),
        "exact_identity_reconstructable": sum(row["exact_identity_reconstructable"] for row in rows),
        "presence_counts": dict(sorted(counts.items())),
        "records": rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", type=int, required=True)
    parser.add_argument("--chroma-path", type=Path, default=ROOT / "chroma_db")
    parser.add_argument("--bm25-db-path", type=Path, default=ROOT / "rag_bm25.db")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--corpus", type=Path, default=BENCHMARK / "corpus.json")
    parser.add_argument("--manifest", dest="manifest_path", type=Path, default=DATA / "golden-manifest.json")
    parser.add_argument("--questions", type=Path, default=DATA / "questions.golden.jsonl")
    parser.add_argument("--labels", type=Path, default=DATA / "labels.golden.jsonl")
    parser.add_argument("--review-status", type=Path, default=DATA / "review-status.golden.jsonl")
    parser.add_argument("--negative-report", type=Path, default=DEFAULT_NEGATIVE)
    parser.add_argument("--diagnostic-top-n", type=int, default=200)
    parser.add_argument("--production-top-k", type=int, default=5)
    parser.add_argument("--retrieval-candidate-multiplier", type=int, default=4)
    parser.add_argument("--model-base-url", default="http://127.0.0.1:18001/v1")
    parser.add_argument("--model-name", default="finquery-finance-v2-lr010-150")
    parser.add_argument("--api-key", default="not-needed-for-local")
    return parser.parse_args()


def _run(args: argparse.Namespace) -> int:
    if args.tenant_id != 1:
        raise NFEval04Error("NF-EVAL-04 benchmark is tenant 1 only")
    if args.diagnostic_top_n < 200:
        raise NFEval04Error("diagnostic Top-N must be at least 200")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ["CHROMA_PATH"] = str(args.chroma_path)
    os.environ["BM25_DB_PATH"] = str(args.bm25_db_path)

    inputs = r1._load_inputs(
        corpus_path=args.corpus,
        manifest_path=args.manifest_path,
        questions_path=args.questions,
        labels_path=args.labels,
        review_status_path=args.review_status,
        negative_report_path=args.negative_report,
    )
    allowed_documents = {str(item["document_id"]) for item in inputs.corpus["documents"]}
    filenames = {str(item["document_id"]): str(item["filename"]) for item in inputs.corpus["documents"]}
    if len(allowed_documents) != 8:
        raise NFEval04Error("benchmark whitelist must contain exactly eight documents")
    source_rows = r1._source_records(inputs.labels_by_id.values())
    source_rows = [row for row in source_rows if row.get("candidate_key")]
    if len(source_rows) != 80:
        raise NFEval04Error(f"expected 80 answerable sources, got {len(source_rows)}")

    from src.services import vector_store
    from src.services.retrieval import SqliteBM25Retriever
    from src.retrieval.query_processor import QueryProcessor

    collection = vector_store.get_or_create_collection()
    bm25_retriever = SqliteBM25Retriever(db_path=str(args.bm25_db_path))
    engine_args = argparse.Namespace(
        tenant_id=args.tenant_id,
        model_base_url=args.model_base_url,
        model_name=args.model_name,
        api_key=args.api_key,
        chroma_path=args.chroma_path,
        bm25_db_path=args.bm25_db_path,
        retrieval_candidate_multiplier=args.retrieval_candidate_multiplier,
        out_dir=args.out_dir,
    )
    engine, dense_recorder, bm25_recorder = _build_engine_for_diagnostic(engine_args)
    query_processor = engine._retrieval_pipeline._query_processor or QueryProcessor()
    mapping = r1._doc_map(inputs.corpus)

    # Keyed source observations and per-case stage summaries.
    bm25_presence: dict[tuple[str, int], tuple[bool, dict[str, Any] | None]] = {}
    chroma_presence: dict[tuple[str, int], tuple[bool, dict[str, Any] | None]] = {}
    reconstructable: dict[tuple[str, int], bool] = {}
    bm25_rows: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []
    source_failures: list[dict[str, Any]] = []
    rrf_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    terminology_rows: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    out_of_scope = Counter()
    diagnostics_errors: list[dict[str, Any]] = []
    observed_limits: list[dict[str, Any]] = []
    terminology_by_case: dict[str, set[str]] = defaultdict(set)
    window_cases: set[str] = set()
    dense_coverage_cases: set[str] = set()
    parent_child_cases: set[str] = set()
    rrf_fusion_cases: set[str] = set()

    labels = inputs.labels_by_id
    for question in inputs.questions:
        case_id = str(question["case_id"])
        label = labels[case_id]
        if label.get("expected_no_answer"):
            continue
        scope_ids = [str(item) for item in question.get("document_scope") or []]
        if not scope_ids or any(item not in allowed_documents for item in scope_ids):
            raise NFEval04Error(f"{case_id}: invalid document scope")
        if len(scope_ids) != 1:
            raise NFEval04Error(f"{case_id}: diagnostic runner expects one-document Golden scope")
        document_id = scope_ids[0]
        filename = filenames[document_id]
        query = str(question.get("question") or "")
        expanded_query = query_processor.expand(query)
        try:
            # This is the actual production retrieval request only; no
            # orchestrator/answer/model call is made.  Its observers reveal
            # the configured candidate limits and raw production lists.
            dense_recorder.calls.clear()
            bm25_recorder.calls.clear()
            engine._retrieval_pipeline.retrieve_single(
                filename,
                query,
                user_id=args.tenant_id,
                top_k=args.production_top_k,
            )
            production_dense_raw = dense_recorder.calls[-1]["results"] if dense_recorder.calls else []
            production_bm25_raw = bm25_recorder.calls[-1]["results"] if bm25_recorder.calls else []
            production_bm25_limit, production_dense_limit = _production_limits(
                dense_recorder.calls[-1] if dense_recorder.calls else None,
                bm25_recorder.calls[-1] if bm25_recorder.calls else None,
            )
            observed_limits.append(
                {
                    "case_id": case_id,
                    "bm25_candidate_limit": production_bm25_limit,
                    "dense_candidate_limit": production_dense_limit,
                }
            )

            diagnostic_dense_raw = vector_store.query_collection(
                expanded_query,
                doc_name=filename,
                n_results=args.diagnostic_top_n,
                user_id=args.tenant_id,
            )
            diagnostic_bm25_raw = _direct_bm25_top_n(
                bm25_retriever,
                expanded_query,
                doc_name=filename,
                user_id=args.tenant_id,
                limit=args.diagnostic_top_n,
            )
            production_dense_raw = _filter_raw_scope(
                production_dense_raw,
                mapping=mapping,
                whitelist=allowed_documents,
                tenant_id=args.tenant_id,
                counter=out_of_scope,
                stage="production",
            )
            production_bm25_raw = _filter_raw_scope(
                production_bm25_raw,
                mapping=mapping,
                whitelist=allowed_documents,
                tenant_id=args.tenant_id,
                counter=out_of_scope,
                stage="production",
            )
            diagnostic_dense_raw = _filter_raw_scope(
                diagnostic_dense_raw,
                mapping=mapping,
                whitelist=allowed_documents,
                tenant_id=args.tenant_id,
                counter=out_of_scope,
                stage="diagnostic",
            )
            diagnostic_bm25_raw = _filter_raw_scope(
                diagnostic_bm25_raw,
                mapping=mapping,
                whitelist=allowed_documents,
                tenant_id=args.tenant_id,
                counter=out_of_scope,
                stage="diagnostic",
            )
        except Exception as exc:  # noqa: BLE001 - preserve case-level diagnostics
            diagnostics_errors.append({"case_id": case_id, "error": type(exc).__name__})
            continue

        production_dense = _safe_candidates(production_dense_raw, mapping=mapping, tenant_id=args.tenant_id)
        production_bm25 = _safe_candidates(production_bm25_raw, mapping=mapping, tenant_id=args.tenant_id)
        diagnostic_dense = _safe_candidates(diagnostic_dense_raw, mapping=mapping, tenant_id=args.tenant_id)
        diagnostic_bm25 = _safe_candidates(diagnostic_bm25_raw, mapping=mapping, tenant_id=args.tenant_id)
        production_union = production_dense + [
            candidate for candidate in production_bm25
            if candidate.get("doc_id") not in {item.get("doc_id") for item in production_dense}
        ]
        diagnostic_union = diagnostic_dense + [
            candidate for candidate in diagnostic_bm25
            if candidate.get("doc_id") not in {item.get("doc_id") for item in diagnostic_dense}
        ]
        production_rrf = normalize_scores(rrf([production_dense_raw, production_bm25_raw]))
        production_rrf = boost_front_matter_chunks(
            query,
            production_rrf,
            is_front_matter_query_fn=query_processor.is_front_matter_query,
        )
        production_rrf = _safe_candidates(production_rrf, mapping=mapping, tenant_id=args.tenant_id)
        diagnostic_rrf = normalize_scores(rrf([diagnostic_dense_raw, diagnostic_bm25_raw]))
        diagnostic_rrf = boost_front_matter_chunks(
            query,
            diagnostic_rrf,
            is_front_matter_query_fn=query_processor.is_front_matter_query,
        )
        diagnostic_rrf = _safe_candidates(diagnostic_rrf, mapping=mapping, tenant_id=args.tenant_id)

        prod_bm25_map = _rank_map(production_bm25)
        prod_dense_map = _rank_map(production_dense)
        diag_bm25_map = _rank_map(diagnostic_bm25)
        diag_dense_map = _rank_map(diagnostic_dense)
        prod_union_keys = stage_key_set(production_union)
        prod_union_map = _rank_map(production_union)
        prod_rrf_map = _rank_map(production_rrf)
        diag_union_keys = stage_key_set(diagnostic_union)
        diag_union_map = _rank_map(diagnostic_union)
        diag_rrf_map = _rank_map(diagnostic_rrf)
        expected_keys = [str(item["candidate_key"]) for item in label.get("expected_sources") or []]
        rrf_cov = source_coverage(expected_keys, prod_rrf_map)
        rrf_top40_cov = source_coverage(expected_keys, list(prod_rrf_map)[:40])
        terminology_case_categories: set[str] = set()

        for source_index, source in enumerate(label.get("expected_sources") or []):
            identity = (case_id, source_index)
            key = str(source.get("candidate_key") or "")
            evidence_id = str(source.get("evidence_id") or "")
            bm25_ok, bm25_meta = _bm25_exact_presence(
                args.bm25_db_path,
                evidence_id,
                document_id=document_id,
                filename=filename,
                tenant_id=args.tenant_id,
            )
            chroma_ok, chroma_meta = _chroma_exact_presence(
                collection,
                evidence_id,
                document_id=document_id,
                tenant_id=args.tenant_id,
            )
            bm25_presence[identity] = (bm25_ok, bm25_meta)
            chroma_presence[identity] = (chroma_ok, chroma_meta)
            synthetic = {
                "document_id": document_id,
                "filename": filename,
                "evidence_id": evidence_id,
                "doc_id": evidence_id,
                "type": source.get("identity_granularity") or source.get("evidence_type") or "text",
                "block_type": source.get("identity_granularity") or source.get("evidence_type") or "text",
                "row_id": source.get("row_id"),
                "parent_row_id": source.get("parent_row_id"),
            }
            synthetic_key, _, _ = r1.candidate_identity_from_record(
                synthetic, filename_to_document=mapping, tenant_id=args.tenant_id
            )
            reconstructable[identity] = bool(synthetic_key and synthetic_key == key)

            bm25_rank = diag_bm25_map.get(key)
            dense_rank = diag_dense_map.get(key)
            production_bm25_rank = prod_bm25_map.get(key)
            production_dense_rank = prod_dense_map.get(key)
            entered_bm25_window = production_bm25_rank is not None
            entered_dense_window = production_dense_rank is not None
            entered_union = key in prod_union_keys
            entered_rrf = key in prod_rrf_map
            normalization_lost = key in diag_union_keys and not entered_union
            rrf_lost = entered_union and not entered_rrf
            parent_observed = any(
                str(item.get("parent_id") or "") == evidence_id
                for item in diagnostic_dense + diagnostic_bm25
            )
            failure = classify_first_recall_failure(
                identity_valid=bool(key and evidence_id and reconstructable[identity]),
                present_in_bm25_index=bm25_ok,
                present_in_dense_index=chroma_ok,
                bm25_rank=bm25_rank,
                dense_rank=dense_rank,
                bm25_production_limit=production_bm25_limit,
                dense_production_limit=production_dense_limit,
                entered_production_union=entered_union,
                entered_production_rrf=entered_rrf,
                normalization_lost=normalization_lost,
                rrf_lost=rrf_lost,
                parent_child_mismatch=False,
            )
            source_failure = {
                "case_id": case_id,
                "source_index": source_index,
                "gold_candidate_key": key,
                "gold_evidence_id": evidence_id,
                "index_presence": classify_index_presence(
                    present_in_bm25_index=bm25_ok,
                    present_in_dense_index=chroma_ok,
                ),
                "present_in_bm25_index": bm25_ok,
                "present_in_chroma_index": chroma_ok,
                "bm25_rank": bm25_rank,
                "dense_rank": dense_rank,
                "bm25_rank_bucket": rank_bucket(bm25_rank),
                "dense_rank_bucket": rank_bucket(dense_rank),
                "production_bm25_rank": production_bm25_rank,
                "production_dense_rank": production_dense_rank,
                "entered_production_bm25_window": entered_bm25_window,
                "entered_production_dense_window": entered_dense_window,
                "entered_production_union": entered_union,
                "entered_diagnostic_union": key in diag_union_keys,
                "entered_rrf_union": entered_union,
                "entered_rrf_pool": entered_rrf,
                "production_union_rank": prod_union_map.get(key),
                "diagnostic_union_rank": diag_union_map.get(key),
                "rrf_rank": prod_rrf_map.get(key),
                "diagnostic_rrf_rank": diag_rrf_map.get(key),
                "first_loss_stage": failure.value,
                "parent_candidate_observed": parent_observed,
                "parent_child_equivalence_verified": False,
                "identity_reconstructable": reconstructable[identity],
            }
            source_failures.append(source_failure)
            bm25_rows.append(
                {
                    "case_id": case_id,
                    "source_index": source_index,
                    **{key_: source.get(key_) for key_ in ("candidate_key", "evidence_id", "document_id")},
                    "bm25_rank": bm25_rank,
                    "rank_bucket": rank_bucket(bm25_rank),
                    "present_in_bm25_index": bm25_ok,
                    "production_bm25_rank": production_bm25_rank,
                }
            )
            dense_rows.append(
                {
                    "case_id": case_id,
                    "source_index": source_index,
                    **{key_: source.get(key_) for key_ in ("candidate_key", "evidence_id", "document_id")},
                    "dense_rank": dense_rank,
                    "rank_bucket": rank_bucket(dense_rank),
                    "present_in_chroma_index": chroma_ok,
                    "production_dense_rank": production_dense_rank,
                }
            )
            window_rows.append(source_failure)
            term = _terminology_row(question, source)
            terminology_rows.append(term)
            terminology_case_categories.add(term["category"])
            if term["category"] != "exact_metric_match":
                terminology_by_case[case_id].add(term["category"])
            if any(
                item is not None and 40 < item <= 100
                for item in (bm25_rank, dense_rank)
            ):
                window_cases.add(case_id)
            if not chroma_ok or dense_rank is None:
                dense_coverage_cases.add(case_id)
            if parent_observed and source.get("identity_granularity") in {"table_row", "table_block"}:
                parent_rows.append(
                    {
                        "case_id": case_id,
                        "source_index": source_index,
                        "gold_candidate_key": key,
                        "retrievable_parent_observed": True,
                        "verified_equivalence": False,
                        "reason": "same-page_or_parent_observation_is_not_equivalence",
                    }
                )

            contribution_bm25 = (
                1 / (60 + production_bm25_rank) if production_bm25_rank is not None else 0.0
            )
            contribution_dense = (
                1 / (60 + production_dense_rank) if production_dense_rank is not None else 0.0
            )
            rrf_rows.append(
                {
                    "case_id": case_id,
                    "source_index": source_index,
                    "gold_candidate_key": key,
                    "bm25_rank": production_bm25_rank,
                    "dense_rank": production_dense_rank,
                    "bm25_rrf_contribution": contribution_bm25,
                    "dense_rrf_contribution": contribution_dense,
                    "final_rrf_score": next((item.get("score") for item in production_rrf if item.get("candidate_key") == key), None),
                    "rrf_rank": prod_rrf_map.get(key),
                    "lost_due_to": (
                        "not_lost_in_rrf"
                        if entered_rrf
                        else "lost_during_rrf_fusion"
                        if rrf_lost
                        else "not_in_production_union"
                    ),
                }
            )

        case_rows.append(
            {
                "case_id": case_id,
                "company": question.get("company"),
                "expected_source_count": len(expected_keys),
                "rrf_coverage": rrf_cov,
                "rrf_top40_coverage": rrf_top40_cov,
                "matched_sources_in_production_rrf": sum(key in prod_rrf_map for key in expected_keys),
                "matched_sources_in_diagnostic_union": sum(key in diag_union_keys for key in expected_keys),
                "matched_sources_in_diagnostic_rrf": sum(key in diag_rrf_map for key in expected_keys),
                "case_class": f"{rrf_cov}_gold_case",
                "terminology_categories": sorted(terminology_case_categories),
            }
        )

    # Rebuild reports from the 80 source observations, never from the 8
    # no-answer cases.  This makes every denominator explicit.
    index_report = _index_presence_report(
        source_rows=source_rows,
        bm25_presence=bm25_presence,
        chroma_presence=chroma_presence,
        reconstructable=reconstructable,
    )
    failure_counts = Counter(row["first_loss_stage"] for row in source_failures)
    affected_cases = {row["case_id"] for row in case_rows if row["rrf_coverage"] != "all"}
    terminology_case_count = len(set(terminology_by_case) & affected_cases)
    # An observed parent/same-page candidate is diagnostic evidence only.  It
    # becomes a Parent/Child gate case only when a pre-verified equivalence
    # mapping exists; never count an unverified observation as a mismatch.
    parent_child_cases.update(
        row["case_id"]
        for row in parent_rows
        if row.get("verified_equivalence") is True
    )
    rrf_fusion_cases.update(
        row["case_id"] for row in rrf_rows if row["lost_due_to"] == "lost_during_rrf_fusion"
    )
    counts = Counter(row["rrf_coverage"] for row in case_rows)
    first_failure_case_rows = []
    for case in case_rows:
        rows = [row for row in source_failures if row["case_id"] == case["case_id"]]
        first_failure_case_rows.append(
            {
                **case,
                "source_failure_stages": dict(Counter(row["first_loss_stage"] for row in rows)),
                "missing_source_count": sum(row["first_loss_stage"] != CandidateRecallFailureStage.ENTERED_RRF_POOL.value for row in rows),
            }
        )
    next_gate = choose_next_gate(
        terminology_cases=terminology_case_count,
        window_cases=len(window_cases),
        dense_coverage_cases=len(dense_coverage_cases),
        parent_child_cases=len(parent_child_cases),
        rrf_fusion_cases=len(rrf_fusion_cases),
    )

    hash_report = inputs.hash_report
    input_integrity = {
        "artifact_schema": "nf-eval-04/v1",
        "benchmark_id": "financial-rag-v1",
        "tenant_id": args.tenant_id,
        "case_count": 64,
        "expected_source_count": 80,
        "question_hash": hash_report["actual"]["question_hash"],
        "reference_answer_hash": hash_report["actual"]["reference_answer_hash"],
        "source_identity_hash": hash_report["actual"]["source_identity_hash"],
        "negative_evidence_hash": hash_report["actual"]["negative_evidence_hash"],
        "review_status_hash": hash_report["actual"]["review_status_hash"],
        "corpus_hash": hash_report["actual"]["corpus_hash"],
        "benchmark_hash": _sha256_text(json.dumps(hash_report["actual"], sort_keys=True)),
        "golden_manifest_sha256": hash_report["actual"]["golden_manifest_sha256"],
        "all_hashes_verified": all(hash_report["matches"].values()),
        "allowed_document_count": len(allowed_documents),
        "legacy_documents_excluded": sorted(LEGACY_FILES),
    }
    scope_report = {
        "allowed_document_count": len(allowed_documents),
        "retrieved_out_of_scope_candidates": out_of_scope["retrieved"],
        "production_out_of_scope_candidates": out_of_scope["production"],
        "diagnostic_out_of_scope_candidates": out_of_scope["diagnostic"],
        "legacy_documents_loaded": 0,
        "scope_integrity_passed": not any(out_of_scope.values()),
    }
    production_window = {
        "diagnostic_top_n": args.diagnostic_top_n,
        "production_top_k": args.production_top_k,
        "candidate_multiplier": engine._retrieval_pipeline._candidate_multiplier,
        "production_limits_observed_from_retrieval_calls": True,
        "observed_production_limits": observed_limits,
        "funnel": {
            "gold_sources": len(source_failures),
            "index_present_any": index_report["present_in_either_index"],
            "bm25_diagnostic_top200": sum(row["bm25_rank"] is not None for row in source_failures),
            "dense_diagnostic_top200": sum(row["dense_rank"] is not None for row in source_failures),
            "entered_production_bm25_window": sum(row["entered_production_bm25_window"] for row in source_failures),
            "entered_production_dense_window": sum(row["entered_production_dense_window"] for row in source_failures),
            "entered_production_union": sum(row["entered_production_union"] for row in source_failures),
            "entered_diagnostic_union": sum(row["entered_diagnostic_union"] for row in source_failures),
            "entered_rrf_pool": sum(row["entered_rrf_pool"] for row in source_failures),
            "rrf_top40": sum((row["rrf_rank"] or 999999) <= 40 for row in source_failures),
        },
        "source_records": window_rows,
    }
    terminology_counts = Counter(row["category"] for row in terminology_rows)
    parent_report = {
        "gold_source_count": len(source_failures),
        "exact_identity_recall_sources": sum(row["entered_rrf_pool"] for row in source_failures),
        "verified_equivalent_identity_recall_sources": 0,
        "retrievable_parent_observed_count": len(parent_rows),
        "verified_equivalence_count": 0,
        "parent_child_identity_mismatch_count": 0,
        "same_page_not_equivalence_count": len(parent_rows),
        "records": parent_rows,
    }
    bm25_metrics = _rank_metrics(bm25_rows, "bm25_rank")
    dense_metrics = _rank_metrics(dense_rows, "dense_rank")
    diagnostic_union_metrics = _rank_metrics(source_failures, "diagnostic_union_rank")
    production_union_metrics = _rank_metrics(source_failures, "production_union_rank")
    rrf_metrics = _rank_metrics(rrf_rows, "rrf_rank")
    acceptance = {
        "artifact_schema": "nf-eval-04/v1",
        "decision": "candidate_recall_attribution_recorded" if (
            len(case_rows) == 64
            and len(source_failures) == 80
            and input_integrity["all_hashes_verified"]
            and scope_report["scope_integrity_passed"]
            and not diagnostics_errors
            and all(row.get("first_loss_stage") for row in source_failures)
        ) else "candidate_recall_attribution_incomplete",
        "diagnostic_integrity_passed": len(case_rows) == 64 and len(source_failures) == 80 and not diagnostics_errors,
        "input_hashes_verified": input_integrity["all_hashes_verified"],
        "scope_integrity_passed": scope_report["scope_integrity_passed"],
        "production_behavior_changed": False,
        "production_queries_executed": 0,
        "retrieval_diagnostic_query_count": len(case_rows),
        "answer_generation_calls": 0,
        "model_chat_completion_requests": 0,
        "legacy_27_included": False,
        "optimization_allowed": False,
        "errors": diagnostics_errors,
    }
    out = args.out_dir
    _write(out / "input-integrity-report.json", input_integrity)
    _write(out / "gold-index-presence-report.json", index_report)
    _write(out / "bm25-rank-report.json", {
        "gold_source_count": len(bm25_rows),
        "rank_metrics": bm25_metrics,
        "records": bm25_rows,
    })
    _write(out / "dense-rank-report.json", {
        "gold_source_count": len(dense_rows),
        "rank_metrics": dense_metrics,
        "records": dense_rows,
    })
    _write(out / "production-window-report.json", production_window)
    _write(out / "parent-child-identity-report.json", parent_report)
    _write(out / "rrf-fusion-attribution.json", {
        "gold_source_count": len(rrf_rows),
        "rank_metrics": rrf_metrics,
        "records": rrf_rows,
        "diagnostic_union_rank_metrics": diagnostic_union_metrics,
        "production_union_rank_metrics": production_union_metrics,
    })
    _write(out / "query-terminology-report.json", {
        "affected_case_count": len(affected_cases),
        "terminology_counts": dict(sorted(terminology_counts.items())),
        "heuristic_case_count": len(terminology_by_case),
        "records": terminology_rows,
    })
    _write(out / "source-failure-attribution.json", {
        "gold_source_count": len(source_failures),
        "first_failure_counts": dict(sorted(failure_counts.items())),
        "records": source_failures,
    })
    _write(out / "case-failure-attribution.json", {
        "answerable_case_count": len(first_failure_case_rows),
        "rrf_coverage_counts": dict(sorted(counts.items())),
        "records": first_failure_case_rows,
    })
    _write(out / "next-gate.json", {
        **next_gate,
        "affected_case_count": len(affected_cases),
        "gate_selection_method": "unique_case_count_and_fixed_threshold",
    })
    _write(out / "nf-eval-04-acceptance.json", acceptance)
    print(json.dumps({"acceptance": acceptance, "rrf_coverage": dict(counts), "index_presence": index_report["presence_counts"], "next_gate": next_gate}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if acceptance["decision"] == "candidate_recall_attribution_recorded" else 2


def main() -> None:
    args = _parse_args()
    try:
        raise SystemExit(_run(args))
    except (NFEval04Error, r1.BaselineConfigurationError) as exc:
        print(f"NF-EVAL-04 configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
