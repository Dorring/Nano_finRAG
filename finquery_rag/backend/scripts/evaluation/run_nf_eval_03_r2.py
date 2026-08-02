"""NF-EVAL-03 R2 candidate-lineage and conditional attribution runner.

The runner observes the unchanged production engine.  It does not alter
retrieval, ranking, answer generation, validation, or repair behavior.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import statistics
import sys
import time
import unicodedata
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation import run_nf_eval_03_r1 as r1
from scripts.evaluation.run_nf_eval_03_baseline import _build_engine
from src.domain.query import QueryRequest
from src.evaluation.nf40_pipeline_observer import AnswerPipelineTrace
from src.evaluation.nf_eval_03_r2 import (
    audit_stage,
    classify_final_coverage,
    first_failure_stage,
    infer_reranker_input_source,
    ordered_candidate_keys,
    validate_candidate_lineage,
)
from src.services.retrieval_config import (
    get_embedding_model_name,
    get_reranker_model,
    get_reranker_name,
)


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "financial_rag_v1"
DATA = BENCHMARK / "data"
DEFAULT_OUT = ROOT / "artifacts" / "evaluation" / "nf-eval-03-r2"
DEFAULT_NEGATIVE = (
    ROOT / "artifacts" / "evaluation" / "nf-eval-02" / "negative-evidence-review-report.json"
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _content_hash(content: Any) -> str:
    if not isinstance(content, str) or not content.strip():
        return ""
    value = unicodedata.normalize("NFC", content).replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _raw_candidate_for_identity(candidate: Mapping[str, Any]) -> dict[str, Any]:
    item = dict(candidate)
    metadata = dict(item.get("metadata") or {})
    item.setdefault("filename", metadata.get("doc_name"))
    item.setdefault("type", metadata.get("type"))
    item.setdefault("block_type", metadata.get("type"))
    item.setdefault("page", metadata.get("page"))
    item.setdefault("parent_id", metadata.get("parent_id"))
    item.setdefault("evidence_id", item.get("doc_id"))
    return item


def _stage_candidate(
    candidate: Mapping[str, Any],
    *,
    mapping: Mapping[str, str],
    tenant_id: int,
    rank: int,
) -> dict[str, Any]:
    raw = _raw_candidate_for_identity(candidate)
    item = r1._annotate(raw, mapping=mapping, tenant_id=tenant_id)
    item["content_hash"] = _content_hash(candidate.get("content"))
    item["parent_candidate_key"] = item.get("parent_id")
    item["stage_rank"] = rank
    # Keep only identity, provenance, scores, and hashes. Never write content.
    allowed = {
        "candidate_key",
        "canonical_document_id",
        "document_id",
        "filename",
        "evidence_id",
        "doc_id",
        "page",
        "type",
        "block_type",
        "parent_id",
        "parent_candidate_key",
        "content_hash",
        "stage_rank",
        "score",
        "fused_score",
        "rerank_score",
        "score_kind",
        "reranker",
    }
    return {key: value for key, value in item.items() if key in allowed}


def _stage_list(
    candidates: Sequence[Mapping[str, Any]],
    *,
    mapping: Mapping[str, str],
    tenant_id: int,
) -> list[dict[str, Any]]:
    return [
        _stage_candidate(candidate, mapping=mapping, tenant_id=tenant_id, rank=index)
        for index, candidate in enumerate(candidates, 1)
    ]


class RecordingReranker:
    """Transparent observer around the configured production reranker."""

    def __init__(self, delegate: Any):
        self._delegate = delegate
        self.name = getattr(delegate, "name", "unknown")
        self.calls: list[dict[str, Any]] = []

    def clear(self) -> None:
        self.calls.clear()

    def rerank(self, query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
        result = self._delegate.rerank(query, chunks, top_k=top_k)
        self.calls.append(
            {
                "top_k": top_k,
                "input": [dict(chunk) for chunk in chunks],
                "output": [dict(chunk) for chunk in result],
            }
        )
        return result


class RecordingDenseQuery:
    """Transparent observer for the injected dense query callable."""

    def __init__(self, delegate: Any):
        self._delegate = delegate
        self.calls: list[list[dict[str, Any]]] = []

    def clear(self) -> None:
        self.calls.clear()

    def __call__(self, **kwargs: Any) -> list[dict[str, Any]]:
        result = self._delegate(**kwargs)
        self.calls.append([dict(item) for item in result or []])
        return result


class RecordingBM25:
    """Transparent observer preserving every BM25 retriever method."""

    def __init__(self, delegate: Any):
        self._delegate = delegate
        self.calls: list[list[dict[str, Any]]] = []

    def clear(self) -> None:
        self.calls.clear()

    def search(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        result = self._delegate.search(*args, **kwargs)
        self.calls.append([dict(item) for item in result or []])
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def _select_calls(
    calls: Sequence[Mapping[str, Any]], *, final_top_k: int
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    if not calls:
        return None, None
    twenty = [call for call in calls if int(call.get("top_k") or 0) == 20]
    finals = [call for call in calls if int(call.get("top_k") or 0) == final_top_k]
    rerank_call = twenty[-1] if twenty else max(calls, key=lambda call: len(call.get("output") or []))
    final_call = finals[-1] if finals else calls[-1]
    return rerank_call, final_call


def _false_score() -> dict[str, bool]:
    return {
        key: False
        for key in (
            "value_correct",
            "currency_correct",
            "unit_correct",
            "scale_correct",
            "period_correct",
            "component_count_correct",
            "component_assignment_correct",
            "text_contract_correct",
            "answer_contract_correct",
        )
    }


def _citation_sources(
    sources: Sequence[Mapping[str, Any]],
    *,
    mapping: Mapping[str, str],
    tenant_id: int,
) -> list[dict[str, Any]]:
    output = []
    for source in sources or []:
        raw = dict(source)
        raw.update(
            {
                "document_id": source.get("filename") or source.get("document_id"),
                "evidence_id": source.get("chunk_id") or source.get("evidence_id"),
                "block_type": source.get("type") or source.get("block_type") or "text",
            }
        )
        key, document, evidence = r1.candidate_identity_from_record(
            raw, filename_to_document=mapping, tenant_id=tenant_id
        )
        output.append(
            {
                "candidate_key": key,
                "canonical_document_id": document,
                "evidence_id": evidence,
                "filename": source.get("filename"),
                "page": source.get("page"),
                "type": source.get("type"),
            }
        )
    return output


async def _run_queries(
    args: argparse.Namespace, inputs: r1.GoldenInputs
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    engine, client = _build_engine(args)
    recorder = RecordingReranker(engine.reranker)
    dense_recorder = RecordingDenseQuery(engine._retrieval_pipeline._dense_query_fn)
    bm25_recorder = RecordingBM25(engine._retrieval_pipeline._bm25_retriever)
    # Both references are replaced only in this isolated evaluator instance.
    engine.reranker = recorder
    engine._retrieval_pipeline._reranker = recorder
    engine._retrieval_pipeline._dense_query_fn = dense_recorder
    engine._retrieval_pipeline._bm25_retriever = bm25_recorder
    mapping = r1._doc_map(inputs.corpus)
    by_doc = {
        str(item["document_id"]): str(item["filename"])
        for item in inputs.corpus["documents"]
    }
    records: list[dict[str, Any]] = []
    for index, question in enumerate(inputs.questions, 1):
        case_id = str(question["case_id"])
        label = inputs.labels_by_id[case_id]
        recorder.clear()
        dense_recorder.clear()
        bm25_recorder.clear()
        trace = AnswerPipelineTrace(
            case_id=case_id,
            trace_id=uuid.uuid4().hex,
            context_hash="",
            context_coverage="not_evaluated",
        )
        request = QueryRequest(
            question=str(question["question"]),
            document_names=tuple(by_doc[str(value)] for value in question.get("document_scope", [])),
            user_id=args.tenant_id,
            conversation_history=(),
            memory_profile=None,
        )
        before = client.chat_completion_requests
        result: dict[str, Any] = {}
        answer_result = None
        error = None
        started = time.perf_counter()
        try:
            answer_result = await engine._orchestrator.answer(
                request, n_results=args.n_results, evaluation_observer=trace
            )
            result = answer_result.to_legacy_dict()
        except Exception as exc:  # noqa: BLE001 - preserve per-case failure
            error = type(exc).__name__
        latency = (time.perf_counter() - started) * 1000
        request_count = client.chat_completion_requests - before
        raw = trace._raw_generation_text or ""
        released = str(result.get("answer") or "")
        raw_available = bool(trace.raw_generation_hash)
        raw_score = r1.score_answer_contract(raw, question, label) if raw_available else _false_score()
        released_score = r1.score_answer_contract(released, question, label)
        debug = result.get("retrieval_debug") or {}
        old_stages = debug.get("candidate_stages") or {}
        rerank_call, final_call = _select_calls(recorder.calls, final_top_k=args.n_results)
        rrf_raw = (rerank_call or {}).get("input") or []
        input_raw = (rerank_call or {}).get("input") or []
        rerank_raw = (rerank_call or {}).get("output") or []
        final_raw = (final_call or {}).get("output") or []
        stages = {
            "dense": _stage_list(
                dense_recorder.calls[-1]
                if dense_recorder.calls
                else [
                    _raw_candidate_for_identity(item)
                    for item in old_stages.get("dense") or []
                ],
                mapping=mapping,
                tenant_id=args.tenant_id,
            ),
            "bm25": _stage_list(
                bm25_recorder.calls[-1]
                if bm25_recorder.calls
                else [
                    _raw_candidate_for_identity(item)
                    for item in old_stages.get("bm25") or []
                ],
                mapping=mapping,
                tenant_id=args.tenant_id,
            ),
            "rrf": _stage_list(rrf_raw, mapping=mapping, tenant_id=args.tenant_id),
            "reranker_input": _stage_list(input_raw, mapping=mapping, tenant_id=args.tenant_id),
            "reranker": _stage_list(rerank_raw, mapping=mapping, tenant_id=args.tenant_id),
            "final": _stage_list(final_raw, mapping=mapping, tenant_id=args.tenant_id),
        }
        input_source = infer_reranker_input_source(
            rrf_keys=ordered_candidate_keys(stages["rrf"]),
            input_keys=ordered_candidate_keys(stages["reranker_input"]),
            input_limit=20,
        )
        lineage = validate_candidate_lineage(
            rrf_candidates=stages["rrf"],
            reranker_input=stages["reranker_input"],
            reranker_output=stages["reranker"],
            final_candidates=stages["final"],
            reranker_input_source=input_source,
            reranker_input_limit=20,
        )
        expected = list(label.get("expected_sources") or [])
        citation_items = _citation_sources(
            result.get("sources") or [], mapping=mapping, tenant_id=args.tenant_id
        )
        citation = r1.citation_breakdown(expected, citation_items)
        coverage = classify_final_coverage(
            expected_sources=expected,
            final_candidates=stages["final"],
            source_matches=r1.source_identity_matches,
        )
        matched_final = sum(
            any(r1.source_identity_matches(source, candidate) for candidate in stages["final"])
            for source in expected
        )
        scope_ids = {str(value) for value in question.get("document_scope", [])}
        all_stage_values = [item for name in stages for item in stages[name]]
        out_of_scope = sum(
            int(item.get("canonical_document_id") not in scope_ids)
            for item in all_stage_values
            if item.get("canonical_document_id")
        )
        citation_out = sum(
            int(item.get("canonical_document_id") not in scope_ids)
            for item in citation_items
            if item.get("canonical_document_id")
        )
        mode = r1._mode(
            answer_result,
            result,
            request_count,
            bool(label.get("expected_no_answer")),
            error,
        )
        repair = r1._repair_flags(result)
        record: dict[str, Any] = {
            "case_id": case_id,
            "company": question.get("company"),
            "bucket": r1._bucket(question),
            "document_scope": list(question.get("document_scope") or []),
            "expected_no_answer": bool(label.get("expected_no_answer")),
            "expected_source_count": len(expected),
            "matched_expected_source_count": citation["matched_expected_source_count"],
            "emitted_citation_count": citation["emitted_citation_count"],
            "correct_emitted_citation_count": citation["correct_emitted_citation_count"],
            "raw_available": raw_available,
            "raw_generation_hash": trace.raw_generation_hash,
            "released_answer_hash": trace.released_answer_hash,
            "answer_execution_mode": mode,
            "model_invoked": bool(request_count),
            "model_request_count": request_count,
            "validation_status": (result.get("validation") or {}).get("status"),
            "context_coverage": coverage,
            "final_matched_gold_source_count": matched_final,
            "reranker_input_source": input_source,
            "reranker_call_count": len(recorder.calls),
            "reranker_input_call_top_k": (rerank_call or {}).get("top_k"),
            "stage_candidate_count": sum(len(values) for values in stages.values()),
            "missing_candidate_identity_count": sum(
                audit_stage(name, values)["missing_identity_count"]
                for name, values in stages.items()
            ),
            "lineage_integrity": lineage,
            "retrieval_stages": stages,
            "legacy_debug_rrf_top40": _stage_list(
                [
                    _raw_candidate_for_identity(item)
                    for item in old_stages.get("rrf") or []
                ],
                mapping=mapping,
                tenant_id=args.tenant_id,
            ),
            "sources": citation_items,
            "page_fallback_count": citation["page_fallback_count"],
            "scope_out_of_scope_count": out_of_scope,
            "citation_out_of_scope_count": citation_out,
            "latency_ms": latency,
            "error": error,
            **repair,
        }
        for prefix, score in (("raw", raw_score), ("released", released_score)):
            for key in (
                "value_correct",
                "currency_correct",
                "unit_correct",
                "scale_correct",
                "period_correct",
                "component_count_correct",
                "component_assignment_correct",
                "text_contract_correct",
                "answer_contract_correct",
            ):
                record[f"{prefix}_{key}"] = bool(score[key])
            available = prefix == "released" or raw_available
            record[f"{prefix}_citation_recall"] = citation["citation_recall"] if available else None
            record[f"{prefix}_citation_precision"] = citation["citation_precision"] if available else None
            record[f"{prefix}_citation_full_recall"] = bool(citation["citation_full_recall"]) if available else False
            record[f"{prefix}_citation_perfect_precision"] = bool(citation["citation_perfect_precision"]) if available else False
            record[f"{prefix}_grounded_pass"] = bool(
                score["answer_contract_correct"] and citation["citation_full_recall"]
            ) if available else False
        records.append(record)
        print(
            f"[{index}/{len(inputs.questions)}] {case_id} mode={mode} calls={request_count} "
            f"rrf={len(stages['rrf'])} reranker_in={len(stages['reranker_input'])} "
            f"reranker={len(stages['reranker'])} final={len(stages['final'])} {latency:.0f}ms",
            flush=True,
        )
    return records, {
        "model_chat_completion_requests": client.chat_completion_requests,
        "case_count": len(records),
    }


def _gold_identity_retrievability(
    inputs: r1.GoldenInputs,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    mapping = r1._doc_map(inputs.corpus)
    documents = {str(item["document_id"]): str(item["filename"]) for item in inputs.corpus["documents"]}
    sources = r1._source_records(inputs.labels_by_id.values())
    chroma_present = bm25_present = either = exact = parent_mismatch = missing = 0
    details = []
    collection = None
    try:
        from src.services.vector_store import get_or_create_collection

        collection = get_or_create_collection()
    except Exception:
        collection = None
    with sqlite3.connect(args.bm25_db_path) as conn:
        for source in sources:
            evidence = str(source.get("evidence_id") or "")
            document = str(source.get("document_id") or "")
            filename = documents.get(document)
            row = conn.execute(
                "SELECT metadata_json, user_id, doc_name FROM chunk_store WHERE doc_id = ?",
                (evidence,),
            ).fetchone() if evidence else None
            in_bm25 = bool(row and int(row[1]) == args.tenant_id and row[2] == filename)
            metadata = {}
            if row:
                try:
                    metadata = json.loads(row[0] or "{}")
                except json.JSONDecodeError:
                    metadata = {}
            in_chroma = False
            chroma_metadata = {}
            if collection is not None and evidence:
                try:
                    got = collection.get(ids=[evidence], include=["metadatas"])
                    in_chroma = bool(got.get("ids"))
                    if in_chroma and got.get("metadatas"):
                        chroma_metadata = got["metadatas"][0] or {}
                except Exception:
                    in_chroma = False
            if in_bm25:
                bm25_present += 1
            if in_chroma:
                chroma_present += 1
            if in_bm25 or in_chroma:
                either += 1
            raw = {
                "doc_id": evidence,
                "filename": filename,
                "type": metadata.get("type") or chroma_metadata.get("type") or "text",
            }
            key, _, _ = r1.candidate_identity_from_record(
                raw, filename_to_document=mapping, tenant_id=args.tenant_id
            )
            exact_match = bool(key and key == source.get("candidate_key"))
            if exact_match:
                exact += 1
            if metadata.get("type") == "table_cell" or chroma_metadata.get("type") == "table_cell":
                parent_mismatch += 1
            if not (in_bm25 or in_chroma):
                missing += 1
            details.append(
                {
                    "case_id": source["case_id"],
                    "source_index": source["source_index"],
                    "candidate_key": source.get("candidate_key"),
                    "evidence_id": evidence,
                    "present_in_bm25": in_bm25,
                    "present_in_chroma": in_chroma,
                    "exact_identity_reconstructable": exact_match,
                    "parent_child_identity_mismatch": bool(
                        metadata.get("type") == "table_cell"
                        or chroma_metadata.get("type") == "table_cell"
                    ),
                }
            )
    return {
        "gold_source_count": len(sources),
        "present_in_chroma": chroma_present,
        "present_in_bm25": bm25_present,
        "present_in_either_index": either,
        "exact_identity_reconstructable": exact,
        "parent_child_identity_mismatch": parent_mismatch,
        "missing_from_current_index": missing,
        "gold_identity_integrity_passed": (
            len(sources) == 80 and either == 80 and exact == 80 and parent_mismatch == 0
        ),
        "details": details,
    }


def _lineage_report(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    per_case = []
    aggregate = Counter()
    anomalies = []
    for record in records:
        stages = record["retrieval_stages"]
        value = record["lineage_integrity"]
        for key in (
            "reranker_input_not_in_rrf_count",
            "reranker_output_not_in_input_count",
            "final_not_in_reranker_output_count",
            "candidate_identity_changed_between_stages",
            "unexpected_candidate_injection_count",
            "unexplained_candidate_drop_count",
            "missing_identity_count",
            "duplicate_candidate_count",
        ):
            aggregate[key] += int(value.get(key, 0))
        anomalies.extend(
            {"case_id": record["case_id"], **item}
            for item in value.get("anomalies") or []
        )
        per_case.append(
            {
                "case_id": record["case_id"],
                "reranker_input_source": record["reranker_input_source"],
                "reranker_input_limit": (
                    None if record["reranker_input_source"] == "rrf_all" else 20
                ),
                "reranker_output_limit": 20,
                "stages": {
                    name: audit_stage(name, values)
                    for name, values in stages.items()
                    if name != "legacy_debug_rrf_top40"
                },
                "legacy_debug_rrf_top40_matches_actual_prefix": ordered_candidate_keys(
                    record["legacy_debug_rrf_top40"]
                )
                == ordered_candidate_keys(stages["rrf"][:40]),
                "lineage_integrity_passed": bool(value["lineage_integrity_passed"]),
            }
        )
    aggregate["lineage_integrity_passed"] = bool(
        all(item["lineage_integrity_passed"] for item in per_case)
    )
    aggregate["case_count"] = len(records)
    aggregate["reranker_input_source_counts"] = dict(
        Counter(str(item["reranker_input_source"]) for item in per_case)
    )
    aggregate["anomalies"] = anomalies
    return dict(aggregate), {"case_count": len(records), "cases": per_case}


def _retrieval_metrics(records: Sequence[Mapping[str, Any]], inputs: r1.GoldenInputs) -> dict[str, Any]:
    labels = [inputs.labels_by_id[item["case_id"]] for item in records]
    directly_comparable = all(
        item["reranker_input_source"] == "rrf_top_n"
        or len(item["retrieval_stages"]["rrf"]) <= 40
        for item in records
    )
    result: dict[str, Any] = {
        "definitions": {
            "rrf_ordered_list_is_captured_full_reranker_input": True,
            "reranker_input_source": dict(
                Counter(item["reranker_input_source"] for item in records)
            ),
            "explicit_multi_source_case_count": sum(item["bucket"] == "multi_source" for item in records),
            "calculation_multi_evidence_case_count": sum(
                item["bucket"] == "calculation" and item["expected_source_count"] > 1
                for item in records
            ),
            "all_multi_evidence_case_count": sum(item["expected_source_count"] > 1 for item in records),
        },
        "stage_metrics_directly_comparable": directly_comparable,
        "stage_metrics_comparability_reason": (
            "Reranker input is the same RRF cutoff"
            if directly_comparable
            else "RRF metrics use Top-40 while production reranker input uses the full RRF list"
        ),
    }
    stage_map = {
        "rrf": (5, 20, 40),
        "reranker": (5, 20),
        "final": (5,),
    }
    for stage, ks in stage_map.items():
        rankings = {item["case_id"]: item["retrieval_stages"][stage] for item in records}
        result[stage] = {
            f"recall_at_{k}": r1._stage_metrics(labels, rankings, k) for k in ks
        }
    return result


def _conditional_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for coverage in ("no_gold_in_final", "partial_gold_in_final", "all_gold_in_final", "no_answer_case"):
        rows = [item for item in records if item["context_coverage"] == coverage]
        output[coverage] = {
            "case_count": len(rows),
            "raw": r1._metric(rows, False),
            "released": r1._metric(rows, True),
        }
    return output


def _execution_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for mode in sorted({item["answer_execution_mode"] for item in records}):
        rows = [item for item in records if item["answer_execution_mode"] == mode]
        output[mode] = {
            "case_count": len(rows),
            "any_gold_in_final_count": sum(item["context_coverage"] in {"partial_gold_in_final", "all_gold_in_final"} for item in rows),
            "all_gold_in_final_count": sum(item["context_coverage"] == "all_gold_in_final" for item in rows),
            "raw": r1._metric(rows, False),
            "released": r1._metric(rows, True),
            "p50_ms": r1._percentile([item["latency_ms"] for item in rows], 0.5),
            "p95_ms": r1._percentile([item["latency_ms"] for item in rows], 0.95),
        }
    return output


def _failure_attribution(records: Sequence[dict[str, Any]], inputs: r1.GoldenInputs) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    counts = Counter()
    for record in records:
        if record["expected_no_answer"]:
            continue
        label = inputs.labels_by_id[record["case_id"]]
        expected = list(label.get("expected_sources") or [])
        expected_keys = [str(source.get("candidate_key") or "") for source in expected]
        valid = all(expected_keys)
        stages = record["retrieval_stages"]
        present = {
            stage: [key for key in expected_keys if key in set(ordered_candidate_keys(stages[stage]))]
            for stage in ("rrf", "reranker_input", "reranker", "final")
        }
        final_count = record["final_matched_gold_source_count"]
        failure = first_failure_stage(
            gold_identity_valid=valid,
            gold_in_rrf=len(present["rrf"]) == len(expected_keys),
            gold_in_reranker_input=len(present["reranker_input"]) == len(expected_keys),
            gold_in_reranker_output=len(present["reranker"]) == len(expected_keys),
            gold_in_final=len(present["final"]) == len(expected_keys),
            final_partial=0 < final_count < len(expected_keys),
            raw_contract_correct=record["raw_answer_contract_correct"],
            released_contract_correct=record["released_answer_contract_correct"],
            raw_value_correct=record["raw_value_correct"],
            released_value_correct=record["released_value_correct"],
            raw_unit_correct=record["raw_unit_correct"],
            released_unit_correct=record["released_unit_correct"],
            raw_period_correct=record["raw_period_correct"],
            released_period_correct=record["released_period_correct"],
            citation_full_recall=record["released_citation_full_recall"],
            execution_mode=record["answer_execution_mode"],
            requires_calculation=bool(
                next(
                    q for q in inputs.questions if q["case_id"] == record["case_id"]
                ).get("requires_calculation")
            ),
            calculation_route_hit=record["answer_execution_mode"] == "deterministic_calculation",
        )
        counts[failure.value] += 1
        rows.append(
            {
                "case_id": record["case_id"],
                "failure_stage": failure.value,
                "context_coverage": record["context_coverage"],
                "expected_source_keys": expected_keys,
                "stage_presence": present,
                "execution_mode": record["answer_execution_mode"],
                "raw_answer_contract_correct": record["raw_answer_contract_correct"],
                "released_answer_contract_correct": record["released_answer_contract_correct"],
            }
        )
    return {"counts": dict(counts), "cases": rows}, {"case_count": len(rows)}


def _no_answer_attribution(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    counts = Counter()
    for record in records:
        if not record["expected_no_answer"]:
            continue
        if record["released_value_correct"]:
            category = "correct_safe_response"
        elif record["answer_execution_mode"] == "llm_generation":
            category = "false_answer_llm"
        elif record["answer_execution_mode"] == "safe_response":
            category = "false_answer_deterministic"
        elif record["released_answer_contract_correct"]:
            category = "validator_false_accept"
        else:
            category = "citation_only_response"
        counts[category] += 1
        rows.append({"case_id": record["case_id"], "failure_stage": category, "execution_mode": record["answer_execution_mode"]})
    return {"counts": dict(counts), "cases": rows}


def _golden_scope(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "allowed_document_count": 8,
        "retrieved_out_of_scope_candidates": sum(item["scope_out_of_scope_count"] for item in records),
        "reranked_out_of_scope_candidates": sum(
            sum(
                int(candidate.get("canonical_document_id") not in set(item["document_scope"]))
                for candidate in item["retrieval_stages"]["reranker"]
                if candidate.get("canonical_document_id")
            )
            for item in records
        ),
        "final_context_out_of_scope_candidates": sum(
            sum(
                int(candidate.get("canonical_document_id") not in set(item["document_scope"]))
                for candidate in item["retrieval_stages"]["final"]
                if candidate.get("canonical_document_id")
            )
            for item in records
        ),
        "citation_out_of_scope_count": sum(item["citation_out_of_scope_count"] for item in records),
        "legacy_27_included": False,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", type=int, required=True)
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-key", default="not-needed-for-local")
    parser.add_argument("--chroma-path", type=Path, default=ROOT / "chroma_db")
    parser.add_argument("--bm25-db-path", type=Path, default=ROOT / "rag_bm25.db")
    parser.add_argument("--n-results", type=int, default=5)
    parser.add_argument("--retrieval-candidate-multiplier", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--corpus", type=Path, default=BENCHMARK / "corpus.json")
    parser.add_argument("--manifest", dest="manifest_path", type=Path, default=DATA / "golden-manifest.json")
    parser.add_argument("--questions", type=Path, default=DATA / "questions.golden.jsonl")
    parser.add_argument("--labels", type=Path, default=DATA / "labels.golden.jsonl")
    parser.add_argument("--review-status", type=Path, default=DATA / "review-status.golden.jsonl")
    parser.add_argument("--negative-report", type=Path, default=DEFAULT_NEGATIVE)
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> int:
    try:
        inputs = r1._load_inputs(
            corpus_path=args.corpus,
            manifest_path=args.manifest_path,
            questions_path=args.questions,
            labels_path=args.labels,
            review_status_path=args.review_status,
            negative_report_path=args.negative_report,
        )
    except r1.BaselineConfigurationError as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _write(
            args.out_dir / "nf-eval-03-r2-acceptance.json",
            {
                "artifact_schema": "nf-eval-03-r2/baseline/v1",
                "decision": "baseline_input_integrity_failed",
                "candidate_lineage_integrity_passed": False,
                "gold_identity_integrity_passed": False,
                "production_behavior_changed": False,
                "optimization_allowed": False,
                "error": str(exc),
            },
        )
        print(f"NF-EVAL-03 R2 configuration error: {exc}", file=sys.stderr)
        return 2

    records, run_info = await _run_queries(args, inputs)
    lineage_aggregate, stage_report = _lineage_report(records)
    gold_report = _gold_identity_retrievability(inputs, args=args)
    retrieval = _retrieval_metrics(records, inputs)
    conditional = _conditional_metrics(records)
    execution = _execution_metrics(records)
    failures, _ = _failure_attribution(records, inputs)
    no_answer = _no_answer_attribution(records)
    scope = _golden_scope(records)
    latencies = [item["latency_ms"] for item in records]
    input_source_counts = dict(Counter(item["reranker_input_source"] for item in records))
    calculation_cases = [
        item
        for item in records
        if bool(next(q for q in inputs.questions if q["case_id"] == item["case_id"]).get("requires_calculation"))
    ]
    calculation_route = {
        "calculation_case_count": len(calculation_cases),
        "execution_mode_counts": dict(Counter(item["answer_execution_mode"] for item in calculation_cases)),
        "deterministic_calculation_count": sum(item["answer_execution_mode"] == "deterministic_calculation" for item in calculation_cases),
        "calculation_route_miss_count": sum(item["answer_execution_mode"] != "deterministic_calculation" for item in calculation_cases),
        "cases": [
            {"case_id": item["case_id"], "execution_mode": item["answer_execution_mode"]}
            for item in calculation_cases
        ],
    }
    errors = [item["case_id"] for item in records if item["error"]]
    hash_report = inputs.hash_report
    baseline = {
        "artifact_schema": "nf-eval-03-r2/baseline/v1",
        "benchmark_id": "financial-rag-v1",
        "case_count": len(records),
        "answerable_count": sum(not item["expected_no_answer"] for item in records),
        "no_answer_count": sum(item["expected_no_answer"] for item in records),
        "tenant_id": args.tenant_id,
        "allowed_document_ids_hash": r1.stable_json_hash(sorted(x["document_id"] for x in inputs.corpus["documents"])),
        **hash_report["actual"],
        "embedding_model": get_embedding_model_name(),
        "reranker": get_reranker_name(),
        "reranker_model": get_reranker_model(),
        "generator_model": args.model_name,
        "generator_endpoint": args.model_base_url,
        "n_results": args.n_results,
        "retrieval_candidate_multiplier": args.retrieval_candidate_multiplier,
        "reranker_input_source": input_source_counts,
        "reranker_input_limit": None,
        "reranker_output_limit": 20,
        "reranker_input_limit_semantics": "full_rrf_list_before_top20_output",
        "parent_expansion_before_rerank": False,
        "candidate_injection_before_rerank": False,
        "production_behavior_changed": False,
        "legacy_27_included": False,
    }
    passed = bool(
        len(records) == 72
        and not errors
        and all(hash_report["matches"].values())
        and bool(lineage_aggregate["lineage_integrity_passed"])
        and gold_report["gold_identity_integrity_passed"]
        and all(item["scope_out_of_scope_count"] == 0 for item in records)
        and all(item["citation_out_of_scope_count"] == 0 for item in records)
    )
    acceptance = {
        "artifact_schema": "nf-eval-03-r2/baseline/v1",
        "decision": "formal_baseline_attributed" if passed else "baseline_lineage_or_identity_incomplete",
        "case_count": len(records),
        "input_hashes_verified": all(hash_report["matches"].values()),
        "candidate_lineage_integrity_passed": bool(lineage_aggregate["lineage_integrity_passed"]),
        "gold_identity_integrity_passed": gold_report["gold_identity_integrity_passed"],
        "scope_integrity_passed": not any(scope[key] for key in (
            "retrieved_out_of_scope_candidates",
            "reranked_out_of_scope_candidates",
            "final_context_out_of_scope_candidates",
            "citation_out_of_scope_count",
        )),
        "production_behavior_changed": False,
        "optimization_allowed": False,
        "model_chat_completion_requests": run_info["model_chat_completion_requests"],
        "execution_mode_counts": dict(Counter(item["answer_execution_mode"] for item in records)),
        "errors": errors,
    }
    out = args.out_dir
    _write(out / "baseline-manifest.json", baseline)
    _write(out / "candidate-lineage-integrity.json", lineage_aggregate)
    _write(out / "gold-identity-retrievability-report.json", gold_report)
    _write(out / "stage-candidate-set-report.json", stage_report)
    _write(out / "retrieval-metrics.json", retrieval)
    _write(out / "conditional-answer-metrics.json", conditional)
    _write(out / "execution-mode-conditional-metrics.json", execution)
    _write(out / "calculation-route-report.json", calculation_route)
    _write(out / "failure-attribution.json", failures)
    _write(out / "no-answer-attribution.json", no_answer)
    _write(out / "case-results.json", {"artifact_schema": "nf-eval-03-r2/baseline/v1", "cases": records})
    _write(
        out / "nf-eval-03-r2-acceptance.json",
        acceptance,
    )
    _write(
        out / "scope-integrity-report.json",
        {
            **scope,
            "scope_integrity_passed": acceptance["scope_integrity_passed"],
        },
    )
    _write(
        out / "latency-report.json",
        {
            "count": len(latencies),
            "p50_ms": r1._percentile(latencies, 0.5),
            "p95_ms": r1._percentile(latencies, 0.95),
            "mean_ms": statistics.mean(latencies) if latencies else None,
            "max_ms": max(latencies) if latencies else None,
        },
    )
    print(json.dumps(acceptance, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps({"retrieval": retrieval, "execution": execution}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 2


def main() -> None:
    raise SystemExit(asyncio.run(_main(_args())))


if __name__ == "__main__":
    main()
