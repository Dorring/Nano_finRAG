"""Run the current production RAG configuration on Financial RAG v1.

This is an evaluation-only runner.  It loads the already promoted Golden
files, passes questions (never labels) to the unchanged production
``RAGOrchestrator`` and scores the saved, redacted observations offline.
The legacy 27-case development set is deliberately not loaded here.

The runner records retrieval stage diagnostics already emitted by the
production retrieval pipeline (dense/BM25/RRF/reranker/final), so Recall@5,
Recall@20 and Recall@40 come from one production query per case.  Raw and
released answer hashes are captured through the existing evaluation observer;
the observer has no effect on production return values.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
import uuid
from typing import Any, Iterable

from scripts.evaluation.benchmark_foundation import stable_json_hash
from scripts.evaluation.freeze_financial_rag_reference_answers import (
    _answer_hash_payload,
    _question_hash_payload,
)
from src.domain.query import QueryRequest
from src.evaluation.manifests import compute_file_sha256
from src.evaluation.nf40_pipeline_observer import AnswerPipelineTrace
from src.services.retrieval_config import (
    get_embedding_model_name,
    get_reranker_model,
    get_reranker_name,
)


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "financial_rag_v1"
DATA = BENCHMARK / "data"
DEFAULT_OUT = ROOT / "artifacts" / "evaluation" / "nf-eval-03"
SCALE_FACTORS = {
    "thousand": Decimal("1000"),
    "million": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "trillion": Decimal("1000000000000"),
    "k": Decimal("1000"),
    "m": Decimal("1000000"),
    "b": Decimal("1000000000"),
    "t": Decimal("1000000000000"),
}
NUMBER_RE = re.compile(r"(?<![A-Za-z])\(?\s*[-+]?\$?\s*\d[\d,]*(?:\.\d+)?\s*%?\s*\)?")
NO_ANSWER_MARKERS = (
    "not disclosed",
    "does not disclose",
    "not available",
    "no information",
    "cannot answer",
    "cannot be determined",
    "insufficient information",
    "无法回答",
    "未披露",
    "没有披露",
)


class BaselineConfigurationError(ValueError):
    """Raised when frozen Golden inputs or scope are invalid."""


class _NullTraceLogger:
    """Prevent the isolated baseline engine from writing production traces."""

    def log(self, **_kwargs: Any) -> None:
        return None


class _CountingCompletions:
    def __init__(self, owner: "_CountingClient", delegate: Any) -> None:
        self._owner = owner
        self._delegate = delegate

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self._owner.chat_completion_requests += 1
        return self._delegate.create(*args, **kwargs)


class _CountingChat:
    def __init__(self, owner: "_CountingClient", delegate: Any) -> None:
        self.completions = _CountingCompletions(owner, delegate.completions)


class _CountingClient:
    def __init__(self, delegate: Any) -> None:
        self.chat_completion_requests = 0
        self.chat = _CountingChat(self, delegate.chat)


@dataclass(frozen=True)
class GoldenInputs:
    questions: tuple[dict[str, Any], ...]
    labels_by_id: dict[str, dict[str, Any]]
    corpus: dict[str, Any]
    manifest: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise BaselineConfigurationError(f"{path}:{line_no}: JSON object required")
        rows.append(value)
    return rows


def _load_inputs(
    *,
    corpus_path: Path = BENCHMARK / "corpus.json",
    manifest_path: Path = DATA / "golden-manifest.json",
    questions_path: Path = DATA / "questions.golden.jsonl",
    labels_path: Path = DATA / "labels.golden.jsonl",
) -> GoldenInputs:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    questions = _read_jsonl(questions_path)
    labels = _read_jsonl(labels_path)
    if manifest.get("golden_ready") is not True:
        raise BaselineConfigurationError("Golden manifest is not marked golden_ready")
    if len(questions) != 72 or len(labels) != 72:
        raise BaselineConfigurationError(
            f"NF-EVAL-03 requires 72 Golden cases, got {len(questions)} questions and {len(labels)} labels"
        )
    label_ids = [str(row.get("case_id")) for row in labels]
    question_ids = [str(row.get("case_id")) for row in questions]
    if len(set(label_ids)) != len(label_ids) or len(set(question_ids)) != len(question_ids):
        raise BaselineConfigurationError("Golden case IDs must be unique")
    if set(label_ids) != set(question_ids):
        raise BaselineConfigurationError("Golden question/label case IDs do not match")
    question_payload = sorted((_question_hash_payload(row) for row in questions), key=lambda item: item["case_id"])
    answer_payload = sorted((_answer_hash_payload(row) for row in labels), key=lambda item: item["case_id"])
    actual_question_hash = stable_json_hash(question_payload)
    actual_answer_hash = stable_json_hash(answer_payload)
    for field, actual in (
        ("question_hash", actual_question_hash),
        ("reference_answer_hash", actual_answer_hash),
    ):
        if manifest.get(field) != actual:
            raise BaselineConfigurationError(
                f"Golden {field} mismatch: manifest={manifest.get(field)} actual={actual}"
            )
    if manifest.get("corpus_hash") != corpus.get("corpus_hash"):
        raise BaselineConfigurationError("Golden corpus hash does not match corpus.json")
    allowed = {str(item.get("document_id")) for item in corpus.get("documents", [])}
    if len(allowed) != 8 or corpus.get("document_count") != 8:
        raise BaselineConfigurationError("Benchmark corpus must contain exactly eight documents")
    for question in questions:
        scope = set(str(item) for item in question.get("document_scope", []))
        if not scope or not scope.issubset(allowed):
            raise BaselineConfigurationError(
                f"{question.get('case_id')}: document_scope is outside the eight-document whitelist"
            )
    for label in labels:
        answerable = not bool(label.get("expected_no_answer"))
        sources = label.get("expected_sources") or []
        if answerable and not sources:
            raise BaselineConfigurationError(f"{label.get('case_id')}: answerable Golden case has no source")
        if answerable:
            for source in sources:
                if not source.get("source_verified") or source.get("candidate_identity_status") != "bound":
                    raise BaselineConfigurationError(f"{label.get('case_id')}: source is not verified/bound")
                if not source.get("candidate_key") or not source.get("evidence_id"):
                    raise BaselineConfigurationError(f"{label.get('case_id')}: source identity is incomplete")
    return GoldenInputs(
        questions=tuple(questions),
        labels_by_id={str(item["case_id"]): item for item in labels},
        corpus=corpus,
        manifest=manifest,
    )


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _normal_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def _number_tokens(text: str) -> list[tuple[Decimal, bool, str | None]]:
    result: list[tuple[Decimal, bool, str | None]] = []
    for match in NUMBER_RE.finditer(text or ""):
        raw = match.group(0).strip()
        negative = raw.startswith("(") and raw.endswith(")")
        token = raw.strip("() ").replace("$", "").replace(",", "")
        is_percent = token.endswith("%")
        token = token.rstrip("% ")
        try:
            value = Decimal(token)
        except (InvalidOperation, ValueError):
            continue
        if negative:
            value = -value
        tail = _normal_text((text or "")[match.end() : match.end() + 14])
        scale = None
        for name in SCALE_FACTORS:
            if re.match(rf"^(?:{re.escape(name)})(?:\b|$)", tail):
                scale = name
                break
        result.append((value, is_percent, scale))
    return result


def _expected_values(answer: dict[str, Any]) -> list[dict[str, Any]]:
    components = answer.get("component_values") or []
    if answer.get("value_type") == "composite" and components:
        return [dict(item) for item in components]
    return [answer]


def _numeric_matches(text: str, answer: dict[str, Any]) -> bool:
    expected_values = _expected_values(answer)
    tokens = _number_tokens(text)
    if not expected_values:
        return False
    used: set[int] = set()
    for expected in expected_values:
        raw_expected = expected.get("canonical_value")
        if raw_expected is None:
            continue
        try:
            target = Decimal(str(raw_expected))
        except (InvalidOperation, ValueError):
            return False
        tolerance = Decimal(str(answer.get("tolerance") or expected.get("tolerance") or "0"))
        matched = False
        for index, (value, is_percent, scale) in enumerate(tokens):
            if index in used:
                continue
            candidates = [value]
            if scale:
                candidates.append(value * SCALE_FACTORS[scale])
            elif expected.get("currency") or expected.get("unit") in {"currency", "financial_volume"}:
                # Generated answers often omit the word "billion" only when
                # they use the frozen display value; compare that raw display
                # value separately below, but do not guess a scale here.
                candidates.append(value * Decimal("1000000"))
                candidates.append(value * Decimal("1000000000"))
                candidates.append(value * Decimal("1000000000000"))
            if is_percent and answer.get("percentage_representation") == "percentage_points":
                candidates.append(value)
            if any(abs(candidate - target) <= tolerance for candidate in candidates):
                used.add(index)
                matched = True
                break
        if not matched:
            display = _normal_text(expected.get("display_value"))
            if display and display in _normal_text(text):
                matched = True
        if not matched:
            return False
    return True


def _answer_correct(text: str, question: dict[str, Any], label: dict[str, Any]) -> bool:
    if label.get("expected_no_answer"):
        lower = _normal_text(text)
        return bool(lower) and any(marker in lower for marker in NO_ANSWER_MARKERS) and not _number_tokens(text)
    answer = dict(label.get("expected_answer") or {})
    if answer.get("canonical_value") is not None or answer.get("component_values"):
        return _numeric_matches(text, answer)
    expected_text = _normal_text(answer.get("text"))
    return bool(expected_text and expected_text in _normal_text(text))


def _source_matches(expected: dict[str, Any], candidate: dict[str, Any]) -> bool:
    expected_id = expected.get("evidence_id")
    candidate_id = candidate.get("evidence_id") or candidate.get("chunk_id") or candidate.get("doc_id")
    if expected_id and candidate_id == expected_id:
        return True
    expected_file = expected.get("filename")
    candidate_file = candidate.get("filename") or candidate.get("document_id") or candidate.get("doc_name")
    if expected_file and candidate_file != expected_file:
        return False
    expected_page = expected.get("page")
    candidate_page = candidate.get("page")
    return expected_page is not None and str(expected_page) == str(candidate_page)


def _matched_sources(expected: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> int:
    return sum(1 for source in expected if any(_source_matches(source, candidate) for candidate in candidates))


_ARTIFACT_TEXT_KEYS = frozenset({
    "text",
    "content",
    "raw_text",
    "chunk_text",
    "document_text",
    "answer",
    "prompt",
    "context",
})


def _sanitize_for_artifact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_for_artifact(item)
            for key, item in value.items()
            if key not in _ARTIFACT_TEXT_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_for_artifact(item) for item in value]
    return value


def _sanitize_retrieval_debug(debug: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_for_artifact(debug)
    return dict(sanitized) if isinstance(sanitized, dict) else {}


def _stage_metrics(labels: list[dict[str, Any]], stage_rankings: dict[str, list[dict[str, Any]]], k: int) -> dict[str, Any]:
    eligible = [label for label in labels if not label.get("expected_no_answer") and label.get("expected_sources")]
    source_total = sum(len(label.get("expected_sources") or []) for label in eligible)
    multi_source = [label for label in eligible if len(label.get("expected_sources") or []) > 1]
    case_hits = 0
    source_hits = 0
    all_hits = 0
    reciprocal: list[float] = []
    for label in eligible:
        case_id = str(label["case_id"])
        ranked = list(stage_rankings.get(case_id, []))[:k]
        expected = list(label.get("expected_sources") or [])
        matched = _matched_sources(expected, ranked)
        case_hits += int(matched > 0)
        source_hits += matched
        if len(expected) > 1:
            all_hits += int(matched == len(expected))
        first_rank = next((rank for rank, candidate in enumerate(ranked, 1) if any(_source_matches(source, candidate) for source in expected)), None)
        reciprocal.append(1.0 / first_rank if first_rank else 0.0)
    return {
        "k": k,
        "case_hit": {"count": case_hits, "denominator": len(eligible), "rate": case_hits / len(eligible) if eligible else 1.0},
        "source_recall": {"count": source_hits, "denominator": source_total, "rate": source_hits / source_total if source_total else 1.0},
        "all_source_coverage": {"count": all_hits, "denominator": len(multi_source), "rate": all_hits / len(multi_source) if multi_source else 1.0},
        "mrr": sum(reciprocal) / len(reciprocal) if reciprocal else 1.0,
    }


def _citation_metrics(label: dict[str, Any], sources: list[dict[str, Any]]) -> tuple[float, float]:
    expected = list(label.get("expected_sources") or [])
    if not expected:
        return 1.0, 1.0
    recall = _matched_sources(expected, sources) / len(expected)
    precision = sum(int(any(_source_matches(source, candidate) for source in expected)) for candidate in sources) / len(sources) if sources else 0.0
    return recall, precision


def _bucket_name(question: dict[str, Any]) -> str:
    categories = set(question.get("category") or [])
    if "no_answer" in categories or not question.get("answerable", True):
        return "no_answer"
    if "calculation" in categories or question.get("requires_calculation"):
        return "calculation"
    if "multi_source" in categories or question.get("requires_multiple_sources"):
        return "multi_source"
    if "table_fact" in categories:
        return "table_fact"
    if "unit_scale_period_trap" in categories:
        return "unit_scale_period_trap"
    return "fact"


def _metrics_for_records(records: list[dict[str, Any]], *, released: bool) -> dict[str, Any]:
    answer_key = "released_answer_correct" if released else "raw_answer_correct"
    citation_key = "released_citation_recall" if released else "raw_citation_recall"
    precision_key = "released_citation_precision" if released else "raw_citation_precision"
    count = len(records)
    answerable = [row for row in records if not row["expected_no_answer"]]
    no_answer = [row for row in records if row["expected_no_answer"]]
    correct = sum(int(row[answer_key]) for row in records)
    numeric = sum(int(row["numeric_correct"]) for row in answerable)
    citation = sum(float(row[citation_key]) == 1.0 for row in answerable)
    precision = sum(float(row[precision_key]) == 1.0 for row in answerable)
    no_answer_correct = sum(int(row["no_answer_correct"]) for row in no_answer)
    return {
        "golden_pass": {"count": correct, "denominator": count, "rate": correct / count if count else 0.0},
        "numeric_accuracy": {"count": numeric, "denominator": len(answerable), "rate": numeric / len(answerable) if answerable else 1.0},
        "citation_recall": {"count": citation, "denominator": len(answerable), "rate": citation / len(answerable) if answerable else 1.0},
        "citation_precision": {"count": precision, "denominator": len(answerable), "rate": precision / len(answerable) if answerable else 1.0},
        "no_answer_accuracy": {"count": no_answer_correct, "denominator": len(no_answer), "rate": no_answer_correct / len(no_answer) if no_answer else 1.0},
    }


def _build_engine(args: argparse.Namespace) -> tuple[Any, _CountingClient]:
    if args.tenant_id != 1:
        raise BaselineConfigurationError("NF-EVAL-03 Golden corpus is tenant 1 only")
    os.environ["CHROMA_PATH"] = str(args.chroma_path)
    os.environ["BM25_DB_PATH"] = str(args.bm25_db_path)
    from openai import OpenAI
    from src.services.rag_engine import RAGEngine

    client = _CountingClient(OpenAI(base_url=args.model_base_url, api_key=args.api_key))
    engine = RAGEngine(
        client,
        model_name=args.model_name,
        use_hybrid=True,
        reranker_name=get_reranker_name(),
        reranker_model=get_reranker_model(),
        retrieval_candidate_multiplier=args.retrieval_candidate_multiplier,
        bm25_db_path=str(args.bm25_db_path),
        trace_db_path=str(args.out_dir / "evaluation-trace.db"),
    )
    # The isolated evaluator must not write to the production trace database.
    engine._orchestrator._trace_logger = _NullTraceLogger()
    return engine, client


async def _run_queries(args: argparse.Namespace, inputs: GoldenInputs) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    engine, client = _build_engine(args)
    filenames = {str(item["document_id"]): str(item["filename"]) for item in inputs.corpus["documents"]}
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for question in inputs.questions:
        case_id = str(question["case_id"])
        label = inputs.labels_by_id[case_id]
        doc_names = [filenames[item] for item in question.get("document_scope", [])]
        trace = AnswerPipelineTrace(
            case_id=case_id,
            trace_id=uuid.uuid4().hex,
            context_hash="",
            context_coverage="not_evaluated",
        )
        request = QueryRequest(
            question=str(question["question"]),
            document_names=tuple(doc_names),
            user_id=args.tenant_id,
            conversation_history=(),
            memory_profile=None,
        )
        case_started = time.perf_counter()
        error: str | None = None
        result: dict[str, Any] = {}
        try:
            answer_result = await engine._orchestrator.answer(
                request,
                n_results=args.n_results,
                evaluation_observer=trace,
            )
            result = answer_result.to_legacy_dict()
        except Exception as exc:  # noqa: BLE001 - preserve per-case diagnostics
            error = type(exc).__name__
        latency_ms = (time.perf_counter() - case_started) * 1000
        answer = str(result.get("answer") or "")
        raw_answer = trace._raw_generation_text or ""
        released_answer_correct = _answer_correct(answer, question, label)
        raw_answer_correct = _answer_correct(raw_answer, question, label) if trace.raw_generation_hash else False
        retrieved = result.get("retrieved_chunks") or []
        sources = result.get("sources") or []
        debug = result.get("retrieval_debug") or {}
        stages = debug.get("candidate_stages") or {}
        scope_ids = set(question.get("document_scope") or [])
        scope_files = {filenames[item] for item in scope_ids}
        stage_out_of_scope = 0
        for stage in stages.values():
            for candidate in stage or []:
                candidate_file = candidate.get("document_id") or candidate.get("filename")
                if candidate_file and candidate_file not in scope_files:
                    stage_out_of_scope += 1
        sources_out_of_scope = sum(int((item.get("filename") or item.get("document_name")) not in scope_files) for item in sources)
        expected_sources = list(label.get("expected_sources") or [])
        raw_citation_recall, raw_citation_precision = _citation_metrics(label, sources)
        released_citation_recall, released_citation_precision = raw_citation_recall, raw_citation_precision
        no_answer_correct = _answer_correct(answer, question, label) if label.get("expected_no_answer") else True
        records.append({
            "case_id": case_id,
            "company": question.get("company"),
            "bucket": _bucket_name(question),
            "document_scope": list(scope_ids),
            "expected_no_answer": bool(label.get("expected_no_answer")),
            "expected_source_count": len(expected_sources),
            "raw_generation_hash": trace.raw_generation_hash,
            "released_answer_hash": trace.released_answer_hash,
            "raw_answer_correct": bool(raw_answer_correct),
            "released_answer_correct": bool(released_answer_correct),
            "numeric_correct": bool(_numeric_matches(raw_answer if trace.raw_generation_hash else answer, label.get("expected_answer") or {})) if not label.get("expected_no_answer") else True,
            "no_answer_correct": bool(no_answer_correct),
            "raw_citation_recall": raw_citation_recall,
            "raw_citation_precision": raw_citation_precision,
            "released_citation_recall": released_citation_recall,
            "released_citation_precision": released_citation_precision,
            "validation_status": (result.get("validation") or {}).get("status"),
            "repair": result.get("repair"),
            "retrieval_debug": _sanitize_retrieval_debug(debug),
            "stage_out_of_scope_count": stage_out_of_scope,
            "sources_out_of_scope_count": sources_out_of_scope,
            "retrieved_count": len(retrieved),
            "latency_ms": latency_ms,
            "error": error,
        })
        print(f"[{len(records)}/{len(inputs.questions)}] {case_id} {latency_ms:.0f}ms", flush=True)
    total_ms = (time.monotonic() - started) * 1000
    return records, {
        "model_chat_completion_requests": client.chat_completion_requests,
        "case_count": len(records),
        "total_elapsed_ms": total_ms,
    }


def _build_report(args: argparse.Namespace, inputs: GoldenInputs, records: list[dict[str, Any]], run_info: dict[str, Any]) -> dict[str, Any]:
    labels = [inputs.labels_by_id[row["case_id"]] for row in records]
    stage_rankings: dict[str, dict[str, list[dict[str, Any]]]] = {name: {} for name in ("dense", "bm25", "rrf", "reranker", "final")}
    for row in records:
        for stage in stage_rankings:
            stage_rankings[stage][row["case_id"]] = list((row.get("retrieval_debug") or {}).get("candidate_stages", {}).get(stage) or [])
    retrieval = {
        stage: {f"recall_at_{k}": _stage_metrics(labels, rankings, k) for k in ([5, 20, 40] if stage in {"dense", "bm25", "rrf"} else [5, 20] if stage == "reranker" else [5])}
        for stage, rankings in stage_rankings.items()
    }
    all_stages = [candidate for row in records for stage in (row.get("retrieval_debug") or {}).get("candidate_stages", {}).values() for candidate in stage]
    out_of_scope = {
        "retrieved_out_of_scope_candidates": sum(row["stage_out_of_scope_count"] for row in records),
        "citation_out_of_scope_count": sum(row["sources_out_of_scope_count"] for row in records),
        "scope_integrity_passed": all(row["stage_out_of_scope_count"] == 0 and row["sources_out_of_scope_count"] == 0 for row in records),
        "stage_candidate_count": len(all_stages),
    }
    answer = {
        "raw": _metrics_for_records(records, released=False),
        "released": _metrics_for_records(records, released=True),
        "raw_available_count": sum(int(bool(row["raw_generation_hash"])) for row in records),
        "validator_status_counts": dict(Counter(str(row["validation_status"]) for row in records)),
        "repair_attempt_count": sum(int(bool(row.get("repair"))) for row in records),
    }
    slices: dict[str, Any] = {}
    for key in ("company", "bucket"):
        values = sorted({str(row.get(key)) for row in records})
        slices[key] = {value: _metrics_for_records([row for row in records if str(row.get(key)) == value], released=True) for value in values}
    latencies = [float(row["latency_ms"]) for row in records]
    manifest = inputs.manifest
    baseline_manifest = {
        "artifact_schema": "nf-eval-03/baseline/v1",
        "benchmark_id": "financial-rag-v1",
        "case_count": len(records),
        "answerable_count": sum(int(not row["expected_no_answer"]) for row in records),
        "no_answer_count": sum(int(row["expected_no_answer"]) for row in records),
        "tenant_id": args.tenant_id,
        "allowed_document_ids_hash": stable_json_hash(sorted(item["document_id"] for item in inputs.corpus["documents"])),
        "corpus_hash": manifest["corpus_hash"],
        "question_hash": manifest["question_hash"],
        "reference_answer_hash": manifest["reference_answer_hash"],
        "source_identity_hash": manifest["source_identity_hash"],
        "index": {"chroma_path_type": "production_default", "bm25_path_type": "production_default"},
        "embedding_model": get_embedding_model_name(),
        "reranker": get_reranker_name(),
        "reranker_model": get_reranker_model(),
        "generator_model": args.model_name,
        "generator_endpoint": args.model_base_url,
        "n_results": args.n_results,
        "retrieval_candidate_multiplier": args.retrieval_candidate_multiplier,
        "production_behavior_changed": False,
        "legacy_27_included": False,
        "golden_manifest_sha256": compute_file_sha256(args.manifest_path),
    }
    return {
        "baseline_manifest": baseline_manifest,
        "retrieval_metrics": retrieval,
        "answer_metrics": answer,
        "slice_metrics": slices,
        "scope_integrity": out_of_scope,
        "latency": {"count": len(latencies), "p50_ms": _percentile(latencies, 0.50), "p95_ms": _percentile(latencies, 0.95), "mean_ms": statistics.mean(latencies) if latencies else None, "max_ms": max(latencies) if latencies else None},
        "run_info": run_info,
        "errors": [row["case_id"] for row in records if row.get("error")],
        "records": records,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> int:
    inputs = _load_inputs(corpus_path=args.corpus, manifest_path=args.manifest_path, questions_path=args.questions, labels_path=args.labels)
    records, run_info = await _run_queries(args, inputs)
    report = _build_report(args, inputs, records, run_info)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "baseline-manifest.json", report["baseline_manifest"])
    _write_json(args.out_dir / "retrieval-metrics.json", report["retrieval_metrics"])
    _write_json(args.out_dir / "answer-metrics.json", report["answer_metrics"])
    _write_json(args.out_dir / "slice-metrics.json", report["slice_metrics"])
    _write_json(args.out_dir / "scope-integrity-report.json", report["scope_integrity"])
    _write_json(args.out_dir / "latency-report.json", report["latency"])
    _write_json(args.out_dir / "case-results.json", {"cases": report["records"]})
    acceptance = {
        "artifact_schema": "nf-eval-03/baseline/v1",
        "benchmark_id": "financial-rag-v1",
        "case_count": len(records),
        "baseline_run_completed": len(records) == 72 and not report["errors"],
        "production_behavior_changed": False,
        "legacy_27_included": False,
        "scope_integrity_passed": report["scope_integrity"]["scope_integrity_passed"],
        "model_chat_completion_requests": run_info["model_chat_completion_requests"],
        "errors": report["errors"],
        "decision": "baseline_recorded" if len(records) == 72 and not report["errors"] and report["scope_integrity"]["scope_integrity_passed"] else "baseline_incomplete",
    }
    _write_json(args.out_dir / "nf-eval-03-acceptance.json", acceptance)
    print(json.dumps({"acceptance": acceptance, "answer": report["answer_metrics"], "latency": report["latency"], "retrieval": report["retrieval_metrics"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if acceptance["baseline_run_completed"] and acceptance["scope_integrity_passed"] else 2


def main() -> None:
    args = _parse_args()
    try:
        raise SystemExit(asyncio.run(_async_main(args)))
    except BaselineConfigurationError as exc:
        print(f"NF-EVAL-03 configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
