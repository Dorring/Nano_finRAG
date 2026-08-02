"""NF-EVAL-03 R1 formal baseline with strict metric semantics."""

from __future__ import annotations
import argparse
import asyncio
import json
import re
import statistics
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.evaluation.benchmark_foundation import corpus_hash, stable_json_hash
from scripts.evaluation.bind_nf_eval_02_sources import _source_value_label
from scripts.evaluation.freeze_financial_rag_reference_answers import (
    _answer_hash_payload,
    _question_hash_payload,
)
from scripts.evaluation.run_nf_eval_03_baseline import (
    _build_engine,
    _percentile,
    _sanitize_retrieval_debug,
)
from src.domain.answer import AnswerPath
from src.domain.query import QueryRequest
from src.evaluation.benchmark_source_binding import source_expected_metric
from src.evaluation.manifests import compute_file_sha256
from src.evaluation.nf40_pipeline_observer import AnswerPipelineTrace
from src.retrieval.candidate_identity import (
    CandidateIdentityError,
    candidate_key as stable_candidate_key,
    identity_from_candidate,
)
from src.services.retrieval_config import (
    get_embedding_model_name,
    get_reranker_model,
    get_reranker_name,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "financial_rag_v1"
DATA = BENCHMARK / "data"
DEFAULT_OUT = ROOT / "artifacts" / "evaluation" / "nf-eval-03-r1"
DEFAULT_NEGATIVE = (
    ROOT
    / "artifacts"
    / "evaluation"
    / "nf-eval-02"
    / "negative-evidence-review-report.json"
)
SCALES = {
    "thousand": Decimal("1000"),
    "million": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "trillion": Decimal("1000000000000"),
    "k": Decimal("1000"),
    "m": Decimal("1000000"),
    "b": Decimal("1000000000"),
    "t": Decimal("1000000000000"),
}
MARKERS = (
    "not disclosed",
    "does not disclose",
    "not available",
    "no information",
    "cannot answer",
    "cannot be determined",
    "insufficient information",
    "not provided",
    "未披露",
    "没有披露",
)
NUM_RE = re.compile(r"(?<![A-Za-z])\(?\s*[-+]?\s*\$?\s*\d[\d,]*(?:\.\d+)?\s*%?\s*\)?")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
SCALE_RE = re.compile(r"\b(thousand|million|billion|trillion|[KMBT])\b", re.I)
CURRENCY_RE = re.compile(r"(?:\bUSD\b|\bUS\$|[$€£¥])", re.I)


class BaselineConfigurationError(ValueError):
    """Frozen baseline input or scope is invalid."""


@dataclass(frozen=True)
class GoldenInputs:
    questions: tuple[dict[str, Any], ...]
    labels_by_id: dict[str, dict[str, Any]]
    corpus: dict[str, Any]
    manifest: dict[str, Any]
    hash_report: dict[str, Any]


@dataclass(frozen=True)
class NumberToken:
    value: Decimal
    percent: bool
    scale: str | None
    currency: str | None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise BaselineConfigurationError(f"{path}:{line_no}: object required")
        rows.append(value)
    return rows


def _source_records(labels: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for label in labels:
        for index, source in enumerate(label.get("expected_sources") or []):
            out.append(
                {
                    "case_id": str(label["case_id"]),
                    "source_index": index,
                    "document_id": source.get("document_id"),
                    "pdf_page": source.get("page"),
                    "row_label": source.get("row_label"),
                    "column_header": source.get("column_header"),
                    "metric": source_expected_metric(label, source, index),
                    "expected_value": _source_value_label(label, index),
                    "candidate_identity_status": source.get(
                        "candidate_identity_status"
                    ),
                    "candidate_key": source.get("candidate_key"),
                    "evidence_id": source.get("evidence_id"),
                    "identity_granularity": source.get("identity_granularity"),
                    "candidate_content_hash": source.get("candidate_content_hash"),
                    "binding_method": source.get("binding_method"),
                    "source_verified": bool(source.get("source_verified")),
                }
            )
    return out


def _load_inputs(
    *,
    corpus_path: Path = BENCHMARK / "corpus.json",
    manifest_path: Path = DATA / "golden-manifest.json",
    questions_path: Path = DATA / "questions.golden.jsonl",
    labels_path: Path = DATA / "labels.golden.jsonl",
    review_status_path: Path = DATA / "review-status.golden.jsonl",
    negative_report_path: Path = DEFAULT_NEGATIVE,
) -> GoldenInputs:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    questions, labels, reviews = (
        _read_jsonl(questions_path),
        _read_jsonl(labels_path),
        _read_jsonl(review_status_path),
    )
    if manifest.get("golden_ready") is not True or any(
        len(x) != 72 for x in (questions, labels, reviews)
    ):
        raise BaselineConfigurationError("Golden inputs must contain 72 ready rows")
    ids = [
        [str(x.get("case_id")) for x in rows] for rows in (questions, labels, reviews)
    ]
    if any(len(set(x)) != 72 for x in ids) or not (
        set(ids[0]) == set(ids[1]) == set(ids[2])
    ):
        raise BaselineConfigurationError("Golden IDs are not aligned and unique")
    documents = list(corpus.get("documents") or [])
    allowed = {str(x.get("document_id")) for x in documents}
    if corpus.get("document_count") != 8 or len(documents) != 8 or len(allowed) != 8:
        raise BaselineConfigurationError(
            "Benchmark corpus must contain exactly eight documents"
        )
    for question in questions:
        scope = {str(x) for x in question.get("document_scope", [])}
        if not scope or not scope.issubset(allowed):
            raise BaselineConfigurationError(
                f"{question['case_id']}: out-of-scope document"
            )
    for label in labels:
        if label.get("expected_no_answer"):
            continue
        for source in label.get("expected_sources") or []:
            if not (
                source.get("source_verified")
                and source.get("candidate_identity_status") == "bound"
                and source.get("candidate_key")
                and source.get("evidence_id")
            ):
                raise BaselineConfigurationError(
                    f"{label['case_id']}: incomplete Source Identity"
                )
    source_records = _source_records(labels)
    negative = (
        json.loads(negative_report_path.read_text(encoding="utf-8")).get("cases")
        if negative_report_path.is_file()
        else None
    )
    if (
        len(source_records) != 80
        or not isinstance(negative, list)
        or len(negative) != 8
    ):
        raise BaselineConfigurationError(
            "Expected 80 sources and 8 negative-evidence records"
        )
    actual = {
        "question_hash": stable_json_hash(
            sorted(
                (_question_hash_payload(x) for x in questions),
                key=lambda x: x["case_id"],
            )
        ),
        "reference_answer_hash": stable_json_hash(
            sorted(
                (_answer_hash_payload(x) for x in labels), key=lambda x: x["case_id"]
            )
        ),
        "source_identity_hash": stable_json_hash(source_records),
        "negative_evidence_hash": stable_json_hash(negative),
        "review_status_hash": stable_json_hash(reviews),
        "corpus_hash": corpus_hash(documents),
        "golden_manifest_sha256": compute_file_sha256(manifest_path) or "",
    }
    matches = {}
    for key, value in actual.items():
        matches[key] = (
            bool(value)
            if key == "golden_manifest_sha256"
            else manifest.get(key) == value
        )
        if not matches[key]:
            raise BaselineConfigurationError(
                f"Golden {key} mismatch: {manifest.get(key)} != {value}"
            )
    return GoldenInputs(
        tuple(questions),
        {str(x["case_id"]): x for x in labels},
        corpus,
        manifest,
        {
            "expected": {
                k: manifest.get(k) for k in actual if k != "golden_manifest_sha256"
            },
            "actual": actual,
            "matches": matches,
        },
    )


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _numbers(text: str) -> list[NumberToken]:
    out = []
    for match in NUM_RE.finditer(text or ""):
        raw = match.group(0).strip()
        clean = (
            raw.strip("() ").replace("$", "").replace(",", "").replace("%", "").strip()
        )
        try:
            value = Decimal(clean)
        except (InvalidOperation, ValueError):
            continue
        if raw.startswith("(") and raw.endswith(")"):
            value = -value
        scale_match = SCALE_RE.search((text or "")[match.end() : match.end() + 24])
        before = (text or "")[max(0, match.start() - 12) : match.start()]
        after = (text or "")[match.end() : match.end() + 12]
        context = before + raw + after
        if re.search(r"(?:\bUSD\b|\bUS\$|\$)", context, re.I):
            currency = "USD"
        elif re.search(r"(?:\bEUR\b|€)", context, re.I):
            currency = "EUR"
        elif re.search(r"(?:\bGBP\b|£)", context, re.I):
            currency = "GBP"
        elif re.search(r"(?:\bJPY\b|¥)", context, re.I):
            currency = "JPY"
        else:
            currency = None
        out.append(
            NumberToken(
                value,
                "%" in raw,
                scale_match.group(1).casefold() if scale_match else None,
                currency,
            )
        )
    return out


def _period_ok(text: str, period: Any) -> bool:
    if period is None:
        return True
    years = list(dict.fromkeys(YEAR_RE.findall(str(period))))
    return (
        all(year in text for year in years) if years else (_norm(period) in _norm(text))
    )


def _metric_ok(text: str, metric: Any) -> bool:
    words = [
        x
        for x in re.findall(r"[a-z0-9]+", _norm(metric))
        if len(x) >= 3
        and x
        not in {
            "and",
            "the",
            "for",
            "from",
            "net",
            "total",
            "income",
            "revenue",
            "revenues",
        }
    ]
    return all(re.search(rf"\b{re.escape(word)}\w*\b", text, re.I) for word in words)


def _component_score(
    text: str, expected: Mapping[str, Any], used: set[int]
) -> dict[str, Any]:
    if expected.get("canonical_value") is None:
        return {
            x: False
            for x in (
                "value_correct",
                "currency_correct",
                "unit_correct",
                "scale_correct",
                "period_correct",
                "component_assignment_correct",
            )
        } | {"token_index": None}
    try:
        target = Decimal(str(expected["canonical_value"]))
    except (InvalidOperation, ValueError):
        return {
            x: False
            for x in (
                "value_correct",
                "currency_correct",
                "unit_correct",
                "scale_correct",
                "period_correct",
                "component_assignment_correct",
            )
        } | {"token_index": None}
    tolerance = Decimal(str(expected.get("tolerance") or "0"))
    display = _norm(expected.get("display_value"))
    display_scale = next(
        (m.group(1).casefold() for m in SCALE_RE.finditer(display)), None
    )
    percentage = (
        _norm(expected.get("unit")) == "percentage"
        or _norm(expected.get("value_type")) == "percentage"
    )
    for index, token in enumerate(_numbers(text)):
        if index in used or (
            percentage
            and not token.percent
            and "%" not in text
            and "percent" not in _norm(text)
            and "margin" not in _norm(text)
        ):
            continue
        actual = token.value * (SCALES[token.scale] if token.scale else Decimal("1"))
        value_ok = abs(actual - target) <= tolerance or bool(
            display and display in _norm(text)
        )
        if not value_ok:
            continue
        used.add(index)
        wanted_currency = str(expected.get("currency") or "").upper()
        currency_ok = (
            not wanted_currency
            or (wanted_currency == "USD" and token.currency == "USD")
            or wanted_currency in text.upper()
        )
        unit = _norm(expected.get("unit"))
        unit_ok = (
            (token.percent or "percent" in _norm(text) or "margin" in _norm(text))
            if unit == "percentage"
            else ("transaction" in _norm(text))
            if unit == "count"
            else (token.currency == "USD" or "$" in text or "usd" in _norm(text))
            if unit in {"currency", "financial_volume", "volume"}
            else True
        )
        return {
            "value_correct": True,
            "currency_correct": currency_ok,
            "unit_correct": unit_ok,
            "scale_correct": token.scale == display_scale if display_scale else True,
            "period_correct": _period_ok(text, expected.get("period")),
            "component_assignment_correct": _metric_ok(text, expected.get("metric")),
            "token_index": index,
        }
    return {
        "value_correct": False,
        "currency_correct": False,
        "unit_correct": False,
        "scale_correct": False,
        "period_correct": _period_ok(text, expected.get("period")),
        "component_assignment_correct": False,
        "token_index": None,
    }


def score_answer_contract(
    text: str, question: Mapping[str, Any], label: Mapping[str, Any]
) -> dict[str, Any]:
    del question
    if label.get("expected_no_answer"):
        ok = (
            bool(_norm(text))
            and any(x in _norm(text) for x in MARKERS)
            and not _numbers(text)
        )
        return {
            "answer_present": bool(text.strip()),
            "value_correct": ok,
            "currency_correct": True,
            "unit_correct": True,
            "scale_correct": True,
            "period_correct": True,
            "component_count_correct": True,
            "component_assignment_correct": True,
            "text_contract_correct": ok,
            "answer_contract_correct": ok,
            "components_matched": 0,
        }
    answer = dict(label.get("expected_answer") or {})
    components = (
        [dict(x) for x in answer.get("component_values", [])]
        if answer.get("value_type") == "composite"
        else [answer]
    )
    if answer.get("canonical_value") is None and not answer.get("component_values"):
        ok = bool(
            _norm(answer.get("text")) and _norm(answer.get("text")) in _norm(text)
        )
        return {
            "answer_present": bool(text.strip()),
            "value_correct": ok,
            "currency_correct": True,
            "unit_correct": True,
            "scale_correct": True,
            "period_correct": True,
            "component_count_correct": True,
            "component_assignment_correct": True,
            "text_contract_correct": ok,
            "answer_contract_correct": ok,
            "components_matched": int(ok),
        }
    used = set()
    parts = [_component_score(text, item, used) for item in components]
    keys = (
        "value_correct",
        "currency_correct",
        "unit_correct",
        "scale_correct",
        "period_correct",
        "component_assignment_correct",
    )
    return {
        "answer_present": bool(text.strip()),
        **{key: all(item[key] for item in parts) for key in keys},
        "component_count_correct": all(item["value_correct"] for item in parts),
        "text_contract_correct": all(item["value_correct"] for item in parts),
        "answer_contract_correct": all(item[key] for item in parts for key in keys),
        "components_matched": sum(int(item["value_correct"]) for item in parts),
    }


def _doc_map(corpus: Mapping[str, Any]) -> dict[str, str]:
    return {
        value: str(item["document_id"])
        for item in corpus.get("documents") or []
        for value in (str(item["document_id"]), str(item["filename"]))
    }


def candidate_identity_from_record(
    candidate: Mapping[str, Any],
    *,
    filename_to_document: Mapping[str, str],
    tenant_id: int,
) -> tuple[str | None, str | None, str | None]:
    raw_document = (
        candidate.get("document_id")
        or candidate.get("filename")
        or candidate.get("doc_name")
    )
    document = filename_to_document.get(
        str(raw_document), str(raw_document) if raw_document else None
    )
    evidence = (
        candidate.get("evidence_id")
        or candidate.get("chunk_id")
        or candidate.get("doc_id")
        or candidate.get("candidate_id")
    )
    block = candidate.get("block_type") or candidate.get("type") or "text"
    if not document or not evidence:
        return None, document, str(evidence) if evidence else None
    # Do not pass explicit None identity fields: the shared identity
    # contract treats a present-but-null row_id as invalid instead of falling
    # back to the stable evidence id for a table row. A table cell still
    # requires an actual parent row and therefore fails closed when absent.
    payload = {
        "tenant_id": tenant_id,
        "document_id": document,
        "block_type": block,
        "evidence_id": str(evidence),
        "candidate_id": str(evidence),
        "doc_id": str(evidence),
        "metadata": {"user_id": tenant_id, "doc_name": document, "type": block},
        "collection_id": candidate.get("collection_id"),
    }
    row_id = candidate.get("row_id")
    parent_row_id = candidate.get("parent_row_id") or candidate.get("parent_id")
    if row_id:
        payload["row_id"] = row_id
    if parent_row_id:
        payload["parent_row_id"] = parent_row_id
    try:
        return (
            stable_candidate_key(identity_from_candidate(payload)),
            document,
            str(evidence),
        )
    except (CandidateIdentityError, TypeError, ValueError):
        return None, document, str(evidence)


def _annotate(
    candidate: Mapping[str, Any], *, mapping: Mapping[str, str], tenant_id: int
) -> dict[str, Any]:
    result = dict(candidate)
    key, document, evidence = candidate_identity_from_record(
        candidate, filename_to_document=mapping, tenant_id=tenant_id
    )
    if key:
        result["candidate_key"] = key
    if document:
        result["canonical_document_id"] = document
    if evidence:
        result["evidence_id"] = evidence
    return result


def source_identity_matches(
    expected: Mapping[str, Any], candidate: Mapping[str, Any]
) -> bool:
    if expected.get("candidate_key"):
        return (
            bool(candidate.get("candidate_key"))
            and candidate["candidate_key"] == expected["candidate_key"]
        )
    if expected.get("evidence_id"):
        return (
            bool(candidate.get("evidence_id"))
            and candidate["evidence_id"] == expected["evidence_id"]
        )
    return bool(
        expected.get("filename")
        and expected["filename"]
        == (candidate.get("filename") or candidate.get("document_id"))
        and expected.get("page") is not None
        and str(expected["page"]) == str(candidate.get("page"))
    )


def _matched_sources(
    expected: Iterable[Mapping[str, Any]], candidates: Iterable[Mapping[str, Any]]
) -> tuple[int, int]:
    count = fallback = 0
    for source in expected:
        for candidate in candidates:
            if source_identity_matches(source, candidate):
                count += 1
                fallback += int(
                    not source.get("candidate_key") and not source.get("evidence_id")
                )
                break
    return count, fallback


def _stage_metrics(
    labels: list[Mapping[str, Any]],
    rankings: Mapping[str, list[Mapping[str, Any]]],
    k: int,
) -> dict[str, Any]:
    eligible = [
        x
        for x in labels
        if not x.get("expected_no_answer") and x.get("expected_sources")
    ]
    source_total = sum(len(x.get("expected_sources") or []) for x in eligible)
    multi = [x for x in eligible if len(x.get("expected_sources") or []) > 1]
    cases = sources = all_hits = 0
    ranks, fallbacks = [], 0
    for label in eligible:
        expected = list(label.get("expected_sources") or [])
        ranked = list(rankings.get(str(label["case_id"]), []))[:k]
        matched, fallback = _matched_sources(expected, ranked)
        cases += int(matched > 0)
        sources += matched
        fallbacks += fallback
        all_hits += int(len(expected) > 1 and matched == len(expected))
        rank = next(
            (
                i
                for i, c in enumerate(ranked, 1)
                if any(source_identity_matches(s, c) for s in expected)
            ),
            None,
        )
        ranks.append(1 / rank if rank else 0)
    return {
        "k": k,
        "case_hit": {
            "count": cases,
            "denominator": len(eligible),
            "rate": cases / len(eligible) if eligible else 1.0,
        },
        "source_recall": {
            "count": sources,
            "denominator": source_total,
            "rate": sources / source_total if source_total else 1.0,
        },
        "multi_evidence_all_source_coverage": {
            "count": all_hits,
            "denominator": len(multi),
            "rate": all_hits / len(multi) if multi else 1.0,
        },
        "mrr": sum(ranks) / len(ranks) if ranks else 1.0,
        "page_fallback_count": fallbacks,
    }


def citation_breakdown(
    expected: Iterable[Mapping[str, Any]], emitted: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    expected, emitted = list(expected), list(emitted)
    if not expected:
        return {
            "expected_source_count": 0,
            "matched_expected_source_count": 0,
            "emitted_citation_count": len(emitted),
            "correct_emitted_citation_count": 0,
            "citation_recall": 1.0,
            "citation_precision": 1.0 if not emitted else 0.0,
            "citation_full_recall": True,
            "citation_perfect_precision": not emitted,
            "page_fallback_count": 0,
        }
    matched = fallback = 0
    for source in expected:
        for candidate in emitted:
            if source_identity_matches(source, candidate):
                matched += 1
                fallback += int(
                    not source.get("candidate_key") and not source.get("evidence_id")
                )
                break
    correct = sum(
        int(any(source_identity_matches(s, c) for s in expected)) for c in emitted
    )
    recall = matched / len(expected)
    return {
        "expected_source_count": len(expected),
        "matched_expected_source_count": matched,
        "emitted_citation_count": len(emitted),
        "correct_emitted_citation_count": correct,
        "citation_recall": recall,
        "citation_precision": correct / len(emitted) if emitted else 0.0,
        "citation_full_recall": recall == 1.0,
        "citation_perfect_precision": (correct / len(emitted) if emitted else 0.0)
        == 1.0,
        "page_fallback_count": fallback,
    }


def _repair_flags(result: Mapping[str, Any]) -> dict[str, bool]:
    repair = result.get("repair") if isinstance(result.get("repair"), Mapping) else {}
    applied, failed = (
        bool(repair.get("was_repaired")),
        bool(repair.get("fallback_used")),
    )
    return {
        "repair_attempted": applied or failed,
        "repair_applied": applied,
        "repair_succeeded": applied,
        "repair_failed": failed,
    }


def _mode(
    answer_result: Any | None,
    result: Mapping[str, Any],
    requests: int,
    no_answer: bool,
    error: str | None,
) -> str:
    if error:
        return "failed"
    if str((result.get("validation") or {}).get("status") or "").casefold() in {
        "blocked",
        "failed",
    }:
        return "blocked"
    if requests:
        return "llm_generation"
    if answer_result is not None and getattr(answer_result, "calculations", ()):
        return "deterministic_calculation"
    if no_answer or str(
        (result.get("answerability") or {}).get("status") or ""
    ).casefold() in {"not_answerable", "insufficient_context"}:
        return "safe_response"
    if getattr(answer_result, "path", None) in {
        AnswerPath.CONVERSATIONAL,
        AnswerPath.NO_RETRIEVAL,
        AnswerPath.NO_DOCUMENTS,
    }:
        return "safe_response"
    return "deterministic_fact"


def _metric(records: list[dict[str, Any]], released: bool) -> dict[str, Any]:
    prefix = "released" if released else "raw"
    rows = records if released else [x for x in records if x["raw_available"]]
    answerable = [x for x in rows if not x["expected_no_answer"]]
    no_answer = [x for x in rows if x["expected_no_answer"]]

    def count(name: str, subset: list[dict[str, Any]] = answerable) -> int:
        return sum(int(x.get(f"{prefix}_{name}", False)) for x in subset)

    denom, source_total = (
        len(answerable),
        sum(x["expected_source_count"] for x in answerable),
    )
    matched, emitted = (
        sum(x["matched_expected_source_count"] for x in answerable),
        sum(x["emitted_citation_count"] for x in answerable),
    )
    correct_emitted = sum(x["correct_emitted_citation_count"] for x in answerable)
    return {
        "case_denominator": len(rows),
        "answerable_case_denominator": denom,
        "raw_available_denominator": sum(int(x["raw_available"]) for x in records)
        if not released
        else None,
        "answer_value_pass": {
            "count": count("value_correct"),
            "denominator": denom,
            "rate": count("value_correct") / denom if denom else 1.0,
        },
        "answer_contract_pass": {
            "count": count("answer_contract_correct"),
            "denominator": denom,
            "rate": count("answer_contract_correct") / denom if denom else 1.0,
        },
        "grounded_pass": {
            "count": count("grounded_pass"),
            "denominator": denom,
            "rate": count("grounded_pass") / denom if denom else 1.0,
        },
        "value_accuracy": {
            "count": count("value_correct"),
            "denominator": denom,
            "rate": count("value_correct") / denom if denom else 1.0,
        },
        "currency_accuracy": {
            "count": count("currency_correct"),
            "denominator": denom,
            "rate": count("currency_correct") / denom if denom else 1.0,
        },
        "unit_accuracy": {
            "count": count("unit_correct"),
            "denominator": denom,
            "rate": count("unit_correct") / denom if denom else 1.0,
        },
        "scale_accuracy": {
            "count": count("scale_correct"),
            "denominator": denom,
            "rate": count("scale_correct") / denom if denom else 1.0,
        },
        "period_accuracy": {
            "count": count("period_correct"),
            "denominator": denom,
            "rate": count("period_correct") / denom if denom else 1.0,
        },
        "component_assignment_accuracy": {
            "count": count("component_assignment_correct"),
            "denominator": denom,
            "rate": count("component_assignment_correct") / denom if denom else 1.0,
        },
        "micro_source_recall": matched / source_total if source_total else 1.0,
        "micro_citation_precision": correct_emitted / emitted if emitted else 1.0,
        "macro_citation_recall": statistics.mean(
            x[f"{prefix}_citation_recall"] for x in answerable
        )
        if answerable
        else 1.0,
        "macro_citation_precision": statistics.mean(
            x[f"{prefix}_citation_precision"] for x in answerable
        )
        if answerable
        else 1.0,
        "full_citation_recall_case_rate": sum(
            int(x[f"{prefix}_citation_full_recall"]) for x in answerable
        )
        / denom
        if denom
        else 1.0,
        "perfect_citation_precision_case_rate": sum(
            int(x[f"{prefix}_citation_perfect_precision"]) for x in answerable
        )
        / denom
        if denom
        else 1.0,
        "no_answer_accuracy": {
            "count": count("value_correct", no_answer),
            "denominator": len(no_answer),
            "rate": count("value_correct", no_answer) / len(no_answer)
            if no_answer
            else 1.0,
        },
    }


def _bucket(question: Mapping[str, Any]) -> str:
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


async def _run_queries(
    args: argparse.Namespace, inputs: GoldenInputs
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    engine, client = _build_engine(args)
    mapping, by_doc = (
        _doc_map(inputs.corpus),
        {str(x["document_id"]): str(x["filename"]) for x in inputs.corpus["documents"]},
    )
    records = []
    for index, question in enumerate(inputs.questions, 1):
        label = inputs.labels_by_id[str(question["case_id"])]
        trace = AnswerPipelineTrace(
            case_id=str(question["case_id"]),
            trace_id=uuid.uuid4().hex,
            context_hash="",
            context_coverage="not_evaluated",
        )
        request = QueryRequest(
            question=str(question["question"]),
            document_names=tuple(
                by_doc[str(x)] for x in question.get("document_scope", [])
            ),
            user_id=args.tenant_id,
            conversation_history=(),
            memory_profile=None,
        )
        before, error, result, answer_result = (
            client.chat_completion_requests,
            None,
            {},
            None,
        )
        started = time.perf_counter()
        try:
            answer_result = await engine._orchestrator.answer(
                request, n_results=args.n_results, evaluation_observer=trace
            )
            result = answer_result.to_legacy_dict()
        except Exception as exc:  # preserve per-case failure
            error = type(exc).__name__
        requests, latency = (
            client.chat_completion_requests - before,
            (time.perf_counter() - started) * 1000,
        )
        raw, released, raw_available = (
            trace._raw_generation_text or "",
            str(result.get("answer") or ""),
            bool(trace.raw_generation_hash),
        )
        raw_score = (
            score_answer_contract(raw, question, label)
            if raw_available
            else {
                x: False
                for x in (
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
        )
        released_score = score_answer_contract(released, question, label)
        debug = result.get("retrieval_debug") or {}
        stages, missing, total = {}, 0, 0
        for stage, values in (debug.get("candidate_stages") or {}).items():
            stages[stage] = []
            for candidate in values or []:
                item = _annotate(candidate, mapping=mapping, tenant_id=args.tenant_id)
                stages[stage].append(item)
                total += 1
                missing += int(not item.get("candidate_key"))
        emitted, citation_missing = [], 0
        for source in result.get("sources") or []:
            raw_source = dict(source)
            raw_source.update(
                {
                    "document_id": source.get("filename") or source.get("document_id"),
                    "evidence_id": source.get("chunk_id") or source.get("evidence_id"),
                    "block_type": source.get("type")
                    or source.get("block_type")
                    or "text",
                }
            )
            key, document, evidence = candidate_identity_from_record(
                raw_source, filename_to_document=mapping, tenant_id=args.tenant_id
            )
            item = dict(source)
            item.update(
                {
                    "candidate_key": key,
                    "canonical_document_id": document,
                    "evidence_id": evidence,
                }
            )
            emitted.append(item)
            citation_missing += int(not key)
        expected = list(label.get("expected_sources") or [])
        citation = citation_breakdown(expected, emitted)
        scope_ids = {str(value) for value in question.get("document_scope", [])}
        scope_out = sum(
            int(item.get("canonical_document_id") not in scope_ids)
            for values in stages.values()
            for item in values
            if item.get("canonical_document_id")
        )
        citation_out = sum(
            int(item.get("canonical_document_id") not in scope_ids)
            for item in emitted
            if item.get("canonical_document_id")
        )
        mode, repair = (
            _mode(
                answer_result,
                result,
                requests,
                bool(label.get("expected_no_answer")),
                error,
            ),
            _repair_flags(result),
        )
        record = {
            "case_id": str(question["case_id"]),
            "company": question.get("company"),
            "bucket": _bucket(question),
            "document_scope": list(question.get("document_scope") or []),
            "expected_no_answer": bool(label.get("expected_no_answer")),
            "expected_source_count": len(expected),
            "matched_expected_source_count": citation["matched_expected_source_count"],
            "emitted_citation_count": citation["emitted_citation_count"],
            "correct_emitted_citation_count": citation[
                "correct_emitted_citation_count"
            ],
            "raw_available": raw_available,
            "raw_generation_hash": trace.raw_generation_hash,
            "released_answer_hash": trace.released_answer_hash,
            "answer_execution_mode": mode,
            "model_invoked": bool(requests),
            "model_request_count": requests,
            "validation_status": (result.get("validation") or {}).get("status"),
            "stage_candidate_count": total,
            "missing_candidate_identity_count": missing,
            "missing_citation_identity_count": citation_missing,
            "page_fallback_count": citation["page_fallback_count"],
            "retrieval_debug": {
                "candidate_stages": _sanitize_retrieval_debug(stages),
                "candidate_count": debug.get("candidate_count"),
                "candidate_multiplier": debug.get("candidate_multiplier"),
            },
            "sources": _sanitize_retrieval_debug({"sources": emitted}).get(
                "sources", []
            ),
            "scope_out_of_scope_count": scope_out,
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
            record[f"{prefix}_citation_recall"] = (
                citation["citation_recall"]
                if prefix == "released" or raw_available
                else None
            )
            record[f"{prefix}_citation_precision"] = (
                citation["citation_precision"]
                if prefix == "released" or raw_available
                else None
            )
            record[f"{prefix}_citation_full_recall"] = (
                bool(citation["citation_full_recall"])
                if prefix == "released" or raw_available
                else False
            )
            record[f"{prefix}_citation_perfect_precision"] = (
                bool(citation["citation_perfect_precision"])
                if prefix == "released" or raw_available
                else False
            )
            record[f"{prefix}_grounded_pass"] = (
                bool(
                    score["answer_contract_correct"]
                    and citation["citation_full_recall"]
                )
                if prefix == "released" or raw_available
                else False
            )
        records.append(record)
        print(
            f"[{index}/{len(inputs.questions)}] {question['case_id']} mode={mode} requests={requests} {latency:.0f}ms",
            flush=True,
        )
    return records, {
        "model_chat_completion_requests": client.chat_completion_requests,
        "case_count": len(records),
    }


def _report(
    args: argparse.Namespace,
    inputs: GoldenInputs,
    records: list[dict[str, Any]],
    run_info: dict[str, Any],
) -> dict[str, Any]:
    labels = [inputs.labels_by_id[x["case_id"]] for x in records]
    stage_names = ("dense", "bm25", "rrf", "reranker", "final")
    rankings = {
        name: {
            x["case_id"]: list(x["retrieval_debug"]["candidate_stages"].get(name) or [])
            for x in records
        }
        for name in stage_names
    }
    retrieval = {
        name: {
            f"recall_at_{k}": _stage_metrics(labels, rankings[name], k)
            for k in (
                (5, 20, 40)
                if name in {"dense", "bm25", "rrf"}
                else (5, 20)
                if name == "reranker"
                else (5,)
            )
        }
        for name in stage_names
    }
    retrieval["definitions"] = {
        "explicit_multi_source_case_count": sum(
            x["bucket"] == "multi_source" for x in records
        ),
        "calculation_multi_evidence_case_count": sum(
            x["bucket"] == "calculation" and x["expected_source_count"] > 1
            for x in records
        ),
        "all_multi_evidence_case_count": sum(
            x["expected_source_count"] > 1 for x in records
        ),
    }
    stage_identity = {
        name: {
            "candidate_count": sum(
                len(x["retrieval_debug"]["candidate_stages"].get(name) or [])
                for x in records
            ),
            "missing_candidate_identity_count": sum(
                sum(
                    not c.get("candidate_key")
                    for c in (x["retrieval_debug"]["candidate_stages"].get(name) or [])
                )
                for x in records
            ),
        }
        for name in stage_names
    }
    identity = {
        "stage_counts": stage_identity,
        "missing_candidate_identity_count": sum(
            x["missing_candidate_identity_count"] for x in records
        ),
        "missing_citation_identity_count": sum(
            x["missing_citation_identity_count"] for x in records
        ),
        "page_fallback_count": sum(x["page_fallback_count"] for x in records),
        "expected_bound_source_count": 80,
    }
    identity["identity_observability_passed"] = (
        all(x["missing_candidate_identity_count"] == 0 for x in stage_identity.values())
        and identity["missing_citation_identity_count"] == 0
        and identity["page_fallback_count"] == 0
    )
    answer = {
        "raw": _metric(records, False),
        "released": _metric(records, True),
        "raw_available_count": sum(x["raw_available"] for x in records),
        "raw_unavailable_count": sum(not x["raw_available"] for x in records),
    }
    execution = {
        "execution_mode_counts": dict(
            Counter(x["answer_execution_mode"] for x in records)
        ),
        "model_invoked_case_count": sum(x["model_invoked"] for x in records),
        "model_invoked_case_ids": [x["case_id"] for x in records if x["model_invoked"]],
        "model_chat_completion_requests": run_info["model_chat_completion_requests"],
    }
    repair = {
        key: sum(x[field] for x in records)
        for key, field in {
            "repair_attempted_count": "repair_attempted",
            "repair_applied_count": "repair_applied",
            "repair_succeeded_count": "repair_succeeded",
            "repair_failed_count": "repair_failed",
        }.items()
    }
    latencies = [x["latency_ms"] for x in records]
    manifest = {
        "artifact_schema": "nf-eval-03-r1/baseline/v1",
        "benchmark_id": "financial-rag-v1",
        "case_count": len(records),
        "answerable_count": sum(not x["expected_no_answer"] for x in records),
        "no_answer_count": sum(x["expected_no_answer"] for x in records),
        "tenant_id": args.tenant_id,
        "allowed_document_ids_hash": stable_json_hash(
            sorted(x["document_id"] for x in inputs.corpus["documents"])
        ),
        **inputs.hash_report["actual"],
        "embedding_model": get_embedding_model_name(),
        "reranker": get_reranker_name(),
        "reranker_model": get_reranker_model(),
        "generator_model": args.model_name,
        "generator_endpoint": args.model_base_url,
        "n_results": args.n_results,
        "retrieval_candidate_multiplier": args.retrieval_candidate_multiplier,
        "candidate_identity_observability_passed": identity[
            "identity_observability_passed"
        ],
        "production_behavior_changed": False,
        "legacy_27_included": False,
    }
    scope = {
        "allowed_document_count": 8,
        "retrieved_out_of_scope_candidates": sum(
            x["scope_out_of_scope_count"] for x in records
        ),
        "reranked_out_of_scope_candidates": sum(
            sum(
                int(item.get("canonical_document_id") not in set(x["document_scope"]))
                for item in (
                    x["retrieval_debug"]["candidate_stages"].get("reranker") or []
                )
                if item.get("canonical_document_id")
            )
            for x in records
        ),
        "final_context_out_of_scope_candidates": sum(
            sum(
                int(item.get("canonical_document_id") not in set(x["document_scope"]))
                for item in (
                    x["retrieval_debug"]["candidate_stages"].get("final") or []
                )
                if item.get("canonical_document_id")
            )
            for x in records
        ),
        "citation_out_of_scope_count": sum(
            x["citation_out_of_scope_count"] for x in records
        ),
        "scope_integrity_passed": all(
            not x["scope_out_of_scope_count"] and not x["citation_out_of_scope_count"]
            for x in records
        ),
        "legacy_27_included": False,
    }
    slices = {
        field: {
            value: {
                "raw": _metric(
                    [x for x in records if str(x.get(field)) == value], False
                ),
                "released": _metric(
                    [x for x in records if str(x.get(field)) == value], True
                ),
            }
            for value in sorted({str(x.get(field)) for x in records})
        }
        for field in ("company", "bucket")
    }
    citation_keys = (
        "micro_source_recall",
        "macro_citation_recall",
        "full_citation_recall_case_rate",
        "micro_citation_precision",
        "macro_citation_precision",
        "perfect_citation_precision_case_rate",
    )
    citation = {
        prefix: {key: answer[prefix][key] for key in citation_keys}
        for prefix in ("raw", "released")
    }
    return {
        "baseline_manifest": manifest,
        "input_integrity": inputs.hash_report,
        "retrieval_identity": {"stages": retrieval, "identity_observability": identity},
        "answer_contract": answer,
        "citation": citation,
        "grounded_pass": {
            "raw": answer["raw"]["grounded_pass"],
            "released": answer["released"]["grounded_pass"],
        },
        "execution_mode": execution,
        "repair": repair,
        "slice": slices,
        "scope": scope,
        "latency": {
            "count": len(latencies),
            "p50_ms": _percentile(latencies, 0.5),
            "p95_ms": _percentile(latencies, 0.95),
            "mean_ms": statistics.mean(latencies) if latencies else None,
            "max_ms": max(latencies) if latencies else None,
        },
        "records": records,
        "errors": [x["case_id"] for x in records if x["error"]],
        "run_info": run_info,
    }


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    parser.add_argument(
        "--manifest",
        dest="manifest_path",
        type=Path,
        default=DATA / "golden-manifest.json",
    )
    parser.add_argument(
        "--questions", type=Path, default=DATA / "questions.golden.jsonl"
    )
    parser.add_argument("--labels", type=Path, default=DATA / "labels.golden.jsonl")
    parser.add_argument(
        "--review-status", type=Path, default=DATA / "review-status.golden.jsonl"
    )
    parser.add_argument("--negative-report", type=Path, default=DEFAULT_NEGATIVE)
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> int:
    try:
        inputs = _load_inputs(
            corpus_path=args.corpus,
            manifest_path=args.manifest_path,
            questions_path=args.questions,
            labels_path=args.labels,
            review_status_path=args.review_status,
            negative_report_path=args.negative_report,
        )
    except BaselineConfigurationError as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _write(
            args.out_dir / "nf-eval-03-r1-acceptance.json",
            {
                "artifact_schema": "nf-eval-03-r1/baseline/v1",
                "baseline_run_completed": False,
                "decision": "baseline_input_integrity_failed",
                "metric_semantics_verified": False,
                "production_behavior_changed": False,
                "error": str(exc),
            },
        )
        print(f"NF-EVAL-03 R1 configuration error: {exc}", file=sys.stderr)
        return 2
    records, run_info = await _run_queries(args, inputs)
    report = _report(args, inputs, records, run_info)
    out = args.out_dir
    for name, value in {
        "baseline-manifest.json": report["baseline_manifest"],
        "input-integrity-report.json": report["input_integrity"],
        "execution-mode-report.json": report["execution_mode"],
        "retrieval-identity-metrics.json": report["retrieval_identity"],
        "answer-contract-metrics.json": report["answer_contract"],
        "citation-metrics.json": report["citation"],
        "grounded-pass-metrics.json": report["grounded_pass"],
        "repair-metrics.json": report["repair"],
        "slice-metrics.json": report["slice"],
        "scope-integrity-report.json": report["scope"],
        "latency-report.json": report["latency"],
        "case-results.json": {
            "artifact_schema": "nf-eval-03-r1/baseline/v1",
            "cases": report["records"],
        },
    }.items():
        _write(out / name, value)
    passed = (
        len(records) == 72
        and not report["errors"]
        and all(report["input_integrity"]["matches"].values())
        and report["retrieval_identity"]["identity_observability"][
            "identity_observability_passed"
        ]
        and report["retrieval_identity"]["identity_observability"][
            "page_fallback_count"
        ]
        == 0
        and report["scope"]["scope_integrity_passed"]
    )
    acceptance = {
        "artifact_schema": "nf-eval-03-r1/baseline/v1",
        "benchmark_id": "financial-rag-v1",
        "case_count": len(records),
        "baseline_run_completed": len(records) == 72 and not report["errors"],
        "input_integrity_passed": all(report["input_integrity"]["matches"].values()),
        "identity_observability_passed": report["retrieval_identity"][
            "identity_observability"
        ]["identity_observability_passed"],
        "page_fallback_count": report["retrieval_identity"]["identity_observability"][
            "page_fallback_count"
        ],
        "scope_integrity_passed": report["scope"]["scope_integrity_passed"],
        "metric_semantics_verified": passed,
        "production_behavior_changed": False,
        "legacy_27_included": False,
        "model_chat_completion_requests": run_info["model_chat_completion_requests"],
        "execution_mode_counts": report["execution_mode"]["execution_mode_counts"],
        "decision": "formal_baseline_recorded"
        if passed
        else "baseline_semantics_incomplete",
    }
    _write(out / "nf-eval-03-r1-acceptance.json", acceptance)
    old = (
        ROOT / "artifacts" / "evaluation" / "nf-eval-03" / "nf-eval-03-acceptance.json"
    )
    if old.is_file():
        old_value = json.loads(old.read_text(encoding="utf-8"))
        old_value.update(
            {
                "status": "superseded_due_to_metric_semantics",
                "superseded_by": "nf-eval-03-r1",
            }
        )
        _write(old, old_value)
    print(
        json.dumps(
            {
                "acceptance": acceptance,
                "execution_mode": report["execution_mode"],
                "answer_contract": report["answer_contract"],
                "retrieval": report["retrieval_identity"],
                "latency": report["latency"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 2


def main() -> None:
    raise SystemExit(asyncio.run(_main(_args())))


if __name__ == "__main__":
    main()
