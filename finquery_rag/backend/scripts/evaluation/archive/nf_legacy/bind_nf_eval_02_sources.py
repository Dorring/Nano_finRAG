"""Bind NF-EVAL-02 Expected Sources to verified indexed identities.

This command is intentionally evaluation-only.  It reads the existing BM25
SQLite store in read-only mode and the already-ingested PDFs.  It does not
modify retrieval, Chroma, BM25, documents, or the frozen reference answers.
Bindings are accepted only when a *unique* candidate satisfies document, PDF
page, metric, value, period and unit/scale checks.  Ambiguous and missing
records remain unresolved instead of being guessed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Mapping

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from scripts.evaluation.benchmark_foundation import (  # noqa: E402
    load_json,
    load_jsonl,
    stable_json_hash,
    write_jsonl,
)
from scripts.evaluation.benchmark_scope import benchmark_document_ids  # noqa: E402
from scripts.evaluation.freeze_financial_rag_reference_answers import (  # noqa: E402
    _answer_hash_payload,
    _question_hash_payload,
)
from src.evaluation.benchmark_source_binding import (  # noqa: E402
    BindingDecision,
    binding_passes,
    choose_candidate,
    golden_promotion_gate,
    negative_review_passes,
    source_expected_metric,
)
from src.retrieval.candidate_identity import candidate_key, identity_from_candidate  # noqa: E402


SCHEMA = "nf-eval-02/golden-evidence-binding/v1"
REVIEWER = "codex_pdf_review"


def _content_hash(content: str) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_pdf_text(path: Path) -> str:
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError("PyMuPDF is required for negative-evidence review") from exc
    document = fitz.open(path)
    try:
        return "\n".join(page.get_text("text") for page in document)
    finally:
        document.close()


def _pdf_search(path: Path, terms: list[str]) -> tuple[dict[str, int], int]:
    text = _load_pdf_text(path)
    lowered = text.casefold()
    hits = {term: lowered.count(term.casefold()) for term in terms if term}
    return hits, sum(hits.values())


def _negative_terms(label: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    review = label.get("no_answer_review") or {}
    original = [str(value) for value in review.get("searched_terms", []) if value]
    concept = original[:1]
    additions = {
        "internal model accuracy": [
            "model accuracy", "accuracy metric", "model performance", "artificial intelligence accuracy",
        ],
        "customer contract amount": [
            "client contract amount", "customer contract value", "individual client", "contract value",
        ],
        "plant level output": [
            "plant output", "production volume", "named plant", "facility output", "plant capacity",
        ],
        "customer renewal rate": [
            "contract renewal", "customer retention", "renewal rate", "customer churn",
        ],
        "customer level revenue": [
            "customer concentration", "individual customer revenue", "customer revenue", "data center customer",
        ],
        "future financial guidance": [
            "financial guidance", "2026 guidance", "fiscal 2026 outlook", "projected 2026 results", "forecast",
        ],
        "customer specific future price": [
            "customer purchase price", "future purchase price", "guaranteed purchase price", "2026 price",
        ],
        "internal employee metric": [
            "employee performance metric", "employee accuracy", "internal performance metric", "employee score",
        ],
    }
    key = concept[0] if concept else ""
    synonyms = list(additions.get(key, []))
    return list(dict.fromkeys(original)), list(dict.fromkeys(synonyms))


def _source_value_label(label: Mapping[str, Any], source_index: int) -> str:
    calculation = label.get("calculation")
    if isinstance(calculation, Mapping):
        operands = calculation.get("operands")
        if isinstance(operands, list) and source_index < len(operands):
            return str(operands[source_index].get("value") or "")
    answer = label.get("expected_answer") or {}
    components = answer.get("component_values")
    if isinstance(components, list) and source_index < len(components):
        return str(components[source_index].get("canonical_value") or "")
    return str(answer.get("canonical_value") or "")


def _candidate_record(
    *,
    doc_id: str,
    content: str,
    metadata: Mapping[str, Any],
    tenant_id: int,
    document_id: str,
    filename: str,
) -> dict[str, Any] | None:
    block_type = str(metadata.get("type") or "text")
    if block_type == "table_cell":
        # Cells are not global candidates; source identity is the parent row.
        return None
    item: dict[str, Any] = {
        "doc_id": doc_id,
        "evidence_id": doc_id,
        "candidate_id": doc_id,
        "content": content,
        "metadata": dict(metadata),
        "tenant_id": tenant_id,
        "document_id": document_id,
        "benchmark_document_id": document_id,
        "filename": filename,
        "page": metadata.get("page"),
        "block_type": block_type,
        "parent_row_id": metadata.get("parent_row_id"),
        "content_hash": _content_hash(content),
    }
    try:
        identity = identity_from_candidate(item)
    except Exception:
        return None
    item["candidate_key"] = candidate_key(identity)
    item["identity"] = {
        "schema_version": identity.schema_version,
        "tenant_id": identity.tenant_id,
        "document_id": identity.document_id,
        "kind": identity.kind.value,
        "source_id": identity.source_id,
        "collection_id": identity.collection_id,
    }
    return item


def _load_index(
    *,
    db_path: Path,
    tenant_id: int,
    documents: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    by_filename = {str(item["filename"]): str(item["document_id"]) for item in documents}
    by_key: dict[str, dict[str, Any]] = {}
    connection = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT doc_id, content, metadata_json, user_id, doc_name "
            "FROM chunk_store WHERE user_id = ?",
            (tenant_id,),
        )
        for doc_id, content, metadata_json, user_id, doc_name in rows:
            if int(user_id) != tenant_id or str(doc_name) not in by_filename:
                continue
            metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else {}
            record = _candidate_record(
                doc_id=str(doc_id), content=str(content or ""), metadata=metadata,
                tenant_id=tenant_id, document_id=by_filename[str(doc_name)], filename=str(doc_name),
            )
            if record is None:
                continue
            by_key.setdefault(str(record["candidate_key"]), record)
    finally:
        connection.close()
    records = list(by_key.values())
    # Some layout-extracted rows point at a synthetic parent that is not
    # stored in BM25.  Make a non-candidate page table lookup available for
    # header/unit checks without adding it to the searchable candidate list.
    for record in records:
        if record.get("block_type") != "table":
            continue
        page_key = f"__page_table__:{record.get('benchmark_document_id')}:{record.get('page')}"
        by_key.setdefault(page_key, record)
    page_text: dict[tuple[str, Any], list[str]] = {}
    for record in records:
        if record.get("block_type") not in {"text", "table"}:
            continue
        key = (str(record.get("benchmark_document_id")), record.get("page"))
        page_text.setdefault(key, []).append(str(record.get("content") or ""))
    for (document_id, page), contents in page_text.items():
        by_key.setdefault(
            f"__page_text__:{document_id}:{page}",
            {"content": "\n".join(contents), "benchmark_document_id": document_id, "page": page},
        )
    return records, by_key


def _load_top20_keys(candidate_dir: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for path in candidate_dir.glob("*.jsonl"):
        for row in load_jsonl(path):
            case_id = str(row.get("case_id") or "")
            key = str(row.get("candidate_key") or "")
            if case_id and key:
                result.setdefault(case_id, set()).add(key)
    return result


def _identity_granularity(candidate: Mapping[str, Any]) -> str:
    block_type = str(candidate.get("block_type") or "")
    return {"table_row": "table_row", "table": "table_block", "front_matter": "chunk"}.get(block_type, "chunk")


def _binding_method(candidate: Mapping[str, Any], top20_keys: set[str]) -> str:
    if str(candidate.get("candidate_key")) in top20_keys:
        return "top20_exact"
    block_type = str(candidate.get("block_type") or "")
    if block_type == "table_row":
        return "parent_table_row_expansion"
    return "full_index_document_page_search"


def _apply_bound_source(
    source: dict[str, Any],
    decision: BindingDecision,
    *,
    top20_keys: set[str],
    reviewed_at: str,
) -> dict[str, Any]:
    result = deepcopy(source)
    result.update({
        "candidate_identity_status": decision.status,
        "candidate_key": None,
        "evidence_id": None,
        "row_id": None,
        "identity_granularity": None,
        "candidate_document_id": None,
        "candidate_pdf_page": None,
        "candidate_content_hash": None,
        "binding_method": None,
        "binding_confidence": None,
        "document_match": False,
        "page_match": False,
        "metric_match": False,
        "value_match": False,
        "period_match": False,
        "unit_scale_match": False,
        "source_verified": False,
        "reviewer": None,
        "reviewed_at": None,
        "review_notes": None,
    })
    if not binding_passes(decision):
        result["review_notes"] = "; ".join(decision.reasons)
        return result
    candidate = decision.candidate or {}
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), Mapping) else {}
    result.update({
        "candidate_identity_status": "bound",
        "candidate_key": candidate.get("candidate_key"),
        "evidence_id": candidate.get("evidence_id"),
        "row_id": metadata.get("row_id"),
        "identity_granularity": _identity_granularity(candidate),
        "identity_limitation": (
            "row_identity_not_available"
            if _identity_granularity(candidate) == "chunk"
            else None
        ),
        "candidate_document_id": candidate.get("benchmark_document_id"),
        "candidate_pdf_page": candidate.get("page"),
        "candidate_content_hash": candidate.get("content_hash"),
        "binding_method": _binding_method(candidate, top20_keys),
        "binding_confidence": "high",
        "document_match": True,
        "page_match": True,
        "metric_match": True,
        "value_match": True,
        "period_match": True,
        "unit_scale_match": True,
        "source_verified": True,
        "reviewer": REVIEWER,
        "reviewed_at": reviewed_at,
        "review_notes": "Matched exact ingested PDF page and indexed candidate value/metric/period/unit; no out-of-scope document.",
    })
    return result


def _review_negative_case(
    *,
    label: Mapping[str, Any],
    pdf_path: Path,
    reviewed_at: str,
) -> tuple[dict[str, Any], bool]:
    terms, synonyms = _negative_terms(label)
    searchable = list(dict.fromkeys(terms + synonyms))
    hits, _total = _pdf_search(pdf_path, searchable)
    # A generic company name is not evidence of a quantitative disclosure.
    excluded_term = terms[-1].casefold() if terms else None
    concept_hits = {
        term: count
        for term, count in hits.items()
        if term.casefold() != excluded_term
    }
    positive = sum(concept_hits.values())
    review = deepcopy(label.get("no_answer_review") or {})
    review.update({
        "automated_pdf_full_text_scan_completed": True,
        "human_negative_evidence_reviewed": False,
        "negative_evidence_reviewed": False,
        "negative_evidence_review_status": "pending_manual_review" if positive == 0 else "pending_positive_match",
        "searched_terms": terms,
        "searched_synonyms": synonyms,
        "searched_sections": list(dict.fromkeys((review.get("searched_sections") or []) + ["notes", "financial statements", "MD&A", "risk factors"])),
        "searched_table_titles": ["operational highlights", "segment information", "consolidated statements", "selected product revenues"],
        "positive_match_count": positive,
        "positive_match_terms": concept_hits,
        "full_document_search_completed": True,
        "reviewer": None,
        "reviewed_at": None,
        "review_notes": (
            "Automated scan found no concept-term match; explicit manual review log is still required."
            if positive == 0 else "Concept-term matches require manual resolution before this no-answer case can be Golden."
        ),
    })
    return review, False


def _load_negative_review_log(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load explicit human review attestations; absence means pending."""

    if path is None:
        return {}
    records = load_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = str(record.get("case_id") or "")
        if not case_id or case_id in result:
            raise ValueError("negative review log requires unique case_id values")
        result[case_id] = record
    return result


def _hash_frozen_inputs(questions: list[dict[str, Any]], labels: list[dict[str, Any]], corpus: Mapping[str, Any]) -> dict[str, str]:
    return {
        "question_hash": stable_json_hash(sorted((_question_hash_payload(item) for item in questions), key=lambda item: item["case_id"])),
        "reference_answer_hash": stable_json_hash(sorted((_answer_hash_payload(item) for item in labels), key=lambda item: item["case_id"])),
        "corpus_hash": str(corpus.get("corpus_hash") or ""),
    }


def bind_sources(
    *,
    corpus_path: Path,
    questions_path: Path,
    labels_path: Path,
    review_path: Path,
    candidate_dir: Path,
    bm25_db: Path,
    pdf_root: Path,
    out_dir: Path,
    tenant_id: int,
    reviewed_at: str | None = None,
    negative_review_log: Path | None = None,
) -> dict[str, Any]:
    corpus = load_json(corpus_path)
    questions = load_jsonl(questions_path)
    labels = load_jsonl(labels_path)
    reviews = load_jsonl(review_path)
    if len(questions) != 72 or len(labels) != 72 or len(reviews) != 72:
        raise ValueError("NF-EVAL-02 requires exactly 72 aligned cases")
    by_question = {item["case_id"]: item for item in questions}
    by_label = {item["case_id"]: item for item in labels}
    by_review = {item["case_id"]: item for item in reviews}
    if set(by_question) != set(by_label) or set(by_question) != set(by_review):
        raise ValueError("questions, labels and review records are not aligned")
    allowed = benchmark_document_ids(corpus)
    documents = [item for item in corpus["documents"] if item["document_id"] in allowed]
    if tenant_id != int(corpus.get("tenant_id", tenant_id)):
        raise ValueError("tenant does not match corpus")
    frozen_hashes = _hash_frozen_inputs(questions, labels, corpus)
    expected_hashes = {
        "question_hash": "3b1433ea7546020ec7ab90a9cb700bb58d90aa040f65e6610d6bc5b1330e97b7",
        "reference_answer_hash": "181c9ed45ae904353431ef6d427b139e1d5f0b8f437c2fdd9f25e2f9b7abb7a7",
        "corpus_hash": "edd4e8c1aac89ea1480dcb723857b487cb6038218d0a924b749eb6ec8fd13448",
    }
    if frozen_hashes != expected_hashes:
        raise ValueError(f"frozen reference hash mismatch: {frozen_hashes}")

    candidates, by_key = _load_index(db_path=bm25_db, tenant_id=tenant_id, documents=documents)
    by_doc_id = {str(item["doc_id"]): item for item in candidates}
    by_doc_id.update({
        key: value
        for key, value in by_key.items()
        if key.startswith(("__page_table__:", "__page_text__:"))
    })
    top20_by_case = _load_top20_keys(candidate_dir)
    manual_negative_reviews = _load_negative_review_log(negative_review_log)
    updated_labels: list[dict[str, Any]] = []
    updated_reviews: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    binding_methods: dict[str, int] = {}
    status_counts = {"bound": 0, "ambiguous": 0, "missing_from_index": 0, "out_of_scope": 0}
    reviewed_at = reviewed_at or date.today().isoformat()
    negative_records: list[dict[str, Any]] = []
    negative_verified = 0

    for case_id in sorted(by_question):
        question = by_question[case_id]
        label = deepcopy(by_label[case_id])
        review = deepcopy(by_review[case_id])
        if bool(label.get("expected_no_answer")):
            corpus_doc = next(item for item in documents if item["document_id"] in question.get("document_scope", []))
            pdf_path = pdf_root / str(corpus_doc["filename"])
            negative_review, _ = _review_negative_case(label=label, pdf_path=pdf_path, reviewed_at=reviewed_at)
            manual = manual_negative_reviews.get(case_id)
            manual_quantitative_matches = (
                int(manual["positive_quantitative_match_count"])
                if manual and manual.get("positive_quantitative_match_count") is not None
                else None
            )
            passed = negative_review_passes(manual)
            if manual:
                negative_review.update({
                    "searched_terms": manual.get("searched_terms", negative_review.get("searched_terms", [])),
                    "searched_synonyms": manual.get("searched_synonyms", negative_review.get("searched_synonyms", [])),
                    "searched_sections": manual.get("searched_sections", negative_review.get("searched_sections", [])),
                    "searched_table_titles": manual.get("searched_table_titles", negative_review.get("searched_table_titles", [])),
                    "full_document_search_completed": bool(manual.get("full_document_search_completed")),
                    "human_negative_evidence_reviewed": bool(manual.get("human_negative_evidence_reviewed")),
                    "negative_evidence_reviewed": bool(manual.get("negative_evidence_reviewed")),
                    "negative_evidence_review_status": "verified" if passed else "pending",
                    "positive_quantitative_match_count": manual_quantitative_matches,
                    "reviewer": manual.get("reviewer") if passed else None,
                    "reviewed_at": manual.get("reviewed_at") if passed else None,
                    "review_notes": manual.get("review_notes") if passed else negative_review.get("review_notes"),
                })
            label["no_answer_review"] = negative_review
            label["ready_for_golden"] = bool(passed)
            review["negative_evidence_review_status"] = "verified" if passed else "pending"
            review["ready_for_golden"] = False
            review["reviewer"] = REVIEWER if passed else None
            review["reviewed_at"] = reviewed_at if passed else None
            negative_verified += int(passed)
            negative_records.append({
                "case_id": case_id,
                "document_id": corpus_doc["document_id"],
                "full_document_search_completed": True,
                "human_negative_evidence_reviewed": passed,
                "negative_evidence_reviewed": passed,
                "positive_match_count": negative_review.get("positive_match_count", 0),
                "positive_quantitative_match_count": negative_review.get("positive_quantitative_match_count"),
                "searched_terms": negative_review.get("searched_terms", []),
                "searched_synonyms": negative_review.get("searched_synonyms", []),
                "reviewed_at": reviewed_at if passed else None,
                "manual_review_log_present": bool(manual),
            })
            updated_labels.append(label)
            updated_reviews.append(review)
            continue

        bound_sources: list[dict[str, Any]] = []
        for source_index, source in enumerate(label.get("expected_sources") or []):
            if str(source.get("document_id")) not in allowed:
                decision = BindingDecision("out_of_scope", None, ("document_not_in_benchmark_whitelist",))
            else:
                scoped = [candidate for candidate in candidates if candidate.get("benchmark_document_id") == source.get("document_id")]
                decision = choose_candidate(
                    label=label, source=source, source_index=source_index,
                    candidates=scoped, by_id=by_doc_id,
                    top20_keys=top20_by_case.get(case_id, set()),
                )
            bound = _apply_bound_source(source, decision, top20_keys=top20_by_case.get(case_id, set()), reviewed_at=reviewed_at)
            bound_sources.append(bound)
            status_counts[decision.status] = status_counts.get(decision.status, 0) + 1
            if decision.status == "bound":
                method = str(bound.get("binding_method"))
                binding_methods[method] = binding_methods.get(method, 0) + 1
            else:
                gaps.append({
                    "case_id": case_id,
                    "source_index": source_index,
                    "document_id": source.get("document_id"),
                    "pdf_page": source.get("page"),
                    "row_label": source.get("row_label"),
                    "status": decision.status,
                    "reasons": list(decision.reasons),
                    "candidate_count": len([candidate for candidate in candidates if candidate.get("benchmark_document_id") == source.get("document_id")]),
                    "candidate_keys": [item[0] for item in decision.scores],
                })
            source_records.append({
                "case_id": case_id,
                "source_index": source_index,
                "document_id": source.get("document_id"),
                "pdf_page": source.get("page"),
                "row_label": source.get("row_label"),
                "column_header": source.get("column_header"),
                "metric": source_expected_metric(label, source, source_index),
                "expected_value": _source_value_label(label, source_index),
                "candidate_identity_status": bound.get("candidate_identity_status"),
                "candidate_key": bound.get("candidate_key"),
                "evidence_id": bound.get("evidence_id"),
                "identity_granularity": bound.get("identity_granularity"),
                "candidate_content_hash": bound.get("candidate_content_hash"),
                "binding_method": bound.get("binding_method"),
                "source_verified": bool(bound.get("source_verified")),
            })
        label["expected_sources"] = bound_sources
        all_bound = bool(bound_sources) and all(item.get("source_verified") for item in bound_sources)
        label["ready_for_golden"] = all_bound
        review["verified_source_count"] = sum(int(item.get("source_verified")) for item in bound_sources)
        review["all_sources_have_candidate_identity"] = all_bound
        review["source_review_status"] = "verified" if all_bound else "pending"
        review["source_reviewed"] = all_bound
        review["ready_for_golden"] = all_bound
        review["reviewer"] = REVIEWER if all_bound else None
        review["reviewed_at"] = reviewed_at if all_bound else None
        updated_labels.append(label)
        updated_reviews.append(review)

    source_count = len(source_records)
    bound_count = sum(int(item["source_verified"]) for item in source_records)
    answerable = sum(not bool(item.get("expected_no_answer")) for item in labels)
    all_cases_ready = all(bool(label.get("ready_for_golden")) for label in updated_labels)
    golden_ready = golden_promotion_gate(
        question_count=len(questions),
        answerable_count=answerable,
        expected_source_count=source_count,
        bound_candidate_identity_count=bound_count,
        negative_evidence_verified_count=negative_verified,
        ambiguous_identity_count=status_counts.get("ambiguous", 0),
        missing_from_index_count=status_counts.get("missing_from_index", 0),
        out_of_scope_identity_count=status_counts.get("out_of_scope", 0),
        unresolved_anomaly_count=len(anomalies),
        all_cases_ready=all_cases_ready and not gaps and not anomalies,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "labels.evidence-bound.jsonl", updated_labels)
    write_jsonl(out_dir / "review-status.evidence-bound.jsonl", updated_reviews)
    _write_json(out_dir / "candidate-binding-report.json", {
        "artifact_schema": SCHEMA, "expected_source_record_count": source_count,
        "bound_candidate_identity_count": bound_count, "source_verified_count": bound_count,
        "binding_methods": binding_methods, "status_counts": status_counts,
        "question_hash": frozen_hashes["question_hash"], "reference_answer_hash": frozen_hashes["reference_answer_hash"],
        "corpus_hash": frozen_hashes["corpus_hash"], "tenant_id": tenant_id,
    })
    _write_json(out_dir / "candidate-binding-case-report.json", {
        "artifact_schema": SCHEMA,
        "cases": [
            {"case_id": item["case_id"], "source_count": len(item.get("expected_sources") or []), "verified_source_count": sum(int(source.get("source_verified")) for source in item.get("expected_sources") or []), "ready_for_golden": bool(item.get("ready_for_golden"))}
            for item in updated_labels
        ],
    })
    _write_json(out_dir / "candidate-identity-gap-report.json", {
        "artifact_schema": SCHEMA, "gap_count": len(gaps), "gaps": gaps,
        "invalid_candidate_key_count": sum(int(not item.get("candidate_key")) for item in source_records),
    })
    _write_json(out_dir / "negative-evidence-review-report.json", {
        "artifact_schema": SCHEMA, "reviewed_count": negative_verified,
        "case_count": len(negative_records), "cases": negative_records,
    })
    _write_json(out_dir / "label-anomaly-report.json", {
        "artifact_schema": SCHEMA, "unresolved_count": len(anomalies), "anomalies": anomalies,
    })
    readiness = {
        "artifact_schema": SCHEMA, "question_count": len(questions), "answerable_count": answerable,
        "expected_source_record_count": source_count, "bound_candidate_identity_count": bound_count,
        "source_verified_count": bound_count, "negative_evidence_verified_count": negative_verified,
        "ambiguous_identity_count": status_counts.get("ambiguous", 0),
        "missing_from_index_count": status_counts.get("missing_from_index", 0),
        "out_of_scope_identity_count": status_counts.get("out_of_scope", 0),
        "unresolved_anomaly_count": len(anomalies), "golden_ready_case_count": sum(int(item.get("ready_for_golden")) for item in updated_labels),
        "golden_ready": golden_ready,
    }
    _write_json(out_dir / "golden-readiness-report.json", readiness)
    acceptance = {
        **readiness, "question_hash": frozen_hashes["question_hash"],
        "reference_answer_hash": frozen_hashes["reference_answer_hash"], "corpus_hash": frozen_hashes["corpus_hash"],
        "benchmark_document_count": len(allowed), "document_whitelist_hash": stable_json_hash(sorted(allowed)),
        "production_behavior_changed": False, "baseline_run": False,
        "golden_promotion_allowed": golden_ready,
    }
    _write_json(out_dir / "nf-eval-02-acceptance.json", acceptance)
    if golden_ready:
        golden_dir = BACKEND / "benchmarks" / "financial_rag_v1" / "data"
        write_jsonl(golden_dir / "questions.golden.jsonl", questions)
        write_jsonl(golden_dir / "labels.golden.jsonl", updated_labels)
        golden_reviews = []
        for review in updated_reviews:
            item = deepcopy(review)
            item["ready_for_golden"] = True
            item["golden_status"] = "golden"
            golden_reviews.append(item)
        write_jsonl(golden_dir / "review-status.golden.jsonl", golden_reviews)
        golden_payload = {
            "benchmark_id": corpus.get("benchmark_id", "financial-rag-v1"), "schema_version": "1.0",
            "question_count": 72, "answerable_count": answerable, "no_answer_count": 8,
            "expected_source_record_count": source_count,
            "question_hash": frozen_hashes["question_hash"], "reference_answer_hash": frozen_hashes["reference_answer_hash"],
            "source_identity_hash": stable_json_hash(source_records), "negative_evidence_hash": stable_json_hash(negative_records),
            "review_status_hash": stable_json_hash(golden_reviews), "corpus_hash": frozen_hashes["corpus_hash"],
            "golden_ready": True,
        }
        _write_json(golden_dir / "golden-manifest.json", golden_payload)
    return acceptance


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    data = BACKEND / "benchmarks" / "financial_rag_v1" / "data"
    runtime = BACKEND / "runtime" / "benchmark" / "financial_rag_v1"
    parser.add_argument("--corpus", type=Path, default=BACKEND / "benchmarks" / "financial_rag_v1" / "corpus.json")
    parser.add_argument("--questions", type=Path, default=data / "questions.reference.jsonl")
    parser.add_argument("--labels", type=Path, default=data / "labels.reference.jsonl")
    parser.add_argument("--review", type=Path, default=data / "review-status.jsonl")
    parser.add_argument("--candidate-dir", type=Path, default=runtime / "review-package" / "source-review-candidates")
    parser.add_argument("--bm25-db", type=Path, default=BACKEND / "rag_bm25.db")
    parser.add_argument("--pdf-root", type=Path, default=runtime / "review-package" / "pdfs")
    parser.add_argument("--out-dir", type=Path, default=BACKEND / "artifacts" / "evaluation" / "nf-eval-02")
    parser.add_argument("--tenant-id", type=int, required=True)
    parser.add_argument("--reviewed-at", default=date.today().isoformat())
    parser.add_argument("--negative-review-log", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(bind_sources(
        corpus_path=args.corpus, questions_path=args.questions, labels_path=args.labels,
        review_path=args.review, candidate_dir=args.candidate_dir, bm25_db=args.bm25_db,
        pdf_root=args.pdf_root, out_dir=args.out_dir, tenant_id=args.tenant_id, reviewed_at=args.reviewed_at,
        negative_review_log=args.negative_review_log,
    ), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
