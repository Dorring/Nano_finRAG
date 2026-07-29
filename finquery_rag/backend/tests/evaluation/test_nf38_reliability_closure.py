"""NF38 R1 tests for explicit scope, labels, tokenizer, and device evidence."""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.evaluation.evaluation import EvaluationCase, ExpectedSource
from src.evaluation.nf38_corpus import CanonicalEvidenceRecord, build_corpus_manifest
from src.evaluation.nf38_evaluator import (
    EvaluationConfigurationError,
    EvaluationDatasetError,
    EvaluationScope,
    TokenizerUnavailableError,
    compute_token_length_report,
    freeze_bm25_pool,
    get_gpu_memory_mb,
    label_hash,
    question_hash,
    validate_labeled_cases,
    validate_scope_corpus,
)


def _record(evidence_id: str = "e1", document_id: str = "a.pdf") -> CanonicalEvidenceRecord:
    return CanonicalEvidenceRecord(
        evidence_id=evidence_id,
        document_id=document_id,
        page=1,
        block_type="text",
        embedding_text="cash and cash equivalents",
        embedding_text_hash="hash",
    )


def _scope(records: list[CanonicalEvidenceRecord]) -> EvaluationScope:
    manifest = build_corpus_manifest(records)
    return EvaluationScope(
        tenant_id=7,
        allowed_document_ids=frozenset(record.document_id for record in records),
        expected_case_count=1,
        expected_corpus_hash=manifest["corpus_hash"],
        expected_evidence_ids_hash=manifest["evidence_ids_hash"],
    )


def _case(*, no_answer: bool = False, sources: bool = True) -> EvaluationCase:
    return EvaluationCase(
        case_id="c1",
        question="What was cash?",
        expected_sources=(ExpectedSource(filename="a.pdf", page=1),) if sources else (),
        expected_no_answer=no_answer,
    )


def _candidate(document_id: str = "a.pdf") -> dict:
    return {
        "doc_id": "e1",
        "metadata": {"doc_name": document_id, "page": 1, "type": "text"},
        "score": 1.0,
    }


def test_nf38_uses_requested_bm25_user_id_and_rejects_out_of_scope_documents():
    seen: list[tuple[int, int]] = []

    def search(query: str, *, k: int, user_id: int):
        seen.append((k, user_id))
        return [_candidate("outside.pdf"), _candidate("a.pdf")]

    pool = freeze_bm25_pool(
        [_case()],
        search,
        scope=_scope([_record()]),
        k=1,
        oversample_k=200,
    )
    assert seen == [(200, 7)]
    assert pool.candidates["c1"][0]["document_id"] == "a.pdf"
    assert pool.out_of_scope_candidate_count == 1
    assert pool.scope_report()["allowed_document_count"] == 1


def test_nf38_records_candidate_shortfall_and_rejects_zero_in_scope_candidates():
    def one_result(query: str, *, k: int, user_id: int):
        return [_candidate("a.pdf")]

    pool = freeze_bm25_pool(
        [_case()],
        one_result,
        scope=_scope([_record()]),
        k=50,
        oversample_k=200,
    )
    assert pool.cases_with_candidate_shortfall == ["c1"]

    def no_result(query: str, *, k: int, user_id: int):
        return [_candidate("outside.pdf")]

    with pytest.raises(EvaluationConfigurationError):
        freeze_bm25_pool([_case()], no_result, scope=_scope([_record()]))


def test_answerable_case_requires_sources_and_no_answer_case_may_have_none():
    with pytest.raises(EvaluationDatasetError):
        validate_labeled_cases([_case(sources=False)], expected_count=1)
    report = validate_labeled_cases([_case(no_answer=True, sources=False)], expected_count=1)
    assert report["no_answer_case_count"] == 1
    assert report["cases_missing_expected_sources"] == 0


def test_expected_case_count_and_duplicate_ids_are_enforced():
    with pytest.raises(EvaluationDatasetError):
        validate_labeled_cases([_case()], expected_count=27)
    duplicate = [_case(), replace(_case(), question="duplicate")]
    with pytest.raises(EvaluationDatasetError):
        validate_labeled_cases(duplicate, expected_count=2)


def test_question_and_label_hashes_are_independent_and_order_stable():
    case = _case()
    changed_question = replace(case, question="What was revenue?")
    changed_label = replace(case, expected_sources=(ExpectedSource(filename="a.pdf", page=2),))
    assert question_hash([case]) != question_hash([changed_question])
    assert label_hash([case]) == label_hash([changed_question])
    assert question_hash([case]) == question_hash([changed_label])
    assert label_hash([case]) != label_hash([changed_label])
    other = replace(case, case_id="c2")
    assert question_hash([case, other]) == question_hash([other, case])


def test_scope_requires_matching_corpus_and_evidence_hashes():
    records = [_record()]
    validate_scope_corpus(_scope(records), records)
    changed = [_record(evidence_id="e2")]
    with pytest.raises(EvaluationConfigurationError):
        validate_scope_corpus(_scope(records), changed)


class _Tokenizer:
    def encode(self, text: str):
        return text.split()


class _ProviderWithTokenizer:
    def get_tokenizer(self):
        return _Tokenizer()


class _ProviderWithoutTokenizer:
    def get_tokenizer(self):
        return None


def test_official_token_report_requires_real_tokenizer_and_records_1024_ratio():
    records = [
        replace(_record(), embedding_text=" ".join(["x"] * 1025)),
        replace(_record(), evidence_id="e2", embedding_text="short"),
    ]
    report = compute_token_length_report(
        records,
        _ProviderWithTokenizer(),
        selected_max_length=1024,
    )
    assert report["token_length_method"] == "real_tokenizer"
    assert report["truncated_count"] == 1
    assert report["truncated_ratio"] == 0.5
    assert report["within_threshold"] is False
    with pytest.raises(TokenizerUnavailableError):
        compute_token_length_report(records, _ProviderWithoutTokenizer(), selected_max_length=1024)


def test_gpu_memory_uses_explicit_non_cuda_device():
    assert get_gpu_memory_mb("cpu") is None
