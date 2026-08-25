"""TV2-07R1 canonical readiness preflight tests."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.evaluation.tv2_07_readiness import (
    TV2ReadinessLabel,
    TV2ReadinessQuery,
    score_readiness_case,
)
from src.evaluation.tv2_07_r1_readiness import (
    TV2IntegratedEvaluationRunner,
    build_tv2_07_r1_preflight,
    load_tv2_07_r1_dataset,
    write_tv2_07_r1_pending_artifacts,
)
from src.runtime.runtime_contract import (
    FinancialQueryRequest,
    FinancialQueryResult,
    ReleaseStatus,
    RuntimeStatus,
    RuntimeVersion,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _canonical_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    categories = [
        "direct_fact",
        "multi_evidence",
        "calculation",
        "qualitative_synthesis",
        "no_answer",
        "wrong_period_trap",
        "retrieval_recovery",
        "multi_turn_context",
    ]
    queries: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for index, category in enumerate(categories):
        case_id = f"canonical-{index}"
        query: dict[str, Any] = {
            "case_id": case_id,
            "question": f"Question {index}",
            "category": category,
            "dataset_provenance": "fresh_company_held_out",
        }
        if category == "multi_turn_context":
            query["input_turns"] = [
                {"turn_id": "t1", "role": "user", "text": "Apple FY2024 revenue?"},
                {
                    "turn_id": "t2",
                    "role": "user",
                    "text": "What about the previous year?",
                },
            ]
        queries.append(query)
        answerable = category != "no_answer"
        labels.append(
            {
                "case_id": case_id,
                "category": category,
                "answerable": answerable,
                "expected_release": answerable,
                "dataset_provenance": "fresh_company_held_out",
            }
        )
    return queries, labels


def _write_complete_manifests(tmp_path: Path, repo_path: Path) -> dict[str, Path]:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"canonical-checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    corpus_sha = "corpus-sha"
    corpus = tmp_path / "corpus.json"
    _write_json(
        corpus,
        {
            "production_v1_modified": False,
            "gold_evidence_generated": False,
            "questions_generated": False,
            "primary_documents": 2,
            "searchable_corpus_sha": corpus_sha,
            "searchable_manifest_sha": "manifest-sha",
        },
    )
    index_config = tmp_path / "index-config.json"
    _write_json(
        index_config,
        {
            "production_v1_overwritten": False,
            "searchable_corpus_sha": corpus_sha,
        },
    )
    index_build = tmp_path / "index-build.json"
    _write_json(
        index_build,
        {
            "production_indices_modified": False,
            "fts": {"built": True},
            "dense": {"built": True},
            "hybrid": {"built": True},
        },
    )
    index_integrity = tmp_path / "index-integrity.json"
    _write_json(
        index_integrity,
        {
            "duplicate_index_ids": 0,
            "metadata_schema_failures": 0,
            "missing_indexed_chunks": 0,
            "orphan_index_entries": 0,
            "provenance_complete_percent": 100.0,
            "searchable_corpus_sha": corpus_sha,
        },
    )
    raw_manifest = tmp_path / "raw-manifest.jsonl"
    parsed_manifest = tmp_path / "parsed-manifest.jsonl"
    _write_jsonl(
        raw_manifest,
        [{"document_id": "doc-1"}, {"document_id": "doc-2"}],
    )
    _write_jsonl(
        parsed_manifest,
        [{"document_id": "doc-1"}, {"document_id": "doc-2"}],
    )
    model = tmp_path / "model.json"
    _write_json(
        model,
        {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "role": "LOCAL_FINANCIAL_SPECIALIST_GENERATOR",
            "precision": "bfloat16",
            "device": "cpu",
            "decoding": {
                "temperature": 0.0,
                "no_thinking_mode": True,
                "no_cot": True,
            },
        },
    )
    wiring_queries = tmp_path / "wiring-queries.jsonl"
    wiring_labels = tmp_path / "wiring-labels.jsonl"
    _write_jsonl(
        wiring_queries,
        [
            {
                "case_id": "wiring-only",
                "question": "fixture",
                "category": "direct_fact",
            }
        ],
    )
    _write_jsonl(
        wiring_labels,
        [
            {
                "case_id": "wiring-only",
                "category": "direct_fact",
                "answerable": True,
                "expected_release": True,
            }
        ],
    )
    return {
        "corpus": corpus,
        "index_config": index_config,
        "index_build": index_build,
        "index_integrity": index_integrity,
        "model": model,
        "raw_manifest": raw_manifest,
        "parsed_manifest": parsed_manifest,
        "wiring_queries": wiring_queries,
        "wiring_labels": wiring_labels,
    }


def test_r1_loader_rejects_wiring_fixture() -> None:
    with pytest.raises(ValueError, match="rejects"):
        load_tv2_07_r1_dataset(
            "tests/fixtures/tv2_07_production_readiness/questions.jsonl",
            "tests/fixtures/tv2_07_production_readiness/labels.jsonl",
        )


def test_preflight_requires_canonical_set_and_does_not_use_consumed_run(
    tmp_path: Path,
) -> None:
    report = build_tv2_07_r1_preflight(
        repo_path=Path("."),
        queries_path=tmp_path / "missing-queries.jsonl",
        labels_path=tmp_path / "missing-labels.jsonl",
        corpus_freeze_path=tmp_path / "missing-corpus.json",
        index_config_path=tmp_path / "missing-index-config.json",
        index_build_path=tmp_path / "missing-index-build.json",
        index_integrity_path=tmp_path / "missing-index-integrity.json",
        model_manifest_path=tmp_path / "missing-model.json",
        min_cases=100,
    )
    assert report.status == "PENDING"
    assert report.ready_to_run is False
    assert "queries_exists" in report.blocking_reasons
    assert report.case_count == 0


def test_preflight_accepts_complete_stratified_fixture_with_lower_test_minimum(
    tmp_path: Path,
) -> None:
    queries, labels = _canonical_rows()
    queries_path = tmp_path / "canonical-queries.jsonl"
    labels_path = tmp_path / "canonical-labels.jsonl"
    _write_jsonl(queries_path, queries)
    _write_jsonl(labels_path, labels)
    manifests = _write_complete_manifests(tmp_path, Path("."))
    report = build_tv2_07_r1_preflight(
        repo_path=Path("."),
        queries_path=queries_path,
        labels_path=labels_path,
        corpus_freeze_path=manifests["corpus"],
        index_config_path=manifests["index_config"],
        index_build_path=manifests["index_build"],
        index_integrity_path=manifests["index_integrity"],
        model_manifest_path=manifests["model"],
        raw_corpus_manifest_path=manifests["raw_manifest"],
        parsed_corpus_manifest_path=manifests["parsed_manifest"],
        wiring_queries_path=manifests["wiring_queries"],
        wiring_labels_path=manifests["wiring_labels"],
        min_cases=8,
    )
    assert report.ready_to_run is True
    assert report.status == "READY_TO_RUN"
    assert report.case_count == 8
    assert report.answerable_cases == 7
    assert report.multi_turn_cases == 1
    assert set(report.category_counts) == {
        "direct_fact",
        "multi_evidence",
        "calculation",
        "qualitative_synthesis",
        "no_answer",
        "wrong_period_trap",
        "retrieval_recovery",
        "multi_turn_context",
    }
    assert report.checks["negative_cases_present"] is True
    assert report.checks["multi_turn_input_present"] is True
    assert report.checks["raw_corpus_manifest_document_count_matches"] is True
    assert report.checks["parsed_raw_document_ids_match"] is True


class _CaptureRuntime:
    def __init__(self) -> None:
        self.requests: list[FinancialQueryRequest] = []

    async def execute(self, request: FinancialQueryRequest) -> FinancialQueryResult:
        self.requests.append(request)
        return FinancialQueryResult(
            status=RuntimeStatus.FAIL_CLOSED,
            answer=None,
            runtime_version=RuntimeVersion.V2,
            release_status=ReleaseStatus.NOT_RELEASED,
        )


def test_multi_turn_runner_requires_factory_and_uses_resolved_query() -> None:
    query = TV2ReadinessQuery(
        case_id="multi-turn",
        question="What about the previous year?",
        category="multi_turn_context",
        input_turns=(
            {"turn_id": "t1", "role": "user", "text": "Apple FY2024 revenue?"},
            {"turn_id": "t2", "role": "user", "text": "What about the previous year?"},
        ),
        dataset_provenance="fresh_company_held_out",
    )
    with pytest.raises(ValueError, match="request_factory"):
        asyncio.run(
            TV2IntegratedEvaluationRunner(
                _CaptureRuntime,
                _CaptureRuntime,
            ).run([query])
        )

    primary = _CaptureRuntime()
    shadow = _CaptureRuntime()

    def request_factory(value: Any) -> FinancialQueryRequest:
        return FinancialQueryRequest(
            request_id=value.case_id,
            user_id="readiness-user",
            session_id=value.case_id,
            original_query=value.question,
            standalone_query="What was Apple FY2023 revenue?",
            query_as_resolved=True,
            conversation_metadata={"resolution_status": "RESOLVED"},
            request_metadata={"readiness_case_id": value.case_id},
        )

    predictions = asyncio.run(
        TV2IntegratedEvaluationRunner(
            lambda: primary,
            lambda: shadow,
            request_factory=request_factory,
        ).run([query])
    )
    assert predictions[0].request["standalone_query"] == "What was Apple FY2023 revenue?"
    assert predictions[0].request["query_as_resolved"] is True
    assert primary.requests[0] is shadow.requests[0]
    assert predictions[0].query.input_turns[1]["text"] == "What about the previous year?"




def test_structured_period_gold_blocks_wrong_period_release() -> None:
    result = FinancialQueryResult(
        status=RuntimeStatus.ANSWER,
        answer="Revenue",
        evidence_ids=["E1"],
        citation_ids=["C1"],
        runtime_version=RuntimeVersion.V2,
        release_status=ReleaseStatus.RELEASED,
        debug_metadata={
            "trace": {
                "bound_evidence_ids": ["E1"],
                "validation_passed": True,
                "release_decision": "RELEASED",
                "release_status": "RELEASED",
                "period": "FY2023",
                "unit": "USD",
                "scale": "million",
            }
        },
    )

    class FixedRuntime:
        async def execute(self, request: FinancialQueryRequest) -> FinancialQueryResult:
            del request
            return result

    query = TV2ReadinessQuery(
        case_id="wrong-period",
        question="Apple FY2024 revenue?",
        category="wrong_period_trap",
        dataset_provenance="fresh_company_held_out",
    )
    prediction = asyncio.run(
        TV2IntegratedEvaluationRunner(
            FixedRuntime,
            FixedRuntime,
        ).run([query])
    )[0]
    label = TV2ReadinessLabel(
        case_id="wrong-period",
        category="wrong_period_trap",
        answerable=True,
        expected_release=True,
        expected_evidence_ids=("E1",),
        expected_citation_ids=("C1",),
        annotation={
            "gold_period": "FY2024",
            "gold_unit": "USD",
            "gold_scale": "million",
        },
        dataset_provenance="fresh_company_held_out",
    )
    scored = score_readiness_case(prediction, label)
    assert "WRONG_PERIOD_RELEASE" in scored["hard_gate_violations"]
    assert "UNSAFE_RELEASES" in scored["hard_gate_violations"]


def test_pending_r1_artifacts_are_explicit_and_keep_production_off(
    tmp_path: Path,
) -> None:
    report = build_tv2_07_r1_preflight(
        repo_path=Path("."),
        queries_path=tmp_path / "queries.jsonl",
        labels_path=tmp_path / "labels.jsonl",
        corpus_freeze_path=tmp_path / "corpus.json",
        index_config_path=tmp_path / "index.json",
        index_build_path=tmp_path / "build.json",
        index_integrity_path=tmp_path / "integrity.json",
        model_manifest_path=tmp_path / "model.json",
    )
    output = tmp_path / "artifacts"
    write_tv2_07_r1_pending_artifacts(
        output,
        preflight=report,
        repo_path=Path("."),
    )
    decision = json.loads((output / "decision.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert decision["decision"] == "HOLD_FOR_QUALITY"
    assert manifest["readiness_evaluation_executed"] is False
    assert manifest["production_runtime"] == "V1"
    assert manifest["v2_authority"] == "OFF"
