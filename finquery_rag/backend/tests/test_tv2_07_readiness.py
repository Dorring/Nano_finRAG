"""TV2-07 frozen readiness evaluation tests."""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from src.evaluation.tv2_07_readiness import (
    TV2IntegratedEvaluationRunner,
    TV2ReadinessDecision,
    TV2ReadinessLabel,
    TV2ReadinessOutcome,
    TV2ReadinessQuery,
    build_tv2_07_manifest,
    load_tv2_07_dataset,
    score_predictions,
    score_readiness_case,
    write_tv2_07_artifacts,
)
from src.runtime import (
    FinancialQueryResult,
    ReleaseStatus,
    RuntimeStatus,
    RuntimeVersion,
)
from tests.test_runtime_router_shadow import _real_v2_fact_runtime


def _result(
    *,
    status: RuntimeStatus = RuntimeStatus.ANSWER,
    answer: str | None = "Revenue: 100 USD million",
    evidence_ids: list[str] | None = None,
    citation_ids: list[str] | None = None,
    route: str = "STRUCTURED_SINGLE",
    repair_count: int = 0,
    validated: bool = True,
) -> FinancialQueryResult:
    trace = {
        "generation_route": route,
        "bound_evidence_ids": evidence_ids or ["E1"],
        "validation_passed": validated,
        "release_decision": "RELEASED" if validated else "NOT_RELEASED",
        "release_status": "RELEASED" if validated else "NOT_RELEASED",
        "repair_count": repair_count,
    }
    return FinancialQueryResult(
        status=status,
        answer=answer,
        evidence_ids=evidence_ids or ["E1"],
        citation_ids=citation_ids or ["citation-E1"],
        runtime_version=RuntimeVersion.V2,
        release_status=(
            ReleaseStatus.RELEASED
            if status is RuntimeStatus.ANSWER and validated
            else ReleaseStatus.NOT_RELEASED
        ),
        debug_metadata={"trace": trace},
    )


class _Runtime:
    def __init__(self, result: FinancialQueryResult | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.requests: list[Any] = []

    async def execute(self, request: Any) -> FinancialQueryResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


class _SlowRuntime:
    async def execute(self, request: Any) -> FinancialQueryResult:
        del request
        await asyncio.sleep(0.05)
        return _result()


class _SlowSyncRuntime:
    def execute(self, request: Any) -> FinancialQueryResult:
        del request
        time.sleep(0.05)
        return _result()


def _query(case_id: str = "case-1") -> TV2ReadinessQuery:
    return TV2ReadinessQuery(
        case_id=case_id,
        question="What was revenue?",
        category="direct_fact",
        metadata={"fixture_key": "fixture"},
    )


def _label(**overrides: Any) -> TV2ReadinessLabel:
    values: dict[str, Any] = {
        "case_id": "case-1",
        "category": "direct_fact",
        "answerable": True,
        "expected_release": True,
        "expected_route": "STRUCTURED_SINGLE",
        "expected_evidence_ids": ("E1",),
        "expected_citation_ids": ("citation-E1",),
        "required_answer_terms": ("100",),
    }
    values.update(overrides)
    return TV2ReadinessLabel(**values)


def test_query_contract_rejects_gold_and_raw_context() -> None:
    with pytest.raises(ValueError, match="Gold field"):
        TV2ReadinessQuery(
            case_id="bad",
            question="Q",
            category="fact",
            metadata={"expected_release": True},
        )
    with pytest.raises(ValueError, match="raw context"):
        TV2ReadinessQuery(
            case_id="bad",
            question="Q",
            category="fact",
            metadata={"conversation_history": []},
        )
    with pytest.raises(ValueError, match="Gold field"):
        TV2ReadinessQuery(
            case_id="bad",
            question="Q",
            category="fact",
            metadata={"fixture": {"gold_answer": "hidden"}},
        )


def test_runner_passes_identical_request_without_labels() -> None:
    primary = _Runtime(_result(status=RuntimeStatus.ANSWER, answer="V1"))
    shadow = _Runtime(_result())
    runner = TV2IntegratedEvaluationRunner(
        lambda: primary,
        lambda: shadow,
        timeout_seconds=1,
    )
    predictions = asyncio.run(runner.run([_query()]))
    assert len(predictions) == 1
    assert primary.requests[0] is shadow.requests[0]
    assert "expected_release" not in predictions[0].request
    assert predictions[0].gold_injection_detected is False
    assert predictions[0].request["request_metadata"]["readiness_case_id"] == "case-1"


def test_runner_records_timeout_and_runtime_errors_per_case() -> None:
    error_runner = TV2IntegratedEvaluationRunner(
        lambda: _Runtime(_result()),
        lambda: _Runtime(error=RuntimeError("boom")),
        timeout_seconds=1,
    )
    error_prediction = asyncio.run(error_runner.run([_query("error")]))[0]
    assert error_prediction.v2.error_code == "RuntimeError"
    assert error_prediction.v2.status == RuntimeStatus.ERROR.value

    timeout_runner = TV2IntegratedEvaluationRunner(
        lambda: _Runtime(_result()),
        lambda: _SlowRuntime(),
        timeout_seconds=0.001,
    )
    timeout_prediction = asyncio.run(timeout_runner.run([_query("timeout")]))[0]
    assert timeout_prediction.v2.error_code == "TIMEOUT"

    sync_timeout_runner = TV2IntegratedEvaluationRunner(
        lambda: _Runtime(_result()),
        lambda: _SlowSyncRuntime(),
        timeout_seconds=0.001,
    )
    sync_timeout_prediction = asyncio.run(
        sync_timeout_runner.run([_query("sync-timeout")])
    )[0]
    assert sync_timeout_prediction.v2.error_code == "TIMEOUT"


def test_scorer_blocks_unanswerable_release_and_wrong_binding() -> None:
    prediction = asyncio.run(
        TV2IntegratedEvaluationRunner(
            lambda: _Runtime(_result()),
            lambda: _Runtime(_result(evidence_ids=["WRONG"])),
        ).run([_query()])
    )[0]
    scored = score_readiness_case(
        prediction,
        _label(answerable=False, expected_release=False),
    )
    assert scored["outcome"] == TV2ReadinessOutcome.UNSAFE_INCORRECT_RELEASE.value
    assert "UNSAFE_RELEASES" in scored["hard_gate_violations"]
    assert "FALSE_BINDING" in scored["hard_gate_violations"]


def test_scorer_distinguishes_correct_and_over_conservative_fail_closed() -> None:
    fail_closed = _result(
        status=RuntimeStatus.FAIL_CLOSED,
        answer=None,
        evidence_ids=[],
        citation_ids=[],
    )
    prediction = asyncio.run(
        TV2IntegratedEvaluationRunner(
            lambda: _Runtime(fail_closed),
            lambda: _Runtime(fail_closed),
        ).run([_query()])
    )[0]
    correct = score_readiness_case(
        prediction,
        _label(answerable=False, expected_release=False, expected_route=None),
    )
    assert correct["outcome"] == TV2ReadinessOutcome.CORRECT_FAIL_CLOSED.value

    over = score_readiness_case(prediction, _label())
    assert over["outcome"] == TV2ReadinessOutcome.OVER_CONSERVATIVE_FAIL_CLOSED.value
    assert over["hard_gate_violations"] == []


def test_scorer_enforces_validation_and_repair_bound() -> None:
    prediction = asyncio.run(
        TV2IntegratedEvaluationRunner(
            lambda: _Runtime(_result()),
            lambda: _Runtime(_result(repair_count=1)),
        ).run([_query()])
    )[0]
    scored = score_readiness_case(prediction, _label())
    assert scored["outcome"] == TV2ReadinessOutcome.CORRECT_RELEASE_AFTER_REPAIR.value

    invalid = asyncio.run(
        TV2IntegratedEvaluationRunner(
            lambda: _Runtime(_result()),
            lambda: _Runtime(_result(repair_count=2)),
        ).run([_query("bad-repair")])
    )[0]
    invalid_scored = score_readiness_case(
        invalid,
        _label(case_id="bad-repair"),
    )
    assert "REPAIR_ATTEMPTS_GT_1" in invalid_scored["hard_gate_violations"]


def test_scorer_detects_false_calculation_execution() -> None:
    result = _result()
    debug = dict(result.debug_metadata)
    debug["trace"] = {**debug["trace"], "calculator_invoked": True}
    result = FinancialQueryResult(
        status=result.status,
        answer=result.answer,
        evidence_ids=result.evidence_ids,
        citation_ids=result.citation_ids,
        runtime_version=result.runtime_version,
        release_status=result.release_status,
        debug_metadata=debug,
    )
    prediction = asyncio.run(
        TV2IntegratedEvaluationRunner(
            lambda: _Runtime(result),
            lambda: _Runtime(result),
        ).run([_query()])
    )[0]
    scored = score_readiness_case(
        prediction,
        _label(
            answerable=False,
            expected_release=False,
            expected_route=None,
            annotation={"calculation_must_not_execute": True},
        ),
    )
    assert "FALSE_CALCULATION_EXECUTION" in scored["hard_gate_violations"]


def test_dataset_is_new_stratified_and_query_label_separated() -> None:
    queries, labels = load_tv2_07_dataset(
        "tests/fixtures/tv2_07_production_readiness/questions.jsonl",
        "tests/fixtures/tv2_07_production_readiness/labels.jsonl",
    )
    assert len(queries) == len(labels) == 22
    assert len({query.category for query in queries}) >= 14
    assert all("expected_release" not in query.to_dict() for query in queries)
    assert all("gold" not in query.metadata for query in queries)


def test_real_tv2_factory_smoke_is_scored_without_gold_injection() -> None:
    query = TV2ReadinessQuery(
        case_id="real-fact",
        question="What was revenue?",
        category="direct_fact",
        metadata={"fixture_key": "real-factory"},
    )
    runtime, _, _, _ = _real_v2_fact_runtime()
    predictions = asyncio.run(
        TV2IntegratedEvaluationRunner(
            lambda: _Runtime(_result(status=RuntimeStatus.ANSWER, answer="V1")),
            lambda: runtime,
        ).run([query])
    )
    label = TV2ReadinessLabel(
        case_id="real-fact",
        category="direct_fact",
        answerable=True,
        expected_release=True,
        expected_route="STRUCTURED_SINGLE",
        expected_evidence_ids=("E1",),
        expected_citation_ids=("citation-E1",),
        required_answer_terms=("100",),
    )
    scored, metrics = score_predictions(
        predictions,
        [label],
        evaluation_scope_complete=False,
        corpus_verified=False,
        canonical_model_verified=False,
    )
    assert scored[0]["outcome"] == TV2ReadinessOutcome.CORRECT_RELEASE.value
    assert metrics.hard_gate_counts["UNSAFE_RELEASES"] == 0
    assert metrics.decision is TV2ReadinessDecision.HOLD_FOR_QUALITY


def test_safety_gate_decision_and_artifacts_are_explicit(tmp_path) -> None:
    query = _query()
    prediction = asyncio.run(
        TV2IntegratedEvaluationRunner(
            lambda: _Runtime(_result()),
            lambda: _Runtime(_result(evidence_ids=["WRONG"])),
        ).run([query])
    )
    label = _label()
    scored, metrics = score_predictions(
        prediction,
        [label],
        evaluation_scope_complete=True,
        corpus_verified=True,
        canonical_model_verified=True,
    )
    assert metrics.decision is TV2ReadinessDecision.BLOCKED_FOR_SAFETY
    assert metrics.hard_gate_counts["UNSAFE_RELEASES"] == 1
    assert metrics.hard_gate_counts["RELEASED_INCORRECT"] == 1

    manifest = build_tv2_07_manifest(
        repo_path=".",
        queries=[query],
        labels=[label],
        runtime_config={"production_runtime": "V1"},
    )
    write_tv2_07_artifacts(
        tmp_path,
        manifest=manifest,
        queries=[query],
        labels=[label],
        scored_cases=scored,
        metrics=metrics,
    )
    expected_files = {
        "manifest.json",
        "dataset-manifest.json",
        "runtime-manifest.json",
        "case-results.jsonl",
        "overall-metrics.json",
        "safety-gates.json",
        "route-breakdown.json",
        "calculation-breakdown.json",
        "recovery-breakdown.json",
        "repair-breakdown.json",
        "v1-v2-comparison.json",
        "latency-summary.json",
        "error-summary.json",
        "decision.json",
    }
    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})


def test_expected_reason_code_is_part_of_fail_closed_gold() -> None:
    result = _result(
        status=RuntimeStatus.FAIL_CLOSED,
        answer=None,
        evidence_ids=[],
        citation_ids=[],
    )
    prediction = asyncio.run(
        TV2IntegratedEvaluationRunner(
            lambda: _Runtime(result),
            lambda: _Runtime(result),
        ).run([_query()])
    )[0]
    scored = score_readiness_case(
        prediction,
        _label(
            answerable=False,
            expected_release=False,
            expected_route=None,
            expected_reason_codes=("CONFLICT",),
        ),
    )
    assert scored["outcome"] == TV2ReadinessOutcome.OVER_CONSERVATIVE_FAIL_CLOSED.value
    assert scored["correctness_checks"]["reason_codes"] is False
