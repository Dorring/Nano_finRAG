from __future__ import annotations

from collections.abc import Iterable

from rag_v2.contracts.evidence import EvidenceBinding
from rag_v2.contracts.plan import SupervisorPlan
from rag_v2.evidence.binder_provider import BinderCallMetadata, BinderProviderError, BinderProviderResult, _binding_from_payload
from rag_v2.evidence.binder_service import BinderRequest, SemanticBinderService
from rag_v2.evidence.transport_retry import TransportRetryPolicy, bind_with_transport_retry, canonical_request_sha


def plan() -> SupervisorPlan:
    return SupervisorPlan.from_dict({
        "intent": "DIRECT_FACT",
        "required_slots": [{"slot_id": "slot_1", "metric": "revenue", "period": "FY2025", "role": "value", "value_type": "numeric", "unit": None}],
        "operation": None,
        "next_action": "RETRIEVE",
    })


def request() -> BinderRequest:
    return BinderRequest("q", "question", plan(), ({"fact_id": "f1", "provenance_complete": True},))


def binding(status: str = "BOUND") -> EvidenceBinding:
    return _binding_from_payload({
        "status": status,
        "slot_bindings": {"slot_1": ["f1"]} if status == "BOUND" else {},
        "missing_slots": ["slot_1"] if status == "MISSING" else [],
        "ambiguous_slots": ["slot_1"] if status == "AMBIGUOUS" else [],
        "invalid_reasons": [],
    })


def metadata(*, success: bool, structured: bool, status: int | None = None, exception: str | None = None) -> BinderCallMetadata:
    return BinderCallMetadata(
        "stub", "qwen3.7-max-preview", "evidence_binder", "strong_general_llm", 1.0,
        success, structured, http_status=status, exception_type=exception,
    )


class SequenceProvider:
    provider_name = "stub"
    model_name = "qwen3.7-max-preview"
    last_call = None

    def __init__(self, outcomes: Iterable[str]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def bind(self, _request):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if outcome == "success":
            self.last_call = metadata(success=True, structured=True)
            return BinderProviderResult(binding(), self.last_call, '{"status":"BOUND"}')
        if outcome in {"missing", "ambiguous"}:
            self.last_call = metadata(success=True, structured=True)
            return BinderProviderResult(binding(outcome.upper()), self.last_call, '{"status":"' + outcome.upper() + '"}')
        if outcome == "schema_invalid":
            self.last_call = metadata(success=True, structured=False)
        elif outcome.startswith("http_"):
            self.last_call = metadata(success=False, structured=False, status=int(outcome.split("_")[1]))
        else:
            self.last_call = metadata(success=False, structured=False, exception=outcome)
        raise BinderProviderError(outcome)


def run(outcomes: list[str], *, sleeps: list[float] | None = None, request_sha_fn=canonical_request_sha):
    sleeps = sleeps if sleeps is not None else []
    provider = SequenceProvider(outcomes)
    result = bind_with_transport_retry(SemanticBinderService(provider), request(), sleep_fn=sleeps.append, request_sha_fn=request_sha_fn)
    return provider, result, sleeps


def test_attempt_1_success_does_not_retry() -> None:
    provider, result, sleeps = run(["success", "success"])
    assert provider.calls == 1
    assert result.semantic_response_count == 1
    assert result.attempt_2 is None
    assert sleeps == []


def test_timeout_then_success_is_transport_recovered() -> None:
    provider, result, sleeps = run(["ReadTimeout", "success"])
    assert provider.calls == 2
    assert result.recovered_by_transport_retry is True
    assert result.semantic_response_count == 1
    assert sleeps == [3.0]


def test_connection_then_success_is_transport_recovered() -> None:
    provider, result, _ = run(["APIConnectionError", "success"])
    assert provider.calls == 2
    assert result.semantic_response_count == 1


def test_503_then_success_is_transport_recovered() -> None:
    provider, result, _ = run(["http_503", "success"])
    assert provider.calls == 2
    assert result.recovered_by_transport_retry is True


def test_two_transport_failures_fail_closed() -> None:
    provider, result, _ = run(["ReadTimeout", "ReadTimeout"])
    assert provider.calls == 2
    assert result.final_provider_completion is False
    assert result.semantic_response_count == 0


def test_schema_invalid_2xx_is_not_retried() -> None:
    provider, result, _ = run(["schema_invalid", "success"])
    assert provider.calls == 1
    assert result.attempt_2 is None
    assert result.semantic_response_count == 0


def test_semantic_statuses_are_not_retried() -> None:
    for status in ("missing", "ambiguous", "success"):
        provider, result, _ = run([status, "success"])
        assert provider.calls == 1
        assert result.attempt_2 is None
        assert result.semantic_response_count == 1


def test_request_sha_mismatch_fails_closed() -> None:
    values = iter(["original", "changed"])
    provider, result, sleeps = run(["ReadTimeout", "success"], request_sha_fn=lambda _request: next(values))
    assert provider.calls == 1
    assert sleeps == []
    assert result.retry_request_sha_matches_original is False
    assert result.semantic_response_count == 0


def test_semantic_response_count_never_exceeds_one() -> None:
    for outcomes in (["success"], ["ReadTimeout", "success"], ["ReadTimeout", "ReadTimeout"]):
        _, result, _ = run(list(outcomes))
        assert result.semantic_response_count <= 1


def test_policy_keeps_sdk_retries_disabled_and_retry_classes_frozen() -> None:
    policy = TransportRetryPolicy()
    assert policy.sdk_max_retries == 0
    assert policy.semantic_attempt_budget == 1
    assert policy.transport_retry_budget == 1
    assert policy.retry_delay_seconds == 3.0
    assert policy.retryable_failures == ("APITimeoutError", "APIConnectionError", "ReadTimeout", "ConnectTimeout", "HTTP_429", "HTTP_502", "HTTP_503", "HTTP_504")


def test_empty_fact_packet_is_deterministic_and_not_a_provider_attempt() -> None:
    empty = BinderRequest("empty", "question", plan(), ())
    provider = SequenceProvider([])
    result = bind_with_transport_retry(SemanticBinderService(provider), empty, sleep_fn=lambda _: None)
    assert provider.calls == 0
    assert result.attempt_1.attempted is False
    assert result.run.skipped_no_fact_supply is True
