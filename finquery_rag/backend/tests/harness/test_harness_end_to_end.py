"""H2A-3E: the whole Harness, end to end, with fakes and no checkpoint.

Two boundaries, one file, because they are the two halves of the same guarantee
and reading them together is what shows it holds:

**What goes in.**  An authoritative runtime state is projected into a context
request, deterministically selected and disclosed into an `AgentContextPackV1`,
rendered into model input, and executed.  Nothing in that chain reads a runtime
authority except the adapter, and nothing outside the compiler decides what the
model may see.

**What comes out.**  A model's output is a candidate until something
deterministic admits it -- the answer path judges it against the boundary's own
requirements, and the ingestion path compares it against the authoritative
source artifact.

They are **not one call chain**, and saying so matters: the specialist answer
becomes a candidate *answer* assessed by the release path, while the table path
becomes a derived *artifact* assessed by a numeric-fidelity verifier.  What they
share is the contract family -- an invocation is described by a `ModelRequestV1`
and answered by a `ModelResponseV1`; model output is unverified until a
deterministic check says otherwise -- and this file exercises both halves in one
place so a reader can see the whole shape at once.

Everything here runs without torch, without a checkpoint, without an API key and
without a network.  That is a result of the phase, not a convenience: the fixed
Linux checkpoint used to be required to exercise anything near a model.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_v2.derived import (
    AdmittedDerivedArtifactV1,
    ModelDerivedArtifactV1,
    TransformationKindV1,
    verify_table_fidelity,
)
from rag_v2.invocation import (
    ModelRequestV1,
    ModelResponseV1,
    ProviderFailureKind,
)
from rag_v2.invocation import ModelProviderError
from src.runtime.trusted_v2_generation import (
    CandidateGenerationCapabilityError,
    TrustedV2GenerationCapability,
)
from src.services import process_tables
from tests.harness.b3_legacy_context_baseline import (
    BASELINE_V3,
    SCENARIOS,
    build_state,
)

TEMPORAL = "Compare revenue year-over-year."


class _FakeProvider:
    """A provider that records what it was asked and answers as configured.

    No transport, no model, no checkpoint -- which is the point.  What is under
    test is the path, not the model at the end of it.
    """

    provider_id = "fake"

    def __init__(self, text: str = "recorded", exc: Exception | None = None) -> None:
        self.text = text
        self.exc = exc
        self.requests: list[ModelRequestV1] = []

    def invoke(self, request: ModelRequestV1) -> ModelResponseV1:
        self.requests.append(request)
        if self.exc is not None:
            raise self.exc
        return ModelResponseV1(text=self.text)


def _deterministic_state():
    """One canonical fact with one source: the structured renderer's shape."""

    from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1

    state = AdaptiveRAGStateV1.new("r", "What was revenue?")
    state.add_evidence(
        [EvidencePacketV1.from_mapping({"evidence_id": "e1", "metric": "Revenue", "value": "391"})]
    )
    state.bound_evidence_ids = ["e1"]
    return state


def _state(scenario: str, **bindings: list[str]):
    state = build_state(scenario)
    if bindings:
        state.bound_slot_bindings = {slot: list(ids) for slot, ids in bindings.items()}
    return state


# --- 1. a normal specialist answer ------------------------------------------------------------


def test_a_normal_specialist_answer_travels_the_whole_path() -> None:
    """Every artefact on the way, asserted against the frozen baseline.

    The prompt at the end is byte-identical to what the boundary produced before
    the compiler took over, so the chain is not merely coherent -- it is the
    same chain the baseline records.
    """

    provider = _FakeProvider(text=BASELINE_V3["multi_fact"]["candidate_answer"])
    capability = TrustedV2GenerationCapability(model_backend=provider)

    result = capability.generate(_state("multi_fact"))

    # The invocation the provider was handed.
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.role == "SPECIALIST"
    assert request.provider_id == "fake"
    assert request.invocation_id.endswith(":candidate-generation")
    assert request.prompt == BASELINE_V3["multi_fact"]["prompt"]

    # The governed context it came from.
    pack = capability.last_context_pack
    assert pack is not None
    assert pack.evidence_ids == ("e1", "e2")
    assert pack.budget.evidence_selected == 2

    # And the candidate that came back.
    assert result.candidate_answer == BASELINE_V3["multi_fact"]["candidate_answer"]
    assert capability.trace_snapshot()["context_compile_count"] == 1
    assert capability.trace_snapshot()["context_render_count"] == 1
    assert capability.trace_snapshot()["context_provider_count"] == 1


# --- 2. multi-support context -----------------------------------------------------------------


def test_multi_support_context_keeps_its_topology_to_the_boundary() -> None:
    """Two witnesses of one figure, carried as one claim and rendered as two items.

    The topology reaches the pack and stops there: the current renderer does not
    present it, so the prompt is unchanged -- which is what keeps a migration
    from silently becoming a prompt change.
    """

    provider = _FakeProvider()
    capability = TrustedV2GenerationCapability(model_backend=provider)

    capability.generate(
        _state("multi_fact", **{"revenue/FY2024": ["e1", "e2"]})
    )

    pack = capability.last_context_pack
    assert pack is not None
    assert [
        (group.handle, group.canonical_slot, group.supports)
        for group in pack.references.support_groups
    ] == [("G1", "revenue/FY2024", ("E1", "E2"))]

    assert provider.requests[0].prompt == BASELINE_V3["multi_fact"]["prompt"]
    assert "G1" not in provider.requests[0].prompt
    assert "revenue/FY2024" not in provider.requests[0].prompt


# --- 3. calculation context -------------------------------------------------------------------


def test_a_calculation_context_carries_its_projection_to_the_boundary() -> None:
    provider = _FakeProvider()
    capability = TrustedV2GenerationCapability(model_backend=provider)

    capability.generate(_state("calculation_with_explanation"))

    pack = capability.last_context_pack
    assert pack is not None
    assert dict(pack.calculation or {}) == {
        "operation": "difference",
        "unit": "USD",
        "value": "8",
    }
    assert "[VERIFIED CALCULATION]" in provider.requests[0].prompt
    assert provider.requests[0].prompt == BASELINE_V3["calculation_with_explanation"]["prompt"]


# --- 4. provider failure ------------------------------------------------------------------------


def test_a_provider_failure_fails_closed() -> None:
    provider = _FakeProvider(
        exc=ModelProviderError(ProviderFailureKind.UNAVAILABLE, "no engine")
    )
    capability = TrustedV2GenerationCapability(model_backend=provider)

    with pytest.raises(ModelProviderError) as raised:
        capability.generate(_state("multi_fact"))

    assert raised.value.kind is ProviderFailureKind.UNAVAILABLE
    assert capability.last_result is None


# --- 5. empty model response ----------------------------------------------------------------------


def test_an_empty_model_response_is_refused_where_it_always_was() -> None:
    """Transport succeeded and produced nothing usable.

    The distinction the phase draws: the provider is right to return empty text
    -- it is a real model output and not a provider failure -- and the boundary
    that knows what it asked for is the one that refuses it.
    """

    provider = _FakeProvider(text="   ")
    capability = TrustedV2GenerationCapability(model_backend=provider)

    with pytest.raises(CandidateGenerationCapabilityError) as raised:
        capability.generate(_state("multi_fact"))

    assert "financial_specialist_empty_candidate" in str(raised.value)
    assert len(provider.requests) == 1, "the model was called; it answered nothing"
    assert capability.last_result is None


# --- 8. no leakage across invocations ---------------------------------------------------------------


def test_a_pack_does_not_leak_across_invocations() -> None:
    """Lifetime is one invocation, asserted across a route change."""

    provider = _FakeProvider()
    capability = TrustedV2GenerationCapability(model_backend=provider)

    capability.generate(_state("multi_fact"))
    assert capability.last_context_pack is not None
    assert len(provider.requests) == 1

    # A route that never compiles anything.  The previous pack must not survive
    # into it -- a stale pack reads as this invocation's context.  One canonical
    # fact with one source routes to the deterministic renderer, which reaches
    # no model at all.
    capability.generate(_deterministic_state())
    assert capability.last_context_pack is None
    assert len(provider.requests) == 1, "the deterministic route must not call a model"


# --- 6 and 7. the model-output boundary ---------------------------------------------------------------
#
# The other half.  The "model" here is reached over HTTP rather than through a
# provider, because ingestion predates the invocation framework -- recorded as
# debt in the phase note rather than papered over.

SOURCE_TABLE = """| Metric | FY2024 | FY2023 |
| --- | --- | --- |
| Revenue | 100 | 90 |
| Cost | 80 | 70 |
"""

ALTERED_TABLE = """| Metric | FY2024 | FY2023 |
| --- | --- | --- |
| Revenue | 101 | 90 |
| Cost | 80 | 70 |
"""

SOURCE_REF = "report.pdf::page_7::table_1"


class _StubRequests:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def post(self, *args: Any, **kwargs: Any) -> Any:
        return _StubResponse(
            {"choices": [{"message": {"content": self.answer}}]}
        )


class _StubResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


def _clean_with(monkeypatch: pytest.MonkeyPatch, table: str) -> dict:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(
        process_tables,
        "requests",
        _StubRequests(f"TABLE SUMMARY:\nA table.\n\nCLEANED TABLE:\n{table}"),
    )
    return process_tables.enhance_table_with_context(
        {"md": SOURCE_TABLE, "bbox": None},
        page_text="",
        page_num=7,
        source_reference=SOURCE_REF,
    )


def test_a_derived_table_artifact_is_admitted_and_becomes_its_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate -> verifier -> admitted -> trusted content, with lineage kept.

    The whole chain is walked explicitly here rather than asserted through the
    function's return, so a reader can see the object that carries the
    authority transition.
    """

    result = _clean_with(monkeypatch, SOURCE_TABLE)

    # Built the way the boundary builds it.  The section it extracts from the
    # answer is stripped of surrounding whitespace, which is not a semantic
    # decision and not what this test is about.
    candidate = ModelDerivedArtifactV1(
        artifact_id=f"{SOURCE_REF}::table-cleaning",
        content=SOURCE_TABLE.strip(),
        transformation=TransformationKindV1.TABLE_CLEANING,
        source_reference=SOURCE_REF,
    )
    admission = verify_table_fidelity(candidate, SOURCE_TABLE)
    admitted = AdmittedDerivedArtifactV1(candidate, admission)

    assert admitted.content == result["content"]
    assert admitted.source_reference == SOURCE_REF
    assert result["admission_record"]["admitted"] is True
    assert result["admission_record"]["source_reference"] == SOURCE_REF


def test_a_rejected_table_artifact_never_becomes_trusted_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The F6 defect, refused at the boundary it used to cross."""

    result = _clean_with(monkeypatch, ALTERED_TABLE)

    assert result["content"] == SOURCE_TABLE
    assert result["admission_record"]["admitted"] is False
    assert result["admission_record"]["reason"] == "NUMERIC_VALUE_CHANGED"
    assert "101" not in result["content"]


# --- the whole surface, one assertion per scenario -----------------------------------------------


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_every_authored_scenario_survives_the_whole_path(scenario: str) -> None:
    """The end-to-end regression, over every shape the boundary is reached with."""

    provider = _FakeProvider(text=BASELINE_V3[scenario]["candidate_answer"])
    capability = TrustedV2GenerationCapability(model_backend=provider)
    result = capability.generate(build_state(scenario))

    assert len(provider.requests) == 1
    assert provider.requests[0].prompt == BASELINE_V3[scenario]["prompt"]
    assert result.candidate_answer == BASELINE_V3[scenario]["candidate_answer"]
    assert result.candidate_generation_id == BASELINE_V3[scenario]["candidate_generation_id"]
