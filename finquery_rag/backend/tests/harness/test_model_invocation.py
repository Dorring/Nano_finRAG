"""H2A-3C: the model-invocation boundary, and what may cross it.

The Context Runtime had a working production path by H2A-3B3, but the model
boundary was still shaped by the one boundary that had crossed it.  This file is
about the boundary itself rather than about B3: a provider receives a
``ModelRequestV1`` and returns a ``ModelResponseV1``, and the renderer that
produced the request is the only thing that decided what went into it.

Three claims are worth separating, because they fail differently:

1. **Shape.**  The contracts have the fields they say they have, and no more.
   Asserted as exact field sets, so adding one is a deliberate edit rather than
   a quiet widening.
2. **Boundary.**  A provider is handed a request and nothing else -- not the
   pack, not evidence, not a RunState, not a calculation, not a metadata bag.
   This is the phase's new trust boundary, and it is checked structurally rather
   than promised: the provider spine does not import the context package at all,
   and the request's field set has nowhere to put an authority object.
3. **Wiring.**  One renderer call and one provider call per invocation, with the
   provider receiving exactly the text the renderer returned.  Asserted by
   substitution, so the claim is about the call path rather than about a
   counter the same code maintains.

Everything here runs without torch or a checkpoint.  That is the point of the
phase as much as the contracts are: the fixed Linux checkpoint used to be
required to exercise anything near the model, and it no longer is.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import pytest

from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1
from rag_v2.context import (
    AgentContextPackV1,
    ContextCompilerV1,
    ContextRequestV1,
    ContextRoleV1,
    ExactTokenCounterV1,
    SpecialistContextPolicyV1,
)
from rag_v2.invocation import (
    CallableContextRendererV1,
    ContextRendererV1,
    InvocationIntegrityError,
    LegacyPromptProviderAdapterV1,
    ModelBindingV1,
    ModelInvocationRuntimeV1,
    ModelProviderError,
    ModelProviderV1,
    ModelRequestV1,
    ModelResponseV1,
    ProviderFailureKind,
)
from src.runtime.trusted_v2_generation import TrustedV2GenerationCapability
from tests.harness.b3_legacy_context_baseline import (
    BASELINE_V2,
    SCENARIOS,
    build_state,
)

RAW_TEXT = "RAW SOURCE TEXT THAT MUST NOT REACH A PROVIDER"


# --- fixtures ---------------------------------------------------------------------------


def _pack(*, bindings: dict[str, list[str]] | None = None) -> AgentContextPackV1:
    state = build_state("multi_fact")
    state.bound_slot_bindings = {
        slot: list(ids) for slot, ids in (bindings or {"revenue/FY2024": ["e1"]}).items()
    }
    return ContextCompilerV1(SpecialistContextPolicyV1()).compile(
        specialist_context_request_of(state)
    )


def specialist_context_request_of(state: Any) -> ContextRequestV1:
    from rag_v2.context import specialist_context_request

    return specialist_context_request(state)


class _RecordingProvider:
    """A provider that records the request and answers.  No torch, no checkpoint."""

    def __init__(self, text: str = "recorded", **response: Any) -> None:
        self.text = text
        self.response = response
        self.requests: list[ModelRequestV1] = []

    def invoke(self, request: ModelRequestV1) -> ModelResponseV1:
        self.requests.append(request)
        return ModelResponseV1(text=self.text, **self.response)


class _FailingProvider:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    def invoke(self, request: ModelRequestV1) -> ModelResponseV1:
        self.calls += 1
        raise self.exc


def _binding(**overrides: Any) -> ModelBindingV1:
    defaults: dict[str, Any] = {
        "provider": _RecordingProvider(),
        "renderer": CallableContextRendererV1(lambda pack: f"rendered:{pack.invocation_id}"),
        "provider_id": "test-provider",
    }
    defaults.update(overrides)
    return ModelBindingV1(**defaults)


# --- 1. shape ---------------------------------------------------------------------------


def test_the_request_is_exactly_the_declared_fields() -> None:
    """An exact field set, because the boundary *is* the field set.

    A request that grew an ``evidence`` or ``metadata`` field would still pass
    every behavioural test in this file, and would have handed providers a route
    around the renderer.
    """

    assert set(ModelRequestV1.__dataclass_fields__) == {
        "invocation_id",
        "role",
        "prompt",
        "provider_id",
        "model_id",
    }


def test_the_response_is_exactly_the_declared_fields() -> None:
    assert set(ModelResponseV1.__dataclass_fields__) == {
        "text",
        "finish_status",
        "citations",
        "usage",
    }


def test_the_binding_is_exactly_the_declared_fields() -> None:
    assert set(ModelBindingV1.__dataclass_fields__) == {
        "provider",
        "renderer",
        "provider_id",
        "model_id",
        "exact_token_counter",
    }


@pytest.mark.parametrize(
    "field", ["retry", "timeout", "tool", "replan", "max_tokens", "temperature"]
)
def test_the_binding_carries_no_execution_policy(field: str) -> None:
    """Three separate concepts, and this is the third one's absence.

    RunBudget bounds how much *work* the runtime performs, ContextBudget bounds
    how much *information* one invocation sees, and a binding is which model and
    how to reach it.  A retry counter here would be one knob changing two of
    those, which is how a context decision becomes a latency decision.
    """

    assert not any(field in name for name in ModelBindingV1.__dataclass_fields__)


def test_a_request_refuses_a_blank_prompt() -> None:
    """A renderer that produced no input is a defect, not an invocation."""

    with pytest.raises(InvocationIntegrityError):
        ModelRequestV1(
            invocation_id="i", role="SPECIALIST", prompt="   ", provider_id="p"
        )


def test_a_response_refuses_a_container_in_its_usage() -> None:
    """A named field is not a licence for a structure stored under it.

    Same rule the pack applies to evidence, for the same reason: a container
    here would be a second object crossing the provider boundary, and this
    boundary is the one thing the phase is about.
    """

    with pytest.raises(InvocationIntegrityError):
        ModelResponseV1(text="a", usage={"nested": {"n": 1}})


def test_a_request_carries_no_configuration_field_it_has_no_producer_for() -> None:
    """H2A-3E removed ``generation``: no producer, no consumer, no test.

    A field with nothing on either end is a claim about a future requirement
    nobody has stated, and the boundary it sat on is the one place a guess is
    most expensive -- a sampling setting nobody chose is indistinguishable at
    the provider from one somebody did.
    """

    assert "generation" not in ModelRequestV1.__dataclass_fields__


def test_a_response_collapses_a_repeated_citation() -> None:
    """A repeated handle is not a second citation, and reporting it as one would
    overstate what the model said it used."""

    response = ModelResponseV1(text="a", citations=("E1", "E1", "E2"))

    assert response.citations == ("E1", "E2")


def test_a_response_may_carry_empty_text() -> None:
    """An empty answer is a real model output; judging it is the boundary's job."""

    assert ModelResponseV1(text="").text == ""


# --- 3. wiring: one renderer, one provider ------------------------------------------------


def test_one_invocation_renders_once_and_calls_the_provider_once() -> None:
    provider = _RecordingProvider()
    rendered: list[AgentContextPackV1] = []

    def render(pack: AgentContextPackV1) -> str:
        rendered.append(pack)
        return "the rendered text"

    binding = ModelBindingV1(
        provider=provider,
        renderer=CallableContextRendererV1(render),
        provider_id="test-provider",
    )
    runtime = ModelInvocationRuntimeV1(binding)
    pack = _pack()

    response = runtime.invoke(pack)

    assert len(rendered) == 1
    assert len(provider.requests) == 1
    assert runtime.render_count == 1
    assert runtime.provider_count == 1

    # The provider received exactly what the renderer produced, and the request
    # carries the pack's identity rather than the pack.
    assert provider.requests[0].prompt == "the rendered text"
    assert provider.requests[0].invocation_id == pack.invocation_id
    assert provider.requests[0].role == pack.role.value
    assert provider.requests[0].provider_id == "test-provider"
    assert response.text == "recorded"


def test_a_renderer_that_raises_still_counts_as_one_render() -> None:
    """The counts rule out a *second* call, so an attempt is what they record."""

    def explode(pack: AgentContextPackV1) -> str:
        raise ValueError("renderer failed")

    binding = _binding(renderer=CallableContextRendererV1(explode))
    runtime = ModelInvocationRuntimeV1(binding)

    with pytest.raises(ValueError):
        runtime.invoke(_pack())

    assert runtime.render_count == 1
    assert runtime.provider_count == 0


def test_a_provider_that_returns_the_wrong_type_is_refused() -> None:
    """A contract violation rather than a provider failure, and it cannot travel
    as runtime truth."""

    class _Wrong:
        def invoke(self, request: ModelRequestV1) -> Any:
            return {"answer_text": "not a ModelResponseV1"}

    runtime = ModelInvocationRuntimeV1(_binding(provider=_Wrong()))

    with pytest.raises(InvocationIntegrityError):
        runtime.invoke(_pack())


def test_the_runtime_refuses_anything_that_is_not_a_binding() -> None:
    with pytest.raises(InvocationIntegrityError):
        ModelInvocationRuntimeV1(object())  # type: ignore[arg-type]


# --- 2. the boundary: no authority reaches a provider ---------------------------------------


def test_a_provider_is_handed_a_request_and_nothing_else() -> None:
    """The new trust boundary, stated over the object a provider receives.

    Every name below is something a provider must never be able to read: the
    pack, the evidence contract, the runtime state, the calculation, the raw
    metadata bag, and the source text that Disclosure Authority keeps out.  The
    assertion is on the request's own attributes, so it holds for any provider
    implementation rather than for the ones written here.
    """

    provider = _RecordingProvider()
    runtime = ModelInvocationRuntimeV1(_binding(provider=provider))
    runtime.invoke(_pack())

    request = provider.requests[0]
    for absent in (
        "pack",
        "context_pack",
        "evidence",
        "evidence_items",
        "evidence_packets",
        "state",
        "run_state",
        "calculation",
        "calculation_result",
        "metadata",
        "source_text",
        "references",
        "support_groups",
    ):
        assert not hasattr(request, absent), absent


def test_the_provider_spine_does_not_import_the_pack_or_the_application() -> None:
    """The structural half of the same claim.

    ``renderer.py`` imports the pack, because a renderer's whole job is one.
    ``provider.py`` must not: a provider author who wants a semantic field has
    to get it through the renderer, which is the only route Disclosure Authority
    governs.  Checked on the syntax tree so a comment naming the pack cannot
    satisfy it.
    """

    tree = ast.parse(
        pathlib.Path("rag_v2/invocation/provider.py").read_text(encoding="utf-8")
    )

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    offenders = {name for name in imported if name.startswith("rag_v2.context")}
    assert offenders == set(), offenders
    assert not {name for name in imported if name.split(".")[0] == "src"}


def test_the_protocol_asks_for_one_annotated_request() -> None:
    """What the Harness requires of a provider, read off the interface itself."""

    parameters = list(inspect.signature(ModelProviderV1.invoke).parameters.values())

    assert [parameter.name for parameter in parameters] == ["self", "request"]
    assert parameters[1].annotation in ("ModelRequestV1", ModelRequestV1)


def test_the_request_carries_no_source_text_even_though_the_prompt_carries_evidence() -> None:
    """The prompt carries evidence; it does not carry what is not disclosed.

    ``source_text`` is absent from the SPECIALIST profile, so a request built
    from a real state cannot contain it however the renderer behaves.
    """

    state = build_state("multi_fact")
    for packet in state.evidence_packets:
        packet["source_text"] = RAW_TEXT
        packet.setdefault("metadata", {})["source_text"] = RAW_TEXT

    from src.generation.specialist_prompt import render_specialist_prompt

    pack = ContextCompilerV1(SpecialistContextPolicyV1()).compile(
        specialist_context_request_of(state)
    )
    provider = _RecordingProvider()
    ModelInvocationRuntimeV1(
        _binding(
            provider=provider,
            renderer=CallableContextRendererV1(render_specialist_prompt),
        )
    ).invoke(pack)

    assert RAW_TEXT not in provider.requests[0].prompt
    assert "Source: doc-1:7" in provider.requests[0].prompt


# --- renderer and provider are separate responsibilities --------------------------------------


def test_the_specialist_renderer_satisfies_the_renderer_protocol() -> None:
    from src.generation.specialist_prompt import render_specialist_prompt

    wrapper = CallableContextRendererV1(render_specialist_prompt)

    assert isinstance(wrapper, ContextRendererV1)
    assert wrapper.renderer_id == "render_specialist_prompt"


def test_a_renderer_that_is_not_callable_is_refused() -> None:
    with pytest.raises(TypeError):
        CallableContextRendererV1("not callable")  # type: ignore[arg-type]


def test_the_binding_refuses_a_provider_without_invoke_or_a_renderer_without_render() -> None:
    """Deployment configuration, so it fails where it is written."""

    with pytest.raises(InvocationIntegrityError):
        _binding(provider=object())

    with pytest.raises(InvocationIntegrityError):
        _binding(renderer=object())

    with pytest.raises(InvocationIntegrityError):
        _binding(provider_id="   ")


def test_the_legacy_adapter_is_a_provider() -> None:
    adapter = LegacyPromptProviderAdapterV1(_EchoBackend())

    assert isinstance(adapter, ModelProviderV1)


# --- §16 error normalization -------------------------------------------------------------------


class _EchoBackend:
    def __init__(self, result: Any = "an answer") -> None:
        self.result = result
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _invoke_adapter(result: Any) -> ModelResponseV1:
    adapter = LegacyPromptProviderAdapterV1(_EchoBackend(result))
    return adapter.invoke(
        ModelRequestV1(
            invocation_id="i", role="SPECIALIST", prompt="text", provider_id="legacy"
        )
    )


def test_a_failing_backend_becomes_an_unavailable_provider() -> None:
    adapter = LegacyPromptProviderAdapterV1(_EchoBackend(RuntimeError("engine died")))

    with pytest.raises(ModelProviderError) as raised:
        adapter.invoke(
            ModelRequestV1(
                invocation_id="i", role="SPECIALIST", prompt="t", provider_id="legacy"
            )
        )

    assert raised.value.kind is ProviderFailureKind.UNAVAILABLE
    assert isinstance(raised.value.__cause__, RuntimeError), "the cause must survive"


def test_a_timeout_is_classified_apart_from_an_unavailable_provider() -> None:
    """A caller may reasonably retry one and not the other; RunBudget decides."""

    adapter = LegacyPromptProviderAdapterV1(_EchoBackend(TimeoutError("too slow")))

    with pytest.raises(ModelProviderError) as raised:
        adapter.invoke(
            ModelRequestV1(
                invocation_id="i", role="SPECIALIST", prompt="t", provider_id="legacy"
            )
        )

    assert raised.value.kind is ProviderFailureKind.TIMEOUT


def test_a_response_that_is_neither_text_nor_a_mapping_is_invalid() -> None:
    with pytest.raises(ModelProviderError) as raised:
        _invoke_adapter(17)

    assert raised.value.kind is ProviderFailureKind.INVALID_RESPONSE


def test_an_empty_answer_is_not_the_provider_s_judgment() -> None:
    """Empty text is a real model output, and the boundary that asked judges it.

    Classifying it here would move a grounding decision into the transport
    layer, where nothing knows what the boundary asked for.
    """

    assert _invoke_adapter("").text == ""
    assert _invoke_adapter({"answer_text": ""}).text == ""


def test_a_mapping_response_keeps_its_citations_and_finish_status() -> None:
    response = _invoke_adapter(
        {
            "answer_text": "391 [E1]",
            "citation_ids": ["E1", "unknown-X"],
            "finish_reason": "stop",
            "tokens_generated": 12,
            "rendered_input_length": 300,
        }
    )

    assert response.text == "391 [E1]"
    assert response.citations == ("E1", "unknown-X")
    assert response.finish_status == "stop"
    assert dict(response.usage or {}) == {"prompt_tokens": 300, "completion_tokens": 12}


def test_a_provider_declared_metadata_bag_does_not_become_runtime_truth() -> None:
    """The legacy boundary merged a backend's own ``metadata`` into the result.

    A provider's private bag is not the runtime's record of the run -- that is
    the rule this contract states -- so it is dropped rather than forwarded.
    """

    response = _invoke_adapter(
        {"answer_text": "a", "metadata": {"latency_seconds": 0.5, "internal": "x"}}
    )

    assert set(ModelResponseV1.__dataclass_fields__) & {"metadata", "raw"} == set()
    assert not hasattr(response, "metadata")


# --- §11 the tokenizer seam ----------------------------------------------------------------------


class _Counter:
    counter_id = "test-only-exact"

    def count(self, text: str) -> int:
        return len(text)


def test_a_binding_carries_an_exact_counter_and_nothing_approximates_it() -> None:
    """The seam, and the rule it participates in.

    A configured token bound with an exact counter is enforced; a configured
    token bound without one is a construction failure.  No word count, character
    count or substitute tokenizer appears anywhere to make a missing counter go
    away -- and a binding that kept no counter would simply be one that cannot
    constrain tokens, which is B3's honest state.
    """

    counter = _Counter()
    binding = _binding(exact_token_counter=counter)

    assert binding.exact_token_counter is counter
    assert isinstance(counter, ExactTokenCounterV1)

    # The rule, applied through the counter the binding supplies.
    ContextCompilerV1(
        SpecialistContextPolicyV1(),
        budget=_bound_budget(),
        token_counter=binding.exact_token_counter,
    )
    with pytest.raises(Exception):
        ContextCompilerV1(SpecialistContextPolicyV1(), budget=_bound_budget())


def _bound_budget() -> Any:
    from rag_v2.context import ContextBudgetV1

    return ContextBudgetV1(max_input_tokens=1000)


def test_a_binding_refuses_a_counter_that_is_not_one() -> None:
    with pytest.raises(InvocationIntegrityError):
        _binding(exact_token_counter=object())


def test_the_local_specialist_binding_configures_no_token_bound() -> None:
    """B3's honest state: no counter, so no bound is enforced or approximated."""

    capability = TrustedV2GenerationCapability(model_backend=_EchoBackend())

    assert capability.binding is not None
    assert capability.binding.exact_token_counter is None
    assert capability.context_compiler.token_counter is None
    assert capability.context_compiler.budget.max_input_tokens is None


# --- §13 the routing helper is routing-only --------------------------------------------------------


def test_the_routing_helper_is_named_for_what_it_does() -> None:
    """It stopped selecting at H2A-3B3; the name had to stop saying it did.

    There is also a second reason now: ``trusted_v2_validation`` has a
    ``_bound_items`` of its own, and two functions of one name in two layers
    doing different things is how a future contributor picks the wrong one.
    """

    from src.runtime import trusted_v2_generation as module

    assert hasattr(module, "_routing_evidence_items")
    assert not hasattr(module, "_bound_items")


def _code_names(path: pathlib.Path) -> set[str]:
    """Every identifier a module's *code* names -- not the ones its prose does.

    A check that a comment mentioning something fails is not checking the code,
    and this file's own subject is a rename whose whole point is that prose
    should be able to say the new name.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names |= {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    return names


def test_no_context_construction_reaches_the_routing_helper() -> None:
    """One consumer, and it is the router.

    The compiler's selection is its own -- it could not call this if it wanted
    to, since ``rag_v2`` imports ``src`` zero times -- and asserting the name
    never appears in ``rag_v2`` *code* is the readable form of that.
    """

    offenders = [
        path.as_posix()
        for path in pathlib.Path("rag_v2").rglob("*.py")
        if "_routing_evidence_items" in _code_names(path)
    ]
    assert offenders == []

    consumers = [
        path.as_posix()
        for path in pathlib.Path("src").rglob("*.py")
        if "_routing_evidence_items" in _code_names(path)
    ]
    assert consumers == ["src/runtime/trusted_v2_generation.py"], consumers


def test_the_router_uses_the_authoritative_admitted_evidence_view() -> None:
    """The helper's output is the admitted evidence, not everything retrieved."""

    from rag_v2.context import admitted_specialist_evidence
    from src.runtime.trusted_v2_generation import _routing_evidence_items

    state = AdaptiveRAGStateV1.new("r", "What was revenue?")
    state.add_evidence(
        [
            EvidencePacketV1.from_mapping({"evidence_id": "e1", "metric": "Revenue"}),
            EvidencePacketV1.from_mapping({"evidence_id": "e2", "metric": "Revenue"}),
        ]
    )
    state.bound_evidence_ids = ["e2"]

    # Identical to the adapter's view, which is what "one implementation" means.
    assert _routing_evidence_items(state) == [
        dict(item) for item in admitted_specialist_evidence(state)
    ]
    assert [item["evidence_id"] for item in _routing_evidence_items(state)] == ["e2"]


# --- §14 trace counts are an observability projection ------------------------------------------------


def test_the_trace_is_rebuilt_and_nothing_reads_it() -> None:
    """No runtime decision depends on a trace copy.

    Two things are checked, and they are the two ways a trace could stop being a
    projection: a consumer holding one could mutate the runtime's state through
    it, or the runtime could consult its own record of what it did.
    """

    capability = TrustedV2GenerationCapability(model_backend=_EchoBackend())
    capability.generate(build_state("multi_fact"))

    first = capability.trace_snapshot()
    first["context_compile_count"] = 999
    first["generation_route"] = "TAMPERED"
    second = capability.trace_snapshot()

    assert second["context_compile_count"] == 1
    assert second["generation_route"] == BASELINE_V2["multi_fact"]["route"]

    # And the decision path never reads it: `generate` is walked rather than
    # grepped, so a comment naming the method cannot satisfy this.
    tree = ast.parse(
        pathlib.Path("src/runtime/trusted_v2_generation.py").read_text(encoding="utf-8")
    )
    generate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "generate"
    )
    referenced = {node.attr for node in ast.walk(generate) if isinstance(node, ast.Attribute)}
    referenced |= {node.id for node in ast.walk(generate) if isinstance(node, ast.Name)}

    assert "trace_snapshot" not in referenced
    assert "last_disclosed_fields" not in referenced


# --- §17 the whole boundary, without torch ------------------------------------------------------


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_the_boundary_reproduces_baseline_v2_with_a_deterministic_provider(
    scenario: str,
) -> None:
    """The runnable architectural proof, and the reason the checkpoint is no
    longer on the critical path for testing the Harness.

    The provider here is a recording stub.  It answers with the same constant the
    frozen baseline's recorder answered with, so the candidate is unchanged --
    and the *prompt* it was handed came through the compiler and the renderer
    exactly as production delivers it.
    """

    import hashlib

    provider = _RecordingProvider(text=BASELINE_V2[scenario]["candidate_answer"])
    capability = TrustedV2GenerationCapability(model_backend=provider)
    result = capability.generate(build_state(scenario))
    frozen = BASELINE_V2[scenario]

    assert len(provider.requests) == 1, "the provider must be called exactly once"
    request = provider.requests[0]

    assert isinstance(request, ModelRequestV1)
    assert request.role == "SPECIALIST"
    assert request.prompt == frozen["prompt"]
    assert (
        hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
        == frozen["prompt_sha256"]
    )

    # The response travelled back through the Harness rather than around it.
    assert result.candidate_answer == frozen["candidate_answer"]
    assert result.candidate_generation_id == frozen["candidate_generation_id"]
    assert capability.trace_snapshot()["context_render_count"] == 1
    assert capability.trace_snapshot()["context_provider_count"] == 1


def test_a_backend_failure_fails_closed_through_the_harness() -> None:
    """Normalizing the error must not soften it.

    The failure still propagates, and the candidate is never produced -- which
    is the behaviour the release path depends on and which this phase must not
    change while it changes how the error is spelled.  The backend here is a
    legacy one, so the *adapter* is what normalizes it; a provider that
    implements the protocol normalizes its own, which is what §2 says a provider
    owns.
    """

    capability = TrustedV2GenerationCapability(
        model_backend=_EchoBackend(RuntimeError("engine died"))
    )

    with pytest.raises(ModelProviderError) as raised:
        capability.generate(build_state("multi_fact"))

    assert raised.value.kind is ProviderFailureKind.UNAVAILABLE
    assert capability.last_result is None


def test_a_providers_own_normalized_failure_is_not_re_wrapped() -> None:
    """The Harness does not second-guess a kind the provider chose.

    Re-wrapping would turn a deliberate classification -- a timeout, say -- into
    a generic unavailability, and the distinction is the whole reason the
    classification exists.
    """

    from rag_v2.invocation import ModelProviderError as Error

    capability = TrustedV2GenerationCapability(
        model_backend=_FailingProvider(
            Error(ProviderFailureKind.TIMEOUT, "the endpoint did not answer")
        )
    )

    with pytest.raises(Error) as raised:
        capability.generate(build_state("multi_fact"))

    assert raised.value.kind is ProviderFailureKind.TIMEOUT


def test_the_capability_binds_an_already_correct_provider_without_adapting_it() -> None:
    """The extension seam: a provider arrives as itself, not wrapped.

    A future financial or DeepSeek binding is constructed this way, and the
    binding is the only thing that changes -- no evidence handling, no compiler,
    no pack.
    """

    provider = _RecordingProvider()
    capability = TrustedV2GenerationCapability(model_backend=provider)

    assert capability.binding is not None
    assert capability.binding.provider is provider
    assert capability.binding.provider_id == "_RecordingProvider"


def test_the_capability_adapts_a_legacy_backend() -> None:
    """And a ``generate(prompt)`` backend arrives wrapped, once."""

    backend = _EchoBackend("an answer")
    capability = TrustedV2GenerationCapability(model_backend=backend)

    assert capability.binding is not None
    assert isinstance(capability.binding.provider, LegacyPromptProviderAdapterV1)
    assert capability.binding.provider.backend is backend

    capability.generate(build_state("multi_fact"))

    assert len(backend.prompts) == 1
    assert backend.prompts[0] == BASELINE_V2["multi_fact"]["prompt"]
