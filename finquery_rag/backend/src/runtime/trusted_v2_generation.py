"""TV2-04 candidate generation and routing capabilities.

The adapters reuse the frozen GeneratorRoutingPolicy and deterministic
calculation renderer.  They create a candidate only; Validator/Release is a
separate TV2-05 boundary.

H2A-3B3.  The specialist's dynamic context is no longer built here.  It is
compiled:

    authoritative state -> adapter -> ContextRequestV1
      -> ContextCompilerV1 -> AgentContextPackV1
      -> pack-based renderer -> the model boundary

Two consequences worth stating where they are visible.  Selection still happens
in this module, because the routing policy needs the admitted items with their
authoritative fields -- including ``entity``, which the SPECIALIST disclosure
profile does not admit -- to decide whether a generator is needed at all; that
selection is now the adapter's one implementation rather than a second copy of
it.  And disclosure no longer happens here at all: there is no ``project`` call
on this path, so this module cannot disagree with the Disclosure Authority about
what a model may see.

H2A-3C moved the model boundary behind a ``ModelBindingV1``.  This module no
longer knows how the pack becomes text or how the model is reached: it compiles,
hands the pack to a ``ModelInvocationRuntimeV1``, and reads a ``ModelResponseV1``
back.  A financial model or a DeepSeek endpoint is a different binding, not a
change to this file -- which is what the phase means by the Harness core not
moving.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from rag_v2.adaptive import AdaptiveRAGStateV1
from rag_v2.context import (
    AgentContextPackV1,
    ContextCompilerV1,
    SpecialistContextPolicyV1,
    admitted_specialist_evidence,
    specialist_context_request,
)
from rag_v2.invocation import (
    ModelBindingV1,
    ModelInvocationRuntimeV1,
)
from src.generation.financial_model_binding import build_financial_model_binding
from src.generation.generator_routing_policy import (
    GeneratorRouteDecision,
    GeneratorRoutingPolicy,
    GeneratorTarget,
    RouteName,
)

from src.domain.calculation import CalculationResult, CalculationStatus
from src.finance.calculation_renderer import render_calculation_result
from src.finance.relational_directory import RelationalOperandDirectory


#: The handles the renderer writes into the prompt and asks the model to cite:
#: ``[E1]``, ``[E2]``, ``[C1]``.  Short on purpose -- a prompt that repeated a
#: sixty-four character digest beside every evidence item would spend its
#: context budget on identifiers.
_CITED_HANDLE = re.compile(r"\[([EC]\d+)\]")


def _resolve_cited_handles(
    answer: str,
    pack: AgentContextPackV1,
) -> tuple[str, tuple[str, ...], dict[str, Any]]:
    """Translate the handles a model cited into the identities they denote.

    The renderer asks the model to cite ``[E1]``; the validator resolves
    citations against canonical fact-store ids.  Nothing reconciled the two, so
    **every** specialist answer that followed the prompt's own instruction was
    rejected as ``GV7_UNKNOWN_CITATION`` -- the model was told to write one
    vocabulary and graded in another.

    This is where the two meet.  The pairing is the pack's, not a guess: a
    handle names one evidence item and the pack carries that item's
    ``citation_id`` beside it.  So the rewrite is denotation-preserving -- it
    replaces a local alias with the global identity of the thing it names, and
    invents nothing.

    A handle the pack does not carry is left **exactly as the model wrote it**.
    Dropping it would hide a fabricated citation, and inventing an id for it
    would manufacture one; leaving it means a hallucinated ``[E9]`` still fails
    validation, which is the outcome that should happen.

    Returns the rewritten answer, the citation ids actually cited, and a record
    of what could not be resolved.
    """

    by_handle: dict[str, str] = {}
    references = getattr(pack, "references", None)
    for reference in getattr(references, "handles", ()) or ():
        citation_id = getattr(reference, "citation_id", None)
        if citation_id:
            by_handle[str(getattr(reference, "handle", ""))] = str(citation_id)

    resolved: list[str] = []
    unresolved: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        handle = match.group(1)
        citation_id = by_handle.get(handle)
        if citation_id is None:
            unresolved.append(handle)
            return match.group(0)
        resolved.append(citation_id)
        return f"[{citation_id}]"

    rewritten = _CITED_HANDLE.sub(_replace, answer)
    return (
        rewritten,
        _stable_unique(resolved),
        {
            "cited_handles_resolved": len(resolved),
            "unresolved_cited_handles": _stable_unique(unresolved),
        },
    )


class CandidateGenerationCapabilityError(RuntimeError):
    """Raised when a candidate-generation contract cannot be satisfied."""

@dataclass(frozen=True)
class CandidateExecutionResult:
    """Structured candidate result; it is never a release decision."""

    candidate_answer: str
    route: str
    route_reason: str
    bound_evidence_ids: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()
    calculation_ids: tuple[str, ...] = ()
    candidate_status: str = "CANDIDATE_READY_FOR_VALIDATION"
    generation_metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_generation_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_answer, str) or not self.candidate_answer.strip():
            raise ValueError("candidate_answer must be non-empty")
        if not isinstance(self.route, str) or not self.route.strip():
            raise ValueError("route must be non-empty")
        if self.candidate_status != "CANDIDATE_READY_FOR_VALIDATION":
            raise ValueError("candidate_status must remain validation-pending")
        object.__setattr__(self, "bound_evidence_ids", _stable_unique(self.bound_evidence_ids))
        object.__setattr__(self, "citation_ids", _stable_unique(self.citation_ids))
        object.__setattr__(self, "calculation_ids", _stable_unique(self.calculation_ids))
        object.__setattr__(self, "generation_metadata", dict(self.generation_metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_answer": self.candidate_answer,
            "route": self.route,
            "route_reason": self.route_reason,
            "bound_evidence_ids": list(self.bound_evidence_ids),
            "citation_ids": list(self.citation_ids),
            "calculation_ids": list(self.calculation_ids),
            "candidate_status": self.candidate_status,
            "generation_metadata": dict(self.generation_metadata),
            "candidate_generation_id": self.candidate_generation_id,
        }


def _stable_unique(values: Iterable[Any]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _routing_evidence_items(state: AdaptiveRAGStateV1) -> list[dict[str, Any]]:
    """The admitted evidence, in state order -- the B3 adapter's own selection.

    H2A-3B3.  This used to be a second implementation of ``_routing_evidence_items``'s
    logic living beside the compiler's copy in ``rag_v2.context.specialist``.
    One of the two had to go, and it was this one: the compiler's is the copy
    the production context path uses, so keeping a second here would have meant
    two answers to "which evidence is admitted" that agree until one is edited.

    What remains is a translation, not a selection.  The routing policy still
    needs the admitted items with their authoritative fields -- it decides
    whether a generator is needed at all, before any context is compiled, and it
    reads ``entity``/``company``/``ticker``, which the SPECIALIST disclosure
    profile does not admit.  So it cannot consume a pack, and this stays its
    input.  The exception type is translated because the two layers name their
    failures differently and a caller here has always caught this one.
    """

    try:
        return [dict(item) for item in admitted_specialist_evidence(state)]
    except TypeError as exc:
        raise CandidateGenerationCapabilityError(
            "candidate_evidence_must_be_mapping"
        ) from exc


def _bound_citation_ids(items: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    return _stable_unique(
        str(item.get("citation_id"))
        for item in items
        if item.get("citation_id")
    )


def _calculation_id(state: AdaptiveRAGStateV1) -> tuple[str, ...]:
    value = getattr(state, "calculation_result_id", None)
    return (str(value),) if value else ()


def _candidate_generation_id(
    answer: str,
    route: str,
    evidence_ids: Iterable[str],
    calculation_ids: Iterable[str],
) -> str:
    payload = {
        "answer": answer,
        "route": route,
        "evidence_ids": list(evidence_ids),
        "calculation_ids": list(calculation_ids),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return f"G1-{digest}"


class DeterministicFactRenderer:
    """Small structured renderer for an admitted single fact.

    The existing calculation renderer remains the canonical calculation
    renderer.  This renderer handles the separate direct-fact candidate shape
    without searching or parsing arbitrary answer text.
    """

    renderer_id = "deterministic_fact_renderer_tv2_04"

    def render(self, item: Mapping[str, Any], *, question: str = "") -> str:
        metric = item.get("metric") or item.get("normalized_metric") or "Result"
        period = item.get("period") or item.get("normalized_period")
        value = item.get("value")
        if value is None:
            value = item.get("parsed_numeric_value")
        unit = item.get("unit") or item.get("currency")
        scale = item.get("scale")
        citation = item.get("citation_id")
        label = str(metric)
        if period:
            label = f"{label} ({period})"
        rendered_value = "N/A" if value is None else str(value)
        suffix = " ".join(str(value) for value in (unit, scale) if value)
        answer = f"{label}: {rendered_value}"
        if suffix:
            answer += f" {suffix}"
        if citation:
            answer += f" [{citation}]"
        return answer


class TrustedV2GenerationCapability:
    """Reuse routing policy and call one candidate generator target."""

    candidate_mode = True

    @staticmethod
    def _binding_for(backend: Any) -> ModelBindingV1:
        """Bind a model backend into the financial model's ``ModelBindingV1``.

        H2A-3C.  ``model_backend=`` accepts either a provider or a legacy
        ``generate(prompt)`` backend, and the difference is one ``isinstance``
        with a large consequence: an object that already implements
        ``ModelProviderV1`` is used as the provider it says it is, and anything
        else is adapted.  A future financial or DeepSeek binding therefore
        arrives through the same parameter as the local specialist, which is
        what "the Harness core does not move" has to mean in practice.

        H2A-3E renamed the parameter from ``specialist=``, which described the
        one backend it was first written for rather than the two kinds it now
        takes.  ``TrustedV2RuntimeResources.specialist`` was left alone: it
        really does hold the specialist, and narrowing that name would have
        been a different change with a different meaning.

        P1.1 moved the body to ``build_financial_model_binding`` and left this
        as the call site.  The shape did not change; what changed is that the
        binding now carries the backend's own ``model_id`` and, once a real
        checkpoint is resident, its exact token counter.  The reasoning for both
        -- and for why no token bound is configured alongside them -- lives with
        the factory rather than being restated here.
        """

        return build_financial_model_binding(backend)

    def __init__(
        self,
        *,
        routing_policy: Any | None = None,
        renderer: Any | None = None,
        model_backend: Any | None = None,
    ) -> None:
        self.routing_policy = routing_policy or GeneratorRoutingPolicy()
        self.renderer = renderer or DeterministicFactRenderer()
        self.model_backend = model_backend
        # The model boundary, built once.  ``None`` when no backend was
        # configured, which is the state every deterministic route runs in.
        self.binding = (
            None if model_backend is None else self._binding_for(model_backend)
        )
        self.invocation = (
            None if self.binding is None else ModelInvocationRuntimeV1(self.binding)
        )
        # One compiler for the capability's lifetime, so "exactly one compile per
        # specialist invocation" is a property of the call path rather than of
        # how often someone happened to construct one.  The binding supplies the
        # exact counter when it has one; B3 configures no token bound, so nothing
        # is enforced and the counter is None.
        self.context_compiler = ContextCompilerV1(
            SpecialistContextPolicyV1(),
            token_counter=(
                None if self.binding is None else self.binding.exact_token_counter
            ),
        )
        self.route_calls = 0
        self.renderer_calls = 0
        self.specialist_calls = 0
        self.context_compile_count = 0
        self.last_decision: GeneratorRouteDecision | None = None
        self.last_result: CandidateExecutionResult | None = None
        #: The governed context compiled for the last specialist invocation, or
        #: ``None`` when the route did not reach one.  Held for the same reason
        #: ``last_result`` is: it is the artifact of this invocation, and a test
        #: that recompiled it itself would be observing a second compile rather
        #: than the one production performed.
        self.last_context_pack: AgentContextPackV1 | None = None
        self._unknown_citation_count = 0
        #: Names of the fields disclosed to the specialist on the last call.
        #: Names only: a trace must never carry evidence content.
        self.last_disclosed_fields: tuple[str, ...] = ()

    @staticmethod
    def _calculation_object(state: AdaptiveRAGStateV1) -> CalculationResult | None:
        value = getattr(state, "_calculation_result_obj", None)
        return value if isinstance(value, CalculationResult) else None

    @staticmethod
    def _relational_directory(
        state: AdaptiveRAGStateV1,
    ) -> RelationalOperandDirectory | None:
        """The ``slot_id -> operand`` projection, built from the plan's slots.

        Built here rather than inside the renderer because the renderer must not
        read the plan: one that could would be able to build a second
        `slot_id -> entity` lookup, and a second lookup is a second authority for
        one fact.

        ``None`` when the state carries no usable plan, which the renderer
        treats as "cannot state a relation" rather than as a reason to guess.
        """

        from rag_v2.contracts.plan import SupervisorPlan

        from .trusted_v2_calculation import build_relational_operand_directory

        raw = getattr(state, "plan", None)
        blob = raw.get("supervisor_plan") if isinstance(raw, Mapping) else None
        if not isinstance(blob, Mapping):
            return None
        try:
            plan = SupervisorPlan.from_dict(blob)
        except Exception:  # noqa: BLE001 - a malformed plan is not a render fault
            return None
        return build_relational_operand_directory(plan)

    def _route(
        self,
        state: AdaptiveRAGStateV1,
        items: list[dict[str, Any]],
        calculation: CalculationResult | None,
    ) -> GeneratorRouteDecision:
        calculation_payload = calculation.to_dict() if calculation else None
        route_hint = str(getattr(state, "intent", "") or "")
        try:
            decision = self.routing_policy.route(
                state.normalized_query,
                items,
                calculation_result=calculation_payload,
                route_hint=route_hint,
            )
        except Exception as exc:
            raise CandidateGenerationCapabilityError("generator_routing_failed") from exc
        # A calculation plan has already been semantically selected by the
        # Supervisor.  Do not allow evidence cardinality or wording heuristics
        # to hand numeric authority to the Specialist.
        if str(state.intent).upper() == "CALCULATION":
            decision = GeneratorRouteDecision(
                route_name=RouteName.CALCULATION_SIMPLE,
                target=GeneratorTarget.DETERMINISTIC_CALCULATOR,
                reason="PLAN_REQUIRES_DETERMINISTIC_CALCULATION",
                requires_c1=True,
            )
        self.route_calls += 1
        self.last_decision = decision
        return decision

    @staticmethod
    def _call_renderer(
        renderer: Any,
        *,
        item: Mapping[str, Any] | None,
        question: str,
        calculation: CalculationResult | None,
        directory: RelationalOperandDirectory | None = None,
    ) -> str:
        if calculation is not None:
            # The existing calculation_renderer is the authoritative path.  It
            # needs the directory to state a relation and returns "" without it,
            # so this passes whatever the caller could build rather than letting
            # the renderer go looking for a plan it must not read.
            return render_calculation_result(calculation, directory=directory)
        if item is None:
            raise CandidateGenerationCapabilityError("renderer_missing_bound_fact")
        method = getattr(renderer, "render", None)
        if callable(method):
            try:
                value = method(item, question=question)
            except TypeError:
                value = method(item)
        elif callable(renderer):
            value = renderer(item)
        else:
            raise CandidateGenerationCapabilityError("renderer_not_callable")
        if not isinstance(value, str) or not value.strip():
            raise CandidateGenerationCapabilityError("renderer_returned_empty_candidate")
        return value.strip()

    def _call_specialist(
        self,
        state: AdaptiveRAGStateV1,
        allowed_citations: tuple[str, ...],
    ) -> tuple[str, tuple[str, ...], dict[str, Any]]:
        """Compile, render, and call the model -- once each.

        H2A-3B3.  This used to assemble the specialist's context by hand:
        ``_routing_evidence_items`` selected, ``project`` disclosed,
        ``project_calculation`` disclosed the calculation, and the three pieces
        went to a backend that rendered them.  Every one of those steps now
        belongs to the Context Runtime, and this method is the wiring that proves
        it takes over:

            authoritative state -> adapter -> ContextRequestV1
              -> ContextCompilerV1 -> AgentContextPackV1
              -> invocation runtime -> ModelRequestV1 -> provider

        The order is the point, and so is the absence of a bypass.  After the
        compile there is no ``project`` call here, no reading of
        ``state.evidence_packets``, and no calculation payload assembled beside
        the pack -- a second projection at this layer would be a second
        disclosure policy, which is the asymmetry H2A-1 existed to remove.

        H2A-3C moved the last two steps behind a binding.  What is left here is
        the compile and the hand-off: this method no longer knows how the text is
        rendered or how the model is reached, which is what lets a financial
        model or a DeepSeek endpoint arrive as a new ``ModelBindingV1`` rather
        than as a change to this file.
        """
        if self.model_backend is None or self.invocation is None:
            raise CandidateGenerationCapabilityError("financial_specialist_not_configured")

        # 1. The adapter reads the runtime's authoritative state.
        request = specialist_context_request(state)
        # 2. The compiler is the only path from those artifacts to model-visible
        #    context.  Compiled once per invocation, here.
        pack = self.context_compiler.compile(request)
        self.context_compile_count += 1
        self.last_context_pack = pack

        # ``last_disclosed_fields`` records which fields crossed, by name only
        # and never by value, so a disclosure question can be answered from the
        # snapshot without putting evidence content into a trace.  Namespaced by
        # artifact type.  The audit found the trace accounting for evidence
        # fields while the calculation payload crossed unrecorded, so "what was
        # this model allowed to see" had an incomplete answer.  Field names only,
        # never values.  It is read from the pack because the pack is what
        # crossed.
        self.last_disclosed_fields = tuple(
            [
                f"evidence.{field}"
                for field in sorted({f for view in pack.evidence for f in view})
            ]
            + [f"calculation.{field}" for field in sorted(pack.calculation or {})]
        )
        self.specialist_calls += 1

        # 3. The invocation runtime renders the pack once and calls the provider
        #    once.  Everything between the pack and the model is now one
        #    component's job, which is what makes "exactly one of each" a
        #    property of a call rather than a claim about a module.
        response = self.invocation.invoke(pack)

        answer = response.text
        raw_citations = response.citations
        metadata: dict[str, Any] = {}
        if not isinstance(answer, str) or not answer.strip():
            raise CandidateGenerationCapabilityError("financial_specialist_empty_candidate")

        # The model answers in the renderer's handle vocabulary; the validator
        # reads the canonical one.  Translate before either is consulted.
        answer, cited_citation_ids, handle_metadata = _resolve_cited_handles(answer, pack)
        metadata.update(handle_metadata)

        raw_citations = raw_citations if isinstance(raw_citations, (list, tuple, set)) else ()
        unknown = [str(item) for item in raw_citations if str(item) not in allowed_citations]
        self._unknown_citation_count += len(unknown)
        metadata.update(
            {
                "provider_citation_ids": [str(item) for item in raw_citations],
                "unknown_generated_citation_ids": unknown,
            }
        )
        # Three tiers, in order of how much the candidate actually declared.
        # The middle one is new: it is what the model cited in its own answer,
        # resolved through the pack.  Before it existed, a model that cited
        # correctly resolved to nothing and fell through to the third tier --
        # which attributes every bound citation to a candidate that named none,
        # so a hallucinating answer and a careful one were indistinguishable.
        declared = tuple(
            str(item) for item in raw_citations if str(item) in allowed_citations
        )
        from_answer = tuple(
            citation_id for citation_id in cited_citation_ids if citation_id in allowed_citations
        )
        citations = declared or from_answer or allowed_citations
        metadata["citation_source"] = (
            "provider_declared"
            if declared
            else "answer_handles" if from_answer else "bound_evidence_fallback"
        )
        return answer.strip(), _stable_unique(citations), metadata

    def generate(self, state: AdaptiveRAGStateV1) -> CandidateExecutionResult:
        # Cleared per invocation: a deterministic route compiles nothing, and a
        # stale pack left over from a previous call would read as this one's.
        self.last_context_pack = None
        items = _routing_evidence_items(state)
        if not items:
            raise CandidateGenerationCapabilityError("generation_requires_bound_evidence")
        calculation = self._calculation_object(state)
        decision = self._route(state, items, calculation)
        evidence_ids = _stable_unique(getattr(state, "bound_evidence_ids", ()))
        allowed_citations = _bound_citation_ids(items)
        calculation_ids = _calculation_id(state)
        metadata: dict[str, Any] = {
            "renderer_id": getattr(self.renderer, "renderer_id", None),
            "route_name": decision.route_name.value,
            "route_target": decision.target.value,
            "route_reason": decision.reason,
            "calculator_invoked": calculation is not None,
        }

        if decision.target is GeneratorTarget.FAIL_CLOSED_PRE_GEN:
            raise CandidateGenerationCapabilityError("generator_route_fail_closed")
        if decision.target is GeneratorTarget.DETERMINISTIC_CALCULATOR:
            if calculation is None or calculation.status is not CalculationStatus.EXECUTED:
                raise CandidateGenerationCapabilityError("calculation_result_not_ready")
            # A relational result is stated from this directory, not from prose.
            # That is what makes the release invariant enforceable rather than
            # hopeful: there is no prose for a wrong relation to hide in.
            answer = render_calculation_result(
                calculation, directory=self._relational_directory(state)
            )
            self.renderer_calls += 1
        elif decision.target is GeneratorTarget.DETERMINISTIC_RENDERER:
            item = items[0] if items else None
            answer = self._call_renderer(
                self.renderer,
                item=item,
                question=state.normalized_query,
                calculation=calculation,
                directory=self._relational_directory(state),
            )
            self.renderer_calls += 1
        elif decision.target is GeneratorTarget.LOCAL_SPECIALIST:
            answer, citations, specialist_metadata = self._call_specialist(
                state, allowed_citations
            )
            metadata.update(specialist_metadata)
            allowed_citations = citations
        else:
            raise CandidateGenerationCapabilityError("unsupported_generator_target")

        result = CandidateExecutionResult(
            candidate_answer=answer,
            route=decision.route_name.value,
            route_reason=decision.reason,
            bound_evidence_ids=evidence_ids,
            citation_ids=allowed_citations,
            calculation_ids=calculation_ids,
            generation_metadata=metadata,
            candidate_generation_id=_candidate_generation_id(
                answer, decision.route_name.value, evidence_ids, calculation_ids
            ),
        )
        self.last_result = result
        state.generation_route = decision.route_name.value
        state.route_reason = decision.reason
        state.candidate_answer = result.candidate_answer
        state.candidate_generation_id = result.candidate_generation_id
        state.candidate_status = result.candidate_status
        state.validation_pending = True
        return result

    def trace_snapshot(self) -> dict[str, Any]:
        result = self.last_result
        pack = self.last_context_pack
        return {
            "generation_route": result.route if result else None,
            "route_reason": result.route_reason if result else None,
            "route_calls": self.route_calls,
            "renderer_invoked": self.renderer_calls > 0,
            "renderer_call_count": self.renderer_calls,
            "specialist_invoked": self.specialist_calls > 0,
            "specialist_call_count": self.specialist_calls,
            # H2A-3B3.  The migration's own observability, and deliberately
            # counts rather than contents: the compile and render counts are what
            # make "one context compilation per specialist invocation" checkable
            # from outside, and the support-group count is the one fact about the
            # topology the current B3 prompt does not carry.  Nothing here can
            # grow into a copy of the pack -- a trace is a record of decisions,
            # not a second context store.
            #
            # H2A-3C classifies all of it as an **observability projection**: no
            # runtime decision reads any of these, and the render count now
            # comes from the invocation runtime that performs the render rather
            # than from a counter kept here that could drift from it.
            "context_compile_count": self.context_compile_count,
            "context_render_count": (
                0 if self.invocation is None else self.invocation.render_count
            ),
            "context_provider_count": (
                0 if self.invocation is None else self.invocation.provider_count
            ),
            "context_evidence_selected": (
                None if pack is None else pack.budget.evidence_selected
            ),
            "context_evidence_dropped": (
                None if pack is None else pack.budget.evidence_dropped
            ),
            "context_support_group_count": (
                None if pack is None else len(pack.references.support_groups)
            ),
            "candidate_ready": result is not None,
            "validation_pending": result is not None,
            "candidate_generation_id": result.candidate_generation_id if result else None,
            "unknown_generated_citation_count": self._unknown_citation_count,
            "disclosed_fields": list(self.last_disclosed_fields),
        }


__all__ = [
    "CandidateExecutionResult",
    "CandidateGenerationCapabilityError",
    "DeterministicFactRenderer",
    "TrustedV2GenerationCapability",
]