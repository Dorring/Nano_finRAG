"""Top-level TrustedRAGRuntimeV2 coordinator."""

from __future__ import annotations

from typing import Any, Mapping

from rag_v2.contracts.plan import Intent, SupervisorPlan
from rag_v2.generation import (GenerationRecoveryPolicyV1,
                               GenericVerifiedPacketRendererV1, RecoveryAction,
                               TrustedGenerationStateMachineV1)
from rag_v2.generation.providers import ProviderRegistryV1

from .contracts import (RuntimeTraceV1, TerminalReason, TrustedRAGQueryV2,
                        TrustedRAGResponseV2)
from .evidence import TrustedEvidenceGateV1
from .routing import GeneratorRoutingPolicyV1
from .semantic_claims import SemanticClaimVerifierV1


class TrustedRAGRuntimeV2:
    """Route-aware coordinator; all release authority remains deterministic."""

    def __init__(self, registry: ProviderRegistryV1, routing_policy: GeneratorRoutingPolicyV1,
                 *, evidence_gate: TrustedEvidenceGateV1 | None = None,
                 renderer: Any | None = None,
                 semantic_verifier: SemanticClaimVerifierV1 | None = None) -> None:
        self.registry = registry
        self.routing_policy = routing_policy
        self.evidence_gate = evidence_gate or TrustedEvidenceGateV1()
        self.renderer = renderer or GenericVerifiedPacketRendererV1()
        self.semantic_verifier = semantic_verifier or SemanticClaimVerifierV1()

    @staticmethod
    def _plan(value: SupervisorPlan | Mapping[str, Any] | None) -> SupervisorPlan | None:
        if isinstance(value, SupervisorPlan):
            return value
        if isinstance(value, Mapping):
            try:
                return SupervisorPlan.from_dict(value)
            except Exception:
                return None
        return None

    @staticmethod
    def _route(plan: SupervisorPlan | None) -> str | None:
        return plan.intent.value if plan else None

    def _response(self, query: TrustedRAGQueryV2, route: str | None, status: str,
                  reason: TerminalReason, *, plan_valid: bool, evidence_available: bool,
                  evidence_source: str | None = None, result: Any = None,
                  primary: str | None = None, fallback: str | None = None) -> TrustedRAGResponseV2:
        attempts = tuple(result.attempts) if result else ()
        attempt_trace = tuple(item.to_dict() for item in attempts)
        codes = tuple(tuple(item.validation_report.failure_codes) if item.validation_report else ()
                      for item in attempts)
        latencies = tuple(item.latency_ms for item in attempts if item.latency_ms is not None)
        models = result.envelope if result and result.released else None
        citations = tuple(models.citation_ids) if models else ()
        provider = models.generator_provider if models else None
        model = models.generator_model if models else None
        used_fallback = len(attempts) > 1
        trace = RuntimeTraceV1(query_id=query.query_id, route=route, supervisor_plan_valid=plan_valid,
                               trusted_evidence_available=evidence_available, evidence_source=evidence_source,
                               generation_attempts=attempt_trace, primary_provider=primary,
                               fallback_provider=fallback, validator_codes=codes,
                               fallback_triggered=used_fallback, released=status == "RELEASED",
                               terminal_reason=reason, latencies_ms=latencies, trace_id=query.trace_id)
        answer = models.answer_text if models else None
        validation_status = result.validation_report.status.value if result and result.validation_report else None
        return TrustedRAGResponseV2(query_id=query.query_id, route=route, status=status, answer_text=answer,
                                    citation_ids=citations, generation_provider=provider,
                                    generation_model=model, used_fallback=used_fallback,
                                    attempt_count=len(attempts), validation_status=validation_status,
                                    terminal_reason=reason, trace_id=query.trace_id, trace=trace)

    def handle(self, query: TrustedRAGQueryV2) -> TrustedRAGResponseV2:
        if query.no_answer:
            return self._response(query, None, "ABSTAINED", TerminalReason.TR7_NO_ANSWER,
                                  plan_valid=False, evidence_available=False)
        plan = self._plan(query.supervisor_plan)
        if plan is None:
            return self._response(query, None, "ABSTAINED", TerminalReason.TR10_OTHER,
                                  plan_valid=False, evidence_available=False)
        route = plan.intent.value
        packet = query.trusted_evidence_packet
        gate = self.evidence_gate.validate(plan, packet, query.query_id)
        if not gate.valid:
            if plan.intent is Intent.CALCULATION:
                reason = TerminalReason.TR8_CALCULATION_NOT_READY
            elif plan.intent is Intent.MULTI_EVIDENCE:
                reason = TerminalReason.TR9_MULTI_NOT_READY
            else:
                reason = TerminalReason.TR2_NO_TRUSTED_EVIDENCE
            return self._response(query, route, "ABSTAINED", reason, plan_valid=True,
                                  evidence_available=False, evidence_source=gate.source)
        config = self.routing_policy.for_route(route)
        if not config.primary:
            reason = TerminalReason.TR8_CALCULATION_NOT_READY if plan.intent is Intent.CALCULATION else TerminalReason.TR9_MULTI_NOT_READY if plan.intent is Intent.MULTI_EVIDENCE else TerminalReason.TR2_NO_TRUSTED_EVIDENCE
            return self._response(query, route, "ABSTAINED", reason, plan_valid=True,
                                  evidence_available=True, evidence_source=gate.source)
        recovery = GenerationRecoveryPolicyV1(
            primary_provider=config.primary,
            fallback_provider=config.fallback,
            action=RecoveryAction.FALLBACK_PROVIDER if config.fallback else RecoveryAction.NO_RECOVERY,
            fallback_budget=1 if config.fallback else 0,
            fallback_on_soft_fail=config.fallback_on_soft_fail,
        )
        machine = TrustedGenerationStateMachineV1(
            self.registry, self._validator(), recovery,
            semantic_verifier=self.semantic_verifier,
        )
        try:
            generation_input = self.renderer.render(packet or {})
            result = machine.run(generation_input)
        except Exception:
            return self._response(query, route, "ABSTAINED", TerminalReason.TR10_OTHER,
                                  plan_valid=True, evidence_available=True, evidence_source=gate.source,
                                  primary=config.primary, fallback=config.fallback)
        if result.released:
            reason = TerminalReason.TR1_RELEASED_FALLBACK if len(result.attempts) > 1 else TerminalReason.TR0_RELEASED_PRIMARY
            return self._response(query, route, "RELEASED", reason, plan_valid=True,
                                  evidence_available=True, evidence_source=gate.source, result=result,
                                  primary=config.primary, fallback=config.fallback)
        if result.state.value == "PROVIDER_ERROR" or (
            len(result.attempts) == 1 and result.attempts[0].terminal_state == "PROVIDER_ERROR"
        ):
            reason = TerminalReason.TR5_PROVIDER_ERROR
        elif len(result.attempts) > 1:
            reason = TerminalReason.TR4_FALLBACK_VALIDATION_FAIL
        else:
            reason = TerminalReason.TR3_PRIMARY_VALIDATION_FAIL_NO_FALLBACK
        return self._response(query, route, "ABSTAINED", reason, plan_valid=True,
                              evidence_available=True, evidence_source=gate.source, result=result,
                              primary=config.primary, fallback=config.fallback)

    @staticmethod
    def _validator():
        from rag_v2.generation import RuntimeGenerationValidatorV1
        return RuntimeGenerationValidatorV1()
