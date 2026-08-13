"""Deterministic one-pass/one-fallback trusted generation state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import monotonic
from typing import Any, Mapping

from .contracts import (GenerationAttemptRecordV1, GenerationInputV1,
                        GenerationValidationReportV1, ValidationSeverity)
from .providers import GeneratorProviderV1, ProviderRegistryV1
from .recovery import GenerationRecoveryPolicyV1, RecoveryAction
from .validator import RuntimeGenerationValidatorV1


class GenerationState(str, Enum):
    READY_FOR_GENERATION = "READY_FOR_GENERATION"
    FIRST_PASS_GENERATED = "FIRST_PASS_GENERATED"
    FIRST_PASS_VALIDATED = "FIRST_PASS_VALIDATED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    RECOVERY_GENERATED = "RECOVERY_GENERATED"
    RECOVERY_VALIDATED = "RECOVERY_VALIDATED"
    RELEASED = "RELEASED"
    ABSTAINED = "ABSTAINED"
    PROVIDER_ERROR = "PROVIDER_ERROR"


@dataclass(frozen=True)
class GenerationRunResultV1:
    state: GenerationState
    released: bool
    envelope: Any
    validation_report: GenerationValidationReportV1 | None
    attempts: tuple[GenerationAttemptRecordV1, ...]
    recovery_action: RecoveryAction
    trace: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state.value, "released": self.released,
                "answer_envelope": self.envelope.to_dict() if self.envelope else None,
                "validation_report": self.validation_report.to_dict() if self.validation_report else None,
                "attempts": [item.to_dict() for item in self.attempts],
                "recovery_action": self.recovery_action.value,
                "trace": [dict(item) for item in self.trace]}


class TrustedGenerationStateMachineV1:
    """Owns budgets and release decisions; providers cannot control state."""

    def __init__(self, registry: ProviderRegistryV1, validator: RuntimeGenerationValidatorV1,
                 recovery_policy: GenerationRecoveryPolicyV1, *, max_attempts: int = 2,
                 fallback_budget: int = 1) -> None:
        if max_attempts != 2 or fallback_budget != 1:
            raise ValueError("R0 freezes max_attempts=2 and fallback_budget=1")
        self.registry = registry
        self.validator = validator
        self.recovery_policy = recovery_policy

    def _trace(self, input_: GenerationInputV1 | None, provider: str | None,
               attempt: int | None, validation: GenerationValidationReportV1 | None,
               action: RecoveryAction, state: GenerationState, released: bool,
               latency: float | None = None) -> dict[str, Any]:
        return {"query_id": input_.query_id if input_ else None, "route": input_.route if input_ else None,
                "provider": provider, "attempt_index": attempt,
                "validation_status": validation.status.value if validation else None,
                "validation_failure_codes": list(validation.failure_codes) if validation else [],
                "recovery_action": action.value, "terminal_state": state.value,
                "released": released, "latency_ms": latency}

    def run(self, generation_input: GenerationInputV1 | None, *, no_answer: bool = False) -> GenerationRunResultV1:
        if no_answer or generation_input is None:
            state = GenerationState.ABSTAINED
            trace = (self._trace(generation_input, None, None, None, RecoveryAction.NO_RECOVERY, state, False),)
            return GenerationRunResultV1(state, False, None, None, (), RecoveryAction.NO_RECOVERY, trace)

        attempts: list[GenerationAttemptRecordV1] = []
        trace: list[Mapping[str, Any]] = []
        primary = self.registry.resolve(self.recovery_policy.primary_provider)
        if primary is None:
            state = GenerationState.PROVIDER_ERROR
            trace.append(self._trace(generation_input, self.recovery_policy.primary_provider, 0, None,
                                     RecoveryAction.NO_RECOVERY, state, False))
            return GenerationRunResultV1(state, False, None, None, (), RecoveryAction.NO_RECOVERY, tuple(trace))

        def call(provider: GeneratorProviderV1, index: int, reason: str | None) -> tuple[Any, GenerationValidationReportV1 | None, float, bool]:
            started = monotonic()
            try:
                envelope = provider.generate(generation_input, {"attempt_index": index, "recovery_reason": reason})
                report = self.validator.validate(generation_input.packet, envelope)
                return envelope, report, (monotonic() - started) * 1000, True
            except Exception as exc:
                return exc, None, (monotonic() - started) * 1000, False

        envelope, report, latency, ok = call(primary, 0, None)
        if not ok:
            attempts.append(GenerationAttemptRecordV1(generation_input.query_id, 0, primary.metadata.provider_id,
                                                       primary.metadata.model_id, None, None, "provider_exception",
                                                       GenerationState.PROVIDER_ERROR.value, latency))
            trace.append(self._trace(generation_input, primary.metadata.provider_id, 0, None,
                                     RecoveryAction.NO_RECOVERY, GenerationState.PROVIDER_ERROR, False, latency))
            action = self.recovery_policy.choose(("PROVIDER_ERROR",))
        else:
            trace.append(self._trace(generation_input, primary.metadata.provider_id, 0, None,
                                     RecoveryAction.NO_RECOVERY, GenerationState.FIRST_PASS_GENERATED, False, latency))
            attempts.append(GenerationAttemptRecordV1(generation_input.query_id, 0, primary.metadata.provider_id,
                                                       primary.metadata.model_id, envelope, report, None,
                                                       GenerationState.FIRST_PASS_VALIDATED.value, latency))
            trace.append(self._trace(generation_input, primary.metadata.provider_id, 0, report,
                                     RecoveryAction.NO_RECOVERY, GenerationState.FIRST_PASS_VALIDATED, False, latency))
            if report and report.status is ValidationSeverity.PASS:
                state = GenerationState.RELEASED
                trace.append(self._trace(generation_input, primary.metadata.provider_id, 0, report,
                                         RecoveryAction.NO_RECOVERY, state, True, latency))
                return GenerationRunResultV1(state, True, envelope, report, tuple(attempts), RecoveryAction.NO_RECOVERY, tuple(trace))
            action = self.recovery_policy.choose(report.failure_codes if report else ("PROVIDER_ERROR",),
                                                 report.status if report else None)

        if action is not RecoveryAction.FALLBACK_PROVIDER or not self.recovery_policy.fallback_provider:
            state = GenerationState.ABSTAINED
            trace.append(self._trace(generation_input, None, None, report if ok else None, action, state, False))
            return GenerationRunResultV1(state, False, envelope if ok else None, report if ok else None,
                                         tuple(attempts), action, tuple(trace))
        trace.append(self._trace(generation_input, self.recovery_policy.fallback_provider, 1,
                                 report if ok else None, action, GenerationState.RECOVERY_REQUIRED, False))
        fallback = self.registry.resolve(self.recovery_policy.fallback_provider)
        if fallback is None:
            state = GenerationState.PROVIDER_ERROR
            trace.append(self._trace(generation_input, self.recovery_policy.fallback_provider, 1, None, action, state, False))
            return GenerationRunResultV1(state, False, None, None, tuple(attempts), action, tuple(trace))

        envelope2, report2, latency2, ok2 = call(fallback, 1, "first_pass_validation_failed")
        if not ok2:
            attempts.append(GenerationAttemptRecordV1(generation_input.query_id, 1, fallback.metadata.provider_id,
                                                       fallback.metadata.model_id, None, None, "fallback_provider_exception",
                                                       GenerationState.PROVIDER_ERROR.value, latency2))
            state = GenerationState.ABSTAINED
            trace.append(self._trace(generation_input, fallback.metadata.provider_id, 1, None, action, state, False, latency2))
            return GenerationRunResultV1(state, False, None, None, tuple(attempts), action, tuple(trace))
        trace.append(self._trace(generation_input, fallback.metadata.provider_id, 1, None,
                                 action, GenerationState.RECOVERY_GENERATED, False, latency2))
        attempts.append(GenerationAttemptRecordV1(generation_input.query_id, 1, fallback.metadata.provider_id,
                                                   fallback.metadata.model_id, envelope2, report2, "first_pass_validation_failed",
                                                   GenerationState.RECOVERY_VALIDATED.value, latency2))
        if report2.status is ValidationSeverity.PASS:
            state = GenerationState.RELEASED
            trace.append(self._trace(generation_input, fallback.metadata.provider_id, 1, report2, action, state, True, latency2))
            return GenerationRunResultV1(state, True, envelope2, report2, tuple(attempts), action, tuple(trace))
        state = GenerationState.ABSTAINED
        trace.append(self._trace(generation_input, fallback.metadata.provider_id, 1, report2, action, state, False, latency2))
        return GenerationRunResultV1(state, False, envelope2, report2, tuple(attempts), action, tuple(trace))
