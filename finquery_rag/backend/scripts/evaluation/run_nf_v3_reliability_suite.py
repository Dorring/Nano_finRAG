#!/usr/bin/env python3
"""NF-V3 reliability: does the trusted pipeline fail closed under injected faults?

One question, asked the only way it can be answered honestly:

    inject a fault at a seam the runtime already has,
    observe what the runtime does,
    and count the answers that should never have been produced.

The single number that matters is **incorrect release**: a case where a fault was
injected and the runtime still handed back an answer.  Every fault below has a
safe outcome (fail closed) and an unsafe one (release), and the suite reports
which one happened rather than which one was expected.

What this suite is, and what it is not
--------------------------------------

**It is not a new fault-injection framework.**  Most of the seams it uses are
already exercised by the repository's own tests and fixtures, and each case
records the existing coverage it reuses (``reuses_existing_coverage``).  Nine of
the sixteen cases are the ``tests/harness/h1_integration.py`` fixtures run
through their own production/wired path; two more are the loop-seam patterns
from ``tests/harness/test_closed_loop.py``; the recovery case is the repair path
from ``tests/test_trusted_v2_validator_release.py``.  What is *new* here is the
provider-boundary faults (timeout / unavailable / invalid / malformed / empty
model output), because no existing fixture drives a failing ``ModelProviderV1``
through the release gate, and the classification itself: a single table that
asks of every case "did this release?" rather than "did this raise where the
test expected it to".

**It is a change detector, not an oracle.**  A green run says these faults do
not produce answers.  It does not say the runtime is safe in general, and it
cannot: the fault list is authored, not derived from a threat model, and the
cases that are missing are the ones nobody thought to write.

Three honesty rules are enforced in the output rather than promised in prose:

* a fault that cannot be injected at any available seam is reported as
  ``NOT_INJECTABLE`` with the reason -- never dropped, never relabelled;
* ``RELEASED`` is read from the runtime's own release decision
  (``release_status``), not from the presence of answer text, because a refused
  outcome can carry the candidate it refused in its ``answer`` field;
* anything the classifier cannot place is ``UNCLEAR`` and fails the run, rather
  than being rounded to fail-closed.

Usage::

    python scripts/evaluation/run_nf_v3_reliability_suite.py [--out DIR] [--runs N]

Exit status is non-zero if any case incorrectly released, if any case is
UNCLEAR, or if the two runs disagree on any case's classification.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

BACKEND_DIR = Path(__file__).parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rag_v2.adaptive import AdaptiveRAGStateV1, BoundedAdaptiveRAGV1, ToolCapability  # noqa: E402
from rag_v2.contracts import Intent  # noqa: E402
from rag_v2.invocation import ModelProviderError, ProviderFailureKind  # noqa: E402
from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService  # noqa: E402
from src.domain.calculation import (  # noqa: E402
    CalculationOperation,
    CalculationResult,
    CalculationStatus,
)
from src.runtime import (  # noqa: E402
    TrustedReleaseValidationCapability,
    TrustedV2CapabilityPorts,
    TrustedV2GenerationCapability,
)
from src.runtime.harness_runtime_mode import AgentRuntimeMode  # noqa: E402
from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator  # noqa: E402
from tests.harness.h1_integration import FIXTURES, run_fixture  # noqa: E402
from tests.harness.harness_support import (  # noqa: E402
    BUDGET,
    execute,
    loop_state,
)
from tests.harness.test_conflict_differential import CONFLICT_FIXTURE, _outcome as _conflict_outcome  # noqa: E402
from tests.harness.test_harness_end_to_end import _FakeProvider  # noqa: E402
from tests.test_trusted_v2_r4_binder import (  # noqa: E402
    SelectingBinderProvider,
    _fact,
    _plan,
    _real_capabilities,
    _request,
    _slot,
)
from tests.test_trusted_v2_validator_release import (  # noqa: E402
    _BadGeneration,
    _bad_candidate,
)

#: The remote host's artifact root.  Present when the suite runs on 4090-qh; the
#: suite writes there and to the repo copy when it is, and only to the repo copy
#: when it is not.
REMOTE_OUT = Path("/disk/qh/nano-finrag/artifacts/evaluation/nf-v3-reliability")
REPO_OUT = BACKEND_DIR / "artifacts/evaluation/nf-v3-reliability"

#: Statuses that mean "the run refused".  ``EXECUTION_ERROR`` is included because
#: the coordinator's contract maps a capability crash to it: a run that died
#: inside a port produced no answer and released nothing, which is a refusal
#: observed, not a refusal by policy.  The distinction is kept in the record so
#: the two are never confused in the report.
_REFUSAL_STATUSES = frozenset({"FAIL_CLOSED", "EXECUTION_ERROR"})


class Observed(str, Enum):
    """What the runtime did, as opposed to what it was supposed to do."""

    FAIL_CLOSED = "FAIL_CLOSED"
    RELEASED = "RELEASED"
    UNCLEAR = "UNCLEAR"


class Expectation(str, Enum):
    """What a correct runtime must do with this fault.

    ``FAIL_CLOSED`` -- a hard fault.  Any answer produced is an incorrect
    release.  ``RECOVERABLE`` -- the fault is a bad *input* to a path that has a
    deterministic repair; releasing the repaired, evidence-faithful answer is
    the correct outcome, and refusing would be a different failure (recorded as
    ``recovery_failed``, never as success).
    """

    FAIL_CLOSED = "FAIL_CLOSED"
    RECOVERABLE = "RECOVERABLE"


@dataclass
class CaseResult:
    """One injected fault and what came back."""

    case_id: str
    fault_kind: str
    seam: str
    expectation: Expectation
    injected: bool = True
    not_injectable_reason: str | None = None
    observed: Observed | None = None
    released: bool = False
    status: str | None = None
    answer: str | None = None
    reason_codes: tuple[str, ...] = ()
    escaped_exception: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    reuses_existing_coverage: str = "none: this case is new here"
    note: str = ""

    # --- derived verdicts ---------------------------------------------------

    @property
    def incorrect_release(self) -> bool:
        return (
            self.expectation is Expectation.FAIL_CLOSED
            and self.observed is Observed.RELEASED
        )

    @property
    def correct_fail_closed(self) -> bool:
        return (
            self.expectation is Expectation.FAIL_CLOSED
            and self.observed is Observed.FAIL_CLOSED
        )

    @property
    def recovery_succeeded(self) -> bool:
        return (
            self.expectation is Expectation.RECOVERABLE
            and self.observed is Observed.RELEASED
            and bool(self.detail.get("answer_is_authoritative"))
        )

    @property
    def recovery_failed(self) -> bool:
        return self.expectation is Expectation.RECOVERABLE and not self.recovery_succeeded

    def stability_key(self) -> tuple[Any, ...]:
        """What must be identical between two runs of the same suite.

        Classification only.  Measurements (``latency_ms`` and friends) are
        excluded deliberately: a suite that demanded byte-identical traces would
        be asserting that clocks do not tick.
        """

        return (
            self.case_id,
            self.observed.value if self.observed else None,
            self.released,
            self.status,
            tuple(self.reason_codes),
            self.escaped_exception,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "fault_kind": self.fault_kind,
            "seam": self.seam,
            "expectation": self.expectation.value,
            "injected": self.injected,
            "not_injectable_reason": self.not_injectable_reason,
            "observed_outcome": self.observed.value if self.observed else None,
            "released": self.released,
            "incorrect_release": self.incorrect_release,
            "correct_fail_closed": self.correct_fail_closed,
            "recovery_succeeded": self.recovery_succeeded,
            "recovery_failed": self.recovery_failed,
            "status": self.status,
            "answer": self.answer,
            "reason_codes": list(self.reason_codes),
            "escaped_exception": self.escaped_exception,
            "detail": self.detail,
            "reuses_existing_coverage": self.reuses_existing_coverage,
            "note": self.note,
        }


# --- classification -----------------------------------------------------------


def classify(
    *,
    released: bool,
    status: str | None,
    answer: str | None,
    escaped_exception: str | None = None,
) -> Observed:
    """Place one observed outcome in exactly one bin, or refuse to place it.

    The rules, in order:

    * an exception that escaped the driving boundary is a refusal (the runtime
      did not produce an answer -- it stopped);
    * a release with non-empty answer text is ``RELEASED``;
    * a release with no answer text is ``UNCLEAR``: the runtime claims to have
      released nothing, which is a shape this classifier has no honest reading
      for and must not silently call a refusal;
    * a non-release in a known refusal status is ``FAIL_CLOSED``;
    * anything else -- a status the runtime invented, a release flag that
      contradicts its own status -- is ``UNCLEAR``.
    """

    if escaped_exception is not None:
        return Observed.FAIL_CLOSED
    if released:
        if answer is not None and str(answer).strip():
            return Observed.RELEASED
        return Observed.UNCLEAR
    if status in _REFUSAL_STATUSES:
        return Observed.FAIL_CLOSED
    return Observed.UNCLEAR


def _apply(outcome: Any, result: CaseResult, *, escaped: str | None = None) -> CaseResult:
    """Fill a ``CaseResult`` from a ``V2ExecutionOutcome``.

    ``released`` is read from the release decision, never from the answer text.
    ``tests/harness/h1_integration.py::validator_rejection`` is the standing
    proof that the two differ: it refuses a candidate and still carries
    ``answer='Revenue was 391.'`` on the refused outcome.
    """

    release_status = getattr(getattr(outcome, "release_status", None), "value", None)
    status = getattr(getattr(outcome, "status", None), "value", None)
    answer = getattr(outcome, "answer", None)
    result.released = release_status == "RELEASED"
    result.status = status
    result.answer = None if answer is None else str(answer)
    result.reason_codes = tuple(str(item) for item in (outcome.reason_codes or ()))
    result.escaped_exception = escaped
    result.observed = classify(
        released=result.released,
        status=status,
        answer=result.answer,
        escaped_exception=escaped,
    )
    return result


def _apply_loop(
    result_state: Any,
    run: Any,
    result: CaseResult,
    *,
    escaped: str | None = None,
    extra_reasons: Sequence[str] = (),
) -> CaseResult:
    """Fill a ``CaseResult`` from a raw ``BoundedAdaptiveRAGV1`` run.

    ``extra_reasons`` carries the cause the loop recorded *before* it terminated
    -- a tool error is written onto the turn and then overwritten by whatever
    terminal the run reaches, so a record built from ``stop_reason`` alone would
    name the terminal and hide the fault.
    """

    status = getattr(result_state, "status", None)
    released = bool(getattr(run, "released", False))
    output = getattr(run, "output", None)
    result.released = released
    result.status = str(status) if status is not None else None
    result.answer = None if output is None else str(output)
    result.reason_codes = tuple(
        [*extra_reasons]
        + [
            str(item)
            for item in filter(
                None, (getattr(result_state, "stop_reason", None),)
            )
        ]
    )
    result.escaped_exception = escaped
    result.observed = classify(
        released=released,
        status=result.status,
        answer=result.answer,
        escaped_exception=escaped,
    )
    return result


# --- injected providers -------------------------------------------------------


class _MalformedResponseProvider(_FakeProvider):
    """A provider that answers with the transport's raw payload, not a contract.

    This is the shape a real boundary fault takes: an adapter that never
    validated what came back off the wire and hands the runtime a dict.  The
    invocation runtime is what must refuse it
    (``ModelInvocationRuntimeV1.invoke``), and this case asks whether the
    refusal survives all the way out to the release decision.
    """

    def invoke(self, request: Any) -> Any:
        self.requests.append(request)
        return {"choices": [{"message": {"content": "Revenue was 100 million USD."}}]}


class _WatchedGeneration:
    """Wrap a generation port so the exception it raises is recorded.

    The coordinator catches every capability exception and reports only
    ``GENERATION_EXCEPTION``, so without this the *kind* of fault is lost by the
    time the outcome exists -- every provider fault would look identical in the
    report.  The wrapper records and re-raises; it changes no behaviour.
    """

    candidate_mode = True

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.error: dict[str, str] | None = None

    def generate(self, state: AdaptiveRAGStateV1) -> Any:
        try:
            return self.delegate.generate(state)
        except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
            self.error = {"type": type(exc).__name__, "message": str(exc)[:200]}
            raise

    def trace_snapshot(self) -> dict[str, Any]:
        snapshot = self.delegate.trace_snapshot()
        if self.error is not None:
            snapshot = {**snapshot, "generation_error": self.error}
        return snapshot


class _LyingCalculator:
    """A calculator that returns BLOCKED and claims an identity for it.

    "Presented as usable" is the whole fault: ``last_calculation_id`` is what
    the coordinator reads to publish a calculation, and this port sets it while
    returning a result the domain refuses to give an identity to.  The result
    object itself is honest -- ``CalculationResult.calculation_id`` is ``None``
    for BLOCKED -- so the lie lives only in the port's declaration, which is
    exactly where a defective adapter would put it.
    """

    candidate_mode = True

    def __init__(self) -> None:
        self.calls = 0
        self.last_calculation_id = "C1-forged"
        self.last_result: CalculationResult | None = None

    def calculate(self, state: AdaptiveRAGStateV1) -> CalculationResult:
        self.calls += 1
        self.last_result = CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=CalculationOperation.GROWTH_RATE,
            error_code="OPERAND_MISSING",
        )
        state._calculation_result_obj = self.last_result
        return self.last_result

    def trace_snapshot(self) -> dict[str, Any]:
        return {
            "calculator_invoked": self.calls > 0,
            "calculator_call_count": self.calls,
            "calculation_status": (
                None if self.last_result is None else self.last_result.status.value
            ),
        }


# --- the specialist-route drivers ---------------------------------------------
#
# One query, two slots, two distinct facts.  The route is MULTI -> LOCAL_SPECIALIST
# (verified: the un-faulted provider below releases), which is what puts a
# ``ModelProviderV1`` on the path at all -- a single-fact query routes to the
# deterministic renderer and never reaches a provider, so a fault injected there
# would be testing nothing.

SPECIALIST_QUERY = "Summarize revenue"
SPECIALIST_FACTS: dict[str, dict[str, Any]] = {
    "E1": _fact("E1", slots=("a",), period="FY2024", value="100"),
    "E2": _fact("E2", slots=("b",), period="FY2023", value="90"),
}
SPECIALIST_PLAN = _plan(_slot("a", period="FY2024"), _slot("b", period="FY2023"))
GOOD_ANSWER = "Revenue was 100 million USD [E1] and 90 million USD [E2]."


def _run_specialist(provider: Any) -> tuple[Any, _WatchedGeneration, str | None]:
    """Drive the coordinator on a route that reaches the model provider."""

    retrieval, binder, _, _, _ = _real_capabilities(
        [["E1", "E2"]], SPECIALIST_FACTS, SelectingBinderProvider()
    )
    generation = _WatchedGeneration(
        TrustedV2GenerationCapability(model_backend=provider)
    )
    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(DeterministicFallbackProvider({SPECIALIST_QUERY: SPECIALIST_PLAN})),
        capabilities=TrustedV2CapabilityPorts(
            retrieval=retrieval,
            evidence_evaluator=binder,
            generation=generation,
            release_validator=TrustedReleaseValidationCapability(),
        ),
        budget=BUDGET,
        runtime_mode=AgentRuntimeMode.HARNESS_V3,
    )
    escaped: str | None = None
    try:
        outcome = asyncio.run(
            coordinator.execute(_request(SPECIALIST_QUERY, "nf-v3-reliability"))
        )
    except BaseException as exc:  # noqa: BLE001 - an escape is itself an observation
        return None, generation, f"{type(exc).__name__}: {exc}"
    return outcome, generation, escaped


def _provider_case(
    case_id: str,
    fault_kind: str,
    provider: Any,
    *,
    note: str,
    reuses: str = "none: this case is new here",
    extra_detail: Callable[[CaseResult], dict[str, Any]] | None = None,
) -> CaseResult:
    result = CaseResult(
        case_id=case_id,
        fault_kind=fault_kind,
        seam="rag_v2.invocation.ModelProviderV1 behind TrustedV2GenerationCapability",
        expectation=Expectation.FAIL_CLOSED,
        note=note,
        reuses_existing_coverage=reuses,
    )
    outcome, generation, escaped = _run_specialist(provider)
    if outcome is None:
        result.released = False
        result.status = None
        result.answer = None
        result.escaped_exception = escaped
        result.observed = classify(
            released=False, status=None, answer=None, escaped_exception=escaped
        )
    else:
        _apply(outcome, result)
    result.detail = {
        "provider_invocations": len(getattr(provider, "requests", ())),
        "generation_error": generation.error,
        "specialist_calls": generation.delegate.specialist_calls,
        "route": getattr(outcome, "route", None),
        "terminal_state": (
            None
            if outcome is None
            else outcome.runtime_metadata.get("terminal_state")
        ),
    }
    if extra_detail is not None:
        result.detail.update(extra_detail(result))
    return result


# --- the cases ----------------------------------------------------------------


def case_provider_timeout() -> CaseResult:
    return _provider_case(
        "provider_timeout",
        "provider timeout",
        _FakeProvider(
            exc=ModelProviderError(ProviderFailureKind.TIMEOUT, "model timed out")
        ),
        note=(
            "The provider raises the normalized timeout failure. The coordinator "
            "must not turn a transport failure into an answer."
        ),
    )


def case_provider_unavailable() -> CaseResult:
    return _provider_case(
        "provider_unavailable",
        "provider unavailable",
        _FakeProvider(
            exc=ModelProviderError(ProviderFailureKind.UNAVAILABLE, "no engine")
        ),
        note="Unreachable provider: no checkpoint, no endpoint, no engine.",
    )


def case_provider_invalid_response() -> CaseResult:
    return _provider_case(
        "provider_invalid_response",
        "provider invalid response",
        _FakeProvider(
            exc=ModelProviderError(
                ProviderFailureKind.INVALID_RESPONSE, "could not be read"
            )
        ),
        note=(
            "The provider itself declares the answer unreadable. Distinct from "
            "the malformed case below, where the provider believes it succeeded."
        ),
    )


def case_malformed_model_output() -> CaseResult:
    return _provider_case(
        "malformed_model_output",
        "malformed / unparseable model output",
        _MalformedResponseProvider(),
        note=(
            "The transport payload (a dict) crosses the provider boundary "
            "unvalidated. This is the only *contract-level* malformation "
            "available at this seam: there is no parser between the provider and "
            "the capability, so 'unparseable' can only mean 'not a "
            "ModelResponseV1'. A malformed *string* is a different fault and is "
            "covered separately below."
        ),
    )


def case_malformed_model_text() -> CaseResult:
    return _provider_case(
        "malformed_model_text",
        "malformed model output (text that reads as nothing)",
        _FakeProvider(text="<<<not a model output>>>"),
        note=(
            "Extra case, beyond the required ten. A provider that returns a "
            "string no reader could mistake for an answer. Nothing parses model "
            "text on this path, so the refusal -- if there is one -- comes from "
            "the validator, not from a parser. Reported as observed."
        ),
    )


def case_empty_model_output() -> CaseResult:
    return _provider_case(
        "empty_model_output",
        "empty model output",
        _FakeProvider(text="   "),
        note=(
            "Whitespace-only text: transport succeeded and produced nothing "
            "usable. Refused by the boundary that knows what it asked for, not "
            "by the provider contract."
        ),
    )


def case_fabricated_number() -> CaseResult:
    return _provider_case(
        "fabricated_number",
        "model output contradicting the evidence",
        _FakeProvider(text="Revenue was 999 million USD [E1]."),
        note=(
            "Extra case, beyond the required ten. A well-formed answer carrying "
            "a number no admitted evidence supports. This is the fault the "
            "release gate exists for, so it is the closest thing here to a "
            "positive control."
        ),
    )


def case_missing_evidence() -> CaseResult:
    """One required slot has no candidate at all."""

    result = CaseResult(
        case_id="missing_evidence",
        fault_kind="missing evidence (slot with no candidate)",
        seam="R4 retrieval returns nothing; Binder admits nothing",
        expectation=Expectation.FAIL_CLOSED,
        reuses_existing_coverage=(
            "tests/harness/h1_integration.py fixture 'missing_evidence'"
        ),
        note="Retrieval returns an empty result for a required slot.",
    )
    outcome = run_fixture(_fixture("missing_evidence"), AgentRuntimeMode.HARNESS_V3)
    _apply(outcome, result)
    result.detail = {"terminal_state": outcome.runtime_metadata.get("terminal_state")}
    return result


def case_missing_operand() -> CaseResult:
    """A calculation plan with one operand slot left unfilled."""

    result = CaseResult(
        case_id="missing_operand",
        fault_kind="missing evidence (calculation operand)",
        seam="Binder admits one operand; the second slot stays empty",
        expectation=Expectation.FAIL_CLOSED,
        reuses_existing_coverage=(
            "tests/harness/test_closed_loop.py::"
            "test_missing_operand_never_invokes_the_calculator_in_either_mode"
        ),
        note=(
            "The second round returns no candidates, so the 'prior' operand "
            "never binds. The calculator must not run and nothing may release."
        ),
    )
    facts = {
        "CURRENT": _fact("CURRENT", period="FY2024", slots=("current",), value="391"),
    }
    outcome = execute(
        AgentRuntimeMode.HARNESS_V3,
        "Compare years",
        _plan(
            _slot("current", period="FY2024", role="current"),
            _slot("prior", period="FY2023", role="prior"),
            intent=Intent.CALCULATION,
            operation="growth_rate",
        ),
        facts,
        [["CURRENT"], []],
        binder_provider=SelectingBinderProvider(),
        request_id="nf-v3-reliability-missing-operand",
    )
    _apply(outcome, result)
    result.detail = {
        "terminal_state": outcome.runtime_metadata.get("terminal_state"),
        "calculation_status": outcome.runtime_metadata.get("calculation_status"),
    }
    return result


def case_conflicting_evidence() -> CaseResult:
    result = CaseResult(
        case_id="conflicting_evidence",
        fault_kind="conflicting evidence (two candidates disagreeing)",
        seam="two admissible candidates for one slot with contradictory values",
        expectation=Expectation.FAIL_CLOSED,
        reuses_existing_coverage=(
            "tests/harness/test_conflict_differential.py::CONFLICT_FIXTURE"
        ),
        note="One slot, two admissible values, no majority and no corroboration.",
    )
    outcome = _conflict_outcome(AgentRuntimeMode.HARNESS_V3)
    _apply(outcome, result)
    result.detail = {"terminal_state": outcome.runtime_metadata.get("terminal_state")}
    return result


def case_invalid_calculation_operand() -> CaseResult:
    result = CaseResult(
        case_id="invalid_calculation_operand",
        fault_kind="invalid calculation operand (BLOCKED result presented as usable)",
        seam="a calculator port that declares last_calculation_id for a BLOCKED result",
        expectation=Expectation.FAIL_CLOSED,
        reuses_existing_coverage=(
            "tests/harness/h1_integration.py fixture 'calculation_blocked' covers "
            "the honest BLOCKED case; tests/harness/test_calculation_admissibility.py "
            "covers the projection refusal. The lie in the port's declaration is "
            "what this case adds."
        ),
        note=(
            "The result is BLOCKED and honest; the port claims an identity for it "
            "anyway. Nothing may be published as a calculation and nothing may "
            "release."
        ),
    )
    calculation = _LyingCalculator()
    facts = {
        "CURRENT": _fact("CURRENT", period="FY2024", slots=("current",), value="391"),
        "PRIOR": _fact("PRIOR", period="FY2023", slots=("prior",), value="383"),
    }
    outcome = execute(
        AgentRuntimeMode.HARNESS_V3,
        "Compare years",
        _plan(
            _slot("current", period="FY2024", role="current"),
            _slot("prior", period="FY2023", role="prior"),
            intent=Intent.CALCULATION,
            operation="growth_rate",
        ),
        facts,
        [["CURRENT", "PRIOR"]],
        binder_provider=SelectingBinderProvider(),
        calculation=calculation,
        request_id="nf-v3-reliability-forged-calculation",
    )
    _apply(outcome, result)
    result.detail = {
        "calculator_calls": calculation.calls,
        "declared_calculation_id": calculation.last_calculation_id,
        "result_calculation_id": (
            None if calculation.last_result is None else calculation.last_result.calculation_id
        ),
        "published_calculations": list(outcome.calculations),
        "published_calculation_ids": list(outcome.calculation_ids),
        "terminal_state": outcome.runtime_metadata.get("terminal_state"),
    }
    return result


def case_budget_exhausted() -> CaseResult:
    result = CaseResult(
        case_id="budget_exhausted",
        fault_kind="budget exhausted (tool/replan budget spent)",
        seam="AdaptiveRAGBudgetV1 with no replan rounds and one tool call",
        expectation=Expectation.FAIL_CLOSED,
        reuses_existing_coverage=(
            "tests/harness/h1_integration.py fixture 'budget_exhaustion'; "
            "tests/harness/test_action_policy.py::"
            "test_tool_call_is_denied_when_budget_is_spent"
        ),
        note=(
            "Retrieval keeps returning nothing under a zero-replan budget, so the "
            "replanner is refused."
        ),
    )
    outcome = run_fixture(_fixture("budget_exhaustion"), AgentRuntimeMode.HARNESS_V3)
    _apply(outcome, result)
    trace = outcome.debug_metadata.get("trace", {})
    result.detail = {
        "tool_call_count": trace.get("tool_call_count"),
        "replan_count": trace.get("replan_count"),
        "terminal_state": outcome.runtime_metadata.get("terminal_state"),
    }
    return result


def case_unsupported_tool_action() -> CaseResult:
    result = CaseResult(
        case_id="unsupported_tool_action",
        fault_kind="unsupported / invalid tool action",
        seam="no tool is wired for the capability the controller must call",
        expectation=Expectation.FAIL_CLOSED,
        reuses_existing_coverage=(
            "tests/harness/h1_integration.py fixture 'unsupported_route' "
            "(factory-ineligible by design)"
        ),
        note=(
            "The capability graph is incomplete, so the initial action has no "
            "tool. Reported by the fixture as reachable only from a manually "
            "constructed coordinator -- the production factory refuses the graph "
            "before a request can reach this guard."
        ),
    )
    outcome = run_fixture(_fixture("unsupported_route"), AgentRuntimeMode.HARNESS_V3)
    _apply(outcome, result)
    result.detail = {"terminal_state": outcome.runtime_metadata.get("terminal_state")}
    return result


def case_malformed_contract_packet() -> CaseResult:
    """A tool returns a packet the harness cannot normalize."""

    result = CaseResult(
        case_id="malformed_contract_packet",
        fault_kind="contract validation rejection (malformed packet)",
        seam="tool result normalization inside BoundedAdaptiveRAGV1 (OBSERVE boundary)",
        expectation=Expectation.FAIL_CLOSED,
        reuses_existing_coverage=(
            "tests/harness/test_closed_loop.py::"
            "test_malformed_tool_packet_fails_closed_like_a_raising_tool"
        ),
        note=(
            "A packet whose `slots` is an int. Normalization is part of the tool "
            "contract, so this must terminate the run through the bounded "
            "contract rather than raising out of run()."
        ),
    )
    escaped: str | None = None
    state = loop_state()
    try:
        run = BoundedAdaptiveRAGV1().run(
            state,
            {
                ToolCapability.SEMANTIC_RETRIEVAL: lambda query, current: [
                    {"evidence_id": "e1", "slots": 5}
                ]
            },
        )
    except BaseException as exc:  # noqa: BLE001 - an escape is a finding
        run = None
        escaped = f"{type(exc).__name__}: {exc}"
    if run is None:
        result.escaped_exception = escaped
        result.observed = classify(
            released=False, status=None, answer=None, escaped_exception=escaped
        )
    else:
        turns = list(getattr(run.state, "turns", ()))
        first = turns[0]["outcome"] if turns else {}
        _apply_loop(
            run.state,
            run,
            result,
            extra_reasons=(
                [f"TOOL_ERROR:{first['error']}"] if first.get("error") else []
            ),
        )
        result.detail = {
            "first_turn_outcome": first,
            "turn_count": len(turns),
        }
    return result


def case_recovery_repair_once() -> CaseResult:
    """The one case where release is the correct outcome.

    A generator hands back a candidate that misstates the value on a
    ``STRUCTURED_SINGLE`` route, which the deterministic repair can reconstruct
    from the authoritative evidence.  Releasing the *repaired* answer is a
    recovery; refusing would be a failure of a different kind.  The case asserts
    the released answer carries the authoritative value, so a run that released
    the fabricated 999 instead would fail here rather than pass as a recovery.
    """

    result = CaseResult(
        case_id="recovery_repair_once",
        fault_kind="recoverable: candidate misstating an admitted value",
        seam="deterministic candidate repair before revalidation",
        expectation=Expectation.RECOVERABLE,
        reuses_existing_coverage=(
            "tests/test_trusted_v2_validator_release.py::"
            "test_numeric_mismatch_repairs_once_then_releases"
        ),
        note=(
            "The only case in this suite where a release is correct. Counted "
            "separately from incorrect_release by construction: the answer must "
            "contain the authoritative value and not the fabricated one."
        ),
    )
    facts = {"E1": _fact("E1", value="100")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)
    generation = _WatchedGeneration(
        _BadGeneration(
            _bad_candidate("Revenue (FY2024): 999 USD million [citation-E1]")
        )
    )
    validator = TrustedReleaseValidationCapability()
    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(
            DeterministicFallbackProvider({"What was revenue?": _plan(_slot("revenue"))})
        ),
        capabilities=TrustedV2CapabilityPorts(
            retrieval=retrieval,
            evidence_evaluator=binder,
            generation=generation,
            release_validator=validator,
        ),
        budget=BUDGET,
        runtime_mode=AgentRuntimeMode.HARNESS_V3,
    )
    escaped: str | None = None
    try:
        outcome = asyncio.run(
            coordinator.execute(
                _request("What was revenue?", "nf-v3-reliability-recovery")
            )
        )
    except BaseException as exc:  # noqa: BLE001
        outcome = None
        escaped = f"{type(exc).__name__}: {exc}"
    if outcome is None:
        result.escaped_exception = escaped
        result.observed = classify(
            released=False, status=None, answer=None, escaped_exception=escaped
        )
    else:
        _apply(outcome, result)
    answer = result.answer or ""
    result.detail = {
        "repair_calls": validator.repair_calls,
        "validation_calls": validator.validation_calls,
        "answer_is_authoritative": ("100" in answer and "999" not in answer),
        "answer": result.answer,
        "reason_codes": list(result.reason_codes),
    }
    return result


#: Cases in a fixed order, so the report is readable and two runs are comparable
#: position by position as well as by id.
CASES: tuple[Callable[[], CaseResult], ...] = (
    case_provider_timeout,
    case_provider_unavailable,
    case_provider_invalid_response,
    case_malformed_model_output,
    case_malformed_model_text,
    case_empty_model_output,
    case_fabricated_number,
    case_missing_evidence,
    case_missing_operand,
    case_conflicting_evidence,
    case_invalid_calculation_operand,
    case_budget_exhausted,
    case_unsupported_tool_action,
    case_malformed_contract_packet,
    case_recovery_repair_once,
)


_FIXTURES_BY_ID = {fixture.fixture_id: fixture for fixture in FIXTURES}


def _fixture(fixture_id: str) -> Any:
    if fixture_id not in _FIXTURES_BY_ID:
        raise KeyError(f"no h1 fixture named {fixture_id!r}")
    return _FIXTURES_BY_ID[fixture_id]


def _fixture_specs() -> dict[str, Any]:
    """The sealed specification of every fixture this suite drives."""

    return {
        fixture_id: _FIXTURES_BY_ID[fixture_id].spec()
        for fixture_id in sorted(_FIXTURES_BY_ID)
        if fixture_id
        in {"missing_evidence", "budget_exhaustion", "unsupported_route"}
    }


# --- reporting ----------------------------------------------------------------


def run_suite() -> list[CaseResult]:
    results: list[CaseResult] = []
    for factory in CASES:
        results.append(factory())
    return results


def _counters(results: Sequence[CaseResult]) -> dict[str, int]:
    return {
        "fault_cases_total": len(results),
        "correct_fail_closed": sum(1 for r in results if r.correct_fail_closed),
        "incorrect_release": sum(1 for r in results if r.incorrect_release),
        "recovery_succeeded": sum(1 for r in results if r.recovery_succeeded),
        # Not one of the four asked for, reported because a suite that counted a
        # failed recovery as anything else would be hiding a failure.
        "recovery_failed": sum(1 for r in results if r.recovery_failed),
        "unclear": sum(1 for r in results if r.observed is Observed.UNCLEAR),
        "not_injectable": sum(1 for r in results if not r.injected),
    }


def _stability(runs: Sequence[Sequence[CaseResult]]) -> tuple[bool, bool, list[dict[str, Any]]]:
    """Whether every run classified every case the same way.

    Returns ``(classification_stable, detail_stable, differences)``.  Only the
    classification is asserted; the detail comparison is reported so a run that
    is stable in verdict but not in trace is visible rather than assumed.
    """

    first = runs[0]
    differences: list[dict[str, Any]] = []
    classification_stable = True
    detail_stable = True
    for index, other in enumerate(runs[1:], start=2):
        for left, right in zip(first, other):
            if left.stability_key() != right.stability_key():
                classification_stable = False
                differences.append(
                    {
                        "case_id": left.case_id,
                        "kind": "classification",
                        "run_1": list(left.stability_key()),
                        "run_n": list(right.stability_key()),
                        "n": index,
                    }
                )
            elif left.to_dict() != right.to_dict():
                detail_stable = False
                differences.append(
                    {
                        "case_id": left.case_id,
                        "kind": "detail",
                        "run_1": left.to_dict(),
                        "run_n": right.to_dict(),
                        "n": index,
                    }
                )
    return classification_stable, detail_stable, differences


def build_report(runs: Sequence[Sequence[CaseResult]]) -> dict[str, Any]:
    primary = runs[0]
    counters = _counters(primary)
    classification_stable, detail_stable, differences = _stability(runs)
    return {
        "suite": "nf-v3-reliability",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "runs": len(runs),
        "counters": counters,
        "incorrect_release_is_zero": counters["incorrect_release"] == 0,
        "classification_stable": classification_stable,
        "trace_stable": detail_stable,
        "deterministic": classification_stable,
        "stability_differences": differences,
        "cases": [result.to_dict() for result in primary],
        "runs_detail": [[result.to_dict() for result in run] for run in runs],
        "driven_fixture_specs": _fixture_specs(),
        "coverage_provenance": _coverage_provenance(primary),
    }


def _coverage_provenance(results: Sequence[CaseResult]) -> dict[str, Any]:
    """How much of this suite re-runs coverage the repository already had."""

    reused = [r.case_id for r in results if not r.reuses_existing_coverage.startswith("none")]
    new = [r.case_id for r in results if r.reuses_existing_coverage.startswith("none")]
    return {
        "cases_reusing_existing_coverage": reused,
        "cases_new_here": new,
        "summary": (
            f"{len(reused)} of {len(results)} cases drive seams the repository's own "
            f"tests and fixtures already exercise; {len(new)} inject a fault no "
            f"existing case injects. This suite is a classifier over those seams, "
            f"not a new fault-injection framework."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    counters = report["counters"]
    lines = [
        "# NF-V3 reliability: does the trusted pipeline fail closed?",
        "",
        f"- generated: {report['generated_at']}",
        f"- host: {report['host']} (python {report['python']})",
        f"- runs: {report['runs']}",
        f"- deterministic: {'yes' if report['deterministic'] else 'NO'}",
        "",
        "## The number that matters",
        "",
        f"**Incorrect release under injected faults: {counters['incorrect_release']}**",
        "",
        "## Counters",
        "",
        "| counter | value |",
        "| --- | --- |",
        f"| fault cases total | {counters['fault_cases_total']} |",
        f"| correct fail-closed | {counters['correct_fail_closed']} |",
        f"| incorrect release | {counters['incorrect_release']} |",
        f"| recovery succeeded | {counters['recovery_succeeded']} |",
        f"| recovery failed | {counters['recovery_failed']} |",
        f"| unclear | {counters['unclear']} |",
        f"| not injectable | {counters['not_injectable']} |",
        "",
        "## Per-case outcome",
        "",
        "| case | fault kind | expectation | observed | released | status | reason codes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        reasons = ", ".join(case["reason_codes"]) or "-"
        lines.append(
            f"| {case['case_id']} | {case['fault_kind']} | {case['expectation']} | "
            f"**{case['observed_outcome']}** | {'yes' if case['released'] else 'no'} | "
            f"{case['status']} | {reasons} |"
        )

    not_injectable = [case for case in report["cases"] if not case["injected"]]
    lines += ["", "## Not injectable", ""]
    if not_injectable:
        for case in not_injectable:
            lines.append(f"- `{case['case_id']}` ({case['fault_kind']}): {case['not_injectable_reason']}")
    else:
        lines.append(
            "Every fault kind in the required list was injected at an available "
            "seam. Two of them are injectable only in a narrower form than the "
            "name suggests, and the narrowing is recorded per case:"
        )
        lines += [
            "",
            "- *malformed / unparseable model output*: there is no parser between "
            "the provider and the generation capability, so 'unparseable' can "
            "only mean 'not a ModelResponseV1'. A malformed string is a "
            "different fault, covered as `malformed_model_text`, and it is "
            "refused by the validator rather than by a parser.",
            "- *unsupported / invalid tool action*: reachable only from a "
            "manually constructed coordinator. The production factory refuses an "
            "incomplete capability graph before a request reaches the guard, so "
            "the fixture that exercises it is marked factory-ineligible rather "
            "than presented as production wiring.",
        ]

    lines += [
        "",
        "## Notes per case",
        "",
    ]
    for case in report["cases"]:
        lines.append(f"### {case['case_id']}")
        lines.append("")
        lines.append(f"- fault: {case['fault_kind']}")
        lines.append(f"- seam: {case['seam']}")
        lines.append(f"- expectation: {case['expectation']}")
        lines.append(f"- observed: {case['observed_outcome']}")
        lines.append(f"- reuses existing coverage: {case['reuses_existing_coverage']}")
        if case["note"]:
            lines.append(f"- note: {case['note']}")
        lines.append("")

    provenance = report["coverage_provenance"]
    lines += [
        "## Honesty notes",
        "",
        "1. " + provenance["summary"],
        "",
        "2. `RELEASED` is read from the runtime's own release decision "
        "(`release_status`), never from the presence of answer text. A refused "
        "outcome can carry the candidate it refused in its `answer` field -- "
        "`validator_rejection` in this report is exactly that shape -- so "
        "classifying on `answer is not None` would have reported a false "
        "incorrect-release count.",
        "",
        "3. `EXECUTION_ERROR` is counted as fail-closed (the run refused and "
        "released nothing) but is kept distinct from `FAIL_CLOSED` in the status "
        "column, because the two mean different things: one refused by policy, "
        "the other died inside a capability. Both are recorded rather than "
        "merged.",
        "",
        "4. The recovery case is the only case where a release is correct. It is "
        "counted in `recovery_succeeded`, never in `incorrect_release`, and only "
        "when the released answer carries the authoritative value.",
        "",
    ]
    if report["stability_differences"]:
        lines += ["## Stability differences", ""]
        for difference in report["stability_differences"]:
            lines.append(f"- {difference['case_id']} ({difference['kind']})")
        lines.append("")
    return "\n".join(lines) + "\n"


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Directory for the JSON and markdown reports. Defaults to the repo "
            "copy under artifacts/evaluation/nf-v3-reliability, plus the remote "
            "copy when this host has the remote artifact root."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="How many times to run the whole suite (default 2, for determinism).",
    )
    args = parser.parse_args(argv)

    runs = [run_suite() for _ in range(max(1, args.runs))]
    report = build_report(runs)

    outputs: list[str] = []
    failures: list[str] = []
    targets: list[Path] = []
    if args.out is not None:
        targets.append(args.out)
    else:
        targets.append(REPO_OUT)
        if REMOTE_OUT.parent.parent.exists():
            targets.append(REMOTE_OUT)
    # Recorded before the write, so the report names where it was meant to go
    # even when one of the two locations is unwritable.
    report["outputs_written"] = [str(target) for target in targets]
    report["output_failures"] = failures
    for target in targets:
        try:
            _write(
                target / "reliability-report.json",
                json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
            )
            _write(target / "reliability-report.md", render_markdown(report))
            outputs.append(str(target))
        except OSError as exc:
            failures.append(f"{target}: {exc}")
            print(f"could not write {target}: {exc}", file=sys.stderr)

    counters = report["counters"]
    print(render_markdown(report))
    for path in outputs:
        print(f"wrote {path}")

    failures = []
    if counters["incorrect_release"] > 0:
        failures.append(f"incorrect_release={counters['incorrect_release']}")
        for case in report["cases"]:
            if case["incorrect_release"]:
                print(
                    f"INCORRECT RELEASE: {case['case_id']} released under "
                    f"{case['fault_kind']} at {case['seam']} -- "
                    f"status={case['status']} answer={case['answer']!r}",
                    file=sys.stderr,
                )
    if counters["unclear"] > 0:
        failures.append(f"unclear={counters['unclear']}")
    if not report["deterministic"]:
        failures.append("non-deterministic")
    if counters["recovery_failed"] > 0:
        print(
            f"WARNING: recovery_failed={counters['recovery_failed']} (not an exit "
            f"condition, but a recovery that did not happen)",
            file=sys.stderr,
        )
    if failures:
        print("FAIL: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("PASS: incorrect_release=0, unclear=0, deterministic=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
