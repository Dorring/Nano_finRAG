# TV2-02 Supervisor and Bounded Runtime Coordinator

- Base: d4cfe1a16db36a12e709277de46857b466ee3fc6
- Scope: Supervisor planning and bounded control-loop wiring
- Production behavior changed: no
- Model calls: no
- GPU calls: no
- Benchmark rerun: no
- Production financial runtime: V1
- Production V2 routing: OFF

## Seal target

TV2-02 SUPERVISOR_BOUNDED_RUNTIME_WIRED = PASS

Supervisor + Plan           = WIRED
Bounded Runtime             = WIRED
Capability execution        = PORT/Fake
R4 + Binder                 = NOT REAL-WIRED
Calculator + Generator      = NOT REAL-WIRED
Validator Release Gate      = NOT REAL-WIRED
Production Runtime          = V1
V2 Routing                  = OFF

## Actual TV2-02 graph

FinancialQueryRequest
        |
        v
TrustedFinancialRuntimeV2
        |
        v
BoundedTrustedV2Coordinator
        |
        +--> SupervisorService.plan(standalone_query)
        |       |
        |       +--> existing SupervisorPlan and validate_plan_v2_01
        |
        +--> existing BoundedAdaptiveRAGV1
                |
                +--> PLAN
                +--> ACT through injected capability ports
                +--> OBSERVE
                +--> EVALUATE
                +--> REPLAN / READY_TO_GENERATE / FAIL_CLOSED

No second LLM planning step is introduced. Supervisor remains a planner only;
the bounded loop remains the policy controller.

## Reused existing components

TV2-02 reuses:

- rag_v2.supervisor.service.SupervisorService
- rag_v2.supervisor.plan_validator.validate_plan_v2_01
- rag_v2.contracts.plan.SupervisorPlan and RequiredSlot
- rag_v2.adaptive.adaptive_state_machine.BoundedAdaptiveRAGV1
- rag_v2.adaptive.adaptive_replanner.BoundedReplannerV1
- rag_v2.adaptive.adaptive_progress.ProgressDetectorV1
- rag_v2.adaptive.adaptive_evaluator.EvidenceStateEvaluatorV1
- rag_v2.adaptive.adaptive_budget.AdaptiveRAGBudgetV1

The SupervisorPlan is stored as the canonical plan in the adaptive state
snapshot. There is no second LLM-generated QueryPlan.

## Capability ports

The coordinator depends only on TrustedV2CapabilityPorts:

- RetrievalCapability: structured action plus adaptive state to candidate
  packets.
- EvidenceEvaluationCapability: deterministic slot and conflict evaluation.
- CalculationCapability: declared only; not called before TV2-04.
- GenerationCapability: declared only; test wiring is opt-in.
- ReleaseValidationCapability: declared only; test wiring is opt-in.

TV2-02 tests use deterministic fakes. The retrieval fake returns packet
mappings, and the existing evaluator determines MISSING_SLOT, WRONG_PERIOD,
MISSING_OPERAND, conflict, and sufficiency. No R4 or Binder implementation is
imported by the coordinator.

## Policy controls

Initial planning is one SupervisorService call. Recovery is selected by the
existing BoundedReplannerV1 reason-code mapping:

MISSING_SLOT      -> SEMANTIC_RETRIEVAL
WRONG_PERIOD      -> STRUCTURED_FINANCIAL_LOOKUP
MISSING_OPERAND   -> STRUCTURED_FINANCIAL_LOOKUP
CONFLICT          -> controlled terminal fail-closed

The coordinator enforces:

- max total tool calls;
- max replan rounds;
- same-tool retry bound;
- no-progress detection;
- invalid-plan fail-closed;
- capability exception as EXECUTION_ERROR.

Policy stops are FAIL_CLOSED with NOT_RELEASED. Software/capability
exceptions are EXECUTION_ERROR with NOT_RELEASED.

## Release boundary

The default coordinator never invokes GenerationCapability or
ReleaseValidationCapability and therefore cannot produce READY_FOR_RELEASE.
The only release test sets allow_test_release explicitly and injects both
deterministic fake ports. This is a unit-test fixture, not production wiring.

When evidence is sufficient but downstream execution is absent, the result is:

V2ExecutionOutcome
  status = FAIL_CLOSED
  release_status = NOT_RELEASED
  reason_codes contains DOWNSTREAM_EXECUTION_NOT_WIRED

TV2-02 does not call TrustedRAGRuntimeV2.handle().

## Trace contract

Every bounded run exposes a V2ExecutionTrace in debug metadata. It contains
structured transitions, tool invocation metadata, reason codes, budget
counters, and terminal state. It does not contain provider raw responses,
private reasoning, or chain-of-thought.

## Environment boundary

TV2-02 uses CPU-safe injected fakes only:

- model calls = false;
- GPU calls = false;
- benchmark rerun = false.

Focused control-loop verification is complete. At the TV2-02 checkpoint the full backend environment was still pending.
TV2-03P0 subsequently verified the canonical Python 3.12.2 environment,
locked dependencies, NumPy/OpenAI/Jose, StrEnum, and the relevant R4/Binder
imports. The pending checkpoint is therefore closed:

FULL_BACKEND_ENVIRONMENT_VERIFICATION = PASS (closed in TV2-03P0)

TV2-03 remains CPU-safe and does not call external models or rerun consumed
benchmarks.

## Production graph remains unchanged

/query and /query/stream
  -> QueryLifecycleService
  -> QueryExecutionService
  -> LegacyFinancialRuntimeAdapter
  -> V1

TV2-02 adds no V2 shadow call, no router, no QueryLifecycleService change, and
no Conversation change. The V2 coordinator is an injectable component only.
