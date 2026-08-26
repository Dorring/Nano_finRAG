# TV2-08  Full Trusted Financial Runtime V2 Production Integration

Status: TV2-08 FULL_V2_PRODUCTION_INTEGRATION = PASS (after the focused
integration and regression gates in this change).

Base: fe2f256 (TV2-07 framework/preflight). This stage promotes the already
constructed TrustedFinancialRuntimeV2 through the existing FinancialQARuntime
port. It is an integration seal, not a model-quality or canonical readiness
benchmark.

## Production modes

| FINANCIAL_RUNTIME_MODE | Official result | V2 behavior |
| --- | --- | --- |
| v1 | Legacy V1 | Explicit rollback/compatibility path |
| shadow | V1 | V2 runs as bounded observation only |
| v2 | Trusted V2 | Official V2 result; no V1 call and no fallback |

The default is v2. The default conversation mode is also on; off and shadow
remain explicit rollback/diagnostic modes.

A V2 deployment must provide a real factory through
configure_trusted_v2_runtime_builder(...) or the
TRUSTED_V2_RUNTIME_BUILDER=module:callable deployment setting. The selected
factory must return TrustedFinancialRuntimeV2; a missing or invalid factory
is a configuration error. The router never constructs a fake runtime and
never silently falls back to V1.

## Shared request path

    /query and /query/stream
            |
            v
    QueryLifecycleService
            |
            v
    FinancialRuntimeRouter
       +----+--------------------+
       |                         |
    v1/shadow primary            v2 official
    LegacyFinancialRuntimeAdapter TrustedFinancialRuntimeV2
            |                         |
            +------------+------------+
                         v
              FinancialQueryResult contract
                         |
                         v
              Session / DialogueState commit

The Conversation layer resolves the turn before the runtime receives it. V2
uses FinancialQueryRequest.standalone_query as its canonical query and does
not receive raw conversation history for financial execution. Existing L1/L2/L3
context management, clarification, CAS, and idempotency remain in the shared
lifecycle.

## Official V2 boundaries

In v2 mode:

- V2 ANSWER, FAIL_CLOSED, and ERROR are official outcomes.
- V2 failure is not retried through V1.
- Binder-admitted evidence_ids, structured citation_ids, and calculator
  calculation_ids are passed through the result contract; provenance is never
  parsed from answer text.
- The official V2 assistant turn and structured provenance are the values
  committed by the existing lifecycle.
- /query/stream remains validated-final SSE: execution and release validation
  finish before the final event is emitted. No token-level generator
  streaming is introduced.

The V1/shadow observation path remains available for diagnostics. Shadow
observations do not write SessionManager or SQLite conversation state.

## TV2-07 relationship

The TV2-07 evaluation framework and TV2-07R1 canonical preflight remain
available for optional offline use. TV2-07R1 was intentionally NOT EXECUTED
and is not a release gate for this integration. No canonical readiness
metrics, canary percentage, or V2-vs-V1 quality claim is made by this seal.

## Verification scope

The TV2-08 gate covers mode resolution, default activation, router authority,
no-fallback behavior, V2 fail-closed mapping, explicit DI, and the existing
TV2/I-series regression suites. It does not rerun consumed benchmarks or
change Supervisor, R4, Binder, Calculator, Generator, Validator, Repair, or
Conversation algorithms.
