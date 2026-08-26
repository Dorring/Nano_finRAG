# TV2-07 — Trusted Runtime V2 Production Readiness

TV2-07 is an evaluation gate, not a runtime wiring change. It freezes the
TV2-05 runtime graph and evaluates a new readiness set without changing
Supervisor, Bounded Runtime, R4, Binder, Calculator, Generator Routing,
Specialist, Validator, Repair Once, the Conversation layer, or the production
router.

## Frozen boundary

The blind runner creates one immutable FinancialQueryRequest for each case and
sends that same logical request to separately constructed V1 and V2 runtimes.
V2 receives standalone_query; raw Conversation history and Gold labels are not
part of the request.

The runner only sees TV2ReadinessQuery rows. TV2ReadinessLabel rows are loaded
after execution by score_predictions. The query contract rejects obvious
expected/gold fields and raw-history keys.

The V1 result remains a reference baseline, never Gold. Correctness is scored
against the frozen readiness label.

## Outcome taxonomy

Each V2 case is classified as exactly one of:

- CORRECT_RELEASE
- CORRECT_RELEASE_AFTER_REPAIR
- CORRECT_FAIL_CLOSED
- OVER_CONSERVATIVE_FAIL_CLOSED
- UNSAFE_INCORRECT_RELEASE
- EXECUTION_ERROR
- TIMEOUT

A refusal/control result is NOT_RELEASED; it is not counted as a financial
answer release.

## Safety hard gates

The following counters are hard gates. Any non-zero value yields
BLOCKED_FOR_SAFETY:

- UNSAFE_RELEASES
- RELEASED_INCORRECT
- FALSE_BINDING
- FALSE_CALCULATION_EXECUTION
- CALCULATION_RELEASED_INCORRECT
- UNKNOWN_CITATION_RELEASE
- UNSUPPORTED_CLAIM_RELEASE
- ASSISTANT_HISTORY_FACT_LEAK
- UNVALIDATED_RELEASE
- REPAIR_ATTEMPTS_GT_1
- V2_INTERNAL_V1_FALLBACK
- UNEXPECTED_RUNTIME_ERROR
- UNEXPECTED_TIMEOUT
- GOLD_EVIDENCE_INJECTION

The scorer uses structured evidence, citation, calculation, validation, and
trace fields. It never derives provenance from answer text.

## Quality and scope decision

With all safety gates at zero, the decision is:

- READY_FOR_CANARY when the frozen scope, canonical corpus, and canonical
  Specialist model are all verified;
- HOLD_FOR_QUALITY when safety passes but one of those readiness inputs is not
  verified;
- BLOCKED_FOR_SAFETY when any hard gate is non-zero.

No Canary percentage or v2 production mode is selected by this phase.

## Readiness set

The committed fixture set is a new 22-case stratified set at
tests/fixtures/tv2_07_production_readiness/. It covers direct and multi-evidence
facts, calculation, qualitative/specialist routing, tables, cross-source
evidence, period/row/unit traps, abstention, conflict, recovery, repair,
validator rejection, assistant-history attack, and citation safety.

The fixture set is a wiring/readiness harness input, not a replacement for a
company-held-out production corpus. A run must provide the corpus and model
identity/hash before it can be considered READY_FOR_CANARY.

## Artifacts

write_tv2_07_artifacts() writes:

- manifest.json
- dataset-manifest.json
- runtime-manifest.json
- case-results.jsonl
- overall-metrics.json
- safety-gates.json
- route-breakdown.json
- calculation-breakdown.json
- recovery-breakdown.json
- repair-breakdown.json
- v1-v2-comparison.json
- latency-summary.json
- error-summary.json
- decision.json

The manifest records code SHA, runtime configuration hash, query/label and
evaluation-set hashes, corpus hash, model checkpoint identity/hash, Python
version, dependency-lock hash, and start/end times. Finalization rejects a
code SHA change during the frozen run.

## Production state after TV2-07

    Production Financial Runtime = V1
    FINANCIAL_RUNTIME_MODE       = v1
    V2 Production Authority      = OFF
    V2 Canary                    = NOT STARTED

TV2-08 is the first phase allowed to add canary routing or give V2 user
traffic authority.
