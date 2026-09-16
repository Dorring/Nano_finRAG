# NF-V3 H1 Harness Core Seal

Status: **BOUNDED_FINANCIAL_AGENT_HARNESS_CORE = SEALED**

H1 turns the existing Trusted RAG pipeline into a bounded financial agent
harness: one closed control loop that plans, acts, observes, evaluates,
replans, calculates, generates, verifies and releases -- with the deterministic
validator still owning the release decision in both runtime modes.

This document is the seal evidence. It records what was claimed, what was
measured, and what the measurement cannot show.

## The audit finding that shaped H1

The H1 brief assumed the harness kernel did not exist and specified a new
`contracts/run.py` / `harness/` / `tools/` tree. It did exist. `rag_v2/adaptive/`
already contained every named component, and it was already on the production
path via `BoundedTrustedV2Coordinator.execute()`:

| Brief | Already present |
| --- | --- |
| `RunState` | `AdaptiveRAGStateV1` (strict superset) |
| `RunBudget` | `AdaptiveRAGBudgetV1` |
| `ActionResolver` | `BoundedReplannerV1` |
| `StopPolicy` | `ProgressDetectorV1` |
| `FinancialHarnessRunner` | `BoundedAdaptiveRAGV1.run()` |
| Tool registry / runtime | `tools: Mapping[ToolCapability, ToolFn]` + the ACT phase |

Building the brief literally would have produced a second parallel execution
kernel. The real gap was narrower and worse: **the loop did not close.** The
coordinator stopped the harness at `READY_TO_GENERATE` and produced the answer
in `_candidate_stage`, outside the loop. The harness's own `GENERATE` /
`VERIFY` / `RELEASE` phases were unreachable in production, and the code said so
itself with a `DOWNSTREAM_EXECUTION_NOT_WIRED` reason code.

H1 therefore extended `rag_v2/adaptive/` rather than duplicating it, and treated
the closed loop -- not a new kernel -- as the deliverable.

## What changed

Runtime mode flag:

- `src/runtime/harness_runtime_mode.py` (new) -- `NF_AGENT_RUNTIME_MODE`,
  `legacy` (default) or `harness_v3`. The coordinator never reads the
  environment; only `trusted_v2_production.py` does.

Harness control plane (`rag_v2/adaptive/`):

- `adaptive_contracts.py` -- `AdaptivePhase.CALCULATE`; failure reason codes for
  calculator, generator and verifier wiring; `state.turns` /
  `record_turn` / `observe_turn` / `action_trace`.
- `adaptive_policy.py` (new) -- `AdaptiveActionPolicyV1` separates *permission*
  (budget, retry, replan) from `BoundedReplannerV1`'s *proposal*.
- `adaptive_state_machine.py` -- CALCULATE as a loop phase; GENERATE, VERIFY
  and RELEASE reachable; unknown status and unwired calculator/generator/
  verifier all fail closed instead of raising out of `run()`.
- `adaptive_budget.py` -- `RESERVED_FIELDS` / `unenforced_settings()`.

Coordinator (`src/runtime/trusted_v2_coordinator.py`):

- `_harness_calculator` captures the in-loop result at the call boundary.
- `_release_verdict` reads a validator result's verdict, never its truthiness.
- `_harness_finalizer` runs the existing candidate/validation path as the
  harness tail; the harness decides *when* generation happens, the deterministic
  validator still decides *whether* an answer may be released.
- The released outcome is rebuilt against the completed state so its trace
  covers GENERATE / VERIFY / RELEASE.
- `V2ExecutionTrace` carries `turns`, `turn_count` and `action_trace`.

## Acceptance criteria

| # | Criterion | Result |
| --- | --- | --- |
| 1 | Existing regression does not degrade | PASS -- failing-test set unchanged from the pre-H1 baseline |
| 2 | New harness tests all green | PASS -- `tests/harness/` 83 passed |
| 3 | `legacy` still runs, and is the default | PASS |
| 4 | `harness_v3` enabled by feature flag | PASS -- `NF_AGENT_RUNTIME_MODE` |
| 5 | Evidence Gate not bypassed | PASS -- and one violation was found and fixed (below) |
| 6 | Calculator contract not bypassed | PASS -- 9 operations unchanged |
| 7 | Runtime Validator not bypassed | PASS -- and one pre-existing bypass was found and fixed (below) |
| 8 | Unsupported action fails closed | PASS -- `unsupported_route` fixture |
| 9 | Budget exhaustion fails closed | PASS -- `budget_exhaustion` fixture |
| 10 | No infinite loop | PASS -- loop guard + trace tripwires |
| 11 | No new agent-framework dependency | PASS -- `pyproject.toml` unchanged |
| 12 | Simple fact query needs no multi-agent | PASS -- `fact_direct` fixture |
| 13 | No subagent / swarm | PASS |

## Two defects found, both in release integrity

**1. A rejected candidate could be released (pre-existing).** The
`allow_test_release` path handed the raw validator to the harness as its
verifier. `TrustedReleaseValidationCapability.validate` returns a
`V2ValidationResult` dataclass, which is truthy regardless of its verdict, so
`bool(verifier(...))` released candidates the validator had just refused. This
was the only path in the coordinator that could release without the validator
agreeing. Fixed by `_release_verdict`, which reads `passed` off the result.

**2. A blocked calculation reached generation (introduced during H1).** When
CALCULATE moved inside the loop, the candidate stage re-read the calculator's
result by id instead of re-validating it, so a `BLOCKED` result could flow into
generation. Fixed by having the candidate stage validate the captured result
through the same path as the legacy branch.

Both are now covered by tests that fail if the defect returns.

## Decision equivalence

`harness_v3` is an ablation, so the claim it must earn is not "it works" but "it
decides the same thing". `tests/harness/equivalence.py` partitions every field
of `V2ExecutionOutcome` once, so that no differential test picks its own list:

- **A. Decision-bearing** -- `status`, `release_status`, `answer`, `route`,
  `reason_codes`, `evidence_ids`, `citation_ids`, `calculation_ids`,
  `calculation_result_id`, `citations`, `calculations`, `validator_status`,
  `claim_provenance`, and `runtime_metadata` minus B and C.
- **B. Harness-only** -- `calculation_in_harness`, and `debug_metadata` in full.
  `harness_v3` executes more phases, so it records more execution.
- **C. Volatile** -- `latency_ms`, duration and per-run identifiers.

The partition exists because hand-picked field lists are what let the first two
divergences survive review. `route` and `reason_codes` were missing from the
first list; `calculations`, `calculation_result_id` and `claim_provenance` were
missing from the second. `test_the_equivalence_contract_covers_the_fields_that_once_diverged`
asserts their presence in the contract.

## Sealed fixtures and integration evidence

Nine fixtures, hashed as a set:

```
sealed fixture digest
75c043585a44c1828dfb58f222ec18d6ff041beb1a5d673e09d495e967c71602
```

They run in both modes through `build_trusted_v2_runtime` -- the factory the
production builder calls -- over the real R4 retriever, the real Semantic
Binder, the real deterministic calculator, the real generator routing and the
real release validator.

| Fixture | legacy | harness_v3 | Equivalent |
| --- | --- | --- | --- |
| `fact_direct` | READY_FOR_RELEASE | READY_FOR_RELEASE | yes |
| `calculation_growth_rate` | READY_FOR_RELEASE | READY_FOR_RELEASE | yes |
| `calculation_blocked` | FAIL_CLOSED | FAIL_CLOSED | yes |
| `calculation_error` | EXECUTION_ERROR | EXECUTION_ERROR | yes |
| `wrong_period_recovery` | READY_FOR_RELEASE | READY_FOR_RELEASE | yes |
| `missing_evidence` | FAIL_CLOSED | FAIL_CLOSED | yes |
| `validator_rejection` | FAIL_CLOSED | FAIL_CLOSED | yes |
| `budget_exhaustion` | FAIL_CLOSED | FAIL_CLOSED | yes |
| `unsupported_route` | FAIL_CLOSED | FAIL_CLOSED | yes |

```
decision equivalence : 9/9
expectation failures : 0
release bypass       : 0
false calc release   : 0
infinite loop        : 0
```

The successful calculation fixture walks the full harness path on real wiring:

```
ACT -> OBSERVE -> EVALUATE -> CALCULATE -> READY_TO_GENERATE -> GENERATE -> VERIFY -> RELEASE
```

`legacy` never enters `CALCULATE`; its calculator stays outside the loop. The
fixture asserts both.

### The runner was falsified before it was trusted

A green report proves nothing if the assertions cannot fail. The same fixture
set was run against `3abc2f1`, the commit immediately before the review fixes
landed, by driving the coordinator directly (the factory did not accept a
runtime mode yet at that commit):

```
decision equivalence : 4/9
```

Five fixtures diverged, on exactly the fields the old assertion sets omitted:

| Fixture | Diverged field | legacy | harness_v3 |
| --- | --- | --- | --- |
| `fact_direct` | `route` | `STRUCTURED_SINGLE` | `DIRECT_FACT` |
| `calculation_growth_rate` | `route` | `CALCULATION_SIMPLE` | `CALCULATION` |
| `calculation_error` | `status` | `EXECUTION_ERROR` | `FAIL_CLOSED` |
| `calculation_error` | `reason_codes` | `['CALCULATOR_EXCEPTION']` | `['CALCULATION_ERROR']` |
| `wrong_period_recovery` | `reason_codes` | `['VALIDATED_RELEASE']` | `['VALIDATED_RELEASE', 'WRONG_PERIOD']` |
| `validator_rejection` | `route` | `STRUCTURED_SINGLE` | `DIRECT_FACT` |

The `route` column is the rebuilt outcome reporting the plan intent instead of
the generation route; the `reason_codes` rows are the released outcome
inheriting the trace's superset, labelling a clean release with a recovery code
that had already been resolved.

## Test suite accounting

The suite reports three numbers, and they add up exactly:

```
4017 collected (pre-H1) = 144 failed + 3835 passed + 38 skipped
4047 collected (H1.1)   =  39 failed + 3865 passed + 143 skipped
```

H1's audit reported "3964 collected", which was simply wrong arithmetic on the
same run -- it omitted the skipped tests. There was no count contradiction; there
was a reporting error, and it is recorded here because the project leans on
exact counts.

The original 144 failures, classified by cause:

| Count | Cause | Disposition |
| --- | --- | --- |
| 104 | Sealed evaluation artifacts under `artifacts/evaluation/<run>/` that were never committed | Now skipped, with the missing path named in the reason |
| 39 | `FINANCIAL_RUNTIME_MODE` defaults to `v2` and `v2` requires `TRUSTED_V2_RUNTIME_BUILDER`; these endpoint tests predate that default and never set the mode | **Not fixed.** See below. |

The 39 are a pre-existing defect introduced by `b84fa3a` and are unrelated to
H1. Setting `FINANCIAL_RUNTIME_MODE=v1` turns 34 of them green, leaving 5 that
fail on semantic-alignment behaviour instead. They are left alone deliberately:
choosing between "these tests should run as v2" and "the tests should pin v1" is
a product decision about the default runtime mode, not a harness fix, and
silently pinning the environment would answer it by accident.

Two collection-time guards were added to `conftest.py`:

- Modules that cannot import without an optional runtime (`chromadb`, `torch`)
  are excluded with a header line naming what is missing. Previously they were
  collection *errors*, and pytest aborts the whole session on a collection
  error -- so a checkout without those runtimes could not run the suite at all.
- A test that fails reading a missing path under `artifacts/` is reported as a
  skip naming that artifact. The skip is decided from the actual error, not from
  the directory being absent: a first attempt skipped whole modules whose
  declared artifact root was missing, and silently swallowed nine tests in those
  same modules that pass without the artifact.

`pytest -m "not requires_artifacts"` deselects the artifact-dependent group.

## What this seal does not claim

- **The harness is not a model-driven ReAct loop.** `BoundedReplannerV1` maps a
  concrete evaluator reason code to one permitted capability. That is
  deterministic bounded replanning by design: financial execution should not be
  handed to free-form model tool selection. What is model-driven is the
  Supervisor's semantic planning and slot requirement.
- **`max_identical_query_retry` is reserved, not enforced.** Nothing reads it.
  Enforcing it at its default of 0 would forbid a same-query retry that legacy
  has always performed, changing retrieval behaviour inside an ablation whose
  only claim is equivalence. It is now listed in
  `AdaptiveRAGBudgetV1.RESERVED_FIELDS`, production warns when an operator sets
  it, and a tripwire test fails if any adaptive decision starts reading it.
- **`UNSUPPORTED_TOOL_ROUTE` is unreachable from the factory.** The factory
  refuses an incomplete dependency graph, so the guard is defence in depth
  against a manually constructed coordinator. The fixture says so.
- **`_candidate_stage` still owns the candidate path.** The finalizer wraps it
  rather than reimplementing it, so no release logic is duplicated -- but the
  harness does not own that code.
- **The R4 evaluation corpora remain uncommitted.** This seal is about the
  harness, not about the evaluation artifacts.

## Reproduction

```bash
cd finquery_rag/backend

# Ablation report
python scripts/runtime/run_nf_v3_h1_harness_integration.py

# Harness suite
python -m pytest tests/harness -q

# Full suite
python -m pytest -q
```

`scripts/runtime/` is matched by the repository's root `.gitignore` rule
`runtime/`, so new files there need `git add -f`.

## Commits

- `fb318da` test(harness): freeze the legacy v2 execution model as a baseline
- `023d54d` feat(harness): surface run turns and name the action policy
- `cbcfa00` feat(harness): make deterministic calculation an explicit harness phase
- `0709636` feat(harness): close the generation loop behind the runtime mode flag
- `715a92c` fix(runtime): validate in-loop calculation results in harness_v3
- `3abc2f1` test(harness): prove mode equivalence at the metadata level too
- `8f5e705` docs(harness): correct two misleading dead-code comments
- `5c6a046` fix(runtime): close review findings in the harness control plane
- `0bb08fc` refactor(harness): share test fixtures and close a second review pass
- H1.1 integration, equivalence contract and suite accounting

## Next

H2 Context & Artifact Runtime. The audit found the agent loop was already 85%
present; the same question now applies to context, which is still conversation
history rather than run state (task state, plan, evidence, calculation, tool
observations, failure state) with evidence artifacts loaded just in time.

Multi-agent remains deferred. `BoundedAdaptiveRAGV1` already retrieves, replans,
repairs, calculates and verifies; adding a Retriever Agent, Calculator Agent and
Validator Agent would be an architectural retreat. The only later candidate is a
complexity gate that spawns ephemeral evidence workers for multi-entity queries,
and those workers would hold no calculation, release or validation authority.
