# NF-V3 H1 — Financial Agent Harness: Plan

Status: **D1–D5 delivered** (see §9 delivery record)
Baseline commit: `b393d3e` (`feat/context-trust-runtime-trace`, in sync with `origin`)
Scope: runtime control plane only. No retrieval, binder, calculator, or generator
algorithm changes.

## 1. Why this document exists

The original H1 brief described building a single-agent harness from scratch
(`RunState`, `ActionResolver`, `ActionPolicy`, `ToolSpec`/`Registry`/`Runtime`,
`BudgetManager`, `StopPolicy`, `FinancialHarnessRunner`, `TrustedAnswerFinalizer`)
under new `rag_v2/contracts/`, `rag_v2/harness/`, and `rag_v2/tools/` packages.

The brief explicitly instructed: audit first, and if the repository does not match
those assumptions, adapt minimally to the real code rather than forcing the design.

**The audit found that the execution harness already exists** — under different
names, in `rag_v2/adaptive/`, and already running in production. Building the
brief as written would have duplicated a live execution kernel.

Correction to an earlier revision of this section, which said "roughly 85% of the
proposed harness already exists". That was reached by matching class names to a
component list, and name correspondence is not capability correspondence:

- `BoundedReplannerV1` is a deterministic `reason_code -> ToolCapability` table,
  not an agent decision layer. No model participates in the loop's choice of next
  action.
- `tools: Mapping[ToolCapability, ToolFn]` is a capability dispatch table, not a
  tool runtime: no per-tool input/output contract, no execution-status versus
  domain-status distinction, no timeout or error taxonomy.

So: the execution harness skeleton existed and was live; the agent intelligence
harness — per-turn context engineering, model-guided replanning, recovery,
durable state — did not. §6 and the seal document list what is still missing.

This document records the real execution model, the deviation from the brief, and
the re-scoped H1 that closes the genuine gap.

## 2. Audit: the real execution path

**This section is the audit record as found at baseline `b393d3e`.** Line numbers
and behaviour described here are pre-H1; D1–D5 changed several of them (§9 records
what changed). Read it as "what was true when the plan was written", not as a
description of the current tree.

### 2.1 Production entry

```
HTTP / runtime router
  → TRUSTED_V2_RUNTIME_BUILDER (env, opt-in; src/main.py:154)
  → build_trusted_v2_runtime_for_request()      src/runtime/trusted_v2_production.py:969
  → build_trusted_v2_runtime()                  src/runtime/trusted_v2_factory.py:31
  → BoundedTrustedV2Coordinator.execute()       src/runtime/trusted_v2_coordinator.py:1438
      → BoundedAdaptiveRAGV1.run()              rag_v2/adaptive/adaptive_state_machine.py:78
```

`TRUSTED_V2_RUNTIME_BUILDER=src.runtime.trusted_v2_production:build_trusted_v2_runtime_for_request`
is the documented deployment default (`.env.example:57`). The non-V2 default path
(`src/services/rag_engine.py` → `CalculationPipeline`) has no supervisor and no harness.

### 2.2 The existing harness

`rag_v2/adaptive/` — "NF-V2-16 bounded adaptive RAG contracts and control plane" —
already implements the control plane the brief asked for:

| Brief item | Existing implementation | Location |
|---|---|---|
| `RunStatus` | `AdaptivePhase` (`PLAN/ACT/OBSERVE/EVALUATE/REPLAN/READY_TO_GENERATE/GENERATE/VERIFY/REPAIR/RELEASE/FAIL_CLOSED`) | `adaptive_contracts.py:24` |
| `RunState` | `AdaptiveRAGStateV1` — strict superset (carries `tool_calls`, `iteration`, `replan_rounds`, `same_tool_retries`, `last_action`, `last_observation`, `progress_signatures`, `transitions`, `stop_reason`) | `adaptive_contracts.py:256` |
| `RunBudget` | `AdaptiveRAGBudgetV1` | `adaptive_budget.py:8` |
| `ToolSpec` / tool enum | `ToolCapability` (6 members) | `adaptive_contracts.py:94` |
| `ToolRegistry` | `tools: Mapping[ToolCapability \| str, ToolFn]`, built per request | `trusted_v2_coordinator.py:1587` |
| `ToolRuntime` | the `ACT` phase: resolve → budget check → execute → normalize exception → no retry | `adaptive_state_machine.py:101-135` |
| `Observation` | `state.last_observation` + `EvidenceEvaluationV1` | `adaptive_contracts.py:229` |
| `ActionResolver` | `BoundedReplannerV1.replan()` — reason-code → capability table | `adaptive_replanner.py:17` |
| `ActionPolicy` | phase-legal transitions + budget gates + `UNSUPPORTED_TOOL_ROUTE` fail-closed | `adaptive_state_machine.py:98-212` |
| `StopPolicy` | `ProgressDetectorV1` (no-progress) + budget exhaustion + terminal decisions | `adaptive_progress.py:9` |
| `FinancialHarnessRunner` | `BoundedAdaptiveRAGV1.run()` | `adaptive_state_machine.py:78` |
| Generation is not a tool | `generator` / `verifier` callbacks after `READY_TO_GENERATE` | `adaptive_state_machine.py:191-205` |

The harness is bounded and fail-closed: a local `guard`
(`adaptive_state_machine.py:95`, `max_total_tool_calls * 4 + 12` at baseline;
`* 6 + 20` after D3 added phases) prevents infinite loops, and every abnormal exit
routes through `_fail()` to `FAIL_CLOSED` with a `ReasonCode`.

### 2.3 The actual gap — the loop does not close

`BoundedTrustedV2Coordinator.execute()` calls the harness, then hands off:

```
supervisor.plan()                                    coordinator.py:1446
  → BoundedAdaptiveRAGV1.run(...)                    coordinator.py:1612
       PLAN → ACT → OBSERVE → EVALUATE → [REPLAN → ACT]*
       … exits at READY_TO_GENERATE
  → _candidate_stage(...)   OUTSIDE the loop         coordinator.py:889, called at :1653
       binder-admission gate → CALCULATE → GENERATE → VALIDATE → RELEASE
```

Consequences, each verified in code:

1. **`GENERATE` / `VERIFY` / `RELEASE` are dead in production.** They are reachable
   only when `self.allow_test_release` is true (`coordinator.py:1603-1609`, `:1685`).
   The production factory sets `allow_test_release=False`
   (`trusted_v2_factory.py:75`); only `tests/test_trusted_v2_coordinator.py:302`
   passes `True`. The harness therefore never generates or releases anything live.

2. **`CALCULATION` is not a harness step.** All six `ToolCapability` values are mapped
   to the same retrieval wrapper (`coordinator.py:1587-1594`). Deterministic
   calculation runs post-loop at `coordinator.py:928`, gated by
   `plan.intent is Intent.CALCULATION`.

3. **There is no run turn trace.** The real turn counter is the local `guard`
   (`adaptive_state_machine.py:94-96`) and is never surfaced. `state.iteration` is a
   tool-call index, not a turn index. The brief's requested
   `{turns, tool_calls, action_trace}` output cannot be produced from the current trace.

4. **The coordinator admits the gap in its own reason codes**:
   `DOWNSTREAM_EXECUTION_NOT_WIRED` (`coordinator.py:1673`, `:1676`).

### 2.4 Two parallel control planes

A second, independent control-plane state machine exists and is **not driven by
anything in production**:

- `rag_v2/orchestration/state.py` — `StateMachine`
  (`RECEIVED → PLANNED → RETRIEVED → MATERIALIZED → BOUND → CALCULATED → GENERATED →
  VALIDATED → RELEASED/ABSTAINED/FAILED`), with `RepairBudget`
  (`retrieval_repair_max`, `generation_repair_max`, `total_tool_steps_max`) and
  `execute(Action)` fail-closed transition validation.
- Grep shows it is referenced only from `tests/rag_v2/test_v2_00_contracts.py`,
  `tests/rag_v2/test_v2_01_supervisor.py`, and
  `scripts/evaluation/run_v2_00_architecture_contract_freeze.py`.

This is a real duplication: the typed `Action` contract
(`rag_v2/contracts/plan.py:19`) is consumed by a state machine nobody drives, while
the live loop uses a different phase enum. Resolving it is **out of H1 scope**
(see §6) but must be recorded.

### 2.5 `TrustedRAGRuntimeV2` is not on the production path

`rag_v2/runtime/runtime.py::TrustedRAGRuntimeV2` is **not imported anywhere under
`src/`**. The only `src/` occurrences are docstring mentions
(`trusted_v2_adapter.py:5`, `trusted_v2_contracts.py:5`, `trusted_v2_coordinator.py:6`),
and `tests/test_trusted_v2_adapter.py:275` asserts
`not hasattr(module, "TrustedRAGRuntimeV2")`.

The brief's assumption that `TrustedRAGRuntimeV2` is the live "trusted generation
finalizer" is **incorrect for the production path**. Its real consumers are
evaluation scripts and tests. The production equivalent is the
generation + validation port pair driven from `_candidate_stage`.

Also note the brief's contract names differ from reality:
`TrustedEvidencePacket` does not exist — the evidence dataclass is
`VerifiedEvidencePacket` (`rag_v2/contracts/evidence.py:143`), while the runtime gate
consumes a **`Mapping`** packet validated by `TrustedEvidenceGateV1`
(`rag_v2/runtime/evidence.py:23`).

### 2.6 Capabilities available for reuse (do not modify)

| Capability | Real entry point | Notes |
|---|---|---|
| Retrieval | `R4RetrievalCapability.retrieve(action, state)` — `src/runtime/trusted_v2_r4.py:664` | core algorithm `CandidateDirectRetriever.retrieve(plan, *, document_scope)` — `src/pdf_retrieval_v4/candidate_direct_retriever.py:169`; request-scoped, rebuilt per request |
| Binding | `SemanticBinderService.bind(request) -> BinderRun` — `rag_v2/evidence/binder_service.py:95`; runtime port `SemanticEvidenceEvaluatorCapability.evaluate(state)` — `src/runtime/trusted_v2_binder.py:803` | LLM-backed; process-cached |
| Calculation | `execute_plan(plan)` — `src/finance/calculation_executor.py:121`; runtime port `DeterministicCalculationCapability.calculate(state)` — `src/runtime/trusted_v2_calculation.py:282` | fully deterministic, **no LLM**, `Decimal` only; 9 ops allowlisted at `rag_v2/contracts/plan.py:32` |
| Supervisor | `SupervisorService.plan(question) -> SupervisorRun` — `rag_v2/supervisor/service.py:43` | LLM-backed; deterministic fallback provider is test-only (`deterministic_fallback.py:11`) |
| Validation | `RuntimeGenerationValidatorV1` — `rag_v2/generation/validator.py` | deterministic release authority |

## 3. Revised H1 — "close the harness loop"

**Goal.** Make the existing harness the single, complete execution kernel for the
trusted financial runtime — including calculation, generation, verification and
release — while preserving the existing deterministic release authority and
changing no domain algorithm.

Target execution model:

```
supervisor.plan()
  → FinancialHarnessRunner  (BoundedAdaptiveRAGV1, extended)
       PLAN → ACT → OBSERVE → EVALUATE
            → [REPLAN → ACT]*                         (existing)
            → [CALCULATE]                             NEW: explicit phase
            → READY_TO_GENERATE
            → GENERATE   (finalizer → existing generation port)
            → VERIFY     (existing deterministic validator)
            → RELEASE | FAIL_CLOSED                   NEW: live, not test-only
  → V2ExecutionOutcome
```

Invariants preserved:

- The LLM (supervisor, binder) may *propose*; it never decides release.
- `TrustedEvidenceGateV1` / binder-admission gate still run before generation.
- The 9 operation allowlist and the deterministic calculator are untouched.
- Budget exhaustion and unsupported routes remain fail-closed.
- No new runtime dependency.

### 3.1 Deliverables

**D1 — Run turn trace.** Surface the loop's real turn counter and per-turn action so
the run trace carries `turns`, `tool_calls`, and `action_trace`. Add the turn index to
`state.transitions` records (or a parallel `state.turns` list) and expose it through
`V2ExecutionTrace`. Keeps existing trace keys intact.

**D2 — Named ActionPolicy.** Extract the transition-legality and budget-gate rules from
`BoundedAdaptiveRAGV1.run()`'s phase dispatch into an explicitly named component
(e.g. `rag_v2/adaptive/adaptive_policy.py`), consumed by the loop. Behaviour must be
bit-identical; this is a naming/seam change so that "resolution and policy are
separate" is true in the artifact, not merely implied. `BoundedReplannerV1` remains
the resolver.

**D3 — Calculation as a harness phase.** Introduce an explicit `CALCULATE` phase
between `EVALUATE`/`READY_TO_GENERATE`, entered when `plan.intent is Intent.CALCULATION`
and the binder-admission gate has passed. It calls the **existing**
`DeterministicCalculationCapability.calculate(state)`. The post-loop calculation in
`_candidate_stage` becomes the fallback behind the flag (see D5), not the primary path.

**D4 — Close the loop with the finalizer.** Wire generation + validation into the
harness's `GENERATE` / `VERIFY` phases for real, so `AdaptivePhase.RELEASE` is a live
production terminal. `allow_test_release` stops gating reachability. The generation
port and the deterministic validator remain the release authority; the harness only
sequences them. Deprecate `DOWNSTREAM_EXECUTION_NOT_WIRED`.

**D5 — Runtime mode flag.** `NF_AGENT_RUNTIME_MODE` ∈ {`legacy`, `harness_v3`},
default `legacy`. `legacy` = today's behaviour (harness stops at `READY_TO_GENERATE`,
`_candidate_stage` produces the answer). `harness_v3` = the closed loop. This makes the
change an ablation: same model, same retrieval, same data, same generator, same
validator — only the runtime differs.

### 3.2 Commit plan

The brief asked for four commits. Re-scoped to the real code:

1. `test(harness): freeze the current execution model as a regression baseline` —
   characterization tests pinning `BoundedTrustedV2Coordinator` behaviour: terminal
   statuses, reason codes, transitions, and the existing trace keys. No production
   code touched.
2. `feat(harness): surface run turns and name the action policy` — D1 + D2.
3. `feat(harness): make deterministic calculation an explicit harness phase` — D3.
4. `feat(harness): close the generation loop behind the runtime mode flag` — D4 + D5,
   plus a legacy-vs-harness integration runner.

Each commit must leave the full suite green. Commit 1 is what makes the rest
reviewable — it converts "the behaviour did not change" from a claim into a test.

### 3.3 Integration runner

Keep `scripts/runtime/run_nf_v2_21_runtime_integration.py` untouched as the legacy
baseline. Add `scripts/runtime/run_nf_v3_h1_harness_integration.py`, which runs the same
fixture through both modes and compares release/abstain decision, final numeric value,
citation semantics, calculation result, and validator result, and additionally reports
harness `turns`, `tool_calls`, `action_trace`, and terminal state.

## 4. Acceptance criteria

H1 is complete only when all hold:

1. Existing tests do not regress (see §5 baseline).
2. New harness tests are green, covering: closed-loop release; calculation inside the
   loop; missing evidence never reaches generation; missing operand never reaches
   calculation; unsupported operation rejected; tool exception normalized and
   fail-closed; budget exhaustion terminal; validator rejection → abstain;
   `action_trace` reflects real order.
3. `legacy` mode reproduces pre-change behaviour exactly (pinned by commit 1).
4. `harness_v3` is independently selectable and produces the same release decisions
   and answer semantics as `legacy` on the existing fixture set.
5. The harness does not bypass the binder-admission gate, the operation allowlist, or
   the deterministic validator.
6. No infinite loop; bounded turns remain enforced.
7. No new runtime dependency (no LangChain/LangGraph/DeepAgents/CrewAI/AutoGen/Agents SDK).
8. No subagent, swarm, or multi-agent construct is introduced.

## 5. Baseline (recorded at `b393d3e`, this environment)

- `python -m pytest tests/rag_v2 tests/test_trusted_v2_coordinator.py tests/test_nf_v2_16_r1_metadata_scope.py -q`
  → **169 passed, 1 failed**.
  The failure is pre-existing and environmental, not a code defect:
  `tests/rag_v2/test_nf_v2_02_top20_financial_fact.py::test_same_candidate_facts_and_fact_ids_are_preserved`
  expects the missing artifact
  `artifacts/evaluation/nf-e2e-09-r0-structured-financial-fact-representation/financial-facts-v1.jsonl.gz`.
- Full collection at `b393d3e`: **4017 collected**, of which **144 failed / 3835
  passed / 38 skipped**, plus 2 collection errors, both from missing optional
  dependencies in this interpreter (`chromadb` for
  `tests/evaluation/test_nf40_cli.py`, `torch` for `tests/test_local_specialist_generator.py`).
  The three counts sum exactly to the collected total.
  Correction: an earlier revision of this section recorded "3964 collected",
  which was bad arithmetic on the same run — it omitted the skipped tests.
  Corrected in §11.
- Tests must be run with `python -m pytest` from `finquery_rag/backend` so that
  `rag_v2` and `src` are importable.
- A bare `pytest` did not complete at all at this baseline: pytest aborts the
  session on a collection error, and the two modules above cannot import in this
  environment. `--continue-on-collection-errors` was needed to see any numbers.
  Fixed in §11.

Any new failure relative to this baseline is a regression.

## 6. Out of scope for H1

- **Package consolidation.** Moving `src/generation/*` into `rag_v2/generation/` is
  H1.5, not H1. Renaming or relocating modules while changing execution semantics
  makes regressions impossible to attribute.
- **Resolving the `rag_v2/orchestration/StateMachine` duplication.** Recorded in
  §2.4; deferred.
- **Retiring `rag_v2/runtime/TrustedRAGRuntimeV2`.** It is already off the production
  path; deleting it is a separate cleanup with its own test fallout.
- Multi-agent, subagents, MCP, A2A, durable checkpointing, context compaction,
  token/dollar budgets, LLM-driven free-form tool selection.

## 7. Risks

| Risk | Mitigation |
|---|---|
| D4 touches the release path — the most sensitive code in the repo | D5 flag keeps `legacy` the default; commit 1 pins legacy behaviour first; D4 is the last commit |
| Calculation moving into the loop changes operand binding context | Reuse `DeterministicCalculationCapability.calculate(state)` unchanged; it already reads `state.bound_evidence_ids` / `bound_slot_bindings` |
| Trace shape changes break evaluation scripts | D1 is additive; existing `V2ExecutionTrace` keys are preserved |
| `harness_v3` and `legacy` diverge on real fixtures | Integration runner (§3.3) compares both on the same fixture in CI |

## 8. Decision record

**Decision: full close.** D1–D5 are all in scope.

Rationale: the central audit finding is that the harness does not own its own loop.
A trace-and-seam-only change would leave that unresolved, and the release-path edit is
already contained by the D5 flag (`legacy` stays the default) and by commit 1, which
pins current behaviour before any production code moves.

Rejected alternative: D1–D2 only (trace + named policy, generation left outside the
loop). Lower risk, but does not close the gap that motivated H1.

Consequence to accept: D4 edits the release path, the most sensitive code in the
repository. Commits are ordered so that this lands last, behind a default-off flag.

## 9. Delivery record

| Commit | Scope | Status |
|---|---|---|
| `fb318da` | characterization baseline + this plan | done |
| `023d54d` | D1 run turn trace, D2 named `AdaptiveActionPolicyV1` | done |
| `cbcfa00` | D3 calculation as a harness phase, D5 runtime mode flag | done |
| `0709636` | D4 closed loop via the harness finalizer | done |
| `715a92c`..`0bb08fc` | two review passes: 23 findings, 16 fixed | done |
| §11 | H1.1 integration runner, sealed fixtures, equivalence contract, seal | done |

Evidence, measured at each step against the same baseline:

- Baseline without any H1 change: **144 failed / 3835 passed / 38 skipped**
  (4017 collected), 2 collection errors from missing optional deps
  (`chromadb`, `torch`).
- After D1+D2: 144 failed / 3791 passed, failing-file breakdown byte-identical.
- After D3+D5: 144 failed / 3806 passed, failing-file breakdown byte-identical.
- After D4: 144 failed / 3822 passed, failing-file breakdown byte-identical.

The 144 are pre-existing and environmental: missing `artifacts/evaluation` frozen
fixtures, and `FINANCIAL_RUNTIME_MODE=v2` needing a configured production factory.
None touch `rag_v2.adaptive` or the coordinator. The exact split is recorded in
§11; H1's per-step evidence is the *failing-file breakdown diff*, which was
byte-identical at every step.

Two deviations from the plan, both recorded rather than absorbed:

1. **D5 moved into commit 3.** D3 cannot land without the flag: entering CALCULATE
   on the production path changes the execution model, which would have broken the
   behaviour pinned by commit 1.
2. **D4 implements the finalizer as a wrapper, not a rewrite.** The plan said to
   wire generation and validation into the harness. It did — but the finalizer
   reuses `_candidate_stage` verbatim rather than reimplementing release logic
   inside the loop, and the coordinator rebuilds the trace against the completed
   state so it covers `GENERATE` / `VERIFY` / `RELEASE`. No release logic is
   duplicated, so the validator remains the single release authority.

One latent hazard was found and closed while wiring D4: the harness's VERIFY phase
read `if verifier is None or verifier(state, output)`, so a generator wired without
a verifier would have released unconditionally. It now fails closed with
`VERIFICATION_NOT_WIRED`. This mattered because `TrustedReleaseValidationCapability.validate`
returns a `V2ValidationResult` object, not a bool — it is always truthy, so a
naive wiring would have released without the validator's verdict being read.

## 10. Review findings and disposition

A three-way review (change set, coordinator, `rag_v2/adaptive`) produced 15
findings. Fixed:

| # | Finding | Where |
|---|---|---|
| 1 | A rejected candidate could RELEASE: the test-release path wired the raw validator into the harness, and `bool(V2ValidationResult)` is always true | `_release_verdict` reads the verdict explicitly |
| 2 | harness_v3 dropped `route`, so a public field reported the plan intent instead of the generation route | outcome rebuild passes `route=` |
| 3 | harness_v3 released with the trace's reason-code superset, labelling a clean release `WRONG_PERIOD` | rebuild uses the candidate's own codes |
| 4 | A BLOCKED/FAILED in-loop calculation reached generation (validation was skipped) | invocation conditional, validation shared |
| 5 | The candidate stage read `capability.last_result`, an undocumented side effect; a conforming calculator without it turned a release into EXECUTION_ERROR | the loop's return value is captured at the call boundary |
| 6 | The in-loop CALCULATE ran *before* evidence admission | gated on `state.bound_evidence_ids` |
| 7 | A raising calculator produced a different failure class per mode | mapped to the legacy terminal |
| 8 | `NF_AGENT_RUNTIME_MODE` leaked into every coordinator construction, so an exported env var broke unrelated tests | only production wiring reads the env |
| 9 | `turns` bypassed `_sanitize_trace_payload`, breaking the no-private-reasoning invariant | added to the normalized field list |
| 10 | `AdaptivePhase(state.status)` raised on an unknown status, escaping `run()` | fails closed with `STRUCTURAL_NOT_READY` |
| 11 | A malformed tool packet raised during normalization, outside the tool `try` | normalization moved inside it |
| 12 | The VERIFY phase recorded no turn, so the trace could not show verification ran or failed | records a turn with the verdict |
| 13 | This document stated the pre-change guard formula as current | labelled §2 as the baseline record |

Reported, not fixed — pre-existing and out of H1 scope:

- `max_identical_query_retry` is declared, plumbed through
  `V2_MAX_IDENTICAL_QUERY_RETRIES`, and enforced nowhere. Left alone
  deliberately: the replanner emits `state.normalized_query` verbatim, so
  enforcing the default of 0 would change legacy retrieval behaviour that H1
  exists to preserve.
- `_capability_trace()` exposes lifetime port counters, so a trace can report
  work done by a previous request on the same coordinator.
  **Resolved in H1.1.** The counters are still lifetime figures -- they live on
  the port -- but nothing asserted that the ports themselves are per request,
  which is the property that makes the two the same thing. The claim in the
  original finding that evaluation scripts reuse a coordinator was wrong:
  `scripts/evaluation/run_tv2_canonical_benchmark.py:105` builds the runtime
  inside its per-question loop. `tests/test_trusted_v2_production_builder.py`
  now builds twice from one `TrustedV2RuntimeResources` and asserts all five
  ports differ while the fact store is still shared, and the coordinator says
  which way it depends on that. The remaining honest limit: a coordinator that
  *is* reused still reports a lifetime in a per-run trace; the invariant is
  pinned at the builder, not enforced at the coordinator.
- `runtime_metadata` is not passed through `_sanitize_trace_payload`, unlike the
  trace itself.

A second cleanup pass then:

- moved the harness test fixtures into `tests/harness/harness_support.py`, so
  the calculation-phase and closed-loop suites share one coordinator wiring
  instead of two copies that could drift;
- extended the mode-equivalence assertions to `calculations`,
  `calculation_result_id` and `claim_provenance`, which had been left
  uncompared — the same gap that had hidden the `route` and `reason_codes`
  divergences earlier;
- added `test_required_calculation_without_admitted_evidence_never_calculates`,
  pinning the admission gate at the harness level rather than only through the
  coordinator.

A second review pass then found 8 more. Fixed:

- `_candidate_stage` had two call sites and only one supplied the captured
  in-loop calculation result, leaving `state.calculation_attempted` and the value
  coupled by convention only. Both supply it now.
- `V2ExecutionTrace.__post_init__` had an `elif` byte-identical to its `else`;
  adding `turns` to the `elif` set was a no-op (the load-bearing change was the
  field tuple). The branches are collapsed.
- `BoundedAdaptiveRAGV1` accepted a `policy` carrying a *different* budget from
  the loop's own, creating two sources of truth for the same limits — the run
  could fail closed while reporting budget it never used. A mismatched policy is
  now rejected at construction.

Accepted or deferred, with reasons:

- **A stronger harness-side admission gate is not implementable there.** The
  coordinator's `_binder_admission_is_authoritative` needs the evaluator adapter;
  the harness only sees `state.bound_evidence_ids`, and `_EvaluatorAdapter.evaluate`
  assigns that field from the adapter, so the two cannot diverge on any path that
  goes through EVALUATE. Divergence requires a resumed or externally supplied
  state. The docstring now says the gate is "evidence was admitted" rather than
  claiming parity with the coordinator's gate.
- **harness_v3 builds the trace twice per released request** — once inside
  `_candidate_stage`, then again against the completed state. The second build is
  what makes the trace cover GENERATE/VERIFY/RELEASE. Removing it means patching
  a frozen outcome on the release path; not worth the risk for a per-request
  constant factor.
- **`_release_verdict` does not run on the production harness_v3 path**, because
  the finalizer replaces the harness's generator and verifier there, and the
  verdict is honoured instead inside `_candidate_stage`. That is correct — the
  release-integrity defect it fixes was reproducible only on the
  `allow_test_release` path, which is where it now applies.

## 11. H1.1 — Integration and seal

H1's core design needed no change. What it lacked was production-level evidence:
the equivalence claim was proven on a coordinator composed inside a test, and the
suite could not report an honest number about itself.

Delivered:

1. **Integration runner** — `scripts/runtime/run_nf_v3_h1_harness_integration.py`
   (§3.3, previously outstanding), plus `tests/harness/h1_integration.py` holding
   the fixtures and the wiring so the script and the test suite cannot drift.
2. **Ten sealed fixtures** — fact, calculation, blocked calculation, raising
   calculator, wrong-period no-progress, missing evidence, retrieval error,
   validator rejection, budget exhaustion, unsupported route. The set is
   content-hashed (`sealed_digest`) and the digest is recorded in the report.
3. **Canonical equivalence contract** — `tests/harness/equivalence.py`, replacing
   per-test field lists. See the seal document for the A/B/C partition.
4. **End-to-end trace proof** — `calculation_growth_rate` reaches
   `ACT → OBSERVE → EVALUATE → CALCULATE → READY_TO_GENERATE → GENERATE → VERIFY → RELEASE`
   on real wiring. `legacy` never enters CALCULATE.
5. **Suite accounting** (below).
6. **Runner rewired onto the production entry point** — the first H1.1 runner
   called the factory with an explicit `runtime_mode`, which meant
   `resolve_agent_runtime_mode()` — the actual production seam, and the only
   place the environment is read — was never exercised. The runner now sets
   `NF_AGENT_RUNTIME_MODE`, goes through `build_trusted_v2_runtime_for_request`,
   and asserts the flag reached the coordinator it produced. Three fixtures whose
   capability the production builder cannot be told to build use the factory with
   one substituted port, and the report states which tier each fixture used.
   Rewiring also surfaced that the WRONG_PERIOD replan does not vary retrieval —
   see the seal document, §6.
7. **Port scope pinned** — the capability snapshots the coordinator reports are
   lifetime figures, which only equal this-run figures because the builder makes
   fresh ports per request. That was true and untested; it now is tested (§10).
8. **Seal** — `docs/showcase/nf-v3-h1-harness-core.md`, tag `nf-v3-h1-harness-core`.

### Cross-check: the runner was falsified against the pre-fix tree

A green report is worthless if its assertions cannot fail. The fixture set was
run against `3abc2f1`, the commit immediately before the review fixes:

```
decision equivalence : 4/9
```

Five fixtures diverged — on `route` (four fixtures), `reason_codes` (two) and
`status` (one) — i.e. exactly the fields the hand-picked assertion lists had
omitted. Full table in the seal document.

### Suite accounting

The "3964 vs 3979" contradiction raised in review was a reporting error on my
side, not an inconsistency in the suite. The counts sum exactly:

| State | Collected | Failed | Passed | Skipped | Errors |
|---|---|---|---|---|---|
| Baseline `b393d3e` | 4017 | 144 | 3835 | 38 | 2 |
| After H1.1 | 4047 | 39 | 3865 | 143 | 0 |

`144 + 3835 + 38 = 4017`. The earlier "3964" omitted the skipped tests.

Both rows were measured in this working tree, which also carries five uncommitted
TV2 evaluation files contributing 14 tests (12 pass, 2 skip). On a clean checkout
the identities are `4033 = 39 + 3853 + 141`.

By cause, the original 144:

| Count | Cause | Disposition |
|---|---|---|
| 104 | Sealed evaluation artifacts under `artifacts/evaluation/<run>/` never committed | now skipped, reason names the missing path |
| 39 | `FINANCIAL_RUNTIME_MODE` defaults to `v2`; `v2` requires `TRUSTED_V2_RUNTIME_BUILDER`; these endpoint tests predate that default (introduced by `b84fa3a`) and never set the mode | **deliberately not fixed** — see below |
| 1 | a test mid-edit during the measurement run | n/a |

Setting `FINANCIAL_RUNTIME_MODE=v1` turns 34 of the 39 green; the remaining 5
fail on semantic-alignment behaviour instead. They are left failing rather than
pinned to `v1` because that choice is a product decision about the default
runtime mode, and an environment pin would make it by accident.

Two collection-time guards were added to `conftest.py`:

- **Optional-runtime modules.** `tests/evaluation/test_nf40_cli.py` (`chromadb`)
  and `tests/test_local_specialist_generator.py` (`torch`) are excluded, with a
  session header line naming the missing runtime. Previously these were
  collection *errors*, and pytest aborts the session on a collection error — so
  on this machine a bare `pytest` ran nothing at all.
- **Missing-artifact reads.** A test that fails reading a path under `artifacts/`
  is reported as a skip naming that artifact. The decision is taken from the
  actual error, not from the directory being absent: a first attempt skipped
  whole modules whose declared `ARTIFACT*` root was missing, and silently
  swallowed nine tests in those modules that pass without the artifact. Caught by
  diffing the skip set against the baseline failure set.

`pytest -m "not requires_artifacts"` deselects the artifact group, which is
marked at collection time when a module's declared artifact root is absent.

### One budget field is now explicitly inert

`max_identical_query_retry` was declared, read from
`V2_MAX_IDENTICAL_QUERY_RETRIES`, and asserted in the V2-16 contract test — but
read by nothing. `BoundedReplannerV1` reuses the same query text for
`MISSING_SLOT`, so enforcing the field at its default of 0 would forbid a retry
that `legacy` has always performed: a retrieval behaviour change arriving
through a budget, inside an ablation whose only claim is equivalence.

It is now listed in `AdaptiveRAGBudgetV1.RESERVED_FIELDS`, production warns via
`unenforced_settings()` when an operator configures it away from its default,
and `tests/harness/test_reserved_budget.py` fails if any module in
`rag_v2/adaptive/` starts reading it — which is the signal to enforce it
deliberately and update the note.
