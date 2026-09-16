# NF-V3 H1 — Financial Agent Harness: Plan

Status: **plan, pending execution**
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

**The audit found that roughly 85% of the proposed harness already exists** — under
different names, in `rag_v2/adaptive/`, and it is already running in production.
Building the brief as written would have duplicated a live execution kernel.

This document records the real execution model, the deviation from the brief, and
the re-scoped H1 that closes the genuine gap.

## 2. Audit: the real execution path

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
(`adaptive_state_machine.py:95`, `max_total_tool_calls * 4 + 12`) prevents infinite
loops, and every abnormal exit routes through `_fail()` to `FAIL_CLOSED` with a
`ReasonCode`.

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
- Full collection: **3964 tests collected, 2 collection errors**, both from missing
  optional dependencies in this interpreter (`chromadb` for
  `tests/evaluation/test_nf40_cli.py`, `torch` for `tests/test_local_specialist_generator.py`).
- Tests must be run with `python -m pytest` from `finquery_rag/backend` so that
  `rag_v2` and `src` are importable.

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
