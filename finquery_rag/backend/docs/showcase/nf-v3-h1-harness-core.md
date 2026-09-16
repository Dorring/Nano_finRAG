# NF-V3 H1 Harness Core Seal

Status: **BOUNDED FINANCIAL AGENT HARNESS CORE = SEALED**

H1 closes the existing trusted execution loop: calculation, generation,
verification and release became harness phases, behind a default-off runtime
mode flag, with the deterministic validator still owning the release decision.

**What this is not.** It is a *bounded, policy-constrained* harness, not an
autonomous one. The execution skeleton was already present; the agent
intelligence half is not. §7 lists exactly what is still missing, and that list
is the honest boundary of this seal.

## 1. The audit finding, stated correctly

The H1 brief assumed no harness existed and specified a new `contracts/run.py` /
`harness/` / `tools/` tree. `rag_v2/adaptive/` already implemented the component
list — `AdaptiveRAGStateV1`, `AdaptiveRAGBudgetV1`, `BoundedReplannerV1`,
`ProgressDetectorV1`, `BoundedAdaptiveRAGV1.run()` — and it was already on the
production path through `BoundedTrustedV2Coordinator.execute()`.

An earlier revision of this document summarised that as "roughly 85% of the
harness already exists". **That framing was too generous, and it was reached by
matching class names to a component list.** Name correspondence is not capability
correspondence. Two of the correspondences are only partly true:

- `BoundedReplannerV1` is not an agent decision layer. It is a deterministic
  `reason_code -> ToolCapability` table. That is a deliberate design choice for a
  financial runtime, but it means no model participates in the loop's choice of
  next action, which is the single largest thing separating this from a modern
  harness.
- `tools: Mapping[ToolCapability, ToolFn]` is a capability dispatch table, not a
  tool runtime. It carries no input/output contract per tool, no execution-status
  versus domain-status distinction, no per-tool timeout or error taxonomy. The
  calculator defect found during H1 (§5) is exactly the failure that distinction
  prevents: `calculate()` returning normally is not the same as the calculation
  having succeeded.

The accurate statement is:

> **The execution harness skeleton existed and was live. The agent intelligence
> harness — per-turn context engineering, model-guided replanning, recovery,
> durable state — did not.**

What was genuinely broken was narrower and worse than either description: the
loop did not close. The coordinator stopped the harness at `READY_TO_GENERATE`
and produced the answer in `_candidate_stage`, outside the loop, so the harness's
own `GENERATE` / `VERIFY` / `RELEASE` phases were unreachable in production. The
code said so itself with a `DOWNSTREAM_EXECUTION_NOT_WIRED` reason code.

## 2. Execution paths

### legacy (default)

```
FinancialQueryRequest
  -> build_trusted_v2_runtime_for_request()
  -> BoundedTrustedV2Coordinator.execute()
       supervisor.plan()
       BoundedAdaptiveRAGV1.run()          exits at READY_TO_GENERATE
       _candidate_stage()                  OUTSIDE the loop
         binder-admission gate -> calculate -> generate -> validate -> release
```

### harness_v3

```
FinancialQueryRequest
  -> NF_AGENT_RUNTIME_MODE=harness_v3
  -> resolve_agent_runtime_mode()          the only environment read
  -> build_trusted_v2_runtime_for_request()
  -> BoundedTrustedV2Coordinator.execute()
       supervisor.plan()
       BoundedAdaptiveRAGV1.run()
         ... -> CALCULATE -> READY_TO_GENERATE
             -> GENERATE  (the finalizer wraps _candidate_stage)
             -> VERIFY    (the validator's verdict, read as a verdict)
             -> RELEASE
```

The finalizer wraps `_candidate_stage` rather than reimplementing it, so no
release logic is duplicated: the harness decides *when* generation happens, the
deterministic validator still decides *whether* an answer may be released.

## 3. Why H1 extended `rag_v2/adaptive/` instead of adding a second kernel

Building the brief literally would have produced two parallel execution kernels
with two `RunState`s, two budgets, two replanners and two stop policies, both
reachable from production. The brief itself authorised the deviation ("do not
force the design; adapt minimally to the real code and explain why"). The
deviation was taken, and it is recorded here rather than presented as the
original plan.

`rag_v2/adaptive/` is now the harness kernel. A second one is not to be added.

## 4. Decision equivalence

`tests/harness/equivalence.py` classifies every field of `V2ExecutionOutcome`
once, so no differential test picks its own list:

| Class | Fields | Rule |
| --- | --- | --- |
| **A** decision-bearing | `status`, `release_status`, `answer`, `route`, `reason_codes`, `evidence_ids`, `citation_ids`, `calculation_ids`, `calculation_result_id`, `citations`, `calculations`, `validator_status`, `claim_provenance`, `plan_id`, `evidence_packet_id`, `latency_metadata`, `runtime_metadata` | must be identical |
| **B** harness-only | `debug_metadata` (the execution trace), and `calculation_in_harness` inside `runtime_metadata` | stripped, never asserted equal, verified against its own invariants |
| **C** volatile | `latency_ms`, `duration_ms`, `elapsed_ms`, `run_id`, `span_id` | removed at any depth |

`plan_id`, `evidence_packet_id` and `latency_metadata` were uncertain, so per the
project's rule they default to *compared*. `unclassified_fields()` returns empty,
and a test asserts it: a new field on the outcome cannot pass silently until it
has been bucketed.

The partition exists because hand-picked lists are how the first two divergences
survived a review pass: `route` was missing from the first list, and
`calculations` / `calculation_result_id` / `claim_provenance` from the second.

## 5. Two defects found, both in release integrity

**1. A rejected candidate could be released (pre-existing).** The
`allow_test_release` path handed the raw validator to the harness as its
verifier. `TrustedReleaseValidationCapability.validate` returns a
`V2ValidationResult` dataclass, which is truthy regardless of its verdict, so
`bool(verifier(...))` released candidates the validator had just refused. It was
the only path in the coordinator that could release without the validator
agreeing.

**2. A blocked calculation reached generation (introduced during H1).** When
CALCULATE moved inside the loop, the candidate stage re-read the calculator's
result by id instead of re-validating it, so a `BLOCKED` result could flow into
generation. This is the concrete case of *invocation success is not domain
success* noted in §1.

Both are fixed, both have tests that fail if they return, and the integration
suite now asserts the second one from the generation capability's own recorded
state rather than from control flow.

## 6. Sealed integration fixtures

Ten fixtures, hashed as a set:

```
sealed fixture digest
9c7cda07004f05ce1bac616c1a7f6bb3570488c052d6299fe193674d1a3a562e
```

They run through the **production entry point** — `NF_AGENT_RUNTIME_MODE` read by
`resolve_agent_runtime_mode`, then `build_trusted_v2_runtime_for_request`, then
the factory — over the real R4 retriever, Semantic Binder, deterministic
calculator, generator routing and release validator. The runner asserts the flag
reached the coordinator it produced, so a mode that silently failed to propagate
cannot pass.

| Fixture | legacy | harness_v3 | Equivalent | Entry point |
| --- | --- | --- | --- | --- |
| `fact_direct` | READY_FOR_RELEASE | READY_FOR_RELEASE | yes | production |
| `calculation_growth_rate` | READY_FOR_RELEASE | READY_FOR_RELEASE | yes | production |
| `calculation_blocked` | FAIL_CLOSED | FAIL_CLOSED | yes | factory + substituted calculator |
| `calculation_error` | EXECUTION_ERROR | EXECUTION_ERROR | yes | factory + substituted calculator |
| `wrong_period_no_progress` | FAIL_CLOSED | FAIL_CLOSED | yes | production |
| `missing_evidence` | FAIL_CLOSED | FAIL_CLOSED | yes | production |
| `retrieval_error` | EXECUTION_ERROR | EXECUTION_ERROR | yes | production |
| `validator_rejection` | FAIL_CLOSED | FAIL_CLOSED | yes | factory + substituted generator |
| `budget_exhaustion` | FAIL_CLOSED | FAIL_CLOSED | yes | production |
| `unsupported_route` | FAIL_CLOSED | FAIL_CLOSED | yes | directly constructed coordinator |

```
decision equivalence : 10/10
expectation failures : 0
release bypass       : 0
false calc release   : 0
infinite loop        : 0
```

Three fixtures cannot use the production entry point, and say so in the report
rather than being presented as production wiring: two need a calculator the
production builder cannot be told to build, one needs a generator that cites
unadmitted evidence, and one needs an absent retrieval port — which the factory
correctly refuses, making `UNSUPPORTED_TOOL_ROUTE` a defence-in-depth guard
rather than a production-reachable path.

### The successful calculation trace

```
ACT -> OBSERVE -> EVALUATE -> CALCULATE -> READY_TO_GENERATE -> GENERATE -> VERIFY -> RELEASE
```

`legacy` never enters `CALCULATE`: its calculator stays outside the loop. The
fixture asserts both.

### The runner was falsified before it was trusted

A green report is worthless if its assertions cannot fail. Run against
`3abc2f1`, the commit immediately before the review fixes, the same fixture set
reports 4/9 and names the divergences — `route` on four fixtures, `reason_codes`
on two, `status` on one — i.e. exactly the fields the hand-picked lists omitted.

### A finding from the rewiring: WRONG_PERIOD does not vary retrieval

The first version of the wrong-period fixture drove its second retrieval round
from a round counter, a test-only affordance. Rebuilt against the real
`CandidateDirectR4Policy`, the two rounds issue **byte-identical derived
queries**: the period constraint the replanner adds is already present in the
alias queries generated from the slot in round one (`"Revenue FY2024 | Revenue |
FY2024"` appears in both). The recovery therefore re-retrieves the same
candidates, re-binds the same wrong-period fact, and loops to budget exhaustion.

This is pre-existing runtime behaviour, untouched by H1, and identical in both
modes (which the fixture now pins). Recovery is not unreachable — the unit suite
drives it with a reader that can advance between rounds — but it is not
reachable through this retriever's query derivation. Recorded, not fixed:
changing retrieval or binder semantics is outside H1's scope by construction.

## 7. What H1 does not have

These are the capabilities of a modern agent harness that this project does
**not** have. They are the forward roadmap, not a defect list.

1. **Per-turn agent context engineering.** Each model call does not rebuild what
   the model sees from run state. Context is still conversation history plus RAG
   evidence plus the plan. There is no `ContextPack` (task goal, phase, verified
   and missing slots, calculation state, recent observations, failure state,
   available capabilities, remaining budget, evidence summaries, artifact
   references).
2. **Artifact / JIT context.** Evidence is carried as packets, not as addressable
   artifacts loaded on demand. Nothing compacts tool output before it re-enters
   context.
3. **Model-guided replanning.** `BoundedReplannerV1` is a deterministic
   reason-code table. No model proposes an action in response to an observation,
   and there is no typed action proposal for a policy to validate.
4. **Durable checkpoint / resume.** Run state is in-process. A crash loses the
   run; there is no resume, no failure taxonomy and no repeated-action detection
   beyond the same-tool retry budget.
5. **Subagent orchestration.** None, deliberately. The loop already retrieves,
   replans, repairs, calculates and verifies; splitting those into agents would
   be a retreat. The only later candidate is a complexity gate spawning ephemeral
   evidence workers that hold no calculation, release or validation authority.

Also absent, and named so it is not mistaken for done: the release path is
single-shot (a rejected candidate fails closed rather than entering the repair
lane), `runtime_metadata` is not passed through `_sanitize_trace_payload`, and
`harness_v3` builds the execution trace twice on a released request.

## 8. Acceptance criteria

| # | Criterion | Result |
| --- | --- | --- |
| 1 | Existing regression does not degrade | PASS — failing-test set unchanged from the pre-H1 baseline |
| 2 | New harness tests all green | PASS — `tests/harness/` 89 passed |
| 3 | `legacy` still runs, and is the default | PASS |
| 4 | `harness_v3` enabled by feature flag | PASS — and the flag path itself is under test |
| 5 | Evidence Gate not bypassed | PASS — one violation found and fixed |
| 6 | Calculator contract not bypassed | PASS — 9 operations unchanged |
| 7 | Runtime Validator not bypassed | PASS — one pre-existing bypass found and fixed |
| 8 | Unsupported action fails closed | PASS — `unsupported_route` |
| 9 | Budget exhaustion fails closed | PASS — `budget_exhaustion` |
| 10 | No infinite loop | PASS — loop guard + bounded-turn invariant |
| 11 | No new agent-framework dependency | PASS — `pyproject.toml` unchanged |
| 12 | Simple fact query needs no multi-agent | PASS — `fact_direct` |
| 13 | No subagent / swarm | PASS |

## 9. Test suite accounting

```
4047 collected (H1.1, before the rewiring) = 39 failed + 3865 passed + 143 skipped
4053 collected (H1.1, final)               = 39 failed + 3871 passed + 143 skipped
4017 collected (baseline)                  = 144 failed + 3835 passed + 38 skipped
```

The counts sum exactly in every row. H1's audit reported "3964 collected", which
was bad arithmetic on the same run — it omitted the skipped tests. There was no
count contradiction, only a reporting error, recorded here because the project
leans on exact counts.

Measured in a working tree that also carries five uncommitted TV2 evaluation
files contributing 14 tests (12 pass, 2 skip). On a clean checkout of this
commit, subtract them: `4039 = 39 + 3859 + 141`. The identities hold either way.

The original 144 failures by cause:

| Count | Cause | Disposition |
| --- | --- | --- |
| 104 | Sealed evaluation artifacts under `artifacts/evaluation/<run>/` never committed | now skipped, reason names the missing path |
| 39 | `FINANCIAL_RUNTIME_MODE` defaults to `v2`; `v2` requires `TRUSTED_V2_RUNTIME_BUILDER`; these endpoint tests predate that default (`b84fa3a`) and never set the mode | **deliberately not fixed** |
| 1 | a test mid-edit during the measurement run | n/a |

Setting `FINANCIAL_RUNTIME_MODE=v1` turns 34 of the 39 green; the remaining 5
fail on semantic-alignment behaviour instead. They are left failing rather than
pinned to `v1` because choosing the default runtime mode is a product decision,
and an environment pin would make it by accident.

Two collection-time guards were added to `conftest.py`:

- Modules that cannot import without an optional runtime (`chromadb`, `torch`)
  are excluded with a header line naming what is missing. Previously these were
  collection *errors*, and pytest aborts the session on a collection error — so
  on this machine a bare `pytest` ran nothing at all.
- A test that fails reading a path under `artifacts/` is reported as a skip
  naming that artifact. The decision is taken from the actual error, not from the
  directory being absent: a first attempt skipped whole modules whose declared
  `ARTIFACT*` root was missing and silently swallowed nine tests in those modules
  that pass without the artifact.

`pytest -m "not requires_artifacts"` deselects the artifact-dependent group.

## 10. One budget field is explicitly inert

`max_identical_query_retry` was declared, read from the environment and asserted
in the V2-16 contract test — but read by nothing. `BoundedReplannerV1` reuses the
same query text for `MISSING_SLOT`, so enforcing the field at its default of 0
would forbid a retry `legacy` has always performed: a retrieval behaviour change
arriving through a budget, in an ablation whose only claim is equivalence.

It is listed in `AdaptiveRAGBudgetV1.RESERVED_FIELDS`, production warns when an
operator configures it away from its default, and a tripwire test fails if any
mode in the control plane starts reading it.

## 11. Capability trace scope

The coordinator reports each port's own `trace_snapshot()` verbatim, and those
are lifetime figures — `validation_calls`, `calculator_call_count`,
`retrieval_rounds` accumulate for as long as the port lives. They describe *this
run* only because `build_trusted_v2_runtime_for_request` builds a fresh port set
per request. `tests/test_trusted_v2_production_builder.py` asserts all five ports
differ between two builder calls over one shared resource set, and the
coordinator states which way it depends on that. Honest limit: a coordinator that
*is* reused across requests would still report a lifetime in a per-run trace —
pinned at the builder, not enforced at the coordinator.

## 12. Reproduction

```bash
cd finquery_rag/backend

python scripts/runtime/run_nf_v3_h1_harness_integration.py   # ablation report
python -m pytest tests/harness -q                            # harness suite
python -m pytest -q                                          # full suite
```

`scripts/runtime/` is matched by the repository root's `.gitignore` rule
`runtime/`, so new files there need `git add -f`.

## 13. Commits

Core:

- `fb318da` test(harness): freeze the legacy v2 execution model as a baseline
- `023d54d` feat(harness): surface run turns and name the action policy
- `cbcfa00` feat(harness): make deterministic calculation an explicit harness phase
- `0709636` feat(harness): close the generation loop behind the runtime mode flag
- `715a92c` fix(runtime): validate in-loop calculation results in harness_v3
- `3abc2f1` test(harness): prove mode equivalence at the metadata level too
- `8f5e705` docs(harness): correct two misleading dead-code comments
- `5c6a046` fix(runtime): close review findings in the harness control plane
- `0bb08fc` refactor(harness): share test fixtures and close a second review pass

H1.1 seal:

- `cc9e412` test(harness): pin the legacy/harness_v3 decision-equivalence contract
- `12cd3c9` feat(harness): run the ablation over sealed fixtures end to end
- `7f4589b` fix(tests): make the suite report an honest number about itself
- `7d39b34` docs(harness): seal NF-V3 H1
- `acc63f0` docs(harness): state which working tree the sealed counts came from
- `8eaa319` test(runtime): pin that capability ports are per request, not per process
- and the H1.1 rewiring commit that moved the runner onto the production entry point

## 14. Next

`H2A Context & Artifact Runtime` — what the agent sees each turn, not more state
machinery. Then `H2B Hybrid Decision Runtime` (deterministic replanner for
slot-missing / operands-ready, model-guided replanner for metric ambiguity and
evidence conflict, typed action proposal, policy validation), then `H3 Recovery &
Durable State`, then `H4 Selective Parallel Workers` only if a complex benchmark
demonstrates the benefit.
