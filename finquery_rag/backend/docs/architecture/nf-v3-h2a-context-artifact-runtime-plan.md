# NF-V3 H2A — Context & Artifact Runtime: Audit and Plan

Status: **AUDIT + PLAN ONLY — no production implementation in this phase**

Baseline: `02d86d7` (`nf-v3-h1-harness-core` is frozen and is not moved by this
work). Predecessor: `nf-v3-h1-2-runtime-baseline-disposition.md`.

Two independent audits of the real code back this document: one over the
conversation/context stack, one over the runtime/adaptive stack. Findings marked
**[verified]** were reproduced by execution during this audit rather than read.

---

## 1. The finding that decides the shape of H2A

H1 established that the execution loop was ~85% present and the real gap was
that it did not close. The analogous question here is *what does the agent
actually see each turn*, and the answer is narrower than the roadmap assumed.

**There is no context runtime to extend. There are five context budgets, three
durable copies of the same evidence, and no authority over either.**

Concretely:

- **Five budgets, different units, none authoritative** — dialogue tokens
  (`ContextBudgetManager`, `target_tokens=4096`), evidence tokens
  (`ContextBuilder`, 900 real tiktoken tokens), raw messages
  (`SessionManager`, 16), tool/replan steps (`AdaptiveRAGBudgetV1`,
  `RepairBudget`), and candidate counts (`CANDIDATE_BUDGET=50`,
  `RERANK_INPUT_BUDGET=100`, `MAX_EVIDENCE_ITEMS=5`). Only the step budgets have
  a hard, observable failure mode (`ABSTAINED`); the rest drop content silently
  and report only through a diagnostics dict.
- **Three per-request copies of the same evidence** — `state.evidence_packets`,
  `R4RetrievalCapability.last_result.candidate_evidence` (verbatim, top-level
  `content`/`source_text`), and
  `SemanticEvidenceEvaluationCapability.last_run.request.facts`. Refreshed every
  retrieval round; nothing reconciles them.
- **The raw source text has no contracted home.** `EvidencePacketV1` has no
  content field. **[verified]** `content` and `source_text` land only in a
  free-form `metadata` bag, and `pdf_page` lands there too.
- **Two conversation stores in one SQLite file** with **incompatible turn-id
  spaces** (`turn_N` vs `session_message_N`), which makes the relevance
  filter's `+10.0` explicit-reference boost and the budget's "always retain
  protected turns" branch **inert in production**.
- **The dialogue summariser is dead in the production path.**
  `_serialize_state` zeroes `recent_turns` on every write and `_deserialize_state`
  rejects non-empty ones, so `update_compressed_history` — which needs ≥8 turns —
  always returns early. `compressed_history` is never populated in production.

None of this is a defect list to fix one by one. It is the evidence that H2A is
a **consolidation** phase, not an additive one.

---

## 2. Answers to the audit questions

### 2.1 What is visible at each adaptive turn?

`AdaptiveRAGStateV1` (`rag_v2/adaptive/adaptive_contracts.py:263-316`) is a
**single mutable object per run**, handed to every port by reference. There is no
per-turn snapshot and no turn-scoped projection: turn N+1 sees everything turn N
wrote, and ports write into it (the evaluator sets `bound_evidence_ids` and
`bound_slot_bindings`; the calculator sets `calculation_result`,
`calculation_result_id`, `_calculation_result_obj`; the generator sets six
fields).

Content-bearing fields include `normalized_query`, the full `plan` dict,
`evidence_packets`, `calculation_result` (with operand `source_text`),
`candidate_answer`, `last_action.query`, and per-turn `query` text. Everything
else is ids, hashes, counters or labels.

### 2.2 What reaches a model prompt?

Three model boundaries are reached from the state, and **no prompt is built
inside `rag_v2/adaptive/`** — it imports no provider:

| Boundary | Reads from state | Projection |
| --- | --- | --- |
| Supervisor | **nothing** — called with `request.standalone_query` before the state exists | static prompt + JSON schema + question |
| Binder | `state.evidence_packets` | **allowlisted** (`binder_fact_view.py:190-223`); raw text, assistant text and nested mappings excluded by construction |
| Specialist generator | `_bound_items(state)` — the packet dicts | **none** — passed through unprojected |

The asymmetry is the finding: **the binder boundary promises raw text cannot
cross, and the specialist boundary has no such promise.** **[verified]** the
packet dict handed to a specialist backend carries
`metadata["source_text"]`. The built-in renderer happens not to read it (it looks
at top level, `local_specialist_generator.py:193`), but that is a property of one
renderer, not of a contract — any other specialist backend renders it.

The verifier and validator are deterministic (`RuntimeGenerationValidatorV1`,
`SemanticClaimVerifierV1`); no model there.

### 2.3 How often is raw retrieval output carried?

Three durable per-request holders (§1), plus loop locals, plus the process-scoped
fact store. Everything downstream of the loop is reference-only: the trace
records ids and counts, and `'content' in json.dumps(trace)` is `False`.

### 2.4 Evidence packet ↔ evidence ids

**One id space by construction.** R4 forces `evidence_id` from
`candidate.get("evidence_id") or candidate.get("fact_id")`
(`trusted_v2_r4.py:695-700`), and that single field flows to the trace, the
outcome, the generator, the calculator, the validator and `claim_provenance`.

Two hazards:
- The canonical fact store derives **four** ids from one physical cell
  (`physical:`, `evidence:`, `candidate:`, `citation:`). Nothing dedupes by
  content — `add_evidence` keys by `evidence_id` only — and
  `EvidencePacketV1.content_hash` is **not** a content identity, since it hashes
  `to_dict()` which includes the id. So progress detection cannot notice a
  repeated round returning the same row under a new key.
- **Five id-resolution helpers, two priority orders** — `evidence_id`-first
  (`coordinator.py:66-71`, `trusted_v2_validation.py:37`) vs `fact_id`-first
  (`trusted_v2_generation.py:83`, `trusted_v2_calculation.py:73`). They agree
  only because R4 aliases the two today.

### 2.5 Does a calculation carry domain validity?

**Yes, and it is a real distinction** — `CalculationResult.status` is
`NOT_APPLICABLE | READY | EXECUTED | BLOCKED | FAILED`
(`src/domain/calculation.py:52-62`, `:186-210`), and the coordinator checks it
correctly (`coordinator.py:1069`).

But the runtime's gate is **id equality, not status**:
- `calculation_result_id` is a *usability proxy*, assigned only on `EXECUTED`,
  and its value comes from `capability.last_calculation_id` — a private
  attribute that is **not in the port contract** (which is only
  `calculate(state) -> Any`).
- The validator checks `candidate.calculation_ids != (state.calculation_result_id,)`,
  so a conforming-but-different calculator that set a non-None id for a BLOCKED
  result would pass.
- **`_structured_calculations` has no status gate at all.** **[verified]** given
  a `BLOCKED` result and a non-None id it emits
  `{'status': 'blocked', 'calculation_id': 'C1-fake', 'error_code': 'INSUFFICIENT_OPERANDS'}`.
  Only the coordinator's upstream fail-closed at `:1069` keeps that out of a
  released outcome — one call site, not a contract.

This is precisely the defect class the H1 review named, still structurally
present.

### 2.6 Where does evidence text live after binding?

Only inside `metadata["content"]` / `metadata["source_text"]` — **[verified]**, a
key no contract declares and that the binder's promoted-key allowlist
deliberately omits. Plus verbatim copies in the R4 last-result and the binder's
last run.

One consequence: **`page` never reaches the public citation.** `pdf_page` lands
in `metadata`, while `_structured_citations` reads `packet.get("page", packet.get("pdf_page"))`
at top level (`coordinator.py:129`). `BoundFact.pdf_page` exists but is never
constructed by the runtime.

### 2.7 What the supervisor and replanner receive

- **Supervisor** — `self.supervisor.plan(request.standalone_query)`. Exactly one
  string. No state, no plan, no evidence, no history. `MAX_CALLS=1` per request.
- **Replanner** — `replan(state, evaluation)`. It reads `state.replan_rounds`,
  `state.normalized_query`, `state.required_slots[*].period/scope/entity`,
  `evaluation.decision/reason_codes/missing_slots`, and its own budget. No
  evidence content, no ids. It declares a `history` kwarg the loop never passes.

### 2.8 What the trace records per turn

```json
[{"turn":1,"action":"SEMANTIC_RETRIEVAL","reason_code":"MISSING_SLOT",
  "query":"What was revenue?","outcome":{"packet_count":1,"evidence_ids":["REVENUE"]}},
 {"turn":2,"action":"CALCULATE","outcome":{"calculation_status":"executed"}},
 {"turn":3,"action":"GENERATE","outcome":{"generated":true,"output_type":"bool"}},
 {"turn":4,"action":"VERIFY","outcome":{"verification_passed":true}}]
```

`query` is the only content; everything else is references, labels and counters.
The trace is a **deep copy of state** (`turns`, `transitions`, `tool_history`),
not an independent projection — so every new state field risks becoming a second
trace field by copy.

### 2.9 Where the proposed artifacts belong

Both land on **existing** contracts; neither needs a new package.

| Proposed | Extend | Why not new |
| --- | --- | --- |
| `EvidenceArtifact` | **`EvidencePacketV1`** (`adaptive_contracts.py:158-216`), with `BindingStatus` lifted from `rag_v2/contracts/evidence.py:13` | It is already the object the loop carries and every port reads, and it already has id, metric, period, unit, value, source, document_id. It lacks exactly two things: a **page** field and a **binding status**. |
| `CalculationArtifact` | **`CalculationResult`** (`src/domain/calculation.py:186-274`), with `supporting_evidence_ids` from `CalculationResultPacket` (`rag_v2/contracts/calculation.py:20-62`) | It already carries operation, operands (with `evidence_chunk_id`, `page`), value, unit, formula version, `status` and `error_code`. Its only missing element is the **id**, computed outside it today. |
| `AgentContextPack` | dataclass in **`src/runtime/trusted_v2_contracts.py`**, compiler next to **`rag_v2/evidence/binder_fact_view.py`** | Those two files already own the model-facing projection contract and the `BINDER_ADMITTED_EVIDENCE` trust rule. A new package would create a third projection vocabulary beside `binder_fact_view.py` and `ContextBuilder`. |

Note `BoundFact` (`rag_v2/contracts/evidence.py:29-49`) is *already* the field
list the proposal describes — and is constructed nowhere in `src/` or `rag_v2/`.
It is the shape to copy, not a second thing to build.

### 2.10 Existing id → content indirection

**Yes, and it is the established pattern**, which is the strongest argument
against inventing a new one: `bound_evidence_ids` → `evidence_packets` resolved
in five separate places; outcome `evidence_ids` → `citations`; binder `F01..Fn`
handles → `fact_id`; `citation_id` → source metadata; claim ids → provenance.

The one place with **no** indirection is the calculation: both the id and the
object are written onto the state, and the object is read back through a
**private attribute** by three modules. That is the inverse of an indirection,
and it is why the status gate is enforceable only at one call site.

---

## 3. File-level plan

Ordered so that each step is independently verifiable and none of them requires
the next.

### Step 1 — One evidence contract, with content and binding status contracted

- `rag_v2/adaptive/adaptive_contracts.py` — add `page` and `binding_status` to
  `EvidencePacketV1` (lift `BindingStatus` from `rag_v2/contracts/evidence.py`),
  and a declared `content_ref` rather than an untyped `metadata` bag key.
- `src/runtime/trusted_v2_coordinator.py:129` — `_structured_citations` reads the
  promoted field, restoring `page` to public citations.
- Keep `metadata` for genuinely opaque candidate fields; stop using it as the
  home for contracted content.

Closes: §2.6 (page lost), §1 (uncontracted content home). Does not touch
retrieval or the fact store.

### Step 2 — Calculation id into the contract, status into the gate

- `src/domain/calculation.py` — move the id in (today `_calculation_id` in
  `trusted_v2_calculation.py:121-139`) so a result identifies itself.
- `src/runtime/trusted_v2_capabilities.py` — add `calculate`'s return contract
  explicitly, so `last_calculation_id` stops being the interface.
- `src/runtime/trusted_v2_coordinator.py:146-164` — `_structured_calculations`
  gates on `status is EXECUTED`, matching the upstream check, so the invariant is
  enforced in two places instead of one.

Closes: §2.5. This is the defect class the H1 review named; the H1 fix is
correct and its invariant is currently held by a single call site.

### Step 3 — One projection for the specialist boundary

- `src/runtime/trusted_v2_generation.py` — project `_bound_items(state)` through
  the same allowlist shape the binder already uses
  (`build_runtime_binder_fact_view`), instead of passing packet dicts through.

Closes: §2.2 — the asymmetric disclosure policy. This is a **correctness and
trust-boundary fix, not an optimisation**, and should land before any token
work.

### Step 4 — One context budget authority

- Decide which of the five budgets is authoritative and make the others
  subordinate or explicit. The recommendation: **`ContextBuilder`'s real-token
  budget owns the evidence that reaches the generator** (it is the only one using
  real tokenisation), and the dialogue budget becomes a named input to it rather
  than an independent cap. `ContextBudgetManager.max_tokens` and
  `summary_max_tokens` are inert today and should be removed or wired.
- `src/runtime/trusted_v2_contracts.py:166` — `runtime_budget` is declared and
  never populated or read. Either wire it into this authority or delete it.

Closes: §1 (five budgets, none authoritative), §2.3's silent drop points.

### Step 5 — `AgentContextPack`

- `src/runtime/trusted_v2_contracts.py` — the dataclass: goal, current phase,
  plan summary, verified/missing slots, evidence **refs**, calculation **refs**,
  recent observations, failure state, available capabilities, remaining budget,
  dialogue summary.
- Next to `rag_v2/evidence/binder_fact_view.py` — the compiler, following the
  same allowlist discipline, and routed through `ContextTrustLevel` so content
  cannot self-attest as `BINDER_ADMITTED_EVIDENCE`.

Explicit rule: **the compiler takes `RunState` and emits a projection. It never
serialises `RunState`.** That is the difference between context engineering and
`json.dumps`.

### Step 6 — Deterministic clearing, and not before

- Drop raw tool results from the active context once an artifact reference
  exists. No LLM compaction in H2A: the deterministic steps above are what the
  financial task actually needs, and compaction is meaningful only once there is
  a long-running context to compact.

### Deliberately not in H2A

Model-guided replanning (H2B), MCP, subagents, multi-agent, durable checkpoint,
ReAct, and any LangChain/LangGraph/DeepAgents migration. Also not: fixing the
dead `compressed_history` path by persisting `recent_turns` — that would break
the `SQLiteConversationStateStore` trust boundary, which forbids answer-bearing
keys.

---

## 4. Evaluation design — H1 vs H2A ablation

The point is **not** to raise accuracy. The expected and acceptable result is
that trust metrics hold flat while active context shrinks. An accuracy jump would
be a reason to distrust the measurement.

Run both configurations over the same fixture corpus, same model, same retrieval:

| Metric | Expected |
| --- | --- |
| end-to-end task success | ≈ unchanged |
| grounded success | ≈ unchanged |
| false binding | 0 → 0 |
| **false release** | **0 → 0 (hard gate)** |
| average input tokens | down |
| P95 input tokens | down |
| raw tool-result tokens retained | down |
| context construction latency | reported; may rise, must be bounded |

Method notes that matter for this to mean anything:

- **Token counts must come from the real tokenizer** used by the model boundary,
  not the regex approximation in `ContextBudgetManager` (which is 1.15×word
  count, not the 1.25 its docstring claims).
- **Measure at the boundary, not the layer.** Today
  `estimated_context_tokens` is reported as "context tokens" but is a
  dialogue-turn estimate; the number that matters is what reaches the generator.
- **The corpus does not exist yet.** H1.1 established that
  `tests/fixtures/tv2_07_production_readiness/` is a labelled scoring set with no
  fact corpus, and that 12 of its 22 cases are not covered by any fixture.
  Building those cases is a prerequisite for the ablation, and is the natural
  first task of H2A implementation rather than a separate phase.

---

## 5. Risks

1. **A naive context layer will double-rewrite the query.** Conversation mode
   defaults to `on`, so the resolver runs first; V1's rewrite is bypassed only
   because `query_as_resolved=True` is set. A layer that runs in addition to the
   resolver rewrites twice.
2. **`compressed_history` is empty in production.** A new summariser must not
   assume L3 is populated, and must not "fix" it by persisting `recent_turns`.
3. **Referenced-turn protection never fires** because of the id-space mismatch.
   Tests pass only because they hand-construct matching ids.
4. **Three candidate-pool tiers already exist for V2.** Adding a fourth
   "keep top-N evidence" stage compounds rather than consolidates.
5. **`ContextBuilder._last_context_evidence` is mutable cross-request state** on
   a cached engine, and in the truncation branch it stores *truncated* text as
   validation evidence.
6. **Dead surface that H2A may mistake for live**: `scope_groups`,
   `_candidate_citation_ids` / `_candidate_calculation_ids` (read, never
   written), `max_identical_query_retry`, `BoundFact`,
   `VerifiedEvidencePacket`, `BoundedReplannerV1(history=...)`,
   `V2ExecutionRequest.runtime_budget`.

---

## 6. What this document does not decide

Per the brief, the directory structure is not prescribed. Step 5's placement is
the one recommendation offered, and it is offered because two existing files
already own the projection contract — not because a standard harness layout says
so. The H1 lesson was that auditing the real implementation before choosing the
abstraction is what prevents a second kernel; the same reasoning applies to
`AgentContextPack`, which is why it is step 5 and not step 1.
