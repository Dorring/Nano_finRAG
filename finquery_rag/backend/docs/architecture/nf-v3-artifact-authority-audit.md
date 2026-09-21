# NF-V3 H2A-2A — Artifact Authority Audit

Status: **AUDIT ONLY.** No structure changed, no canonicalisation moved, no
`ArtifactStore` written, no readiness work.

Baseline: `bc227bb`. Sealed predecessor: `nf-v3-h2a1-context-trust-r2`.

Every entry below is read from code with citations. Where the code does not
answer the question, it says **NOT DETERMINED** rather than filling the gap with
architecture intent.

---

## The question this audit exists to answer

For each piece of domain state: **who owns the truth, who copies it, who
references it, who projects it — and which of the current verifications are
independent of the thing they verify.**

The second half matters as much as the first. Collapsing duplicate truth is the
goal of H2A-2; a guard derived from the same source as its object can only ever
agree with itself, and that pattern has already appeared three times in this
project. Merging two truths must not merge away the only detached check.

---

## A. Evidence

### A1 · Lifecycle, retrieval result → public citation

| # | Object | Where | Relation |
| --- | --- | --- | --- |
| 1 | fact-store record | `trusted_v2_production.py:332` | **OWNS** |
| 2 | `R4RetrievalResult.candidate_evidence` | `trusted_v2_r4.py:486`, `:702-713` | **COPIES** (shallow) |
| 3 | retrieval return | `trusted_v2_r4.py:698`, `:724` | **COPIES**; a second **OWNED** copy on `last_result` |
| 4 | tool bridge | `trusted_v2_coordinator.py:698-720` | **REFERENCES** |
| 5 | `EvidencePacketV1` | `adaptive_state_machine.py:188,192` | **PROJECTS**; unlisted keys swept into `metadata` |
| 6 | `state.evidence_packets` | `adaptive_contracts.py:460-464` | **COPIES** (re-serialized), keyed **by `evidence_id` only** |
| 7 | binder fact view | `trusted_v2_binder.py:112-127` | **COPIES** shallowly; `metadata` shared |
| 8 | model-facing binder view | `binder_fact_view.py:143-305` | **PROJECTS** (allowlist) |
| 9 | `BinderRun.binding`/`.validation` | `binder_service.py:59,63` | **OWNS** the decision; ids only |
| 10-13 | bound ids → state → generation | `trusted_v2_binder.py:1092`, `coordinator.py:657`, `generation.py:98` | **COPIES** ids; generation re-copies packets |
| 14 | validation items | `trusted_v2_validation.py:151`, `:183-195` | **COPIES**, and **SYNTHESIZES** `source_text` |
| 15-18 | citation → outcome → adapter → response | `coordinator.py:74-143`, `contracts.py:332`, `adapter.py:153`, `response_mapper.py:108` | **PROJECTS** then **COPIES** four times |

Public citations carry only `evidence_id`, `chunk_id`, `citation_id`, `filename`,
`page`, `type` — not metric, period, value, unit, source or document_id.

### A2 · Every place the full packet is stored

- `state.evidence_packets` — read by eight modules
- `R4RetrievalCapability.last_result` — `trusted_v2_r4.py:702-713`
- `StructuredFactStore._by_candidate` — **process-scoped, outlives the request**
  (`trusted_v2_production.py:313`, reachable via `_RESOURCE_CACHE:898`)
- `SemanticEvidenceEvaluationCapability.last_run` — retains full fact copies for
  the capability lifetime (`trusted_v2_binder.py:917` via `BinderRun.request.facts`)
- four call-scoped copies that outlive their call

### A3 · Fields with more than one place of independent truth

**`page` — three copies inside the packet alone.** `EvidencePacketV1.page`
(`adaptive_contracts.py:218`) **and** `metadata["page"]` **and**
`metadata["pdf_page"]`, because neither is in `consumed` (`:232-236`). Plus the
R4 candidate's `pdf_page`, the binder view emitting both, and the citation.

**`value` — three independent rewrites.** `packet.value` (stringified),
`metadata["parsed_numeric_value"]`, and a third normalization at
`trusted_v2_production.py:397-406`.

**`metric`, `period`, `source`** — each present as a typed field plus retained
`normalized_*`/`raw_*` metadata plus a fact-store alias plus a binder promotion
list.

**`binding status` — four representations with no reconciliation:**
`BinderRun.binding.status`, `BinderRun.validation.final_status`,
`evaluation.decision`, and `state.bound_evidence_ids` as an admission proxy.

**`content_fingerprint`** — a pure property; no independent copy exists.

### A4 · `BoundFact` is declared but never constructed

Defined at `rag_v2/contracts/evidence.py:29`, exported, referenced by
`VerifiedEvidencePacket`. **The only construction site in the repo is a test**
(`tests/rag_v2/test_v2_00_contracts.py:44`); `VerifiedEvidencePacket` likewise.
The live representation is `EvidencePacketV1` plus plain `Mapping` dicts.

---

## B. Calculation

### B5 · `CalculationResult` → public response

Owned at construction; `last_result` and `state._calculation_result_obj`
**reference** it; `last_calculation_id` and `state.calculation_result_id`
**copy** the id; `state.calculation_result` **copies the full internal
diagnostics dict** (`to_dict()`, `trusted_v2_calculation.py:348`). The public
projection uses `to_public_dict()` (`coordinator.py:177-178`) and is gated on the
object, not the caller's ids.

### B6 · Identity derivation

`CalculationResult.calculation_id` (`src/domain/calculation.py:232-264`) is the
**owner** — a pure sha256 property returning `None` unless admissible. Everything
else is a copy. `state.calculation_result_id` is **written from two call sites**
(`trusted_v2_calculation.py:349` and `trusted_v2_coordinator.py:1108`).

Two narrow openings where copies could disagree — both unreachable today because
state and capabilities are per-request, and both stated because per-request is a
construction property, not a contract:

1. the mirrors are mutable copies of an id the result already owns;
2. the BLOCKED early return sets `last_calculation_id = None` but does **not**
   clear the three state fields, so a state carrying a prior calculation would
   keep it.

`_structured_calculations` never checks the object's id against the caller's
`ids`; the published id comes from the object, the outcome-level id from the list.

### B7 · **P0 — the specialist boundary is ungoverned for calculation payloads**

`trusted_v2_generation.py:292`:

```python
calculation_payload = calculation.to_dict() if calculation else None
```

passed at `:310`. Verified independently: `to_dict()` carries each operand's
**full `source_text`**, where `to_public_dict()` substitutes a 240-character
`evidence_excerpt`.

```
internal to_dict operands[0] : [..., 'source_text', ...]   raw text present
public   to_public_dict      : [..., 'evidence_excerpt', ...]  bounded
```

**The asymmetry is in one call.** The evidence items immediately above it are
projected (`:300-303`); the calculation payload receives no profile. There is no
`EvidenceDisclosureProfile` member for calculations, and `project()` operates on
an evidence-field mapping, so the payload is **outside the authority's reach by
construction**. `last_disclosed_fields` (`:307-309`) is computed only from the
projected evidence, so the trace is silent about the calculation payload.

**Mitigation that does not change the finding:** the prompt renderer reads only
`value`, `unit`, `operation` (`local_specialist_generator.py:198-200`), so the
raw text crosses into the provider call but is not interpolated. This is exactly
the "safe because two shapes disagree" pattern `disclosure.py` was written to
replace.

**Reachability.** `_route` unconditionally overrides to `DETERMINISTIC_CALCULATOR`
when `state.intent == "CALCULATION"` (`:242-248`), `state.intent` is set once
from `plan.intent.value`, and `_calculation_result_obj` has one writer inside
`calculate()`, which refuses a non-CALCULATION plan. So on the coordinator path
the branch is **live code the shipping route does not take** — resting on two
independent facts staying true. It is reached by any caller constructing a state
directly, and the disclosure test suite does not exercise it: the recording
specialist **accepts and discards** `calculation_result`
(`test_model_facing_disclosure.py:174-181`).

**Disposition: P0 trust-boundary remediation for H2A-2**, not routine debt. The
invariant A claims is "model-facing domain data passes a controlled projection";
this is a model-facing domain object that does not.

---

## C. Provenance / identity

### C8 · Three identities

All four ids come from one helper (`trusted_v2_canonical_fact_store.py:56-58`)
hashing the **same** `(document_id, table_id, row_id, cell_id)`; only a 9-character
prefix differs.

| Field | Kind | Note |
| --- | --- | --- |
| `evidence_id` | provenance identity, **consumed as instance identity** | keys the state dict and the public citation |
| `fact_id`, `candidate_id`, `candidate_key`, `citation_id`, `physical_source_id` | **five names for one digest** | nothing enforces prefix/role consistency |
| `content_fingerprint` | **semantic content identity** | explicitly excludes ids, document, source, page, metadata |
| `document_id`, `page` | provenance identity | `page` is also published as a display field |

`evidence_id` serving two roles is the significant one: it *is* the sha of the
source location, so a re-extraction that changes `row_id` renames the fact even
when the content is identical.

### C9 · `metadata.pdf_page` is still populated — confirmed

Decided by **omission** from `consumed` (`adaptive_contracts.py:232-236`); neither
`page` nor `pdf_page` is listed, so both are retained at `:257`. Verified: input
`{'page': 12, 'pdf_page': 12}` yields `page == 12` **and**
`metadata == {'page': 12, 'pdf_page': 12}`.

Consequence: two readers use `dict.get(k, default)`, which returns the default
only when the key is **missing** — so the fallback is dead at
`trusted_v2_coordinator.py:129` and `trusted_v2_calculation.py:110`. The same trap
is documented for `from_mapping` (`:244-249`) and was not applied at those two
readers. The binder view reads both correctly.

---

## D. Independent Verification Surfaces

### D10 · What is genuinely independent

| Surface | Independent of |
| --- | --- |
| `.sealed/labels.jsonl` (out-of-repo, operator-held) | the entire codebase |
| readiness gold labels (`tests/fixtures/tv2_07_production_readiness/labels.jsonl`) | the runtime |
| **R1 canonical loader rejecting consumed datasets** (`tv2_07_r1_readiness.py:38-42,764-773`) | any previously used input — **the strongest anti-circularity guard found** |
| `SourceContamination` blacklist (`test_tv2_canonical_eval_integrity.py:202-210`) | the fact store |
| RC freeze manifest written by a prior process (`rc_freeze.py:1-30`) | the run it gates |
| `ShadowComparator` V1-vs-V2 differential | each other — two implementations |
| byte-frozen artifact hashes; jieba two-implementation parity | the code under test |
| pre-refactor parity baseline (`artifacts/refactor/phase2-behavior-parity.json`) | the current tree |
| phase 4 external API contract, including its excluded-field list | the tests that assert it |

### D11 · Circular or self-referential checks

| # | Check | Why |
| --- | --- | --- |
| 1 | `build_tv2_canonical_eval_set.py:494-497` | reads gold out of the same fact store the runtime retrieves from |
| 2 | `test_tv2_canonical_eval_integrity.py:134-194` | recomputes the builder's own formula over the builder's own operands |
| 3 | `equivalence.py:223-268` | **the historical circular payload guard — documented and replaced.** Residual: the classification tuples are hand-written and excuse a field by name |
| 4 | `h1_integration.py` `expect_*` fields | live inside the fixture the test runs |
| 5 | `VIEW_SHA` literals in 8+ scripts | a copy of a constant asserted against itself |
| 6 | `phase4-api-contract.json` | generated from the tests that assert it |
| 7 | readiness `SEMANTIC_MISMATCHES`/`LABEL_ALIASES` | derived from the run, so the guard is self-referential |
| 8 | disclosure profile test | expectation transcribed from the prompt under test |
| 9 | determinism/round-trip tests (7 files) | re-invoke the primitives they check |
| 10 | assertions against a capability's own recorded state | the capability records what it did; the test asserts it did it |
| 11 | `test_slot_conflict_gate.py:181-246` | calls the production `_fact_value_key` to pin `_fact_value_key` |

**Read this list as the risk map for H2A-2.** Items 1-2 and 7 live in the
readiness path this phase is about to change. Item 11 is in the conflict gate E
just sealed. Removing them is not required now; knowing which checks will lose
their independent footing when truth is consolidated is.

---

## The authority table

| State | Current authority | Duplicate / mirror | Consumer | Proposed disposition |
| --- | --- | --- | --- | --- |
| Evidence content | `state.evidence_packets` (copies of `EvidencePacketV1`) | R4 `last_result`; `StructuredFactStore._by_candidate`; binder `last_run` | binder, generator, calculator, validator, provenance | **one authoritative store; others reference** |
| Page | `EvidencePacketV1.page` | `metadata["page"]`, `metadata["pdf_page"]`, candidate `pdf_page`, fact-store `page` alias | citation, binder view | `metadata` copy becomes **ingest-only**; two dead fallbacks replaced |
| Source / document | `packet.document_id`, `packet.source` | `metadata["source_id"]`, `physical_source_id`, 5-way citation fallback | citation, binder | decide one owner; fallbacks collapse |
| Binding status | `BinderRun.validation.final_status` | `BinderRun.binding.status`, `evaluation.decision`, `bound_evidence_ids` proxy | evaluator, generation | **not a duplicate to remove — four different questions.** Rename to say which |
| Semantic identity | `content_fingerprint` (property) | none | progress, `evidence_hashes` | **keep** |
| Calculation object | `CalculationResult` (`state._calculation_result_obj`) | `to_dict()` copy in `state.calculation_result` | generator, validator, public projection | keep object; the diagnostics copy is what leaks |
| Calculation id | `CalculationResult.calculation_id` | `last_calculation_id`, `state.calculation_result_id` (written twice) | generation, validator | **derive/reference**; remove the two-writer mirror |
| Slot fact | single fact per slot | — | calculator | evolve to `SlotBinding` in 2C |
| **Specialist calculation input** | **none — raw `to_dict()`** | — | **model boundary** | **P0: projection or bounded excerpt** |
| `BoundFact` / `VerifiedEvidencePacket` | declared, never constructed | — | nothing | delete or implement, not both |

No question mark in this table is answered by intent. Three remain literally
undecided and are named as decisions, not findings: which source/document field
owns the truth; whether the four binding-status representations are four
questions or one (the code says four, no comment confirms it); and the P0
disposition.

---

## NOT DETERMINED

- **Whether the raw calculation payload has ever crossed in a real run.** The
  static trace says the branch is unreachable via `_route`; no runtime artifact
  was searched for a historical occurrence. Needs `trace_log.db` queried for
  `specialist_calls > 0` alongside `calculator_invoked`.
- **Whether the prefix-only distinction among the five id names is intentional.**
  No comment states it.
- **Which disclosure profile, if any, calculation payloads should have.** None
  exists; no design document was found stating a decision.
- **Whether the readiness gold labels are truly independent of `FIXTURE_SPECS`.**
  Both appear authored by the same hand; would need per-file commit history.

---

## H2A-1 architecture invariant, formalised

Three defects in this project — the H1.1 `runtime_metadata` omission, the C.1
numeric-gate tautology, and the equivalence payload guard — share one shape:

```
producer source
   ├── builds the object
   └── builds the expectation
             ↓
     guaranteed agreement
```

> **A guard must be derived independently of the object or representation it
> guards.**

The correct structure is:

```
source A ──→ object under test

source B / external contract / independent oracle ──→ guard expectation
```

This is why D11 is part of this audit and not an appendix: H2A-2 exists to remove
duplicate truth, and **removing a duplicate is not the same as removing an
independent check**. Before any copy is deleted, the question to answer is
whether that copy is also the last detached way to detect the owner being wrong.
