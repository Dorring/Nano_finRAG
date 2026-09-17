# NF-V3 H2A-2C — Multi-Evidence SlotBinding: Implementation Note

Status: **AUDIT + DESIGN.** Written before any edit, per the brief. Baseline:
`503672f` (H2A-2B closed).

---

## 1 · The live binding representation

Not designed from `BoundFact` / `VerifiedEvidencePacket`. Those declare a
one-fact-per-slot shape but are **not** what the live path binds:
`EvidenceBinding.slot_bindings` is already `Mapping[str, tuple[str, ...]]`
(`rag_v2/contracts/evidence.py:89`), and the runtime carries
`last_bound_slot_bindings: dict[str, tuple[str, ...]]`
(`src/runtime/trusted_v2_binder.py:64`). The shape has always been able to hold
several ids per slot. Only the *contract* forbids it.

| Stage | Type | Canonical value | Evidence id | Survives multiple? |
| --- | --- | --- | --- | --- |
| retrieval candidates | `state.evidence_packets` (mappings) | `value`/`parsed_numeric_value` | `fact_id`/`evidence_id` | **yes** |
| evaluation | `SemanticEvidenceEvaluationCapability` | via `_fact_value_key` | `last_bound_evidence_ids` | **no** — see §2 |
| binder | `BinderRun.binding: EvidenceBinding` | not carried | `slot_bindings[slot]` (tuple) | shape yes, contract no |
| slot representation | `slot_bindings[slot_id] = (id,)` | — | 1-tuple | **no** |
| calculator | operand binding | `value`+`unit`+`scale` | operand `evidence_id` | no (by design) |
| generation | `CandidateExecutionResult` | rendered text | `bound_evidence_ids`, `citation_ids` | **no** |
| validator | `GenerationValidationReportV1` | — | allowed citation ids | — |
| citation/provenance | `citation_ids` | — | one per bound fact | **no** |

## 2 · The cardinality collapse, exactly

Not one line — five, in two layers.

**(a) The contract — the hard one.** `rag_v2/evidence/binding_validator.py:68`:

```python
if any(len(ids) != 1 for ids in binding.slot_bindings.values()):
    reasons.append("bound_fact_cardinality_mismatch")
```

A `BOUND` binding with two ids for one slot is `INVALID`. `EvidenceBinding` itself
is permissive (`contracts/evidence.py:103-109` rejects only duplicate ids *within*
a slot) and the provider schema is an array (`evidence/prompt.py:43-46`) — only
the prompt's *example* is single-element. So multi-support is unrepresentable
because of a **validator decision**, not because the data model cannot hold it.

**(b) The consensus return.** `src/runtime/trusted_v2_binder.py:458`:

```python
return winner_values[0][0], len(winner_values), winner_key
```

`_consensus_fact_for_slot` **already computes the whole corroborating set** —
grouped by canonical value key, deduplicated by physical source — and returns one
id plus its *count*. The multiplicity is computed and thrown away; the count is
only ever reported (`consensus_size_by_slot`).

**(c) The three repair sites**, all writing a 1-tuple:

| Line | Function | Note |
| --- | --- | --- |
| `trusted_v2_binder.py:484` | `_repair_semantic_binding` | the main repair |
| `trusted_v2_binder.py:677` | `_repair_ambiguous_binding` | "collapse equal-valued duplicates", under a comment that says *"``BOUND`` permits exactly one fact per slot"* |
| `trusted_v2_binder.py:696` | `_repair_ambiguous_binding` | the ambiguous-slot branch |

And upstream, `src/runtime/trusted_v2_binder.py:906` names the assumption
outright: *"Providers occasionally return every equivalent duplicate row for a
slot instead of the one-fact-per-slot shape **required by the domain contract**.
Treat that narrow cardinality violation as a repairable provider-shape
problem."* Multiple equivalent rows are currently a **bug to be repaired**.
H2A-2C makes them the thing.

### The one path where multiplicity already survives

`_repair_semantic_binding` returns early for intents other than
`DIRECT_FACT`/`CALCULATION` (`:470-471`). `Intent` has three members — the third
is `MULTI_EVIDENCE`. So for a `MULTI_EVIDENCE` plan the repair never runs: the
validator's `bound_fact_cardinality_mismatch` is swallowed by
`structural_repairable` (`:911-917`), no repair is attempted, and the raw
multi-id binding falls through to `:1046-1052`, publishing
`last_bound_slot_bindings = {"slot": ("E1","E2")}` for one slot. That is the only
live path today where more than one id per slot reaches generation — and it is
exactly the shape the readiness label names and the runtime cannot otherwise
express.

### What `MULTI` actually requires

`src/generation/generator_routing_policy.py:110-115`:

```python
if "MULTI" in norm_hint or len(evidence_items) > 1:
    return GeneratorRouteDecision(route_name=RouteName.MULTI, ...)
```

`evidence_items` is `_bound_items(state)` — the Binder-admitted ids. So `MULTI`
means **more than one admitted evidence id**, which a single slot with two
supports produces. No new route is needed and no route name has to be invented;
`STRUCTURED_SINGLE` requires `len(evidence_items) == 1` (`:127-132`), which is
why `cross_source` currently routes there.

**Consumers that assume exactly one:** `_operand_for_slot`
(`src/runtime/trusted_v2_calculation.py:186` — `if len(ids) != 1: return None`,
which becomes a `BLOCKED` calculation); the deterministic renderer
(`trusted_v2_generation.py:379` renders `items[0]` and ignores the rest); the
three repair sites above; and `validate_binding`'s
`duplicate_fact_across_slots` (`:59-60`), which forbids one fact id in two slots.

**Consumers that only need one canonical value:** the calculator's operand
binding, and `DeterministicFactRenderer.render`.

**Consumers that already carry multiplicity — nothing to build:** 
`build_claim_provenance` emits `bound_evidence_ids` as a tuple per slot and
collects *all* the slot's citations (`trusted_v2_provenance.py:115-131`);
`_structured_citations` iterates `_stable_unique(evidence_ids)`
(`coordinator.py:102-142`); `V2ValidationResult.bound_evidence_ids` carries
`admitted_ids` (`validation.py:169`); `trace_snapshot` publishes all four
`last_*` attributes. The released surfaces can already say "these three
evidence items support this claim" — they are simply never handed more than one.

**Dead, and left alone:** `validation.py:322-323` reads
`state._candidate_citation_ids` / `_candidate_calculation_ids`, neither of which
is set anywhere in `src/`. Recorded, not repaired — 2D territory.

## 3 · What is already right, and must be reused rather than rebuilt

`_consensus_fact_for_slot` already implements the whole of §5's conflict policy
and §7's support identity:

- groups admissible facts by `_fact_value_key` (2B shared quantity semantics —
  so `1000 million` and `1 billion` are one group);
- drops facts whose `physical_source_id` has already been counted, *"so
  duplicate extraction rows cannot manufacture a false majority"*;
- requires a strict majority when there is more than one group, and returns
  `None` on a tie or on a winner of size 1 against any rival.

This is the support-set identity §7 asks for, and it is **not**
`content_fingerprint`-based: two independent sources stating the same figure
share a fingerprint and still count as two, which is exactly the corroboration
that must survive. Nothing in 2C adds a second notion of support.

## 4 · The change

1. **`binding_validator`**: `BOUND` requires `len(ids) >= 1` per slot (empty is
   still a violation), plus a new `slot_supports_disagree` reason: every id in
   one slot must share a canonical quantity identity. Conflict stays
   fail-closed, and it is now enforced *at the contract* rather than only in the
   consensus path.
2. **`_consensus_fact_for_slot`**: return the winning id **set**, keeping the
   existing `(winner, support_count, value_key)` behaviour for its other
   callers.
3. **Repair path**: bind the winning set, not `winner_values[0][0]`. The
   `structural_repairable` branch at `:913` stops firing for multi-support and
   keeps firing for genuine structural faults.
4. **Route**: `MULTI` when any slot carries more than one support. This is the
   existing readiness meaning of the route; no new route is invented.
5. **Calculator**: unchanged in behaviour — one canonical operand per slot,
   taken from the binding's canonical value. Multi-support must not reach
   arithmetic.
6. **No new status enum value.** `BOUND` with `len(supports) > 1` is the whole
   distinction; `CORROBORATED` would be naming, not semantics.

## 5 · What the brief's categories map to

- **A** (candidates disagree) — `_consensus_fact_for_slot` returns `None` →
  `EVIDENCE_CONFLICT` → fail closed. **Unchanged.**
- **B** (corroborating sources) — one canonical value + N support refs → route
  `MULTI`. **New.**
- **C** (several semantic slots for one claim) — already modelled: one slot per
  operand, `slot_bindings` is keyed by slot. **Unchanged**, and deliberately not
  conflated with B.

## 6 · The `multi_evidence` finding — a fixture defect, not a runtime gap

The sealed gold, read exactly (`labels.jsonl` / `questions.jsonl`, unchanged):

| case | answerable | expected_release | expected_route | expected_evidence | required terms |
| --- | --- | --- | --- | --- | --- |
| `cross-source-001` | true | true | MULTI | X1, X2 | `100` |
| `multi-evidence-001` | true | true | MULTI | Q1, Q2 | **`Revenue`** |
| `qualitative-001` | false | false | MULTI | Q1, Q2 | — |

The harness fixture for `multi_evidence` (`tests/benchmark/tv2_readiness_cases.py:63`)
supplies **one** slot `margin/operating_margin/FY2024` with `Q1 = "100"` and
`Q2 = "80"` — and **no Revenue fact at all**.

`qualitative` is a *separate* fixture object, not the same one: it carries
identical slots, identical facts and identical routes, and only the query text
differs, while its label asks for the opposite outcome (abstain). So the two
cases present the same evidence under opposite labels, and at most one of them
is reachable — the mismatch count was guaranteed non-zero before any runtime
behaviour was considered.

Two independent reasons `multi_evidence` cannot satisfy its own label, whatever
SlotBinding does:

1. `required_answer_terms` is checked only on release (`scoring.py:104`), and the
   answer for a slot whose metric is `operating_margin` cannot contain
   `"Revenue"`. No binding change can produce that term from this fixture.
2. The label wants Q1 and Q2 as **supporting evidence for one claim**, but the
   fixture gives them different values for one slot. Under the frozen conflict
   contract (§5), that is category **A**, and fail-closed is the correct
   outcome — which is what the runtime already does, and what the *sealed*
   `multi-evidence-002` asks for with M1/M2.

So `multi_evidence` is not blocked by a missing capability in SlotBinding. It is
blocked by a harness fixture that contradicts the label it was written to drive
— a self-inflicted wiring defect H1 already recorded in the registry note
("This fixture also gave the two candidates different values, which the conflict
gate correctly refuses").

`cross_source` has no such problem: X1 and X2 both state `100`, differ in
physical source, and need only §4's change. **That one closes on the runtime
side alone.**

**This is a decision for review, not one to take silently**: the sealed labels
are untouched either way, but correcting the fixture edits the instrument that
measures the phase. Options are recorded in the report; nothing is changed in
this commit.

## 7 · Not in this phase

Per the brief: `EvidencePacketV1.page`/metadata duplicates, dead page readers,
`state.calculation_result_id`, prefix-only id naming, `BoundFact` /
`VerifiedEvidencePacket` cleanup (2D); source-authority weighting (deferred,
unchanged); the unreachable validator ratio predicate (stays documented and
inactive).

## 8 · The second blocker: the MULTI route has no evidence-bearing answer

Found by probing, after §6. It is independent of §6 and it bounds what 2C can
deliver.

Both sealed labels require `expected_route = MULTI`. Reproduced on the
pre-2C runtime with a corrected `cross_source`-shaped fixture (one Revenue slot,
X1 and X2 both `100`, distinct physical sources):

```
route     STRUCTURED_SINGLE      (label: MULTI)
evidence  ['X1']                 (label: X1 and X2)
citations ['citation-X1']        (label: citation-X1 and citation-X2)
released  RELEASED               (correct)
answer    'Revenue (FY2024): 100 USD million [citation-X1]'
```

That is the genuine cardinality failure §2 predicts, reproduced on an
independently corrected input. **This part is 2C's to fix.**

But the route the label asks for is a different path. `MULTI` is selected by
`len(evidence_items) > 1` (`generator_routing_policy.py:110-115`) with
`target=GeneratorTarget.LOCAL_SPECIALIST`, and in this harness the specialist is
`_FixtureSpecialist`, whose `generate` returns the literal constant
`"fixture specialist answer"` (`tests/harness/h1_integration.py:575-583`).
Probed with a two-slot fixture that routes MULTI:

```
route     MULTI
released  NOT_RELEASED           reason SCV_METRIC_UNSUPPORTED
answer    'fixture specialist answer'
```

So **on the MULTI route the harness answer contains no metric, no value and no
evidence text.** Consequences for the two target labels, whatever SlotBinding
does:

| Label | requires | MULTI route gives |
| --- | --- | --- |
| `cross-source-001` | answer contains `100` | `"fixture specialist answer"` |
| `multi-evidence-001` | answer contains `Revenue` | `"fixture specialist answer"` |

`required_answer_terms` is checked only on release (`scoring.py:104`), and
`SemanticClaimVerifierV1` additionally rejects the constant as
`SCV_METRIC_UNSUPPORTED` — so a MULTI release would *also* flip to
not-released, turning a correct release into a release mismatch.

Routing either case to `MULTI` therefore trades one failure for two. **The two
readiness gaps cannot be closed by binding alone**; the harness's specialist
stand-in has to be able to carry evidence, or the routing target for a
single-canonical-value multi-support slot has to stop being the specialist.

### The sequencing consequence

This is not a separate issue to park: it decides the order of the remaining
commits. Both target cases release correctly *today* because they keep one
support and therefore route `STRUCTURED_SINGLE`, which the deterministic
renderer serves. Teaching the binding to keep both supports makes
`len(evidence_items) == 2`, which routes `MULTI`, which hands the answer to the
constant stub — so the release that is currently correct would flip to
`NOT_RELEASED` on `SCV_METRIC_UNSUPPORTED`, `over_conservative` would climb back
off zero, and the benchmark would get *worse* while the underlying capability
got better.

Verified by probe on the two-slot fixture above, which reaches the same code
path today.

So the binding change is correct and required, but it must not land before the
`MULTI` target is able to produce a grounded answer, or the phase's own
instrument will report a regression that is not one.

That second option is a real design question and belongs to §2's B-vs-C
distinction: `cross_source` is *one* slot with *two supports* — one canonical
value, which the deterministic renderer already renders correctly — while the
existing `MULTI` target assumes several distinct values that need synthesis.
Conflating the two is precisely what §2 warns against, and the routing policy
currently does conflate them. It is a decision for review, not one to take while
implementing binding.

**Nothing is changed on the strength of this section.** It is recorded so the
readiness target can be read correctly: with the §6 fixture corrected and 2C
implemented, `cross_source` will still mismatch on `route`, and that mismatch
will be a harness capability gap, not a runtime defect.
