# NF-V3 H2A-2C — Multi-Evidence Slot Binding

Status: **CLOSED.** Predecessor: `nf-v3-h2a-2b-1-shared-canonicalization.md`.
Audit/plan: `nf-v3-h2a-2c-slot-binding-note.md`.

H2A-2C's purpose was to remove the effective "one usable fact per slot"
cardinality model, so a slot can be supported by several independent evidence
items without exposing competing values downstream, and so the two remaining
readiness gaps (`cross_source`, `multi_evidence`) close against unchanged
sealed gold.

---

## 1 · Two independent defects, not one

They looked like one limit and are two, and they were fixed in that order for a
reason that only became visible once the first was fixed.

### A — route and execution target were the same decision

`RouteName` and `GeneratorTarget` were already separate enums. The *selection
rule* was not:

```python
if "MULTI" in norm_hint or len(evidence_items) > 1:
    return GeneratorRouteDecision(route_name=RouteName.MULTI,
                                  target=GeneratorTarget.LOCAL_SPECIALIST, ...)
```

`len(evidence_items) > 1` was read as MULTI, and MULTI as "the Local Specialist
writes it". That answered three different questions with one number: evidence
cardinality, semantic arity, and execution capability. `cross_source` is one
canonical Revenue value with two independent sources — nothing to synthesise —
and it was handed to a free-form generator.

The same coupling existed a second time, in the calculation branch
(`has_explanation_terms or len(evidence_items) > 1`), where it was inert only
because `trusted_v2_generation.py:_route` overrides it for a CALCULATION plan. A
policy that is correct only when something above it disagrees is not correct.

**Now:** the route names the evidence/task shape; the target answers a capability
question — *can the deterministic structured path state this admitted state
honestly?* A state where every item belongs to one semantic slot and states one
canonical quantity goes to the deterministic renderer however many sources
support it. Several distinct facts still go to the specialist, because one
rendering of them would have to drop or choose. Qualitative prose keeps the
specialist because the deterministic path has no capability for it at all.

### B — the binding contract said one fact per slot

`EvidenceBinding.slot_bindings` is `Mapping[str, tuple[str, ...]]` — it always
could hold several ids. The invariant forbade it:

```python
if any(len(ids) != 1 for ids in binding.slot_bindings.values()):
    reasons.append("bound_fact_cardinality_mismatch")
```

**No new `SlotBinding` class was introduced, and none was needed.** The problem
was never a missing type; it was a structure whose invariant had been narrowed
to 1:1 by hand. The fix is the invariant, not a wrapper.

---

## 2 · The final semantics

**Semantic slot identity.** Composed from the repository's existing canonical
identities — `canonical_metric_id`, `canonical_period_id`, `canonical_entity_id`,
`canonical_scope_id` — plus the 2B canonical quantity. Deliberately the same
identities `_fact_matches_slot` uses to decide which slot a fact belongs to, so
the routing rule and the binder mean the same thing by construction. Entity and
scope are not optional: a quantity that matches across two issuers is two facts,
not one.

**Binding invariant.** One semantic slot; one canonical financial quantity; one
or more independently admitted support ids. This is *not* "multiple competing
facts" — competing facts are the conflict case and still fail closed.

**Support identity.** Independent corroboration is decided by physical-source
identity, reusing the policy E already had. A fact whose `physical_source_id`
has already been counted does not add a witness. It is deliberately **not**
`content_fingerprint`: two independent sources corroborating one figure share a
fingerprint on purpose, and deduplicating by it would erase the corroboration
being kept.

**Consensus.** Unchanged and reused, not rewritten: group admissible facts by
canonical quantity, deduplicate by physical source, require a strict majority
when there is more than one group, and return `None` on a tie or an
uncorroborated winner. What changed is only its *return*: it used to compute the
whole corroborating set and hand back one id plus a count it discarded.

**Primary and repair paths move together.** Four repair sites wrote 1-tuples
(`_repair_semantic_binding`, both branches of `_repair_ambiguous_binding`, the
entity branch). Fixing only the validator would have made normal binding
multi-support while repair silently collapsed back to one — path-dependent
semantics.

**Calculator.** A multi-support slot yields **one** canonical operand and **one**
calculator invocation. Support multiplicity is provenance, not operand
multiplicity; a corroborated figure must not be summed or averaged with itself.
The representative id is chosen by admission order and cannot change the
arithmetic because the validator proved the group agrees.

**Provenance.** `ClaimProvenance.bound_evidence_ids` and the public `citations`
array preserve every admitted support, each with its own source location.
A *representative* inline citation in the rendered answer text is permitted; the
structured surfaces are authoritative.

---

## 3 · Readiness history — the harness correction is its own event

Recorded separately, because collapsing the two would claim a runtime fix that
did not happen.

1. The sealed gold (`questions.jsonl` / `labels.jsonl`) **predates** the H2A
   fixture specs.
2. The original `multi_evidence` fixture was **impossible by construction**: one
   `operating_margin` slot with `Q1 = 100`, `Q2 = 80` and no Revenue fact, while
   its label wants release, route MULTI, evidence `[Q1, Q2]` and an answer
   containing "Revenue". No runtime could satisfy both. `qualitative` carried
   structurally identical evidence under the opposite label.
3. The fixture was corrected (one Revenue slot, `1 million` / `1000 thousand`
   from distinct physical sources) **without touching the gold**.
4. On the pre-2C runtime the corrected fixture reproduced the genuine gap:
   `released`, route `STRUCTURED_SINGLE`, evidence `['X2']`, citations
   `['citation-X2']` — one of two stated supports kept.
5. The 2C changes closed it.

`over_conservative` moving 1 → 0 belongs to step 3, not to the runtime: that
counter was recording the harness's mistake.

---

## 4 · Readiness

| | before 2C | after 2C |
| --- | --- | --- |
| `task_success` | 17 | **19** |
| `false_release` | 0 | **0** |
| `over_conservative` | 1 | **0** |
| `semantic_mismatch` | 2 | **0** |
| `label_alias` | 3 | **3** |
| `unexplained_mismatch` | 0 | **0** |
| `SEMANTIC_MISMATCHES` | 2 entries | **`{}`** |

The registry shrank because the defects were fixed. The labels were never
edited, and the length assertion is kept (as `== 0`) so a new entry cannot be
added without a test noticing.

---

## 5 · Independent-oracle methodology

Held throughout, and it is the reason the readiness result means something:

- Expected support sets come from sealed gold (`expected_evidence_ids`) and from
  authored fixtures — never read back out of the binding, consensus or
  generation helpers.
- The eight reconciled trust tests each had their expected set derived by hand
  from the fixture's metric/period/entity/scope/quantity/`physical_source_id`.
  The newest production output was printed for diagnosis only.
- The fixture-consistency guards were verified to *fail* on the old fixture
  before being trusted.
- The V1 vocabulary guard was verified to fail on a one-digit change.

---

## 6 · Carried into H2A-2D

Not defects; authority questions this phase deliberately did not answer.

1. **Representative inline citation.** The answer text names one source
   (`[citation-X1]`) while `citations` and `claim_provenance` carry both. Fine
   while the claim value renders once. If a product wants `[1][2]` inline, that
   is a rendering/public-contract decision.
2. **`outcome.calculations[*].supporting_evidence_ids` does not exist**, while
   the calculation's claim provenance carries the full set. Which structure owns
   calculation provenance, and whether the public dict should copy it, is a 2D
   authority question. No duplicate field was added for symmetry.
3. `EvidencePacketV1.page` vs `metadata["page"]` / `metadata["pdf_page"]`.
4. `CalculationResult.calculation_id` vs `state.calculation_result_id`.
5. `BoundFact` / `VerifiedEvidencePacket` — declared, never constructed; delete
   or adopt.
6. Prefix-only id domains — NOT DETERMINED pending the 2D identity audit.

## 7 · Non-defects found and left alone

The validator's ratio-incompatibility predicate is unreachable and was before
H2A-2B. `validation.py` reads `state._candidate_citation_ids` /
`_candidate_calculation_ids`, neither of which is set anywhere in `src/`. Both
are recorded in place and belong to 2D.
