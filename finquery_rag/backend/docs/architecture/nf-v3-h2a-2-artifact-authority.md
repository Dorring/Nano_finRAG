# NF-V3 H2A-2 — Artifact Authority

Status: **CLOSED.** Predecessor: `nf-v3-h2a-2b-1-shared-canonicalization.md`.

H2A-2's question was not "how do we remove duplicate fields" but **which
structure owns each domain truth**, and how to tell the different kinds of
duplication apart. Both answers are below.

---

## 1 · The four shapes of "duplicate" — the phase's main result

Every apparent duplication H2A-2 found turned out to be one of four things, and
conflating them is what made earlier cleanup attempts unsafe. The taxonomy is
first-class, not a summary:

| Shape | Example | Disposition |
| --- | --- | --- |
| **Duplicate truth** | `metadata["page"]` vs `EvidencePacketV1.page` | migrate to a single authority |
| **Mutable mirror** | stored `calculation_result_id`, `last_calculation_id` | replace with a derived reference |
| **Legitimate projection** | trace / public `calculation_result_id` | **retain** |
| **Distinct identity domains** | `physical:` / `evidence:` / `candidate:` / `citation:` | **retain the domains**; forbid implicit substitution |

The last one is the counter-intuitive case: four ids over *one* digest payload
are not four copies of one identity. Sharing a suffix is a property of the
payload, not an equality rule.

---

## 2 · Final authority matrix

| Domain truth | Final authority | Derived refs / projections | Compatibility inputs / aliases | Removed / demoted |
| --- | --- | --- | --- | --- |
| Financial quantity semantics | `rag_v2/contracts/financial_semantics.py` | — | V1 scale tables (guarded, not delegated) | 3 duplicate vocabularies; `_identity_text` |
| Runtime page provenance | `EvidencePacketV1.page` | `_structured_citations` entries | `metadata["page"]`/`["pdf_page"]`, **ingest only** | metadata read in the binder fact view |
| Evidence instance identity | evidence identity domain | — | `fact_id`, `candidate_id`, `candidate_key`, `chunk_id` (same-domain field aliases) | — |
| Physical-source identity | `physical_source_id` / physical domain | — | — | — |
| Content semantic identity | `content_fingerprint` | `evidence_hashes` | — | comma-stripping text helper |
| Candidate identity | candidate domain | — | — | — |
| Citation identity | citation domain | public `citations` | — | `citation:v2:` |
| Slot binding | `EvidenceBinding.slot_bindings` | `_operand_for_slot` (one representative) | — | `len(ids) == 1` invariant; 4 repair 1-tuple writes |
| Calculation identity | `CalculationResult.calculation_id` | state property, capability property | — | two stored mirrors |
| Calculation identity (surfaces) | — | trace id, public outcome id | — | — |
| Operand provenance | `operands[].evidence_chunk_id` | — | — | — |
| Claim-support provenance | `ClaimProvenance.bound_evidence_ids` | `outcome.evidence_ids` | — | — |
| Inline citation | presentation reference | — | — | — |

**Slot binding invariant:** one semantic slot + one canonical quantity + N
independently admitted supports.

**Operand vs claim support are different truths**, not a big one and a small
copy: the operand records which evidence supplied *a value*; the claim records
what supports *the claim*. `outcome.calculations[*].supporting_evidence_ids` was
deliberately **not** added — it would have been a third durable truth.

---

## 3 · Identity domains

| Domain | Meaning | Live representation | Authority boundary |
| --- | --- | --- | --- |
| physical | physical/source occurrence | `physical:<digest>` | physical-source/provenance layer |
| evidence | admitted/runtime evidence | `evidence:<digest>` + same-domain aliases | evidence/binding layer |
| candidate | candidate before admission | `candidate:<digest>` | candidate/generation layer |
| citation | citation/provenance reference | `citation:<digest>` | provenance/public citation layer |

**A shared digest suffix does not imply cross-domain substitutability.** A
`citation:<same digest>` cannot authorize itself as `evidence:<same digest>`;
pinned by negative tests at both the contract and the resolver.

`atomic:<digest>` belongs to the **retrieval semantic graph** — adjacent to, not
part of, this four-domain model, and it never reaches the two runtime evidence
resolvers.

### Resolvers

Both are documented as **SAME-DOMAIN FIELD-ALIAS RESOLVERS**, not identity-domain
matchers, and each has exactly one production caller:

| Resolver | Domain | Caller |
| --- | --- | --- |
| `coordinator._structured_evidence_identity` | evidence | `coordinator.py:93` (`_structured_citations`) |
| `provenance._identity` | evidence | `provenance.py:64` (`build_claim_provenance`) |

**`candidate_id` / `candidate_key` appearing in the evidence ladder does not mean
the candidate domain may serve as evidence identity.** On these admitted-evidence
packets they are historical field *names*; their values are evidence-domain. The
domain is decided by the producer/field contract, not by what a field is called.

**`fact_id` = historical evidence-identity field alias.** The live fact store
emits `"fact_id": evidence_id`; no live TV2 producer places a physical-,
candidate- or citation-domain value there. Not a physical-fact domain, despite
the name.

**Alias disagreement policy** (same-domain precedence, *not* cross-domain
conversion):

```
one alias present                     -> resolves normally
several present, agreeing             -> resolves normally
several present, disagreeing          -> evidence_id wins (canonical name)
```

---

## 4 · Independent-oracle matrix

| Authority / contract | Independent oracle |
| --- | --- |
| Financial quantity semantics | authored semantic-value matrix |
| Evidence conflict admission | independent conflict oracle |
| Content fingerprint / progress | authored equivalence/change cases |
| Page lineage | authored page + poisoned metadata sentinel `42` |
| Multi-support binding | independent readiness gold + authored fixture semantics |
| Calculation identity | literal `C1-b55ba5d38df1276c` + independent contract re-derivation + sensitivity |
| Candidate citation authorization | candidate-authored refs vs independently admitted Binder refs |
| Candidate calculation authorization | candidate-authored literal ids vs `CalculationResult` authority |
| Identity domains | authored fact payload + explicit domain expectations + cross-domain negatives |

> **No migrated authority is verified solely by equality with one of its former
> mirrors, or by production code generating both actual and expected values.**

This is the phase's methodology, and it is what made the four mirror removals
safe: the frozen `C1` vector never changed across any of them.

---

## 5 · Removed

| Contract | Reason |
| --- | --- |
| `citation:v2:` | no producer, parser, loader or export; a comment and a fixture string are not a compatibility contract |
| `BoundFact` | dead exported type; test-only construction |
| `VerifiedEvidencePacket` | dead exported type; test-only construction |

Not listed as architecture components. One test that had been written *through*
`VerifiedEvidencePacket` also asserted a live `AnswerEnvelope` invariant; the
dead half was removed and the live assertion kept.

---

## 6 · Intentionally deferred

**Ratio incompatibility predicate** — unreachable by boolean structure
(`incompatible` is assigned `answer_ratio_units`, making the condition
`A and K and not (A and K)`). It reads live authoritative sets, so this is dead
*logic*, not dead state. Activating it would introduce a validator `HARD_FAIL`
that has never fired, which is a decision about validator policy rather than
about Artifact Authority. **Post-H2A validator-hardening debt.**

**Representative inline citation** — the rendered answer may name one
representative source while `ClaimProvenance` and the structured citations carry
every admitted support. This is presentation/product policy, not Artifact
Authority debt. Forcing `[citation-X1][citation-X2]` into the text would be a
rendering-scope expansion.

Neither blocks the H2A-2 seal.
