# NF-V3 H2A-1 — Context Trust Contracts Seal

Status: **SEALED**

Frozen at `027082262549e5338509dfaa06353eb49c102818`
(`feat/context-trust-runtime-trace`, clean tree, 0 ahead / 0 behind `origin`).

H2A-1 is not a feature phase. It is five trust contracts that decide **what the
runtime is allowed to believe**, at the boundaries where believing the wrong
thing produces a confident wrong answer:

    E  Evidence Admission      is this slot settled, or do its sources disagree
    A  Model Disclosure        what may a model see
    B  Calculation Admission   is this calculation usable
    C  Provenance / Numeric    is this number a claim, or a location
    D  Content Identity        is this the same evidence, or a second copy

Each was a *latent* boundary defect that a repair exposed, not a planned
feature. The pattern that produced all five is the same: **existence was being
mistaken for trustworthiness.**

---

## E — Evidence Admission

**Invariant.** A slot is bound only when its admissible candidates agree. One
candidate being bindable is not the same claim as the slot being settled.

**Defect.** `_consensus_fact_for_slot` already drew the right distinctions —
exact duplicate counted once, corroborating sources agree, strict majority wins,
`None` on a tie — and the caller read its `None` as "no repair found, proceed"
instead of "this slot is unresolved".

**Verified consequence.**

```
slot revenue
  ├── CF1 = 100
  └── CF2 = 999
-> decision=SUFFICIENT, reason_codes=(), bound=('CF1',)
```

**Fix.** `_unresolved_conflict_slots` reports a slot only when more than one
candidate is admissible **and** no consensus exists, at the binding layer rather
than in the validator or a coordinator branch — so a call path added later
cannot treat conflicted evidence as trusted bound evidence.

**Result: false releases 3 → 0.**

---

## A — Model Disclosure

**Invariant.** No model-facing component consumes raw retrieval or source
objects without an explicit projection contract. Deny by default.

**Defect.** The same evidence had two disclosure policies. The Binder's was an
allowlist; the Specialist's was nothing — it received whole evidence packets
whose `metadata` carries the extracted source text. It was not *rendering* that
text, because its prompt reads `ev["source_text"]` at the top level and the text
is nested one level below. Two independent things had to disagree for it to be
safe, and neither was a decision.

**Fix.** `rag_v2/evidence/disclosure.py` — one authority, role-specific
allowlists, four profiles, one implementation:

| Profile | Boundary |
| --- | --- |
| `BINDER` | V2 slot selection (its established surface, unchanged) |
| `SPECIALIST` | the eleven fields its prompt actually reads |
| `V1_ANSWER` | the V1 rollback runtime's assembled context |
| `INGEST_TABLE` | document ingestion, the one profile that *permits* raw text |

`SPECIALIST` deliberately excludes `source_text` and `page`: adding either would
broaden model exposure while claiming to unify it.

**One boundary is classified rather than governed**: `query_processor.py:252`
passes truncated conversation turns. It is a *dialogue* boundary with no
retrieval object and no evidence packet, so the evidence authority has nothing
to project there — and it is not cleanly source-free either, since assistant
turns can quote documents. Named so it is not counted as covered.

---

## B — Calculation Admission

**Invariant.** `calculation exists != calculation is admissible`. A non-None
calculation id must never by itself make a result usable.

**Defect.** The id was computed *beside* the result from a call-site status
check, so any other producer could pair a `BLOCKED` result with a non-None id.
`_structured_calculations` gated on the ids it was handed:

```
before   BLOCKED result + caller id "C1-fake"
         -> [{'status': 'blocked', 'calculation_id': 'C1-fake'}]
after    -> []
```

**Fix.** `CalculationResult.is_admissible` (status is `EXECUTED` — the existing
enum, no new vocabulary) and `CalculationResult.calculation_id`, which returns an
id only when the result is admissible. The digest is byte-identical to the one
computed before. The projection gates on the result's own identity.

The coordinator's upstream guard stays as defence in depth.

---

## C — Provenance and Numeric Role

**Invariant.** Source location survives from retrieval to public citation, and a
number that *locates* evidence is not a number the answer *claims*.

**Two coupled defects, the second hidden by the first.**

Defect 1 — `EvidencePacketV1` had no `page` field, so the extractor's value was
demoted into `metadata` and the citation projection, which reads the top level,
found nothing:

```
pdf_page = 7  ->  public citation page = None
```

for every citation the system had produced.

Defect 2 — repairing that re-enabled `CalculationRenderer`'s `doc-X, p.N`
suffix, which had never once rendered because there was never a page to render.
`GV3_NUMERIC_FIDELITY` then read the `1` in `p.1` as an unsupported financial
claim and hard-failed every answer citing a page.

**Fix.** `page` is a first-class packet field, typed, unrenumbered, with `0`
preserved and absence preserved as absence. Numeric fidelity reads the
calculation's structured values rather than the rendered string: for a
calculation answer the routing policy forces the deterministic calculator, so
the answer *is* the rendering (verified byte-identical) and its claim surface is
exactly those values. No locator string is reconstructed and nothing is removed
from text.

**Post-seal correction: the first C.1 made this check vacuous.** The seal was
originally issued at `551ceb7`, whose validator replaced the answer-text scan
with a structured claim set built from the same expressions `_supported_numbers`
uses. Claims were therefore a subset of supported *by construction*, and the
answer text was never read whenever a packet carried a calculation:

```
answer "Growth rate: 2.09% (fabricated 99999)"   [packet with a calculation]
before C.1   HARD_FAIL GV3 "unsupported material number(s): ['99999']"
at 551ceb7   PASS, findings=[]
```

That is not latent. `TrustedRAGRuntimeV2` is reachable by configuration with a
model-backed `CALCULATION` route, and there a model-written answer contradicting
the canonical result released where it previously failed. The justification
written for it -- that the routing policy forces a `CALCULATION` plan onto the
deterministic calculator -- holds for the TV2 coordinator and is false in
general.

The gate is now live again, by scanning the answer in full and *supporting* the
locator values the renderer legitimately emits, read from the operand's own
`page` field. Neither previous problem returns: no locator string is
reconstructed, so the validator still does not know the renderer's format, and
nothing is removed from text, so a data-controlled document name still cannot
suppress a claim. Tests cover both directions -- a fabricated number hard-fails,
a rendered `p.7` passes.

**Honest scope of C.1 itself.** It is hardening, not a demonstrated exploit of
the *string-removal* version it replaced: that version could not be made to
produce a wrong outcome, because on a calculation route the claims are operand
values that are structurally supported either way. What C.1 changed is the
*coupling* -- the validator no longer knows the renderer's format, so a
formatter change cannot drift them apart and no document name can influence
which numbers are checked. The blind spot above was introduced by C.1, not by
the version before it, and this seal originally overstated C.1 by not recording
it.

The tag `nf-v3-h2a1-context-trust` was moved to this commit. Its previous
position (`551ceb7`) contained the blind spot, and a seal tag that identifies
code failing the seal's own claims is worse than a moved tag.

**A moved tag is not an immutable tag.** Having moved once, `nf-v3-h2a1-context-trust`
can no longer be described as immutable in the strict sense. A second tag was
created that has never moved and will not:

```
nf-v3-h2a1-context-trust-r2 -> fda328a   (immutable, final)
```

`nf-v3-h2a1-context-trust` is now a historical alias with a prior position;
`-r2` is the name to cite.

---

## D — Evidence Content Identity

**Invariant.** Instance identity, semantic content identity and provenance
identity are three different questions.

**Defect.** `content_hash` covered `to_dict()`, which includes `evidence_id` — a
serialized-record hash wearing a content-hash name. Verified before fixing:

```
round 1                        progress=True
round 2, same content new id   progress=True
round 3, same content new id   progress=True
```

The loop could not tell it was re-retrieving evidence it already had.

**Fix.** A consumer audit established the value had **not** escaped into any
persisted, sealed, cached or published contract — one functional reader, inside
a single run — so it is corrected in place. `content_hash` is **removed**;
`content_fingerprint` is the sole semantic-content identity. A second identity
beside a wrong one just leaves the wrong one to be reached for again.

The fingerprint covers metric, period, entity, scope, value, unit, currency and
scale. It deliberately excludes `evidence_id`, `citation_id`, `document_id`,
`source`, `page` and `metadata`: two documents reporting the same figure are the
same content and *different* provenance, and E's consensus works by counting
them as distinct sources agreeing.

**Exact limit, stated so it is not overclaimed.** `content_fingerprint`
normalizes **presentation** variance — instance id, case, whitespace, commas. It
does **not** yet unify representation-equivalent financial scales:

```
1000 million   !=  1 billion        (currently different fingerprints)
```

Closing that needs the canonical value semantics and the packet to share a
layer. They do not: C's canonicalisation lives in `src/runtime/`, the packet in
`rag_v2/`, and the layering forbids importing upward. That is an H2A-2 Artifact
Contract concern, and pulling it forward would turn a trust seal into a
domain-layer refactor. **D does not implement full financial semantic
canonicalisation, and this seal does not claim it does.**

---

## Frozen metrics

Raw counts from the seal run. No historical figure is reused.

```
collected          4150
full regression    4007 passed + 143 skipped + 0 failed

E conflict                   14 passed
A disclosure                 16 passed
B calculation admissibility  21 passed
C provenance / numeric       15 passed
D content identity            7 passed

V1 regression (mode=v1)     694 passed + 7 skipped

readiness (22 cases)
  task_success       17
  false_release       0
  semantic_mismatch   2      cross_source, multi_evidence
  label_alias         3      conflict, no_answer, qualitative
  unexplained         0
  over_conservative   1      multi_evidence
```

**`false_release = 0`** is the metric that matters. `label_alias` is a separate
category on purpose: those three cases abstain exactly as their label requires,
and the two vocabularies name the outcome differently — counting them as
failures would hide the real gaps.

---

## Known non-blocking gaps

Capability and cardinality limits, not correctness failures.

| # | Gap | Nature |
| --- | --- | --- |
| 1 | `cross_source` | one fact per slot — two corroborating evidence ids cannot both bind, so `MULTI` is never selected |
| 2 | `multi_evidence` | same limit; the label wants two pieces of support for one claim |
| 3 | scale-equivalent fingerprint | `1000 million` vs `1 billion` differ; needs a shared canonicalisation layer |
| 4 | one-fact-per-slot cardinality | the binding validator already names it: `bound_fact_cardinality_mismatch` |

## Compatibility duplicates — H2A-2 audit input

Neither is an H2A-1 failure. Both are places where a value lives twice.

| Duplicate | Authoritative | Mirror |
| --- | --- | --- |
| page | `EvidencePacketV1.page` | `metadata.pdf_page` |
| calculation id | `CalculationResult.calculation_id` | `state.calculation_result_id` |

Plus the three durable representations of the same evidence the H2A audit found
(`state.evidence_packets`, `R4RetrievalCapability.last_result`, and the binder's
last run).

---

## Correction history

| Revision | Commit | What changed |
| --- | --- | --- |
| original | `551ceb7` | sealed; C.1 recorded as complete |
| corrected | `0270822` | GV3 blind spot fixed; tag moved here |

The correction is recorded rather than silently folded in, because the defect
was introduced by a change the original seal called complete, and a seal that
edits its own history without saying so is not evidence.

## What this seal does not establish

- **No independent review.** Every "verified" here means verified by execution
  during implementation. The connector that would allow a second reader never
  came up, so no claim in this document has had one.
- **`query_processor.py:252` is classified, not governed** (see A).
- **Raw Source Exposure Ratio is not measurable.** The outcome is reference-only
  by design; the metric needs the loop instrumented.
- **Token figures are unavailable.** tiktoken is not installed, so the readiness
  baseline records `null`, not zero.
- **The H2A-4 ablation has no corpus.** `tests/fixtures/tv2_07_production_readiness/`
  is labelled with no fact corpus; H2A-0 supplied executable fixtures for all 22
  cases, but a larger frozen corpus is needed before efficiency claims.

## Next

**H2A-2 Artifact Authority**, and its first task is an audit, not a class: pull
every duplicate and compatibility state, and answer for each who owns the truth,
who is a reference, who is a projection, and who should be deleted. Only then
decide whether `EvidenceArtifact`, `CalculationArtifact` or `SlotBinding` are
needed.

`nf-v3-h1-harness-core` remains at `02d86d7` and is not moved by this seal.
