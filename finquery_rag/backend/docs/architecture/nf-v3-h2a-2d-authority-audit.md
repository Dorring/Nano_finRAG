# NF-V3 H2A-2D-1 — Final Authority & Reachability Audit

Status: **AUDIT ONLY.** No production code changed, no migration started.
Baseline: `dfb5f7e` (H2A-2C closed and pushed).

The question is not "how do we remove duplicate fields" but **which structure
owns each domain truth**, now that H2A-2B and H2A-2C have changed the runtime
contracts. A duplicate can only be removed once an owner is named and an
independent guard exists that does not derive from the owner.

---

## 1 · The matrix

| Domain truth | Current authority | Duplicate / mirror | Projection / reference | Proposed 2D action | Independent guard after removal |
| --- | --- | --- | --- | --- | --- |
| Evidence instance identity | `packet["evidence_id"]` | four prefixed ids over one digest (§7) | `_structured_evidence_identity` ladder | **Consolidate** — one digest, one identity | **NONE YET** — see §9 blocker |
| Evidence content identity | `content_fingerprint` | — | — | none (clean) | existing (2B contract tests) |
| Evidence provenance (source) | `physical_source_id` | — | — | none (clean) | existing (conflict oracle) |
| Page provenance | `EvidencePacketV1.page` | `metadata["page"]`, `metadata["pdf_page"]` | `coordinator._structured_citations` | **Make metadata ingest-only**; close the binder fallback (§2) | **WEAK** — the existing test pins the duplicate |
| Calculation identity | `CalculationResult.calculation_id` (property) | `state.calculation_result_id`, `outcome.calculation_result_id`, trace mirror | `_calculation_id(state)` | **Demote the mirrors to derived** | **NONE YET** — needs an authored digest test |
| Calculation operand provenance | `CalculationResult.operands[].evidence_chunk_id` | — | `to_public_dict()` | **Keep** — it is a different truth, not a duplicate (§4) | existing (`test_multi_support_provenance`) |
| Claim support provenance | `ClaimProvenance.bound_evidence_ids` | — | `outcome.evidence_ids`, `outcome.citations` | **Keep** — projections, not copies | sealed readiness gold |
| Candidate citation ids | `last_citation_ids` → `CandidateExecutionResult.citation_ids` | `state._candidate_citation_ids` (**no producer**) | — | **Resolve vacuous guard first** (§5) | needs authoring |
| Candidate calculation ids | `CalculationResult.calculation_id` → `calculation_result_id` → `calculation_ids` | `state._candidate_calculation_ids` (**no producer**) | — | as above | needs authoring |
| Dead contract types | — | `BoundFact`, `VerifiedEvidencePacket` | — | **Delete** (§6) | needs an import guard |
| Inline citation | presentation | — | answer text | **Document only** (§8) | n/a |
| Ratio-unit gate | — | unreachable predicate | — | **Post-H2A debt** (§9) | n/a |

---

## 2 · Page provenance

**The hypothesis holds, with one contradiction.**

`EvidencePacketV1.page` is the runtime authority: a declared field, part of
instance identity, excluded from the content fingerprint, promoted out of
`metadata` by H2A-1C precisely because the citation projection reads the top
level.

`metadata["page"]` and `metadata["pdf_page"]` survive as **retained ingest
inputs**, not by design: `page` and `pdf_page` are absent from
`EvidencePacketV1.from_mapping`'s `consumed` set, so the raw ingest keys ride
along into the metadata bag. Nothing in the V2 runtime reads them for
correctness except the contradiction below. The name-fallback in `from_mapping`
(`raw["page"] if raw.get("page") is not None else raw.get("pdf_page")`) is the
**correct** ingest reconciliation of the two spellings, and is the only place
they are reconciled.

**Contradiction — `build_runtime_binder_fact_view`.** It resolves fields across
`[fact, structural_context, metadata]` and continues past `None`
(`binder_fact_view.py:110-118`), and both `"page"` and `"pdf_page"` are in the
binder allow-list (`disclosure.py:83-86`). A packet whose page is absent writes
`"page": None`, so **the metadata value wins**. This is the one production
runtime read of page that comes from `metadata`. It is on the live binder path.

Two further dead reads, recorded not fixed: `coordinator.py:129` and
`trusted_v2_calculation.py:110` both fall back to `pdf_page`, which no current
writer can produce because `to_dict()` always emits `"page"`. And
`local_specialist_generator.py:181` reads a page the SPECIALIST profile never
discloses, so it always resolves to the literal `1`.

## 3 · Calculation identity

`CalculationResult.calculation_id` is the owner — a *property*, derived from the
operation, formula version, value, unit and each operand's
`(name, value, evidence_chunk_id)`, so it cannot disagree with the result's own
admissibility.

`state.calculation_result_id` is a **derived compatibility mirror**, written from
`capability.last_calculation_id` at two sites and read by two. It stores nothing
`CalculationResult` does not already own. `outcome.calculation_result_id` is a
positional `[0]` mirror of `outcome.calculation_ids`, and the trace carries a
third. Four stored copies of one derivable string.

## 4 · Calculation support provenance — two truths, not one duplicate

This is the finding that changes the shape of 2D, and it contradicts the framing
carried out of 2C-3.

- `CalculationResult.operands[].evidence_chunk_id` records **which evidence
  supplied this operand's value**. `_operand_for_slot` takes `ids[0]` — one
  representative per slot, deliberately, because the calculator runs once on the
  canonical quantity.
- `ClaimProvenance` for a calculation claim records **the union of every
  admitted support of every participating slot** (`trusted_v2_provenance.py:135`),
  which is strictly wider whenever a slot has more than one support.

These answer different questions — *where did this number come from* versus
*what supports this claim* — so the smaller set is not a collapsed copy of the
larger one, and **adding `supporting_evidence_ids` to the public calculation
dict would create a third durable truth rather than repair a loss.** The 2C-3
question is answered: `CalculationResult` owns the calculation fact and result;
`ClaimProvenance` owns claim-support; the public projection should stay a
projection.

## 5 · Candidate citation / calculation ids — classification **C**

**Verdict: missing producer — a vacuous validator guard.** Highest priority in
this audit, and worse than "dead code".

`state._candidate_citation_ids` and `state._candidate_calculation_ids` are read
at `trusted_v2_validation.py:322-323` and **written nowhere**: no assignment, no
`setattr`, no `__dict__` mutation, no dataclass field on `AdaptiveRAGStateV1`, no
deserialization path, no test fixture. The sibling `_calculation_result_obj` *is*
written, which sharpens the contrast.

Because the state class has an ordinary `__dict__`, `getattr(state, name, ())`
returns `()` rather than raising — so `_candidate()` silently yields
`citation_ids=()`, and the guard it feeds:

```python
if not set(candidate.citation_ids) <= set(allowed_citations):
    raise ValueError("candidate citation metadata is not Binder-admitted")
```

is **trivially satisfied for every input**. The invariant it names has never been
tested. The run dies one step later in `_coerce_envelope` (missing structured
citation IDs), which is the failure actually observed.

Scope: this reads only on the string-candidate branch, which is wired only when
the generation port is *not* candidate-mode; production runs candidate-mode, so
the branch is also unreachable in production. That makes it **C with a strong A
component** — a correctness hole in a guard, on a path with no producer.

The real authorities already exist: `last_citation_ids`
(`trusted_v2_binder.py:1089`) for citations, and
`CalculationResult.calculation_id` via `_calculation_id(state)` for calculations.

## 6 · `BoundFact` / `VerifiedEvidencePacket` — dead exported types

Declared in `rag_v2/contracts/evidence.py`, re-exported in
`rag_v2/contracts/__init__.py`, constructed **only** in
`tests/rag_v2/test_v2_00_contracts.py`. No production import, no `isinstance`, no
serialization, no FastAPI route, no published schema (only stale artifact files
carrying the *string* name). `EvidenceBinding` / `BindingStatus`, in the same
module, are live authority.

**They are not H2A-2D's artifact authority and should not be adopted as one.**
Deletion is the 2D-4 candidate; `VerifiedEvidencePacket` in particular validates
a shape (`calculation_result.supporting_evidence_ids ⊆ allowed_citation_ids`)
against a contract that no longer describes the runtime.

## 7 · Prefix-only id families

**Confirmed: one object, four prefixes.** In
`trusted_v2_canonical_fact_store.py:218-221` the four ids are built from the
**same** `(document_id, table_id, row_id, cell_id)` payload through the same
`_record_id` digest — only the literal prefix differs. One physical fact then
carries five field names (`candidate_key`, `candidate_id`, `fact_id`,
`evidence_id`, `citation_id`) and four prefixed identities, which downstream
papers over with a first-match ladder
(`coordinator.py:66-71`, `provenance.py:29-34`).

Also found: two different producers both emit a `candidate_key` under different
prefix schemes (`candidate:v1:<sha>` in `candidate_identity.py:87` vs
`candidate:<sha>` in the fact store), and `citation:v2:<sha>` is documented in a
comment but has **no producer anywhere**.

Genuinely distinct domains also exist (`atomic:` cell vs `narrative:` text block
vs `table:`/`row:`/`cell:`), so this is not "all prefixes are duplication" — it
is specifically the fact-store quad.

## 8 · Representative inline citation

`inline citation = presentation`; `ClaimProvenance` + structured `citations` =
authority. No validator treats the inline citation as the complete support set —
`scoring.py` and the validator both read `outcome.citation_ids`, not the answer
text. **Document only.** Forcing `[citation-X1][citation-X2]` into the answer
would be a rendering-scope expansion.

## 9 · Ratio predicate and independent-oracle blockers

**Ratio predicate:** dead by *structure*, not by empty input — `incompatible` is
assigned `answer_ratio_units`, making the condition `A and K and not (A and K)`.
The sets it reads (`answer_units`, `known_unit_tokens`, `representation_of`) are
live and authoritative. Classified as **post-H2A validator debt**, not 2D.

**Independent-oracle blockers — flagged before any migration:**

| Duplicate | Why no independent oracle survives today |
| --- | --- |
| `metadata["page"]` / `pdf_page` | `test_page_provenance.py:59-60` asserts `packet["page"] == packet["metadata"]["pdf_page"]` — it **pins the duplication**, so it cannot guard its removal. A lineage fixture authored from an ingest mapping, independent of the projection, has to exist first. |
| `state.calculation_result_id` | No test recomputes `C1-<digest>` from authored operands. The digest is deterministic, so such an oracle is authorable — it does not exist yet. |
| Prefix-id quad | No guard asserts that one physical fact has one identity. |
| `BoundFact` / `VerifiedEvidencePacket` | No import guard; the only test constructs them. |

Per H2A-2C's methodology, a duplicate must not be removed before its independent
guard exists — otherwise the removal is verified by the thing it removed.

## 10 · Recommended sequence

- **2D-2 — Evidence / Provenance Authority.** Close the binder `metadata`-page
  fallback; make `page`/`pdf_page` explicit ingest-only inputs. Author the
  lineage oracle *first*.
- **2D-3 — Calculation / Candidate Authority.** Resolve classification C: give
  the citation-admission guard the real authority it was written for, or retire
  the branch with the reason recorded. Demote the three mirrors of
  `calculation_id`. Author the digest oracle first.
- **2D-4 — Dead Contract Cleanup + H2A-2 Seal.** Prefix-id consolidation (with
  its guard), `BoundFact` / `VerifiedEvidencePacket` deletion, seal.

**NOT DETERMINED:** whether the binder `metadata`-page fallback is reachable in
production practice — the precedence is confirmed, but no end-to-end fixture
exercising it was found. Whether any generation fixture pairs a string-returning
generator with the real validator under `allow_test_release=True`; the branch
appears untested, but this was not exhaustively ruled out.
