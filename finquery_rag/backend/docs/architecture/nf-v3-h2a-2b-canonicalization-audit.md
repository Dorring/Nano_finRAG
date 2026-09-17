# NF-V3 H2A-2B — Shared Financial Canonicalization: Audit

Status: **AUDIT ONLY.** No code moved. The migration is the next step.

Baseline: `491e4b9`.

---

## 1 · The layering, established rather than assumed

```
rag_v2            ->  src/           0 imports      (rag_v2 is strictly lower)
src/runtime       ->  rag_v2        28 imports      (legal)
```

So the shared layer **must live inside `rag_v2`**, or in something below it. A new
shared package outside `rag_v2` would have to be imported by `rag_v2`, which
would invert the current direction for every existing edge.

## 2 · Every canonicalisation site

| # | Definition | Location | Layer | On | Changes magnitude |
| --- | --- | --- | --- | --- | --- |
| 1 | `_fact_value_key` | `src/runtime/trusted_v2_binder.py:262` | runtime | typed fields | **yes** — folds scale |
| 2 | `_canonical_scale` | `src/runtime/trusted_v2_binder.py:244` | runtime | typed field | no (lookup) |
| 3 | `resolve_scale_keyword` | `src/pdf_retrieval_v4/semantic_scale_resolver.py:59` | **retrieval** | keyword text | no (lookup) |
| 4 | `_identity_text` | `rag_v2/adaptive/adaptive_contracts.py:31` | adaptive | **arbitrary text** | **yes** — strips commas |
| 5 | `content_fingerprint` | `rag_v2/adaptive/adaptive_contracts.py:273` | adaptive | typed fields | no (hashes them) |

Consumers: `_fact_value_key` → conflict consensus (`_consensus_fact_for_slot`,
`_unresolved_conflict_slots`). `_identity_text` → `content_fingerprint` →
`ProgressDetector` and `evidence_hashes`.

## 3 · What the audit found that changes the plan

**Three findings, in order of how much they reshape the phase.**

### 3.1 The scale vocabulary is in the retrieval layer, not the domain

`_fact_value_key` (runtime) calls `resolve_scale_keyword` from
`src/pdf_retrieval_v4/` — a *retrieval* module. Moving `_fact_value_key` down into
`rag_v2` therefore requires the **scale vocabulary to move down as well**, or to
be injected from above. It cannot come along as-is: `rag_v2` may not import
`src/pdf_retrieval_v4`.

That makes the move larger than "extract a function". It is "extract a function
*and* its vocabulary, and re-point the retrieval layer at the extracted copy".

### 3.2 `rag_v2` already has its own scale vocabulary — a third one

```
rag_v2/generation/validator.py:22
  _UNIT_RE = r"\b(?:millions?|billions?|thousands?|percent|percentage|ratio|shares?|dollars?)\b|%"

rag_v2/runtime/semantic_claims.py:90
  r"millions?|billions?|thousands?|percent(?:age)?|ratio|shares?|dollars?)\b|%"
```

Both enumerate the same scale words as `_SCALE_MAP`, with no multipliers and no
shared source. So there are **at least two independent notions of what a scale
word is**, in different layers, used for different checks.

The exit gate says "one shared financial-value semantic implementation exists".
That is not a move; it is a **consolidation of three**, and the phase should say
so rather than move one and leave two.

### 3.3 The requirements are not fully compatible as stated

`_UNIT_RE` places `percent` and `ratio` in the same alternation as
`million`/`billion`. The required semantics say **ratio ≠ percentage** and
**unknown scale stays distinct**. A single "scale" vocabulary that treats `ratio`
and `million` as the same kind of token cannot express either distinction.

This needs a decision before the migration: either three vocabularies
(scale magnitude / unit kind / representation kind) or one vocabulary with an
explicit kind per entry. It is a design question the audit can pose but not
answer from the code, because the code currently conflates them.

## 4 · The trap the brief names, confirmed

`_identity_text` is an **arbitrary-text** helper that changes magnitude:

```
'1,500' vs '1500'  -> same identity   correct (grouping separator)
'1,5'   vs '15'    -> same identity   WRONG (locale decimal -> different number)
'1,50'  vs '150'   -> same identity   WRONG
'100'   vs '100.0' -> different       WRONG for a typed value, right for prose
```

It is applied to `value`, `unit`, `currency`, `scale`, `metric`, `period`,
`entity`, `scope` — a mix of numeric and purely textual fields. The last line
above is the sharpest statement of the problem: the *same* helper is wrong in one
direction for numbers and right for prose, because it does not know which it has.

The fix is not a smarter `_identity_text`. It is that a **typed numeric field must
not go through a text helper at all** — the two must be separate paths, and only
the typed path may change magnitude.

## 5 · Proposed shape (for review, not implemented)

```
rag_v2/<shared>/
    scale vocabulary + kind      (moved down from pdf_retrieval_v4, extended
                                  with the magnitude/unit/representation kinds)
    canonical_financial_value(metric, period, value, unit, currency, scale)
        -> a canonical, hashable representation
    normalize_text(value)        (conservative; never changes magnitude)

rag_v2/adaptive  content_fingerprint  -> canonical_financial_value + text
src/runtime      _fact_value_key      -> delegates to canonical_financial_value
src/pdf_retrieval_v4                  -> imports the vocabulary from rag_v2
```

`conflict consensus` and `content_fingerprint` then share one primitive, which is
the central exit invariant.

**Numeric lexical rule** (step 4 of the brief), independently pinned:

```
1,500      -> 1500      strict grouping (groups of 3, separator only between digits)
1,500.25   -> 1500.25
1,5        -> reject / literal   (not a valid group: 1 digit after the separator)
1,50       -> reject / literal
1,5000     -> reject / literal
```

Not `value.replace(",", "")`. The accepted forms are a decision, and the brief is
explicit that the project contract decides them; the audit does not find a stated
one, so this is **NOT DETERMINED** and must be settled before the parser is
written.

## 6 · What this phase must not do

Per the brief, and confirmed by the audit as separable:

- cardinality (`cross_source`, `multi_evidence`) — H2A-2C;
- `EvidencePacketV1.page` / `metadata` duplicates — H2A-2D;
- `state.calculation_result_id` mirror — H2A-2D;
- the five prefix-only id names — H2A-2D;
- `BoundFact` / `VerifiedEvidencePacket` — H2A-2D.

Expected readiness after this phase is unchanged: `false_release 0`,
`semantic_mismatch 2`, `label_alias 3`, `unexplained 0`.

## 7 · Two questions the audit cannot answer

1. **Which numeric lexical forms the project contract accepts.** No stated rule
   was found. `1,500` is treated as `1500` today and `1,5` as `15`; those are
   different claims and only one of them is defensible.
2. **Whether `percent`/`ratio` belong to the same vocabulary as
   `million`/`billion`.** The code puts them in one alternation; the required
   semantics say they are different kinds. Deciding this changes the shape of the
   shared primitive, so it is a design decision, not an extraction.
