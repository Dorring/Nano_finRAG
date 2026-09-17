# NF-V3 H2A-2B-1 — Shared Canonicalization Migration

Status: **IMPLEMENTED.** Predecessor: `nf-v3-h2a-2b-canonicalization-audit.md`
(audit only, baseline `491e4b9`). Contract freeze: `8d09b07` (H2A-2B-0).

The audit's two `NOT DETERMINED` items were decided before any code moved, and
the decisions are the contract `. /tests/rag_v2/test_financial_semantics_contract.py`
pins.

---

## 1 · The shared module

`rag_v2/contracts/financial_semantics.py` — the lowest existing layer inside
`rag_v2`, and the only one that imports nothing but itself. Imports stdlib only
(`re`, `dataclasses`, `decimal`, `enum`, `typing`).

Not a single parser. Retrieval still resolves scale keywords from table
captions, the validator still scans answer text with its own regex. What is
shared is the *semantics*: which words exist, which dimension each belongs to,
and what multiplier a magnitude carries. A consumer may choose its lexical form
without choosing its meaning.

### Dependency graph

| Edge | Before | After |
| --- | --- | --- |
| `rag_v2` → `src/` | 0 | **0** |
| `src/runtime` → `rag_v2` | 28 | 29 |
| `src/pdf_retrieval_v4` → `rag_v2` | 1 | 2 |

`rag_v2 -> src/` is still zero, which is the property the audit established and
the one that must not move. `src/runtime/trusted_v2_binder.py` **lost** its
`src.pdf_retrieval_v4.semantic_scale_resolver` import — the edge the audit
found (runtime reaching into the *retrieval* layer for a domain vocabulary) is
gone. `src/pdf_retrieval_v4/semantic_scale_resolver.py` gained the shared
vocabulary, which is step 10 of the brief verbatim.

## 2 · Three dimensions

| Dimension | Answers | Vocabulary |
| --- | --- | --- |
| `MagnitudeScale` | how large | base, thousand, million, billion, trillion |
| `MeasurementUnit` | measured in what | USD, EUR, GBP, JPY, CNY, shares |
| `RepresentationKind` | read how | absolute, percent, ratio |

The three vocabularies are asserted disjoint, and every synonym is asserted to
resolve through its own dimension only.

`percent` and `ratio` are representation kinds, **not** magnitudes. The old
alternations put them beside `million`/`billion`; that conflation is why they
could not express either required distinction. A ratio is not its percentage
form: the conversion exists, no contract authorises it, and folding them would
merge a ratio with a percentage point.

## 3 · Accepted numeric grammar

Strict English/US financial syntax; no locale support.

```
1500           -> 1500          1,500        -> 1500
1500.0         -> 1500          1,500.25     -> 1500.25
12,345,678     -> 12345678      +1500 / -1500
(1,500)        -> -1500         $1,500       -> 1500
1.5e3          -> 1500          1,500(a)     -> 1500
```

Preserved because it already existed and is relied on: explicit signs,
accounting parentheses, scientific notation, a surrounding currency symbol,
footnote markers. The grammar was not broadened past those.

**Rejected — not canonicalizable:**

```
1,5    1,50    1,5000    12,34,567    1,,500    1,500,    ,1500    1500.    1.5.3
```

The rule behind the list: an unrecognised representation is never *guessed*
into a number. `1,5` is 1.5 in one locale and 15 by comma-deletion, and neither
is recoverable from the text, so the text keeps a literal identity. The previous
implementation answered 15 — a number ten times larger, arrived at by deleting a
character the author wrote deliberately.

**Unknown vocabulary.** `magnitude_of` returns `None`, never `BASE`.
`adjusted-billion` keeps its literal identity instead of silently becoming a
billion. Base magnitude is reachable only from an *absent* scale — a stated fact
about the record, not a fallback.

## 4 · The shared quantity primitive

`canonical_quantity()` / `quantity_identity()` — one function, called by both
conflict consensus and the content fingerprint, so the two agree by construction
rather than by maintenance.

```
1000 million USD  ==  1 billion USD        consensus   and   fingerprint
1.000 billion USD ==  1000.0 million USD   consensus
1 billion USD     !=  1 billion EUR        both
0.12 ratio        !=  12%                  both
1,5               !=  15                   both
```

## 5 · Old canonicalizers and maps

**Removed or delegated:**

| Site | Treatment |
| --- | --- |
| `src/pdf_retrieval_v4/semantic_scale_resolver.py::_SCALE_MAP` | deleted; reads `MAGNITUDE_SYNONYMS` |
| `rag_v2/generation/validator.py::scale_words` | deleted; reads `magnitude_of` |
| `rag_v2/runtime/semantic_claims.py::_unit_supported` aliases | deleted; reads `magnitude_of` |
| `rag_v2/generation/validator.py::_UNIT_RE` | alternation built from the shared vocabulary |
| `rag_v2/runtime/semantic_claims.py::_UNIT` | financial half built from the shared vocabulary; physical units stay local |
| `rag_v2/adaptive/adaptive_contracts.py::_identity_text` | deleted; text fields use `text_identity`, numeric fields use `quantity_identity` |
| `rag_v2/adaptive/adaptive_temporal.py::_number` | `replace(",", "")` replaced by `canonical_decimal` |
| `src/runtime/trusted_v2_binder.py::_fact_value_key` | delegates to `quantity_identity` |

**Two defects the consolidation exposed, both reported rather than silently
repaired:**

1. `validator.py`'s local table mapped `thousand` to **1** while mapping
   `million` to 1000000 — a word meaning a magnitude one order off from what the
   same word means everywhere else. It is corrected by delegation (the shared
   authority says 1000), and no test depended on the old value.
2. `validator.py`'s ratio-unit predicate is **unreachable** — verified, and it
   was unreachable before this phase: `incompatible` is the same set as
   `answer_ratio_units`, so the guard reads `A and K and not (A and K)`. Recorded
   in place. Making it live would introduce a HARD_FAIL that has never fired on
   any answer, which is a validator decision this phase was not asked to make.

**Guarded, not delegated** — `tests/architecture/test_financial_vocabulary_agreement.py`:

| Site | Why not delegated |
| --- | --- |
| `src/finance/primitive_tools.py::_SCALE_FACTORS` | `src/finance` and `src/validation` are pure V1 layers ("dependency-free", "imports only from `src.domain` and stdlib") and neither imports `rag_v2` anywhere. Pointing them at the development-shadow V2 package is the same class of inversion the audit closed for `rag_v2 -> src/`, and the brief does not authorise it. |
| `src/validation/claim_extractor.py::_SCALE_MAP` | as above |
| `src/finance/unit_normalizer.py::_SCALE_WORDS` | as above |
| `src/finance/structured_operand_binding.py::_SCALE_WORDS` | as above |
| `src/evaluation/nf_opt_07.py::_SCALE` | as above |

The guard asserts, for every table: a word the shared semantics defines may not
mean anything else in V1 (verified to fail on a one-digit change to
`primitive_tools`), no scale table may list a representation kind or a
measurement unit, and no known-scale list may be missing a shared magnitude.

These tables also carry words the shared contract deliberately does not own —
bare suffixes (`k`, `bn`), Chinese magnitude units, the `zillion` sentinel, and
`%`-as-fraction folding — which is why they are consumers with their own
vocabulary rather than a second authority.

## 6 · ProgressDetector

```
round 1   1 billion USD,    evidence A
round 2   1000 million USD, evidence B   -> same fingerprint, no progress
round 1   1 billion USD
round 2   1.2 billion USD                -> different identity, progress
```

Policy is unchanged; only the identity it reads moved onto the shared quantity.

## 7 · Readiness

`task_success 17 · false_release 0 · over_conservative 1 · semantic_mismatch 2 ·
label_alias 3 · unexplained_mismatch 0` — `0 / 2 / 3 / 0` unchanged.

## 8 · Regression

`4111 passed, 143 skipped` — the full suite, against a `4019 passed, 143
skipped` baseline. No seal tag moves.
