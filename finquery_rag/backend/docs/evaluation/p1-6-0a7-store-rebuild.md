# P1.6-A7/A8 — the rebuilt store

`build_ixbrl_fact_store.py` writes `financial-facts-ixbrl-v1.jsonl` from the iXBRL of the
eight filings. **The existing store is untouched** — rebuilding is one delta and switching
retrieval to it is another, kept separable so each can be measured alone.

```
documents            8 / 8
records              19,795         (existing store: 20,394)
distinct concepts    2,552
company level        7,574          dimension_count == 0
dimensioned         12,221          the scope the existing store dropped entirely
unknown contexts         0
aligned to canonical 1,030
```

Each record carries `concept`, `canonical_concept`, `dimensions` (a list, `[]` meaning
company level), `dimension_count`, `period`, `unit`, `value`, and the filing provenance.
`dimension_count` being explicit is the point: "the coordinate identifies one value" is
now a property of the data rather than something a guard has to enforce after the fact.

## Acceptance: the file answers the stratum

`verify_ixbrl_fact_store.py` reads **the file on disk**, not the reasoning that produced
it — a store that resolves correctly in memory and is written wrongly is a real failure
mode.

```
entity resolutions        47 / 47
expected-value checks     26 / 26 matched
problems                  0
```

The three cases the old store could not answer at all (`compare-010`, `rank-005`,
`crossdiff-005`) resolve, and so do the two whose old values came from the wrong table.

## Two defects found by verifying the artifact

**`canonical_concept` alone loses the disambiguation.** The first pass grouped by
`(entity, canonical_concept, period)` and eight resolutions came back ambiguous —
Tesla holding both 3,794 and 3,855, Coca-Cola both 13,107 and 13,137 — because
`ProfitLoss` and `NetIncomeLoss` both map to `net_income` and the field cannot say which
wins. The store preserves the *source* concept; the alignment's ordering has to be applied
**at query time**. Storing the canonical name and keying on it would have reintroduced
exactly the ambiguity this work removed.

That is a design constraint on step 2, not a bug to patch: retrieval must resolve through
`CONCEPT_ALIGNMENT`, in order.

**Instant contexts were labelled `ASOF<date>`.** A balance-sheet fact has no duration, so
its context has no `period_start`, and it was falling through to `ASOF2025-09-27` while the
same company's income statement was `FY2025`. One company's balance sheet and income
statement ended up under different periods, so a query for `FY2025` found one and missed
the other. An instant that coincides with the document's `report_period_end` **is** that
fiscal year; fixed, and the acceptance run now shows `FY2025` throughout.

Both were invisible until the artifact was read back. Neither would have surfaced in a
memory-only check, which is why the check reads the file.

## Schema note for step 2

The rebuilt records carry `candidate_key`, `fact_id` and `provenance_complete` but not
`citation_id`, which `StructuredFactStore` requires by default. Loading it with
`require_citation_id=False` works, and whether the runtime needs a citation id from this
store is a step-2 question rather than something to guess at now.

## Status

Store rebuilt and verified; nothing switched. The existing store is unchanged, no fixture
has been changed, and retrieval still reads the old store.
