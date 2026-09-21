# P1.6-A6 — every canonical value read back out of its filing

The check `concept_alignment` needed before any store switch: resolving to a single
undimensioned fact is not the same as the filing stating that number.

```
distinct (canonical quantity, entity) pairs   31
verified against the filing text              31 / 31
```

The 47 entity-resolutions collapse to 31 distinct pairs — Apple's net income is the same
fact whether the case is `compare-002`, `compare-004`, `compare-009` or `crossdiff-002` —
so 31 is the population, not a sample.

## It found a real inversion in the alignment

Tesla's comprehensive income came back as `4,825` where the statement says `4,886`. The
cause is the same one that governs `net_income`, applied the wrong way round:

```
ComprehensiveIncomeNetOfTaxIncludingPortionAttributableToNoncontrollingInterest  4,886
ComprehensiveIncomeNetOfTaxAttributableToNoncontrollingInterest                     61
ComprehensiveIncomeNetOfTax                                                      4,825
```

`4,886 = 4,825 + 61` — the noncontrolling interest, the same 61 that separates Tesla's
consolidated net income of 3,855 from its parent-only 3,794. The alignment preferred the
parent-only concept, so the "fix" for net income had been applied to one quantity and not
to its twin. Corrected, and both now prefer the consolidated figure for the same stated
reason.

This is the second time in this line of work that reading a value back out of the source
caught something the machinery agreed with itself about. It is the reason the verification
was insisted on before the switch rather than after.

## Two methods that did not work, recorded

**`section_type` is not usable as a filter.** Apple's and Microsoft's documents carry no
`INCOME_STATEMENT` block at all — 1,373 of 2,548 blocks across the eight filings are
`UNKNOWN`.

**`document.json`'s `blocks` are a normalised subset, not the document.** Apple's
document has 62 of them and neither its balance sheet text nor its income statement. A
first pass searched those and reported 12 values as "not stated" — including Apple's total
liabilities and Coca-Cola's total assets, both of which are plainly on the page. The
verification reads `primary.html` stripped to text instead, which is the filing itself.

That failure is worth naming: a check that reports a value as unstated because *its own*
view of the document is partial is worse than no check, because "not stated" reads as a
finding about the filing.

## Where the stratum stands

| | store coordinates | canonical concepts |
|---|---|---|
| cross-entity cases usable | 7 / 20 | 20 / 20 |
| entity resolutions ambiguous | — | 0 / 47 |
| values verified against the filing | 6 / 47 | **31 / 31 distinct pairs** |

Both gates the switch was waiting on are now met: the stratum resolves, and what it
resolves to is what the filings state.

## Status

No store changed, no fixture changed, retrieval unchanged. Artifacts at
`artifacts/evaluation/p1-6-a6-verification/` and
`artifacts/evaluation/p1-6-a5-canonical/`.
