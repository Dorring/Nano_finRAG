# P1.6-0H — what is measurable now, and what waits for P1.6-A

> **Superseded on one point.** This document was written at the dry-run stage, and its
> closing line — *no fixture has been changed* — was true when it was written. The
> migration was subsequently **applied**: the rebuilt iXBRL store removed the blocker
> described below, and the stratum went in as one batch. The measurements of what the
> store supports remain as written and are still the reason `p1-6-0h-measurability`'s
> option 3 was the one taken.
>
> Authoritative record of what actually shipped: `p1-6-0h-fixture-migration.md`, the
> per-case diff at `benchmarks/tv2_canonical_v1/migrations/p1-6-0h.json`, and the
> version manifest at `benchmarks/tv2_canonical_v1/benchmark-version.json`.
>
> The history is kept rather than edited because "dry-run, then apply" is the correct
> order and the pre-application measurement is what justified applying it.

Option A executed. Two facts decide the seal's denominator, and the second one was not
known when A was chosen.

## 1. Cross-entity: 7 of 20 cases the store supports

A case is **store-supported** when, for every entity in it, the coordinate
`(entity, metric, FY2025)` holds exactly one distinct value *and* that value is the
company-level figure read from the filing. Measured against the re-derived stratum:

```
SUPPORTED (7)   compare-004  compare-005  compare-006  compare-008
                crossdiff-001  crossdiff-002  rank-003

NOT (13)        the rest
```

The failures are concrete, not judgement calls:

```
compare-010  Apple Operating income      n_distinct=0   concept absent
rank-005     Microsoft Operating income  n_distinct=3   segment values
compare-009  JPMorganChase Net income    n_distinct=3   the Corporate column
rank-002     Tesla R&D                   n_distinct=4
crossdiff-004 The Coca-Cola Long-term debt n_distinct=5
compare-003  Tesla Total assets          n_distinct=0
```

A limitation of this test, recorded because it matters: it compares **magnitudes**, so a
sign defect passes. `rank-002`'s Apple R&D is stored as `(34,550)` where the filing says
`34,550` and the test does not catch it. Every one of the 7 supported cases was checked
by hand for sign and none has one, but the test alone is not sufficient.

## 2. The other three strata are not clean either

This is the part that changes A's premise. Measured over all 120 cases
(`p1-6-a-scoping/coordinate-ambiguity-survey.json`):

| stratum | cases whose gold sits at an ambiguous coordinate |
|---|---|
| adversarial_abstention | 0 / 25 |
| arithmetic_calculation | 9 / 35 |
| factual_lookup | **21 / 40** |
| cross_entity_comparison | 12 / 20 |

For a factual-lookup case this is not fatal the way it is for a deterministic one — a
prose answer can still come out right — but it means a failure **cannot be attributed**.
The system may have picked a different stored value than the gold and been wrong, or been
right; the benchmark cannot tell. Examples of the affected golds:

```
Apple         / Deferred                              3 distinct values
JPMorganChase / U.S. GSEs and government agency       9
NVIDIA        / Colette M. Kress                      5   (a person's name as a metric)
The Coca-Cola / Intersegment                          7
```

`NVIDIA / Colette M. Kress` is a compensation-table row: the extractor filed a named
executive as the metric. That is the same defect as `rank-005`'s `Interest rate
contracts`, at a different scale.

## What this means for the seal

**A seal run today would rest on `adversarial_abstention` (25 cases, clean) plus two
strata in which roughly a quarter to a half of the cases cannot be attributed.** That is
not a clean denominator, and the seal is the artifact the phase is judged on.

The options are the same shape as before, but the numbers have changed:

1. **Seal on abstention alone** (25 clean cases) — honest, narrow, and probably too thin
   to support a phase-level claim.
2. **Seal with the compromised cases excluded** — 19 factual_lookup + 26 arithmetic + 25
   abstention = 70 cases, each of which passed the same unique-coordinate test used for
   cross-entity. The exclusions are then a stated property of the run, not a silent one.
3. **Do P1.6-A first and seal once, on everything.** Slowest, and the only one where the
   seal's denominator is the benchmark rather than a subset of it.

## Status

- **7 cross-entity cases are ready to use now**, with their re-derived values already
  verified against the filings.
- **13 are pending**, and the blocker is the store, not the design.
- **No fixture has been changed.** `plan-fixtures-v7.jsonl`, `gold-evidence-v1.jsonl` and
  `canonical-eval-v1.jsonl` are untouched, preimage at
  `artifacts/evaluation/p1-6-0h-migration-preimage/`.
- **P1.6-A is upstream of the cross-entity stratum and of roughly half of factual_lookup
  and a quarter of arithmetic.** It is no longer a post-seal item.

## Files

```
docs/evaluation/p1-6-0h-measurability.md   this document
docs/evaluation/p1-6-0g-rederivation.md    the stratum, ready to apply
scripts/evaluation/migrate_cross_entity_v8.py   the migration (dry-run only)
artifacts/evaluation/p1-6-a-scoping/coordinate-ambiguity-survey.json
artifacts/evaluation/p1-6-0h-migration-preimage/
```
