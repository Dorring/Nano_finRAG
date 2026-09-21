# P1.6-0H — the re-derived stratum is in the benchmark

Applied and verified. Twenty cross-entity cases, one batch.

```
fixture vs store, every entity of every case
  AGREE 47 / 47    MISMATCH 0    EMPTY 0
  plan metric == gold metric == question metric : no mismatches
  fact_ids resolve into the rebuilt store        : no misses
```

## What moved

```
plan-fixtures-v7.jsonl  43f740f7ab70 -> plan-fixtures-v8.jsonl  7463b45a5bfc
gold-evidence-v1.jsonl  7831eef2d33a ->                          3d2a0c5b7839
canonical-eval-v1.jsonl 9c8ca9ceea21 ->                          227f0341d94a
```

Preimage of all three at `artifacts/evaluation/p1-6-0h-migration-preimage/`; the
per-case before/after log at `artifacts/evaluation/p1-6-0h-migration/migration-log.json`.

Every value was read out of a filing and checked against it (P1.6-A6, 31 of 31 distinct
pairs). The store was used only to resolve `fact_id`, and it is the **rebuilt** one —
resolving against the old store would have attached provenance to whichever flattened row
happened to hold the number.

## Expectations that changed

Eleven cases changed their expected outcome, not just their metric. Listing them because
a fixture migration that alters answers should say so:

| case | was | now |
|---|---|---|
| compare-001 | Tesla higher | Coca-Cola higher |
| compare-002 | Visa higher | Apple higher |
| compare-003 | Coca-Cola higher | Tesla higher |
| compare-005 | Apple higher | JPMorganChase higher |
| crossdiff-001 | `-120` | `69025` |
| crossdiff-002 | `-338` | `10178` |
| crossdiff-003 | `-3596` | `230567` |
| crossdiff-004 | `-18752` | `22517` |
| crossdiff-005 | `246` | `6785` |
| rank-002 | Microsoft > Tesla > Apple | **Apple > Microsoft > Tesla** |
| rank-004 | JPM > Coca-Cola > Visa > Microsoft | **JPM > Microsoft > Coca-Cola > Visa** |
| rank-005 | Apple > Coca-Cola > Microsoft | **Apple > Microsoft > Coca-Cola** |

`rank-002` is the one that mattered most: the old gold came from a mis-signed Apple R&D,
so the runtime's released answer was wrong and scored correct. `rank-004` is the
balance-type correction from `p1-6-0a10-expense-sign.md`.

## `compare-007` carries its fiscal years

```json
"fiscal_year_by_entity": {"JPMorganChase": "FY2025", "Pfizer": "FY2024"}
```

present on both the gold and the eval record, because the corpus holds one FY2024 filing
and comparing it against a FY2025 filer is legitimate only if the case says so. A runner
that aligned the periods would score a false result either way.

## A verification note

The verification reads the three files **back off disk** and asks the store the same
question retrieval will ask. Fixture and store were produced by different scripts from
different sources — the fixture from pages read by hand, the store from iXBRL — so their
agreement is evidence rather than a tautology. That is the same reason the store rebuild
was verified by reading its own output rather than the reasoning behind it.

## Status

Fixtures migrated and verified. **Production retrieval still reads the old store** — the
canonical wrapper exists and resolves the whole stratum, but no runtime path constructs
it. That wiring is the remaining half of step 3.
