# P1.8-B3 — CORRECTION: the coverage gap is column and row identity

**This supersedes the diagnosis in `p1-8-b2`.** B2 said the 17 unreachable cases
were scoped questions with no scope in the store. That framing was wrong and
would have sent the next round at the wrong problem. What is actually missing is
narrower, better-evidenced, and more actionable.

No code changed. No backend stopped.

## What B2 got wrong

B2 tested each gold value with "is it a company-level (`dimension_count == 0`)
iXBRL fact?" and concluded 17 of 22 were unreachable because the store had no
scope. Two errors:

1. **The wrong criterion.** For a question that *names* its scope, the company-level
   test is not the test. `"Apple's Services FY2025"` is class S; its answer is
   correctly not a company-level fact. Failing that test does not make it
   unreachable.
2. **The wrong conclusion.** Measured directly: **all 22 gold facts are present
   in the legacy store, 22/22.** Nothing is missing from the data. B2's
   "the value is in the store, the scope is not" was half right — the scope is
   not there — but it is not what blocks these cases.

## What is actually true

`table_id` is null on all 20,394 records and there is no column field. The store
emits **one fact per printed value**, with the row label as the metric. Three
consequences, each measured:

### C1 — the metric is not the column

```
coordinate (Apple, Services, FY2025)  -- four records, one store row each:
   82,314   page 27   "Services | 82,314 | 71,050 | 60,345"     a revenue row
   75.4%    page 27   "Services | 75.4%  | 73.9%  | 70.8%"      a margin row
  109,158   page 32   "Services | 109,158 | 96,169 | 85,200"    a different table
   26,844   page 32   "Services | 26,844 | 25,119 | 24,855"     cost of revenue
```

`Services` names the row in every case. What separates a revenue figure from a
gross-margin percentage is **which column** it was printed under, and the store
does not record it. This is the same gap B0 reported as `ONE_ROW_COLUMNS / 9`
and judged "not recoverable at all".

### C2 — the metric is not the row

```
coordinate (Apple, Deferred, FY2025)  -- three records:
   (1,804)  row_id 28a572b24c98cb  row_index 3   page 43
   (139)    row_id b6db5deb889146  row_index 7   page 43
   604      row_id 2f0d3bfc83d894  row_index 11  page 43   <- the gold
```

Three physical rows, one table fragment, one page, all labelled `Deferred`. The
heading that distinguishes them — the section they sit under — was not captured.
`row_id` does separate them, but it is a hash of position, not of meaning.

### C3 — the metric is not the table

```
coordinate (The Coca-Cola Company, Operating income, FY2025):
   13,762  row a5da46248aad27  page 63  frag 4b97983e…  "Operating Income | 13,762 | 9,992 | 11,311"
   13,426  row c82e0d7a363533  page 87  frag 9a90ac6d…  "Operating income | $ | 13,426 $ | 12,536 $ | 11,868"
```

Same row label, two tables, two filings-sections. This is sum-007.

## The measured consequence

Applying a dimension-authority join — map each legacy value to its iXBRL fact
and keep only the `dimension_count == 0` ones — to the 22 gate-blocked cases:

```
recoverable   2 / 22   (aapl-001 -> 604, correct; jpm-007 -> $332,754, MISSES
                        the gold's $342,393, i.e. the join picked the wrong one)
```

And testing the gold's own coordinate for a unique value, which is the question
the guard actually asks:

```
UNIQUE / CONSENSUS   4     aapl-003, jpm-006, nvda-024, growth-005
CONFLICTING         18
```

Every lever measured this phase, consolidated:

```
                                  cases     running total   coverage
Gate-ON baseline                    --          46/95        48.4%
gate ontology extension             +4          50/95        52.6%
canonical-candidate seam (B1)       +3          53/95        55.8%
                                ------          -----
target                              --          57/95        60.0%
```

**Thirteen of the twenty-two gate cases are blocked by the same guard that
blocks the nine `EVIDENCE_CONFLICT` cases**, and for the same reason. The gate is
not a separate, cheap lever — it is the same wall seen from upstream.

## Why "column identity" is the honest name for the wall

Look at what the question text gives you for the four recoverable cases versus
the eighteen that are not:

```
aapl-003  "the figure for Services in FY2025"     -- names a row and a scope
nvda-024  "the Direct Customer A value"           -- names a row and a scope
aapl-001  "what was the reported Deferred"        -- names a row; three rows share it
jpm-008   "the figure for U.S. GSEs and ..."      -- names a row; 8 values under it
tsla-034  "the State value"                       -- names a row; 2 values under it
```

The recoverable ones are recoverable because their coordinate happened to
collapse to one value. The rest need the store to say **which column** and
**which row**, and it cannot.

## What this changes about the next step

B2 recommended scoping a store rebuild to carry `statement_or_table_scope`. That
recommendation survives, but the field list was incomplete. The measured
minimum, from the three classes above, is:

```
column_identity            which column of the row a value was printed under
row_identity               which physical row, beyond a positional hash
statement_or_table_scope   which table or section the row sits in
```

`row_id` already exists and does distinguish rows (C2), so `row_identity` is a
naming problem — the row needs a label that means something, not just a hash.
`column_identity` does not exist at all and is the binding one: it is what
separates Apple's Services revenue from its Services margin.

All three are properties of the source, available at extraction, and the iXBRL
layer already demonstrates the corpus carries them (125 axes, 1,083 members, and
the exact values that distinguish the KO rows: 13,762 dim-0 against 13,426
investee-dimensioned).

## What I did not do

```
did not modify   any file under src/, or the store, benchmark, gate, binder,
                 validator, finalizer, retriever or sidecar
did not stop     the backend or any GPU process
did not apply    the B1 seam, the gate ontology, or any migration
```

## Status of the targets

```
Release Coverage     46/95 = 48.4%   (target >= 60%, needs 57)   MEASURED, UNMET
Released Accuracy    all 46 released verdict "correct" in g-new   MEASURED, holds
Incorrect Release    0 in every arm                              MEASURED, holds
Correct Refusal      25/25 = 100%                                MEASURED, holds
Citation P/R         92.4% / 98.5% by logical fact                NOT INDEPENDENTLY
                                                                 VERIFIED HERE
```

The citation row is quoted from `p1-7-release-coverage.md`, not reproduced. An
attempt to recompute it from `g-new`'s `citation_ids` returned 0/46, which means
the ids in that artifact are in a different space from the ones the gold and the
store use — the attempt was wrong, not the figure. `g-new` does show **46/46
released cases carrying at least one citation**, which is all that artifact
supports. Anyone needing the citation target confirmed should recompute it with
`report_system_metrics.py`, which is the tool that produced the quoted numbers.

Coverage is the target that is not met, and it is not reachable from the
retrieval path — three levers, measured, compose to 53/95. Reaching 57 requires
column identity in the store, which is extraction.
