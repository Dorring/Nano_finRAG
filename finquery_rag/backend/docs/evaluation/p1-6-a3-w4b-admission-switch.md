# P1.6-A3-W4-B — the switch, the rebuild, and what blocked closure

The switch is in (`5d08ca5`), the rebuild ran, and **the accounting blocks closure**. That
is the gate working, not the switch failing.

```
before 26977   after 19925   net -7052
added 0        removed 7052   unchanged 19925

removed: LEGACY_FALSE_POSITIVE            37   <- exactly as predicted
removed: VALID_BUT_OUT_OF_SCOPE_GEOMETRY 171   <- exactly as predicted
removed: SOURCE_AMBIGUOUS                541   <- exactly as predicted
removed: UNCLASSIFIED_REMOVED           6303   <- BLOCKS
ADDED                                        0  <- BLOCKS
```

## The decision layer migrated correctly, and it is measurable

```
build_canonical_fact_store:  atomic_facts_seen 26311
```

**26,311 is the shadow's V2-admitted count exactly** — `unchanged_admitted` 19,925 +
`net_gain` 6,386. The emitter admits precisely what the W4-A7 shadow predicted, cell for
cell, and the 749 predicted removals landed as three exact numbers.

So admission authority did switch, and the shadow accounting predicted the store's decision
layer rather than merely correlating with it.

## BLOCK 1 — 6,303 removals outside every adjudicated class

```
26,977 - 19,925 = 7,052 removed
 7,052 = net_loss 7,367 - 315   (legacy-admitted cells the old store never held)
```

Of those 7,052, only the 749 that W4-A5's lens called *plausibly column-local* were ever
adjudicated. The remaining 6,303 are the cells W4-A5 explicitly could **not** confirm or
refute — facts whose period the cell's own column does not name, which W4-A5 refused to call
corrections because a legitimate period can come from a row, a row-group, or the table's own
context.

They are now storage removals with no class attached. The gate says that blocks, and it is
right to: the migration is removing 6,303 facts for a reason nobody has stated.

**What they need is the classification W4-A5 deliberately deferred** — the same A/B/C/D
treatment W4-A8 gave the 1,649, applied to the whole `producer_loss` rather than the
plausibly-column-local subset.

## BLOCK 2 — 6,386 gains admitted, then dropped by a second gate

```
atomic_facts_seen       26311
skipped_missing_period   6386     <- exactly the gains
emitted_facts           19925
```

This is the boundary that was named in advance: shadow accounting and Store serialisation
are separated by a layer of code, and a decision can be right and still not survive it.

`build_canonical_fact_store` computes `_canonical_period(normalized_period, period_end,
period_semantics)` and skips a fact with none — and the 6,386 gains are **precisely** the
cells whose legacy period is absent, because that absence is why the old rule dropped them.
So the older, legacy period gate sits downstream of the new admission decision and vetoes
every fact the new decision adds.

`# Store V2 has no normalized_period` was registered in W4-A4 as a schema observation. It
is now a migration blocker: admission is decoupled from `TemporalKind`, and the store is
still coupled to the *legacy* period.

Whether that layer should read the A3 binding instead is a decision, not a fix — it is the
same question W5 was created to answer, arriving one phase early because W4-B tried to
persist a decision whose provenance the schema cannot yet carry.

## BLOCK 1 — RESOLVED

All 7,052 removals now carry a class, and `UNCLASSIFIED_REMOVED = 0`:

```
LEGACY_FALSE_POSITIVE                691   ( 9.8%)   the only correction
SOURCE_AMBIGUOUS                    4188   (59.4%)   withheld, not deleted
VALID_BUT_OUT_OF_SCOPE_GEOMETRY     2173   (30.8%)   unsupported geometry
                                      ---
                                     7052

UNCLASSIFIED_REMOVED                    0
V2_PRODUCER_GAP                         0
```

The audit's scope was widened from the plausibly-column-local subset to the **whole** loss
set, which is what W4-B needed and W4-A6 did not. The families that only appear at that
scope and their adjudications:

```
Incorporated by Reference       B   an exhibit index -- the legacy axis read a filing
                                    date out of it as if it were a reporting period
Approved June/February/...      B   a date of approval, not a report period
Twelve Months Ended /           C   a period phrase whose year is not in the column
  Year ended December 31,           -- the join gap, at a scope the column cannot reach
2025 versus 2024 /              C   a comparison column names two periods, so it
  Increase/(decrease)               declares no single one
Total / Rate / Average balance/ C   a scope column -- the period comes from the table
  Selected metrics
Fair value measurements /       C   a measurement-basis column in a table whose period
  Available-for-sale securities     is the table's own
```

**One finding worth its own line, because it was a defect in the audit and not in the
system.** The first run of the widened audit reported `V2_PRODUCER_GAP = 16` — four
JPMorganChase columns reading `As of or for the year ended December 31, (in millions, …)`,
a period phrase with no year anywhere in the column. They are class **C**, not a gap. The
adjudication table was keyed on the column text alone, so an entry written for
`FULL_DATE_REFUSED` — where the full date *is* present and the phrase is what the old
predicate refused for being long — also matched `MONTH_DAY_NO_YEAR`, where the year is
missing and the same words mean something else. Entries are now scoped to the bucket they
were adjudicated for.

A false A is worse than a missing one: it blocks a phase for a defect that is not there,
and it would have sent the next step hunting a producer gap in four columns that only need
a year the filing never gives them.

## BLOCK 2 — still open

`ADDED = 0`. The 6,386 gains are admitted by the new decision (`atomic_facts_seen 26311`
proves it) and then dropped by `build_canonical_fact_store`'s `skipped_missing_period`,
because that layer still requires a **legacy** period and the gains are precisely the cells
that have none.

Unchanged from the first accounting, and unchanged deliberately: it is a decision about the
schema, not a fix.
