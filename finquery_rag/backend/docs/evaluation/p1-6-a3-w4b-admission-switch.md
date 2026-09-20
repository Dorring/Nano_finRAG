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

## What did not happen

Nothing was changed to make the numbers close. The three removal classes stayed apart, the
749 landed exactly, and both blockers are reported rather than absorbed. Per the gate, W4-B
does not close.
