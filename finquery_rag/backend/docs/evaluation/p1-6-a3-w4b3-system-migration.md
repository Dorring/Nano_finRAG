# P1.6-A3-W4-B3 — the migration validated, and the one thing it moved that nobody asked for

B3 asks whether moving the store from the legacy admission to the A3 admission changed any
*answer*, and whether every change the evidence supports. It is green.

```
before 26977   after 26311   net -666
added 6386     removed 7052  unchanged 19925

UNCLASSIFIED_ADDED     0
UNCLASSIFIED_REMOVED   0

removed: LEGACY_FALSE_POSITIVE            691   exact
removed: SOURCE_AMBIGUOUS                4188   exact
removed: VALID_BUT_OUT_OF_SCOPE_GEOMETRY 2173   exact

added:   PARTIAL / YEAR                  3368   exact
added:   RESOLVED / DAY                  3018   exact
```

The three removal classes are kept apart and are not summed anywhere. Only the first is a
correction; the other two are facts withheld because the source does not settle them, and
calling all three "removed incorrect facts" would be a claim none of them supports.

## The 47 slots did not move at all

```
37  RESOLVED_COMPANY_LEVEL     -> RESOLVED_COMPANY_LEVEL
 5  NO_FACT                    -> NO_FACT
 5  NO_COMPANY_LEVEL_FACT      -> NO_COMPANY_LEVEL_FACT

existing resolved regression  0
resolved value changed        0
newly resolved                0

wrong_scope        0
wrong_metric       0
value_mismatch     0
dangerous_authority 0
scaffold_authority 0
```

6,386 facts entered the store, 7,052 left, and **not one of the 47 slots changed status or
value**. Compared slot by slot against the pre-migration store under the same authority,
not by distribution — two flips in opposite directions leave the same distribution and
would have hidden here.

**This is strong evidence of safety and weak evidence of benefit, and the difference
matters.** The 47 slots are company-level facts in primary statements. The migration's mass
is elsewhere:

```
additions by table_role
    OPEN                 3680
    DANGEROUS_NEGATIVE   1723
    PRIMARY               620
    WEAK_NON_PRIMARY      363
```

90% of what the migration adds sits in tables that cannot speak for the company, so it
*cannot* reach these slots. The 47-slot gate did its job — it is the instrument that would
have caught the migration breaking something — but it was never the instrument that would
show the migration helping. Nothing here should be read as "the 6,386 gains were worth
having", because this gate cannot say that either way.

## Every addition is explained, and takes its period from its own column

Each addition names the shadow decision that admitted it (`UNCLASSIFIED_ADDED = 0`), and
each takes its period from **its own column**:

```
6110  every source cell is in the fact's own column
 276  row-scoped declaration (INLINE_PERIOD_DATA_ROW)
```

Zero admitted with no source cell, zero bound from a neighbour's column. This is the one
property the store cannot show after the fact — it keeps the *identity*, not the binding
that produced it, which is W5's job — so it is read from the shadow, where the binding is
still recorded.

By method and kind:

```
YEAR_ONLY_PERIOD               3368      new_kind YEAR     3368
ADJACENT_YEAR_JOIN             2159      new_kind point    2742
HEADER_ROW_SELECTION_EXTEND     583      new_kind UNKNOWN   276
INLINE_PERIOD_DATA_ROW          276
```

All 6,386 were `legacy_kind: UNKNOWN`, which is the A3 thesis stated as a number: the
legacy rule saw no temporal semantics at all in these cells and dropped them.

## The legacy resolver, against the new YEAR facts

The resolver is deliberately untouched, so what it does with the new year-only facts is a
question about a seam rather than about this phase.

```
period_binding_status / granularity
    PARTIAL / YEAR        5008     3368  period_end empty
                                  1640  period_end is a bare year
    RESOLVED / DAY       21303

YEAR facts carrying a full date in `period_end`                     0
CONFLICT or UNRESOLVED facts in the store                           0
YEAR facts the resolver may match on `period_end`                1640
    of those, whose `period_end` year disagrees with its identity      0
YEAR facts it cannot match at all (no `period_end`)              3368
```

`resolve_store_v2_slot.py` matches `str(period_end)[:4]` against the year a question asks
for. So the check is on what the store *claims*, because `[:4]` would make a fabricated
`2025-12-31` match the same year as an honest `2025` and the difference would be invisible
in the slot outcome. **No YEAR fact carries a date.** `YEAR(2025)` stayed `2025`; nothing
was rendered into `2025-12-31`; `CONFLICT` and `UNRESOLVED` could not be promoted into the
store because `_canonical_period` returns `None` for them.

The 3,368 that the resolver cannot match are the pre-registered downstream capability
debt, and they are exactly the number pre-registered: the resolver has nothing to match a
year-only fact *on* yet. That is a fact it has not been taught to read, not a fact it got
wrong.

## What B3 found that it was not looking for: `period` moved on 2,593 unchanged cells

The field-by-field comparison of the 19,925 cells present on both sides — "unchanged has
to mean unchanged, not merely still there" — found `period` different on 2,593 of them.
**`period_end` moved on none of them**, and that is the invariant the whole result rests
on: `period_end` is the cell's physical period and the resolver's input; `period` is a
rendering. A rendering may change only where the thing rendered did not, and the audit
blocks if it does.

The change is W4-B2's, not the admission switch's — comparing the pre-B2 post-switch store
against this one gives `period moved 0`. It is one line of the bridge: the emitted fact now
takes `normalized_period` from the binding rather than from the legacy axis.

```
RESPELLED  2010   same precision, different string
NARROWED    563   the old value stated a day the new one does not
WIDENED      20   the new value states a day the old one did not

narrowed cells whose own `period_end` held a full date   0
```

* **NARROWED (563)** — all `NON_PRIMARY`. Every one is of the form
  `As of December 31 / Term loans by origination year / 2020`, where the legacy had built
  `2020-12-31` and the new store keeps `2020`. The test that makes it safe is that **none**
  of them had a full date in their own `period_end`: the legacy `period` was contradicting
  its own `period_end`, and the bridge removed the contradiction. 1,646 cells in total went
  from contradicting themselves to agreeing.
* **RESPELLED (2,010)** — mostly `FY2024` → `2024`, the store's own `FY` convention against
  the year the source actually wrote. The rest are exhibit-index date columns in
  `NON_PRIMARY` tables, where neither value is a financial period.
* **WIDENED (20)** — the only direction that could be a fabrication, so it is bounded by a
  gate rather than described. All 20 are Visa, all maturity schedules, all in
  `NON_PRIMARY`/`UNKNOWN` tables, **none in a table that may speak for the company**. The
  day is source-grounded rather than inferred:

  ```
  header row 1: "For the Years Ending September 30,"
  header row 2: "2026"
  -> 2026-09-30, HEADER_ROW_SELECTION_EXTEND, source_cells: both rows
  ```

  That is the A3 rule working, not bending: `As of December 31` + a vintage column is
  inherited *context* and correctly stays `PARTIAL`, while `For the Years Ending
  September 30,` is a *declaration* of the fiscal year end and correctly resolves.

The 251 cells where `period` agreed with `period_end` and no longer does are the exhibit
indexes — where both old and new values are filing dates in a table that is not a financial
statement.

## What is still debt, unchanged by this phase

* **SECTION scope / SECTION_BRACKET** for row-major equity statements. `VALID_BUT_OUT_OF_SCOPE_GEOMETRY` is
  where it is counted, and it must not be called a correction.
* **The resolver's period matching** does not consume `YEAR(2025)` or granularity. 3,368 facts wait on it.
* **Row-label numeric extraction** (`Balance at January 1` → `1`).

## Verdict

```
W4-A   admission semantics + truth audit     DONE
W4-B1  removal truth closure                 DONE
W4-B2  period identity persistence           DONE
W4-B3  system migration validation           DONE
```

W5 is full provenance persistence: `method`, `source_cells`, `target_scope`, and the
temporal-kind provenance — *why* a period identity is believed, where W4-B2 persisted only
*what* it is.
