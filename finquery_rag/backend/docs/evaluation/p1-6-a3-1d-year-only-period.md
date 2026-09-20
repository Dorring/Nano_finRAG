# P1.6-A3-1d — Coca-Cola's equity statement: a year, and no date

Diagnosis only. **No binding change, no parser change.**

```
year 2025  row 1 col 3  colspan=3  covers logical columns [3,4,5]
year 2024  row 1 col 6  colspan=3  covers logical columns [6,7,8]
year 2023  row 1 col 9  colspan=3  covers logical columns [9,10,11]
```

`2025` must not become `2025-12-31`. What follows is what the source does and does not
support.

## 1. Scope — a group the source declares, over columns that are not replicas

```
col 3: ['4,302', '9', '( 9 )', '4,302', '19,801']
col 4: ['4,302', '9', '( 9 )', '4,302', '1,760']
col 5: []
```

The columns under the year span **do not hold the same values**, so they are not replicas of
one logical column, and the span is not three independent period columns either. The source
declares **one year over a group**, so `target_scope = CELL_GROUP` — the group being the
columns its own cell spans.

**The one-row offset between columns 3 and 4 is a separate defect and not part of this
contract.** The sibling income statement shows the identical offset (`col 4` is `col 3`
shifted down one row) and binds and emits 108 records today. Whatever that misalignment is,
it is not what period provenance is about, and folding it in here would be the same mistake
as filing Pfizer under header selection.

## 2. Semantics — the source states a year and nothing about its shape

```
duration phrase in the table's own evidence: NONE

evidence text: 'Equity Attributable to Shareowners of The Coca-Cola Company /
  Number of Common Shares Outstanding / Dividends (per share — $ 2.04, $ 1.94 and
  $ 1.84 in 2025, 2024 and 2023, respec…'
```

No `years ended`, no `for the years`. The table title carries none either. **A statement of
changes in equity conventionally covers a year, and that convention is not evidence.**

So the honest binding is:

```
normalized_period = 2025
granularity       = YEAR
temporal_kind     = YEAR / UNRESOLVED
method            = YEAR_ONLY_PERIOD
```

**Not `duration`.** A bare year with nothing beside it is a year whose temporal shape the
source does not state, and writing `duration` would be a guess wearing a contract's clothes
— the same failure as fabricating `2025-12-31`, one step less obvious.

This is the case where the source falls on the conservative side, and the contract has to
have somewhere to put it that is not a lie.

## 3. Matching — deterministic compatibility, not string equality

```
slot FY2025        <-> source YEAR(2025)      compatible
slot 2025-12-31    <-> source YEAR(2025)      NOT automatically equivalent
```

`FY2025` names a fiscal year and `YEAR(2025)` names a year; they are the same claim at the
same granularity. `2025-12-31` names a day, and a year does not pin a day without further
source evidence. What the resolver needs is a **period compatibility test that knows about
granularity** — not a fabricated date fed back into string equality.

## 4. Inheritance — per cell, from the year cell, inside the group

Every numeric cell under the span records:

```
target cell        <column c, row r>
source cell        the 2025 header cell at row 1, col 3
method             YEAR_ONLY_PERIOD
```

No cell under the 2024 span may cite the 2025 cell, and no cell may inherit across the
group boundary. The provenance is a per-cell back-reference, which is what makes it
checkable rather than asserted.

## The contract, complete for all four shapes

```
                         scope        granularity   temporal_kind
ADJACENT_YEAR_JOIN        COLUMN       DAY           point / duration
HEADER_ROW_SELECTION_EXTEND  COLUMN    DAY           point / duration
INLINE_PERIOD_DATA_ROW    ROW          DAY           point
YEAR_ONLY_PERIOD          CELL_GROUP   YEAR          YEAR (unresolved)
```

```
PeriodBindingV2
  normalized_period
  granularity
  temporal_kind
  source_cells[]
  target_scope
  method
```

**Three of the six fields — `granularity`, `target_scope`, `method` — were forced by the
three shapes diagnosed after the first**, which is why wiring `ADJACENT_YEAR_JOIN` when it
passed would have meant rewriting it twice.

## Status

Nothing changed. Evidence at
`artifacts/evaluation/p1-6-a3-1d-yearonly/year-only-period.json`.
