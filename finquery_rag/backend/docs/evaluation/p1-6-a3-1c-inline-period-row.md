# P1.6-A3-1c — Pfizer's inline period, on the data row

Diagnosis only. **No binding change, no parser change.** A3-1b2 removed this table from the
header-selection repair because its period does not come from a header block above it. This
asks the three questions that decide whether `INLINE_PERIOD_DATA_ROW` is a contract or a
special case.

## The structure

Pfizer's statement of changes in equity, 38 rows. Exactly **four** rows carry a period
expression, and each is a dated row with values:

```
row  4   'January 1, 2022'     10 numeric cells   9,471 | 473 | 90,591 …
row 15   'December 31, 2022'   10 numeric cells   9,519 | 476 | 91,802 …
row 26   'December 31, 2023'   10 numeric cells   9,562 | 478 | 92,631 …
row 37   'December 31, 2024'   10 numeric cells   9,593 | 480 | 93,603 …

18 rows  VALUES_ONLY           the movement rows, carrying no period of their own
```

The period expression sits in **column 0 — the row label itself**: `Balance, December 31,
2022`. The balance rows are dated; the movement rows between them are not.

The sibling control, Pfizer's balance sheet, has **zero** such rows — its dates are all in
the header. The shape is specific to the roll-forward statement.

## 1. Scope — the row, and only the row

`target_scope = ROW`. Each dated row's period belongs to **that row's own ten numeric
cells**.

**Downward inheritance is not merely unsupported, it is wrong.** The structure is

```
Balance, January 1, 2022   | values      <- instant
  movement rows            | values      <- no period
Balance, December 31, 2022 | values      <- instant
```

A rule that carried the date down would stamp `January 1, 2022` onto the movement rows
below it. Those movements happened **during 2022** — their period is the year *ending* at
the next balance row, not the instant at the previous one. **The obvious direction is the
incorrect one.**

What would be needed to bind the movement rows is a statement-level semantic — *which
balance closes this span* — and **that is not evidence in the row geometry**. It is domain
inference, and it does not belong in a period-provenance contract that promises to say which
cells a period came from. The source supports the row; it does not support the span.

## 2. Direction — row to its own cells, never column

The period reaches the ten numeric cells on the same row. It never reaches the column: the
column's members are dated `January 1, 2022`, `December 31, 2022`, `December 31, 2023` and
`December 31, 2024` on four different rows, so a column-scoped binding would be incoherent
as well as unjustified.

## 3. Identity — nameable, cell by cell

Every fact bound this way can carry:

```
method            INLINE_PERIOD_DATA_ROW
source_cell_id    the column-0 cell of its own row      (row 4, col 0)
period expression 'January 1, 2022'
granularity       DAY
temporal_kind     point            an instant is what a balance is
target cells      that row's ten numeric cells
```

## What this means for the contract

`target_scope` is the field the roll-forward forces, and it is exactly the field the
existing `PeriodBinding` does not have:

```
PeriodBindingV2
  normalized_period
  granularity
  temporal_kind
  source_cells[]
  target_scope    COLUMN | ROW | CELL_GROUP
  method
```

`ADJACENT_YEAR_JOIN` and `HEADER_ROW_SELECTION_EXTEND`, both already validated in shadow,
are `COLUMN`. This one is `ROW`. **Adding the field after wiring the column repairs would
mean rewriting them**, which is the reason the wiring was deferred.

## One consequence to carry forward

The dated rows' labels are `Balance, December 31, 2022` — **the period expression is part
of the row label**. A metric matcher comparing row labels against a fixture's metric will
have to strip a trailing period expression the same way it already strips a trailing unit
caption. That is a metric-identity consequence of binding these rows, not a period one, and
it belongs with the `NO_FACT` work rather than here.

## Status

Nothing changed. Evidence at
`artifacts/evaluation/p1-6-a3-1c-inline/inline-period-row.json`, which carries every row's
kind, its period cells and its numeric cells.
