# P1.6-A3-1b2 — selector union, in shadow

**Nothing applied.** A3-1b tried to *replace* the header selector with a structural one and
regressed Pfizer: its equity statement labels each period inline on a row that is
structurally a data row, so a test that rejects data rows rejects rows the old selector
found usefully. This is the same structural test used as an **extension** instead.

```
new_header_rows = old_header_rows | structurally_discovered_period_rows
```

Never a removal. The union is a superset of the old set by construction, so no load-bearing
row can be lost — which is what makes this `HEADER_ROW_SELECTION_EXTEND` rather than a
rewrite with a regression attached.

## The gate

```
                              old           new          union       union dated
v_fy2025#9951     BS      [0,2,3]      [1,2,3,41,42,43]  ...same...      12/12   FIXED
pfe_fy2024#24395  EQ      [0,1,2,3,4,15,26]  [1,2,3]     [0,1,2,3,4,15,26] 18/75   NOT WORSE
ko_fy2025#11388   BS      [0,1,2,3]    [1,2,3]        [0,1,2,3]          0/9    unchanged
aapl_fy2025#5498  BS      [0,1,2,3]    [1,2,3]        [0,1,2,3]          9/12   unchanged
ko_fy2025#10778   IS      [0,1]        [1]            [0,1]              9/12   unchanged
msft_fy2025#15431 BS      [0,1,2,3,4]  [1,4]          [0,1,2,3,4]        0/9    unchanged
tsla_fy2025#7803  BS      [0,1,2,3]    [1,2,3]        [0,1,2,3]          9/12   unchanged
```

- **Visa keeps its fix** — 0/12 to 12/12, the month-day row selected and its years joined.
- **Pfizer is not worse** — the union restores exactly the set the old rule had, so its 18
  dated columns are untouched.
- **No control loses a dated column**, and none loses a header row: `new` alone dropped
  row 0 from four of them and rows 2 and 3 from Microsoft's, and the union retains all of
  them.

The union also picks up rows 41-43 of Visa's balance sheet that neither selector had — the
structural scan reaching period rows deeper in the table. That is a gain, not a change in
scope: those rows are in the union because they are period rows, and Visa's dated count
goes to 12 of 12.

## Status

`HEADER_ROW_SELECTION_EXTEND` is validated in shadow and is the method this repair will
carry. Nothing is applied yet.

**Pfizer is no longer part of this repair.** A3-1a2 filed it under `HEADER_ROW_NOT_SELECTED`
and that attribution is wrong: its period provenance belongs to **the data row itself**,
not to a header block above it. It is renamed `INLINE_PERIOD_DATA_ROW` and diagnosed
separately, through the row's own period expression, without passing through the header
selector again.

## The provenance taxonomy this establishes

Every period, however it is recovered, reports **which actual HTML cells it came from**,
its value, its granularity and its method:

```
DIRECT_HEADER
ADJACENT_YEAR_JOIN             JPM / KO  BS + CF    shadow validated
HEADER_ROW_SELECTION_EXTEND    Visa BS              shadow validated
INLINE_PERIOD_DATA_ROW         Pfizer EQ            renamed, not yet diagnosed
YEAR_ONLY_PERIOD               Coca-Cola EQ         not started
```

A period that cannot be produced says so with a status, not with `None`.
