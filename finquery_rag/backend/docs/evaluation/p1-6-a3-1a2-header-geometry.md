# P1.6-A3-1a2 — remaining header geometry

Diagnosis only. **No period binding change, no reconstruction rule change.** A3-1a repaired
four of the nine zero-record primary statements by joining a month-day fragment to the
bare-year columns beside it and left five. This asks not how to reach those five but **which
grid geometry breaks the period provenance**, so A3-1b can be attributable repairs rather
than one heuristic grown until it fits.

## The five, and what the geometry says

```
table             rows x cols   header_idx        dated cols   verdict
ko_fy2025#12533 EQ     47 x 12   [0,1,2,3,21]               0   YEAR_ONLY_HEADER
nvda_fy2025#8766 CF    45 x 18   [0,1,2,3]                 15   NOT A PERIOD DEFECT
pfe_fy2024#24395 EQ    38 x 75   [0,1,2,3,4,15,26]         18   HEADER_ROW_NOT_SELECTED
tsla_fy2025#10407 CF   55 x 18   [0,1,2,3]                 15   NOT A PERIOD DEFECT
v_fy2025#9951    BS    56 x 12   [0,2,3]                    0   HEADER_ROW_NOT_SELECTED
```

### Visa — the month-day row was never selected

```
row 1   col 3   cs=9  covers [3..11]  NOT-SEL   text='September 30,'
row 2   col 3          sel             years 2024, 2025
header_idx [0, 2, 3]
```

Row 1 holds the date and row 2 the years, and **`header_idx` kept row 2 and skipped row 1**.
The mechanism is in the selection rule: a row in the first four is taken when it is a `<th>`
or contains no numbers, and `September 30,` contains one — and it carries no year and no
duration phrase, so the second scan does not pick it up either. The month and the years were
never in the same header row to begin with; the row that had the month was dropped.

Pfizer's equity statement is the same shape with a different cause: its month-and-year rows
sit at indices 4, 15, 26 **and 37**, and the second scan is bounded at `grid[:32]`, so the
2024 header row is simply out of range.

### Coca-Cola's equity statement — a year-only header

```
row 1  col 3  cs=3  covers [3,4,5]  sel  text='2025'
row 1  col 6  cs=3  covers [6,7,8]  sel  text='2024'
MONTH rows: []
```

The years are present, selected, and propagated across their spans. **There is no month-day
anywhere in the grid.** Nothing is split and nothing was dropped — the table gives years and
the binder needs a full date. This is not the A3-1a shape and the A3-1a rule cannot and
should not reach it.

### NVIDIA and Tesla — not period defects at all

Both bind **15 of 18 columns**. NVIDIA's date reads `Jan 26, 2025` in a selected row;
Tesla's reads `Year Ended December 31,` in row 1 with the years in row 2. Their period
binding is largely working.

**A3-0's conclusion was too broad.** It found that every cell dying at the temporal-kind
guard was the first divergence across all nine, and that is still true — but for these two
the columns *do* carry a period and the kind is still `bucket`, so the loss is **downstream
of the binding**, in how the kind is derived, not in whether a period was recovered. These
two are a second root cause that the period work will not touch, and treating them as part
of A3-1 would have hidden it.

## The attribution table

```
table             first loss   root cause                              repair
ko  BS  cash flow temporal     split date across sibling columns       adjacent-year join  [A3-1a]
jpm BS  cash flow temporal     split date across sibling columns       adjacent-year join  [A3-1a]
v   BS            temporal     month-day row not selected as header    header-row selection
pfe EQ            temporal     header row at 37, scan stops at 32      header-row selection
ko  EQ            temporal     year-only header, no month-day          year-only period contract
nvda CF           temporal     not a period defect: 15 columns bind    separate root cause
tsla CF           temporal     not a period defect: 15 columns bind    separate root cause
```

**Nine tables, four repairs, and one of them is not a repair.** That is the case for not
having grown one rule.

## What A3-1b may now be

Not one change. The three period repairs share a contract and are separately attributable:
every reconstruction must answer **which actual HTML cells this period came from**, and name
its method — `ADJACENT_YEAR_JOIN`, `HEADER_ROW_SELECTION`, `YEAR_ONLY_PERIOD`. A table whose
period still cannot be produced must say so with a status, not with `None`.

The two that are not period defects must be **removed from A3's scope and carried as their
own root cause**, not folded into the period repair because they happen to fail at the same
guard.

## Status

Nothing changed. Evidence at `artifacts/evaluation/p1-6-a3-1a2-geometry/header-geometry.json`,
which carries each table's full grid evidence — carriers, spans, the columns each covers and
the selected header rows.
