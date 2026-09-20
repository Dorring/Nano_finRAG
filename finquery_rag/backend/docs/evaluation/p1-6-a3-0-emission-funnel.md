# P1.6-A3-0 — emission funnel audit

Measurement only. **No emitter change, no row-classifier change, no schema change.** Nine
oracle-verified primary statements produce no store records while their tables hold
hundreds of valued cells, and this pass finds the first place that happens rather than
guessing at it.

## The funnel, and where it breaks

Every stage, for the nine zero-record tables against three that work:

```
                              cells  col>0  sem_row  fin_row  numeric  axis  temporal  metric  emitted
aapl  balance sheet  (ok)       504    462      462      297      102   102       100     100      100
ko    income stmt    (ok)       264    242      242      220      112   112       108     108      108
-----------------------------------------------------------------------------------------------
jpm   balance sheet             456    418      418      363      150   150         0       0        0
jpm   cash flow                1026    969      969      782      255   255         0       0        0
ko    balance sheet             369    328      328      280      142   142         0       0        0
ko    cash flow                 492    451      451      374      194   194         0       0        0
ko    equity                    564    517      517      385      162   162         0       0        0
nvda  cash flow                 810    765      765      612      194   194         0       0        0
pfe   equity                   2844   2806     2806     1850      228   228         0       0        0
tsla  cash flow                 990    935      935      629      192   192         0       0        0
v     balance sheet             672    616      616      451      150   150         0       0        0
```

**The first divergence is the temporal-kind guard, and for these nine tables it is total:
not one cell survives it.**

It is not the row classifier. `financial_data_rows` is 33-46 on every zero-record table —
the rows are admitted, and admitted in numbers comparable to the controls. It is not the
metric path either: `has_metric_path` is 0 only because nothing reaches it.

```
    temporal_kind_unknown    120, 134, 162, 202, 146
    temporal_kind_bucket     255, 194, 194, 192
    temporal_kind_category    30,   8,  26
    temporal_kind_segment      4
```

Those four kinds are exactly what `emit_atomic_facts` refuses: it admits `point`,
`duration` and `comparison` and nothing else. The controls lose 2 and 4 cells here; these
lose everything.

## Why the temporal kind is wrong

The kind is classified **per column** from the column's header and its cells' period
binding. Side by side:

```
Apple balance sheet   col 3-6   normalized_period = '2025-09-27'   -> point   -> 236 cells emit
Coca-Cola balance     col 3-8   normalized_period = None           -> unknown ->   0 cells emit
```

Both tables are over-segmented — Apple's two-year balance sheet comes out as 12 columns
and Coca-Cola's as 9 — and both have their header polluted with the first column's text
(`'September 27, 2025 / ASSETS: / Current assets:'`). Neither of those is the
discriminator. The discriminator is that **Apple's date survives whole and Coca-Cola's does
not**:

```
Apple      header_path  'September 27, 2025 / ASSETS: / Current assets:'   one cell, full date
Coca-Cola  header_path  'December 31, / ASSETS / Current Assets'          |
                        '2025 / ASSETS / Current Assets'                  +- the date split
                        '2025 / ASSETS / Current Assets'                     across columns
```

Coca-Cola's `December 31,` and `2025` land in **separate columns**, so no column's header
assembles a complete date, so `normalized_period` is `None` for every data column, so the
temporal kind is `unknown` or `category`, so every cell is refused.

```
the date is split across grid columns
  -> no column carries a complete date
  -> cell.normalized_period is None
  -> _classify_column_temporal returns unknown / category / bucket
  -> emit_atomic_facts' temporal guard rejects every cell
  -> 0 atomic facts -> 0 store records
```

**This is a parse-layer column-header defect.** The temporal logic is behaving correctly on
the input it is given, and the row classifier is behaving correctly. Neither is where the
fix goes.

## What the nine have in common

The signature is a **date broken across cells**, and it is not confined to one statement
family or one filer: three balance sheets, four cash flow statements and two equity
statements, across six companies. Visa and JPMorganChase lose their balance sheet this way
while keeping their income statement; Coca-Cola loses three of its five.

It is also not "the last table of a statement": Apple and Microsoft lose nothing.

## Controls

The audit re-runs each table's emission loop so the guards can be counted separately, and
checks its total against `adapt_document_tables`' for the same table. No table disagreed,
so the repetition is faithful and the counts above are the real pipeline's, not a
re-implementation's.

## What A3-1 has to fix, and what it must not

The drop point is proven: the column header does not carry a whole date, so
`normalized_period` is unset. A3-1 fixes **that** — not the temporal classifier, not the row
classifier, not the metric path builder, and not the store schema.

The safety requirement stands and is not optional. Nine primary statements must stop
producing zero records, and **the dangerous `NON_PRIMARY` tables must not gain anything**:
if emission is widened enough that a note's cells start reaching Store V2, the fix has
undone the authority layer from underneath. The re-run must be scored on both.

## Status

Nothing changed. Evidence at `artifacts/evaluation/p1-6-a3-0-funnel/emission-funnel.json`,
which carries every table's full funnel, not just the nine.
