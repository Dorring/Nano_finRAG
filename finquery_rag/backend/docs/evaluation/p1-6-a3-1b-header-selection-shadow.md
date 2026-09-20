# P1.6-A3-1b — header-row selection, in shadow

**Nothing applied.** The shadow does not pass its own gate, and that is the finding.

## What was tried

A3-1a2 traced two tables whose period never binds because a real header row was never
selected. The obvious repair is to widen `grid[:32]`, and the instruction was not to — it
would fix Pfizer and fail again on the next filing with an 80-row header block. What
replaces the bound is a **structural test for what a data row is**:

```
a bare year or a month-day NAMES A TIME
a number counts something

December 31, and 2024 are periods.  10,270 is a value.
A row with two or more value-like numbers outside its label column is a data row.
```

With that test the scan needs no upper bound, and `September 30,` — dropped by the old
selector for containing a digit — is recognised as a period token without a year beside it.

## What it did

```
[target]  v_fy2025#9951     BS   old [0,2,3]              ->  0/12 dated
                                 new [1,2,3,41,42,43]     -> 12/12 dated    FIXED
[target]  pfe_fy2024#24395  EQ   old [0,1,2,3,4,15,26]   -> 18/75 dated
                                 new [1,2,3]              ->  0/75 dated    REGRESSED
[target]  ko_fy2025#11388   BS   old [0,1,2,3]           ->  0/9  (unchanged)
[control] aapl_fy2025#5498  BS   old [0,1,2,3] -> new [1,2,3]     9/12 both
[control] ko_fy2025#10778   IS   old [0,1]     -> new [1]         9/12 both
[control] msft_fy2025#15431 BS   old [0,1,2,3,4] -> new [1,4]     0/9  both
[control] tsla_fy2025#7803  BS   old [0,1,2,3] -> new [1,2,3]     9/12 both
```

**Visa is fixed.** `September 30,` at row 1 is selected, row 2's years join it, and all
twelve columns bind a period — the table that A3-1a2 showed had a month-day row nobody
selected.

**Pfizer is made worse.** The old selector's multi-block behaviour — rows 4, 15, 26, each
`Balance, December 31, 2022` with the equity balances on the same line — is exactly what the
structural test now rejects, because those rows carry value-like numbers and **are data
rows**. The test is right about what they are and wrong about what the selector needs.

**The controls' header sets changed.** Row 0 is dropped from four of them and rows 2 and 3
from Microsoft's. The dated counts are unchanged in every case, so nothing observable moved
— but the gate says an unexplained change to a control's header set is a failure, and it is
one.

## What this means

**A3-1a2's attribution was partly wrong.** Pfizer's equity statement was filed under
`HEADER_ROW_NOT_SELECTED` alongside Visa's. They are not the same shape:

```
v_fy2025#9951     a header row that exists and was not selected     -> selection defect
pfe_fy2024#24395  data rows that label their own period inline      -> neither selection
                                                                       nor a split date
```

Pfizer's rows are data, the old selector admitted them by accident of the year test, and the
period they need comes from **their own inline label**, not from a header block above them.
That is a third shape, and it was mis-filed because both tables failed at the same guard.

## What the next attempt has to be

The structural test is the right idea used as the wrong kind of thing. It should **extend**
the selector — admit rows the old rule missed — and never **filter** it, because the old
rule's inclusions are load-bearing and one of them is a data row that happens to carry its
own period.

So: keep `old ∪ new` as the header set, re-measure Visa and the controls, and treat Pfizer
separately as an inline-period shape. That is a different change from this one and should
be measured on its own terms, not folded in to make one commit look complete.

## Status

Nothing changed. Evidence at
`artifacts/evaluation/p1-6-a3-1b-shadow/header-selection-shadow.json`.
