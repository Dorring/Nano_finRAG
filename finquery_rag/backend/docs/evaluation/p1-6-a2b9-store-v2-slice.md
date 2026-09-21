# P1.6-A2B-9 — Canonical Fact Store V2, one filing

The vertical slice, against the four criteria the plan set — none of which is "resembles the
legacy store".

```
source        SEC_19617_000162828026008131   (jpm_fy2025)
source sha256 4d9febdbc2038dcd
records       11,819
with column_header + row_id + cell_id   11,819 / 11,819
with source anchors                      6,417 / 11,819
store sha256  e9f4649363da4d44
reproducible  True
```

## The criteria

| criterion | result |
|---|---|
| producer in this repository | `run_nf_v2_17a4_parse.py` + `trusted_v2_canonical_fact_store.py`, both here and both named on every record |
| source locatable | every record carries the filing's accession, its `primary.html` sha256, and its table / row / cell ids |
| structural fields not lost | `column_header`, `row_label`, `table_fragment_id`, `row_id`, `cell_id` present on **11,819 / 11,819** |
| audited facts rebuild | 4 of 5 exact; the fifth is characterised below |
| same source + same code → same bytes | verified by building twice in the run: `True` |

## The audited facts

Read out of the filing by hand in P1.6-A2B-7, and checked back through the store:

```
ok   Net income   57,048  under Total      at 2025
ok   Net income   58,471  under Total      at 2024
MISS Net income   49,552  under Total      at 2023
ok   Net income    4,520  under Corporate  at 2025
ok   Net income   10,601  under Corporate  at 2024
```

The pair that matters is exact: `4,520` under **Corporate** and `57,048` under **Total**,
each at its own year. That is the dimension `compare-009` turns on, present in the store
rather than in a parse-time intermediate.

**The miss is a header defect, not a missing fact.** `49,552` is in the store fifteen times,
at `period_end 2023`, with `row_label` `Net income` — the value and the period are right. What
is wrong is its `column_header`:

```
49,552  'As of December 31 / 2023'          <- this column
58,471  'As of December 31 / Total / 2024'
57,048  'As of December 31 / Total / 2025'
```

The group label `Total` is present on the 2025 and 2024 columns of that table and absent on
the 2023 one, so `col_headers` is not carrying the group across all its columns. This is the
inconsistency flagged in A2B-7 as "worth a pass" and now pinned to a specific cell.

It matters for anything keying on the column label, and it does not affect values, periods
or the Corporate/Total separation above. It should be fixed in `col_headers` before the
slice is widened — and it is worth noting that it was found by an acceptance check that
failed, which is the point of having one that can.

## What the slice does not claim

**Not a benchmark migration.** One filing, and the fixture, the denominator and the legacy
store are all untouched. The manifest says so explicitly.

**Not a record-count claim.** 11,819 is not compared to the legacy store's anything. That
artifact has no producer, no parsed source and no reproducible corpus here, and the authority
reset records that it cannot define correctness.

**Not all values verified.** Five audited facts, chosen because they were read from the
filing. The other 11,814 have not been read back, and the slice does not assert they are
right — only that they are structurally locatable and reproducible, which is what makes
reading them back possible.

## Status

One new script. The parse and builder changes from A2B-2/6/8 stand. No fixture, denominator,
legacy store, view, index or production behaviour changed.
