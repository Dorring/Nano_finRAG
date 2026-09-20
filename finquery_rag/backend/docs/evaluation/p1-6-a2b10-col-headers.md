# P1.6-A2B-10 — the column header now carries the group label

Step 1 of 1 → 3 → 2. One defect, found by an acceptance check that failed, fixed and
re-checked.

## The defect

`49,552` sat under `As of December 31 / 2023` while its siblings in the same segment table
read `... / Total / 2024` and `... / Total / 2025`. The group label was lost on one column.

The cause is in the filing's layout, not in the value. A group header like `Total` is a
single cell with a `colspan`, and the run of columns it labels is wider than the colspan
reaches — the three years are laid out as

```
As of or for the |  | Corporate (colspan 15) |  | Reconciling Item (colspan 9) |  | Total (colspan 12) |  |
                   | 2025 |  | 2024 |  | 2023 | | 2025 |  ...  | 2023 |  |  | 2025 | ... | 2023 |  |
```

so reading the header *at a column* returns a year with no group for whichever columns the
colspan stops short of.

## The fix

`col_headers` now carries each header row's text forward across blank columns, bounded by the
next non-empty cell in the same row — so a new group or a new year resets it. That is a
property of how a header row labels a span, rather than a special case for this table.

## The check it was found by

```
before                          after
MISS Net income  49,552 Total    ok  Net income  49,552  under Total      at 2023
                                 ok  Net income  57,048  under Total      at 2025
                                 ok  Net income  58,471  under Total      at 2024
                                 ok  Net income   4,520  under Corporate  at 2025
                                 ok  Net income  10,601  under Corporate  at 2024
```

**5 of 5**, against values read out of the filing by hand. The five were chosen because they
came from the source, and the one that failed is the reason the check exists.

## It is not only cosmetic

The record count moved, 11,819 → 11,713. That is expected and worth stating: the header feeds
`html_period_semantics`, which feeds the axis binding, and `emit_atomic_facts` admits a row
only when the temporal kind is point, duration or comparison. So a corrected header changes
which facts are admitted — 106 fewer here — and the acceptance check is what says the change
went the right way.

Header quality was checked rather than assumed:

```
empty column_header                     0
longest headers                         cover-page securities descriptions,
                                        long and legitimate
top headers                             real table names and period spans
```

## Status

One parse change. The store slice was rebuilt and re-verified: 11,713 records, all with
`column_header` + `row_id` + `cell_id`, reproducible, 5 of 5 audited facts. No fixture,
denominator, legacy store, view, index or production behaviour changed.

Next: step 3, retrieval on canonical candidates.
