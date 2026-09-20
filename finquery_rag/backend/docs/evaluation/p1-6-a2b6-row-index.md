# P1.6-A2B-6 — the row classifier was looking up the wrong cells

Step 2, reframed as agreed: not "why does the re-parse differ from the store", but "why do
rows that are plainly financial fail to become `metric_row`". The answer is a key mismatch,
and the architecture suspicion was wrong.

## The suspicion, tested and rejected

The hypothesis was that `metric_row` might be gated on `section_type != UNKNOWN`, which
would have made "I do not know which statement this is" stand in for "this is not a
financial row". It is not:

```python
# 8. Metric row: has a label AND has numeric values in value columns
elif has_num and normed_label and normed_label not in _NON_METRIC_LABELS:
    row_type = "metric_row"
```

The classifier reads **only the row's label and whether it has numerics**. No section, no
table type, no statement. So the 645 `UNKNOWN` tables were a red herring, and `metric_row`
being zero had to have a different cause.

## The cause

```python
# classify_table_rows
cells_by_row = {}
for cell in cells:
    ri = int(cell.get("row_index") or 0)
    cells_by_row.setdefault(ri, []).append(cell)

for row in rows:
    ri = int(row.get("row_index") or 0)      # <- parsed rows had no `row_index`
    row_cells = cells_by_row.get(ri, [])
```

Cells carry `row_index`. Parsed rows did **not**. So `int(row.get("row_index") or 0)` is `0`
for every row, and every row is handed `cells_by_row[0]` — the first row's cells, usually a
header row with no numerics.

`_has_numeric` is therefore false for essentially every row, the chain falls past
`metric_row`, and ordinary line items land on `unknown`. The 448 that survived are the rows
that happened to be handed numeric cells.

Measured before the fix, which is also the evidence the cause is the right one:

```
rows 5,643
rows with a numeric non-first column   3,825     <- the classifier could not see them
financial-data rows                      448
```

## One field

`parse_table` now emits `row_index` on each row. Nothing else changed.

```
                         before     after
metric_row                    0     2,873
unknown                   3,255       536
financial-data rows         448     3,321

atomic facts              1,543    11,993
  with column_header       100%      100%
  with ixbrl_anchors        58%       55%
```

## What this is not

**It is not "closer to the store".** 11,993 now *exceeds* the legacy store's 3,885 atomic
records, and under the authority reset that is not a success signal in either direction.
The store is not the reference; the filing is. A pipeline that produces more facts than the
orphan artifact is not thereby better, and one that produces fewer is not thereby worse.

What supports this fix is that it is a **key mismatch with a named mechanism**, whose
removal changes exactly the rows it should and leaves the rest alone: `spacer`, `total`,
`subtotal`, `column_header`, `note` and `section_header` counts are all unchanged, and only
`metric_row` and `unknown` move.

## What still needs checking, against the filing

- Whether the 11,993 facts are *correct* — the count moving is not evidence that the rows
  are right, only that they are now seen. The disputed pair and a `Net income` row should be
  read back out of the filing, the way the earlier strata were.
- The `ixbrl_anchors` share fell from 58% to 55%, which is a smaller ratio over a much
  larger population. Worth a look, not a concern yet.
- The three duplicated `table_id`s from A2B-2 are still there and still produce the
  duplicate-key failure at the store step.

## Status

Two source changes in this step — `row_index` on parsed rows, and the four from A2B-2. No
fixture, denominator, store, view, index or production behaviour changed. The legacy store
remains frozen and untouched.
