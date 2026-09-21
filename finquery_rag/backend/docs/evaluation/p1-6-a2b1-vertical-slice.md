# P1.6-A2B-1 — the vertical slice: source structure separates the disputed facts

The question this step exists to answer, on one filing, before any full rebuild:

> does the source-aware parse recover the structural dimension the store lost?

**Yes.** On JPMorganChase, the `compare-009` pair is separated by the filing's own column
header, and the store currently files both under one table and one row.

## What the parse now yields

Two changes at the parse layer, both small and both verified:

1. **Block anchors** — each block records the `ix:nonFraction`/`ix:nonNumeric` facts it
   contains (`ixbrl_anchors`), applying the same filter `ix_facts` applies.
   Verified on Microsoft: **1,710 anchors recorded, 1,710 resolve**, against 1,829 emitted
   facts of which the 119 uncovered are 29 header metadata facts and 90 `...TextBlock`
   wrappers.
2. **Cell anchors** — `grid_rows` captures each grid cell's anchors **before the element is
   dropped**, and `parse_table` carries them onto the cell record. This is what makes a
   *cell* anchorable rather than only the table, and the column is the dimension that was
   missing.

JPMorganChase: 679 blocks, **123,120 cells, 13,682 cell anchors.**

## The exit gate

```
table table_f6415249a02006fe5c6a0468      <- the segment table

  4,520   row_label 'Net income'   column_header 'As of December 31 / Corporate / 2025'
 57,048   row_label 'Net income'   column_header 'As of December 31 /  Total     / 2025'

and 57,048 again in
    'Statements of income and comprehensive income / Year ended December 31 / 2025'
    'Statements of cash flows / Year ended December 31 / 2025'
```

Against what the store holds today:

```
store: 4,520  -> table:8c4dc164...  row:def74505...   (no column)
store: 57,048 -> table:8c4dc164...  row:def74505...   (no column)   <- identical
```

So the two are **the same table and the same row** in the store — they always were, because
they are two cells of one row — and the dimension that separates them is the **column**,
which the store does not carry and the parse now does.

This is the thing P1.6-A has been trying to establish since the beginning, and it is now
established at the source rather than argued from the outside.

## What this does not yet do

The anchor stops at the parser's output. The chain the plan describes —

```
parsed fact -> atomic fact -> canonical store record -> candidate view -> index
```

— is unbuilt past the first arrow. Nothing has been re-parsed, no store has been rebuilt,
no view or index touched.

So this slice proves **the dimension is recoverable**, not that it is carried. The next
step is the propagation and its retention accounting:

```
A  parser cell anchors                     13,682   (JPM)
B  atomic facts retaining an anchor        ?
C  store records retaining an anchor       ?
D  indexed candidates retaining an anchor  ?
```

with every drop explained, and `A -> B` being the first hop — the one that has to preserve
`column_header`, `row_label`, `table_id` and `cell_id` rather than only a fact id.

## Naming

The source is SEC HTML with Inline XBRL, and the pipeline reads `primary.html`. Nothing
here reconstructs a PDF; the PDFs under `data/raw_pdfs` are a separate corpus. Earlier
documents in this line say "PDF source truth" and "PDF reconstruction" in places and
should be read as **SEC HTML / Inline XBRL structural provenance**.

What is being recovered is the lineage between a filing's tables, blocks, rows, columns and
its tagged facts — which is a stronger position than inferring table structure from a PDF,
because the structure and the tags are both stated by the document.

## Status

Parser changes only. No fixture changed, no denominator changed, no store rebuilt, no
production switch, no dense index. The existing store is untouched.
