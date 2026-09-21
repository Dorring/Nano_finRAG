# P1.6-A2B-3 — the 1,543 vs 7,322 gap: my comparison was wrong, and a real signal behind it

Asked to explain why a re-parse produces 1,543 atomic facts where the store holds 7,322 for
the same filing. The first answer is that the two numbers were never comparable.

## The comparison was a category error

```
store fact_type            atomic 9,316 | narrative 6,516 | row_matrix 3,715
                           bucket 438 | comparison 409

build_canonical_fact_store emits    atomic_financial_cell
```

**The store contains no `atomic_financial_cell` records at all.** So the benchmark store was
not built by the builder I was comparing against — it comes from a stage that emits the
whole semantic evidence catalogue (`atomic_fact`, `narrative_evidence`, `row_matrix`,
`bucket_fact`, `comparison_fact`, `semantic_row`, `logical_table`), not only atomic cell
facts. The `fact_type` vocabulary differs, which is the tell I should have checked first.

For JPMorganChase the store's 7,322 split as 3,885 atomic + 1,815 narrative + 1,273
row_matrix + 246 bucket + 103 comparison — so the like-for-like quantity was never 7,322.
Comparing a one-emitter output against a seven-emitter output and calling the difference a
reproduction gap was my mistake, and it is the kind that would have sent the next step in
the wrong direction.

## The real signal, which stands on its own

Measuring the funnel from tables to atomic facts is still worth having done, because it
found something that does not depend on the comparison:

```
tables 679 | parsed cells 123,120
rows 5,643
financial-data rows 448
```

```
row_type:  unknown 3,255 | spacer 1,287 | total 441 | column_header 351
           group_header 236 | note 34 | section_header 32 | subtotal 7
           metric_row  0

table section_type:  UNKNOWN 645 / 679 | INCOME_STATEMENT 13 | BALANCE_SHEET 9
                     CASH_FLOW 4 | NOTES 3 | BUSINESS 3 | MDA 1
```

`emit_atomic_facts` admits a row only when its `row_type` is `metric_row`, `subtotal` or
`total`. **`metric_row` is zero, and 645 of 679 tables carry no section.** So ordinary line
items — `Net income`, `Revenue`, the rows a financial question is actually about — are
classified `unknown` and produce nothing; only rows the classifier calls `total` or
`subtotal` survive, which is where the 448 comes from.

That is a finding about the parse, not about the comparison: **the section inference is
failing on 95% of tables, and the row classifier cannot promote a plain line item without
it.** Whatever built the store faced the same code, so either it fed the classifier
something this path does not, or the store's `atomic` records come from a stage whose
admission rule is not this one. Both are worth knowing before a re-parse is trusted.

## What this changes

The next step is **not** "close a 5,779-fact gap". It is:

1. identify the stage that actually produced `financial-facts.jsonl`, since every
   comparison so far has been against the wrong one;
2. find out why section inference returns `UNKNOWN` for 645 of 679 tables — a parser input,
   a registry field, or the inference itself;
3. only then measure a re-parse against the right reference.

Items 1 and 2 are both small and both cheap to answer, and neither needs a full rebuild.

## Status

No behaviour changed by this investigation; the four source edits from A2B-2 stand. The
existing store is untouched and no document has been rewritten on disk.
