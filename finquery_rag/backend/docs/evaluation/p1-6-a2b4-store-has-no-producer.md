# P1.6-A2B-4 — the benchmark store has no producer in this repository

Step 1 of the three: identify the stage that actually produced `financial-facts.jsonl`.
The answer is that it is not here.

## What was searched

The store's records carry fields no builder in the tree emits:

```
candidate_id  candidate_key  cell_id  citation_id  citation_ids  citation_origin
content  currency  document_id  document_name  entity  evidence_id  fact_id
fact_type  metric  page  pdf_page  period  periods  physical_source_id
provenance_complete  raw_content  raw_value  row_id  row_ids  scale  source_id
source_text  source_traceback  table_fragment_id  table_id  unit  value  values
```

Compare `build_canonical_fact_store`, the only builder here:

```
... normalized_metric  normalized_period  parsed_numeric_value  period_semantics
    statement_type  ticker  metadata_origin  fact_type='atomic_financial_cell' ...
```

Different schema, different `fact_type` vocabulary. So the search was narrowed to the two
strings that are unique to the store's own shape:

```
grep -rl "citation_origin"                             -> nothing
grep -rl "deterministic_physical_source_descriptor"    -> nothing
```

**No file under `/disk/qh/nano-finrag` contains either**, across every `.py` in the tree.
The values appear only inside the store's own data.

## What that means

`financial-facts.jsonl` is an **input artifact whose producer is absent**. Not merely
un-run — not present. That is the same class of finding as the parsed documents behind it,
and it is now the third instance in this line of work:

```
the parsed documents the store was built from   absent
the corpus that can reproduce them              absent   (the on-disk corpus yields 0 facts)
the code that built the store                   absent
```

So the store is not reconstructible here by any route. It can be read, queried, measured and
replaced — and it has been, repeatedly — but it cannot be *derived*, and no re-parse can be
checked against it because there is no reference implementation of the thing that made it.

This is worth stating plainly because it changes what "rebuild the store" can mean. It
cannot mean "run the pipeline that produced it", because that pipeline is not here. It can
only mean "build a new store by a route this repository can execute and defend" — which is
what the iXBRL rebuild did, and which is why that route was chosen from the filings' own
Inline XBRL rather than from the existing corpus.

## Consequence for step 2

The section-inference failure is still worth understanding — 645 of 679 tables classifying
as `UNKNOWN` and `metric_row` coming out at zero would matter to any builder in this tree.
But it can no longer be framed as "why does the re-parse differ from the store", because the
store was made by something with different admission rules. It is a question about *this*
pipeline, and it should be answered as one.

## Status

No behaviour changed. The four source edits from A2B-2 stand. Existing store untouched.
