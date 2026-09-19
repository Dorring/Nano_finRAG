# P1.6-A — scoping the store's concept and dimension recovery

P1.6-A is upstream of the cross-entity stratum, of ~half of `factual_lookup` and of ~a
quarter of `arithmetic_calculation`. This records what the defect is, what it would take
to fix, and one project risk found while looking.

## The defect, precisely

`build_canonical_fact_store` (`src/runtime/trusted_v2_canonical_fact_store.py:168`) does:

```python
metric = _text(payload.get("metric_path")) or _text(payload.get("leaf_metric"))
```

It prefers the **breadcrumb path** over the clean leaf label. So Apple's operating-income
row is filed as:

```
metric = 'Operating expenses: / Research and development'     # breadcrumb / section / leaf
```

`leaf_metric` and `header_path` exist upstream but are **not carried into the record** —
both are `None` in the store. The column context is folded into a display string and
lost as a field.

## What survives in the store

The store is the only surviving artefact of the extraction (see the risk below), so
recovery has to work from what it holds. It holds more than it looks:

```
metric       'Operating expenses: / Research and development'
content      'Research and development | 34,550 | 31,370 | 29,915'
source_text  'Research and development | 34,550 | 31,370 | 29,915'
row_id       'row:ffae2d2ba1ed...'
cell_id      None
leaf_metric  None
header_path  None
```

Two things are recoverable **without touching a PDF**:

1. **The clean concept label.** The last ` / ` segment of `metric_path` is the leaf
   (`Research and development`), and `content` opens with the same row label. Either
   gives a concept string that is comparable across filings — which is what the
   cross-entity stratum needs and currently cannot get.
2. **The row's sibling values.** `content` carries the whole row
   (`Research and development | 34,550 | 31,370 | 29,915`), so the fact's place within
   its row is recoverable.

Two things are **not**:

3. **The column.** `compare-009` needs to know that `4,520` sits under `Corporate` and
   `57,048` under `Total`. Nothing in the record says so; that requires the PDF, which
   the P1.6-0D tooling already reads.
4. **Row-level scope for segment tables.** Where the segments are rows, `content` does
   carry the discriminating label (`Gross profit total automotive & services and other
   segment`), so this half is recoverable in principle — but the label is the row's own
   text, which is also what became the metric, so it needs care not to re-derive the
   same ambiguity.

Note that `metric_path` is not always a section breadcrumb. In NVIDIA's compensation
table it is the **column headers**:

```
metric = 'KEVAN PAREKH / CHRIS KONDO / WANDA AUSTIN / ALEX GORSKY / ANDREA JUNG / /s/ Arthur D. Levi'
```

That is why `NVIDIA / Colette M. Kress` is filed as a metric. Any normalisation has to
handle both shapes, and the two are not distinguishable from the string alone.

## The project risk

**The parsed documents that produced the benchmark store are not on the host.**

`build_canonical_fact_store` takes `parsed_documents` — the canonical parser's output —
and there are exactly **63** `document.json` files on the machine, all under
`data/financial_corpus_v2/normalized/SEC/`, which is the *SEC HTML* corpus. The
benchmark store's 8 filings are the *PDF* set under `data/raw_pdfs/`, and no parsed
documents for them exist.

So the store cannot be rebuilt by re-running the builder. Either:

- the PDF parsing stage is re-run to regenerate the parsed documents (the pipeline
  exists — `pdf_retrieval_v4`, `pdf-sr-v2` artefacts are on the host — but whether it
  reproduces this corpus is unverified), **or**
- P1.6-A works as a post-hoc enrichment layer over the existing store, accepting that
  the column dimension is only recoverable by going back to the PDFs.

This is worth resolving before P1.6-A is scheduled, because the two answers are
different sizes of job.

## Scale, for whoever schedules it

```
7,887 coordinates, 1,433 (18.2%) hold more than one comparable value
2,601 coordinates (33%) hold no readable quantity at all   -- a separate defect

cases whose gold sits at an ambiguous coordinate:
  adversarial_abstention    0 / 25
  arithmetic_calculation    9 / 35
  factual_lookup           21 / 40
  cross_entity_comparison  12 / 20
```

The 2,601 unreadable coordinates are not part of this scope and are unexamined; they may
be a larger defect than the one being fixed.

## Proposed shape

Subject to the rebuild question being answered:

1. **Concept layer** — derive a clean `concept` field from `metric_path`'s leaf (or
   `content`'s leading label), keep the raw `metric_path` for provenance, and use
   `concept` as the retrieval key. This alone would make `Apple / Operating income`
   exist, and is testable against the 20-case re-derived stratum without any PDF work.
2. **Scope layer** — surface a `scope` field where it is recoverable from the store, and
   backfill the column-scoped cases from the PDFs with the P1.6-0D tooling. Facts whose
   scope cannot be established stay flagged rather than guessed, and the operand guard
   keeps refusing them.
3. **Measure after each step** against the same survey
   (`p1-6-a-scoping/coordinate-ambiguity-survey.json`) so the effect on the 18.2% is a
   number rather than a claim.

Step 1 is well-defined and independent; step 2 depends on the rebuild answer.

## Status

- **No fixture has been changed.** Migrating the re-derived stratum waits on this, by
  decision.
- **No store has been changed.** The 7 store-supported cross-entity cases stay usable
  as they are.
