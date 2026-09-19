# P1.6-A2 — the store is HTML-derived, and iXBRL is the clean path

Written after being corrected: I had concluded the benchmark store was PDF-derived from
the fact that its eight `document_name`s match the eight PDF filenames. That was
inference presented as evidence, and the store's own content contradicts it.

## The store is HTML-derived

```
markers across 20,394 facts
  pipe tables          13,025   (64%)
  "Section: X > Y"      2,677
  "## " headers           509

content samples
  'Section: Apple Inc. Yes☒No☐ Indicate by check mark whether the Registrant has
   submitted electronically every Interactive Data File required to be submitted'
```

Pipe-delimited tables, DOM breadcrumbs, markdown headers and cover-page checkbox glyphs
are SEC EDGAR HTML artefacts. PDF text extraction produces none of them. `fact_type`
splits as `atomic 9,316 · narrative 6,516 · row_matrix 3,715 · bucket 438 · comparison 409`
— the 9,316 the review had in mind.

Every store document also has a matching filing in the HTML corpus:

```
aapl_fy2025  SEC_320193_000032019325000079      nvda_fy2025  SEC_1045810_000104581025000023
jpm_fy2025   SEC_19617_000162828026008131      pfe_fy2024   SEC_78003_000007800325000054
ko_fy2025    SEC_21344_000162828026010047      v_fy2025     SEC_1403161_000140316125000089
msft_fy2025  SEC_789019_000095017025100235
```

## Why A1 failed, restated

`P1.6-A1a` tried to clean the breadcrumb path to a leaf and made ambiguity **worse**
(20.02% → 23.14% on the resolved subset). The reason is now visible: the breadcrumb is
not a mangled concept name, it is the *only* surviving record of the fact's context. To
strip it is to delete the disambiguating information and then complain about collisions.
`metric_path` has two incompatible shapes — a section breadcrumb, and a run of column
headers (NVIDIA's compensation table) — and neither can be turned into a concept without
the context that was thrown away upstream.

The fix is not a better string parser. It is to stop recovering context from a flattened
string and take it from the source that has it.

## The clean path: iXBRL

Each `normalized/SEC/*/document.json` carries `ixbrl_facts` — for JPM, 8,442 of them with
1,215 distinct concepts:

```
{"concept": "us-gaap:NetIncomeLoss",
 "context": {"period_start": "2025-01-01", "period_end": "2025-12-31",
             "period_semantics": "ANNUAL", "duration_days": 364},
 "context_ref": "c-1", "unit": ..., "raw_value": "57,048"}
```

`us-gaap:NetIncomeLoss` is the **same string at every filer** — the taxonomy is the clean,
cross-company concept that the store's per-filing `metric` strings cannot be. The same
concept, filtered by period, gives exactly the value verified from the filing:

```
us-gaap:NetIncomeLoss, period 2025-01-01..2025-12-31
   57,048   ctx=c-1     the company's FY2025 net income
   55,681   ctx=c-1     a dimensioned variant (a different context)
```

`context_ref` is the **scope dimension** — 2,216 distinct contexts for JPM. That is the
missing axis: segment, jurisdiction, class of instrument, all of it, carried by the
filing itself rather than reconstructed from a display string.

## What is missing, and what it costs

- **Context definitions are not in `document.json`** — only `context_ref` and a count.
  The dimensions live in `primary.html`'s `ix:header`. A re-parse must read the raw HTML.
- **Not every presented line is tagged.** Subtotals and some rows carry no `us-gaap:`
  fact, so coverage will be partial by construction. Every tag is real; not every row is
  tagged.
- **Tagged values can differ from presented ones.** This is the `(34,550)` class of
  problem: the tagged value, its sign convention and its scale are the filer's XBRL
  choices, not necessarily what the table shows. Any store rebuilt on iXBRL needs its
  values reconciled against the benchmark gold rather than assumed equal.
- **`raw_value` is empty for 64 of 8,442 JPM facts** (0.8%) — fine, but worth checking
  per filer.

## What this means for the plan

The two branches the review laid out were "rebuild parsed docs / fact store correctly" or
"existing-store augmentation + targeted PDF backfill", to be chosen by a feasibility
probe. The probe answers more strongly than either: **the source HTML is on disk, and it
carries a standard concept taxonomy and explicit dimensions.** Neither branch is the
right frame — the rebuild should be *from iXBRL*, and the flattening the store suffers is
a property of the table-parsing path, not of the corpus.

So P1.6-A should be scoped as:

1. **Re-parse the HTML** for the eight filings, preserving `ix:header` contexts, and
   emit facts as `(concept, dimensions, period, unit, value)`.
2. **Reconcile against the benchmark gold** — the 7 store-supported cross-entity cases
   and their verified values are the fixture for this.
3. **Measure** with the same survey, so the effect on the 18.2% is a number.

P1.6-A1a's `derive_fact_concept` stays as a shadow diagnostic — it is what produced the
negative result that pointed here — but the concept layer it implements is superseded.

## Residual, registered and not worked

`2,601 / 20,394` facts sit at coordinates with no readable quantity. Unexamined; may be
larger than the defect being fixed. Not part of P1.6-A.

## Status

No store changed, no fixture changed. `derive_fact_concept` is committed shadow-only and
does not affect retrieval.
