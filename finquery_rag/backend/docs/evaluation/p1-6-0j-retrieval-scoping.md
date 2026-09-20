# P1.6-0J — re-keying retrieval is not a re-index, and the difference matters

Scoping the piece the last attempt showed is needed: retrieval must return canonical
candidates. What follows is what the retrieval path actually is, what a re-index would
cost, and the design problem that makes the obvious version the wrong one.

## What the retrieval path is

```
candidate-metadata.sqlite
  81,576 rows / 40,788 distinct views / 4 lanes
  (candidate_raw_bm25, candidate_raw_dense,
   candidate_structured_bm25, candidate_structured_dense)

  candidate_key   v2fact:0764d4f80e53c7b79a326c13c17762b9
  retrieval_text  "Document: aapl_fy2025\nPage: 1\n(Registrant's telephone
                   number, including area code)\n    "
  metadata_json   {metric_paths, periods, row_ids, pdf_page, ...}
```

The index is built by `CandidateViewIndexBuilder.build(view_pairs)` from candidate views,
which `trusted_v2_fact_index.build_fact_store_candidate_views` derives from store records.
So the builder exists and a re-index is mechanically available.

## The problem with indexing the iXBRL store directly

`retrieval_text` is **the document's own prose**. That is what BM25 and the dense lanes
match on. An iXBRL fact has no prose:

```
Apple | us-gaap:NetIncomeLoss | FY2025 | 112010
```

Indexing the rebuilt store as-is would produce ~19,795 views whose entire matchable text
is an entity, a concept tag, a period and a number. Retrieval quality would fall off a
cliff, and the benchmark would measure that rather than the system.

So "rebuild the index from the canonical store" is the obvious move and the wrong one. The
text-bearing surface and the concept-bearing surface are not the same population:

- a **legacy view** is a table cell or a narrative chunk, with the document's words
- an **iXBRL fact** is a tagged value, with a standard concept and a scope
 
They correlate but neither contains the other. That is the dimensional-recovery problem
from P1.6-A reappearing one layer up — the same question of which fact a piece of text
refers to, now asked about retrieval instead of about the store.

## What the options actually are

**A — index the iXBRL store alone.** Small, mechanical, and it discards the prose.
Retrieval would be worse and the benchmark would say so.

**B — keep the views, attach each one's canonical fact.** Retrieval keeps its text, and a
candidate carries a canonical concept and value where the correspondence is known. The
cost is the correspondence itself: matching a cell or a narrative chunk to a tagged fact
is exactly the judgement the whole line of work has been refusing to guess at, and
getting it wrong would silently attach the wrong concept to a retrieved passage.

**C — keep retrieval, resolve at the operand.** Retrieval returns what it returns; when a
slot is bound, the operand is read from the canonical store by
`(entity, canonical quantity, period)` rather than from the candidate. This is the
smallest change and it is what the wrapper was trying to do — the difference is that it
must happen **at the operand step**, not underneath `facts_at_coordinate`, which the guard
also calls and which made the last attempt refuse everything.

C is the one worth costing first. Its risk is provenance: the operand would carry a
canonical value under a legacy evidence id, and the release validator checks the rendered
answer against the structured result — so an operand whose value and citation disagree
about which store they came from is a defect waiting to be found later.

## What I have not done

Nothing has been built. This is reconnaissance and sizing, because the first attempt at
this failed by implementing before the requirement was precise, and the second would too.

The question to settle before any of A, B or C is written:

> **Which fact does a retrieved passage refer to** — and where is that decided?

That is not a retrieval question. It is the same one P1.6-A answered for the store, asked
about the index, and it should be answered the same way: from the filing, with the
passage's own coordinates, rather than by joining two populations on a matching value.
