# Financial RAG Benchmark v1

This is the annotation workspace for the next evaluation expansion.  It does
not change production retrieval, generation, ranking, validation, or serving.

## Corpus policy

The catalog contains eight issuer investor-relations sources selected for
different document structures.  PDFs are acquired locally into the ignored
`runtime/benchmark/financial_rag_v1/` directory after the direct issuer URL is
verified and the downloaded SHA-256 is recorded in the catalog.  PDFs and
unreviewed extraction output are never committed.

## Label isolation

`annotations.jsonl` is local-only until each record has two reviewers.  It
contains page, section, table/row metadata and gold values.  The blind runner
will later receive a generated `questions.jsonl` without expected fields; its
sealed scorer will receive the reviewed labels separately.

## Taxonomy target

| Slice | Target |
| --- | ---: |
| fact | 20 |
| table_fact | 20 |
| calculation | 15 |
| multi_source | 15 |
| no_answer | 10 |
| unit_period_trap | 10 |

No numeric target is accepted as Golden until the source page, section, and
row/column reference are manually verified by a second reviewer.

## Validate

```bash
python -m scripts.evaluation.validate_financial_benchmark \
  --catalog eval_data/financial_benchmark_v1/documents.json \
  --labels runtime/benchmark/financial_rag_v1/annotations.jsonl
```
