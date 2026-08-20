# NF-V2-17A4 Parser Audit

Input: immutable SEC authoritative HTML/iXBRL from the A3 raw corpus.
Parser path: run_nf_v2_17a4_parse.py.
HTML stack: lxml.etree.HTMLParser(recover=True, no_network=True, huge_tree=True).
PDF conversion, MinerU, PyMuPDF, Camelot, LLM calls, indexing, and
question generation were not used.

## Existing project paths

- finquery_rag/backend/src/pdf_retrieval_v4/table_html_parser.py
  accepts HTML table fragments and preserves rowspan/colspan grids; it
  has limited section and period metadata support.
- finquery_rag/backend/src/services/mineru_parser.py is an optional
  PDF/MinerU adapter and was not used for A4.
- The A4 path adds deterministic HTML/iXBRL normalization, controlled
  section taxonomy, table/row/cell identities, period-column binding,
  numeric/currency/scale fields, and structure-aware chunks.

## A4 output contract

NormalizedFinancialDocumentV2 preserves canonical filing identity,
raw SHA, normalization configuration SHA, typed blocks, tables, rows,
cells, and iXBRL facts/contexts. ParsedFinancialCorpusV2 preserves
tables, chunks, section/content types, period semantics, and provenance
back to normalized and raw documents.

## Limitations

Period and section extraction are deterministic heuristics. Ambiguous
or unavailable period headers remain UNKNOWN/AMBIGUOUS; no Q2-wide
quarter inference is performed. Full XBRL taxonomy reasoning and
page-oriented PDF representation are deferred. The aggregate counts
are coverage measurements, not a claim of global accounting-semantic
accuracy.
