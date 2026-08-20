# NF-V2-17A4-R1 Parser Quality Cross-Validation

Base: `3380592c0ea6bc32aa11b9884772fbec291afd65`

## Result

Decision: **PARSER_ARCHITECTURE_ACCEPTED**

The 60-document corpus remains on the native SEC HTML/Inline-XBRL path. Raw
HTML is immutable. A4 produced searchable typed blocks and chunks with zero
orphan provenance. The 13 A4 warnings are classified as section-taxonomy-only
and non-blocking.

## Routing

- SEC HTML/Inline-XBRL: deterministic native parser; structured facts are authoritative.
- Native PDF: existing MinerU/PyMuPDF adapter.
- HTML fallback: only for explicit prose/table/header failure conditions, with parser provenance and conflict quarantine.

MinerU/PyMuPDF is not installed in this environment and no faithful HTML
renderer was selected, so the 12-filing PDF/MinerU cross-validation is
recorded as **not run**. Native table metrics are preserved; no unsupported
parser superiority claim is made.

## Text retention

Source prose is measured by A4 native block extraction over immutable raw HTML
(whitespace-normalized for comparison), normalized prose by normalized blocks,
and searchable text by exact TEXT chunk content. See
`text-retention-a4-r1.json`.

## Safety

No A5 indexing, retrieval tuning, question generation, model calls, or
raw-source mutation occurred.
