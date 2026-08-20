# NF-V2-16 R1 architecture

User query → query/scope planner → authorization + explicit metadata hard
filters → existing Chroma/SQLite retrieval → shared hard-filter union → soft
section/content boosts → reranker subset check → EvidenceStateEvaluatorV1 →
bounded ReplannerV1 → real second retrieval → trusted evidence / fail-closed.

The R1 layer is an adapter, not a new index.  PostgreSQL/document registry
remain authoritative; index metadata may be rebuilt.  `created_at` and
`ingested_at` are operational only.  Financial time uses explicit report or
filing metadata.  The Financial Specialist is never used as an answerability
judge.  No general NLI is introduced.
