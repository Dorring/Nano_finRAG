# NF-V2-02 — Top20 FinancialFact Expansion

Development-shadow, query-independent expansion from the sealed SADA Top5 view to frozen SADA Top20. The existing SFFM-V1 `materialize_candidate(candidate, atomic_index)` path and FinancialFactV1 contract were reused unchanged. No model, retrieval, reranker, Binder, Calculator, Generator, Validator, PDF reparse, question-aware extraction, or downstream replay was run.

- Top20 candidates: 776 unique / 1440 occurrences
- Raw/deduplicated facts: 614 / 445
- Provenance-complete facts: 445
- Relation failures: 0
- Fabricated cross-candidate facts: 0
- Historical compatible cohort: Top5 39/46; Top20 42/46
- Top5→Top20 newly recovered: 0
- Calculation fact supply complete: 6/11
- V2 direct-fact supply: 48/56
- Materialization question/Gold reads: 0 / 0
- Decision: `True`; next gate `v2_03_semantic_evidence_binder`
- Production switch allowed: `false`
