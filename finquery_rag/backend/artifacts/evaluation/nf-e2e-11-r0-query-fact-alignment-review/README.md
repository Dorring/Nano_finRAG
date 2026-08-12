# NF-E2E-11 R0 — Query–FinancialFact Canonical Alignment Review

Development-shadow, read-only review on the sealed NF-E2E-10 selection state. No model, retrieval, reranker, PDF reparse, DFS execution, alias modification, or E2E replay was performed.

- FinancialFactV1: 169 facts; full query-level provenance 39/46; contract SHA `7a253b443962c5f372dd897c49c057a19b553e92314faadc31eefc82b27b54eb`
- Query signals: document, metric, and period are available for 46/46, but query and fact paths do not share an alias/canonical namespace.
- DS3 metric mismatches: 28; canonical-recoverable under the frozen definitions: 5/28.
- DS4 period mismatches: 3; canonical-recoverable: 0/3.
- DS7 exact-tuple ambiguities: 7; provenance-safe deterministic dedup: 0/7.
- Projected Ready upper bound after canonical-only recovery: 6/46; projected provenance-safe upper bound: 5/46.
- Decision: `query_fact_alignment_recovery_warranted=false`; next gate `end_to_end_method_freeze`.
- Production switch allowed: `false`.
