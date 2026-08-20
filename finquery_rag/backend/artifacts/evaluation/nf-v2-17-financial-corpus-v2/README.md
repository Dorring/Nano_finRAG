# NF-V2-17A5 Corpus Quality Freeze

A4/A4-R1 outputs were consumed without modifying raw, normalized, or parsed data.

- Searchable records: 95154 (input chunks 95506; empty records excluded 352).
- FTS5/BM25: built at `/mnt/disk/mxf/projects/Qhhhhhhaaa/financial_corpus_v2/indexes/financial-corpus-v2/bm25/index.sqlite`.
- Dense: built with the frozen all-MiniLM-L6-v2 configuration. Hybrid uses fixed RRF k=60 and no tuning.
- Provenance: 100%; orphan index entries: 0.
- Fresh-blind reservation: 12 documents, no questions or Gold.
- Decision: **CORPUS_V2_FROZEN_AND_INDEXED**.

UNKNOWN period annotations remain candidate-only and cannot satisfy explicit period hard filters. Production V1 indices were not modified.
