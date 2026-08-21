# NF-V2-18A Retrieval Recovery

This is a development-only retrieval experiment on the consumed NF-V2-17 120-question regression set. The original B3 fresh-blind outputs and scores are immutable.

The run evaluates:
- A0 frozen B3 retrieval baseline
- A1 BM25 query repair
- A2 enriched TABLE_ROW serialization
- A3 row-level all-MiniLM-L6-v2 dense candidates
- A4 deterministic TABLE to TABLE_ROW expansion
- A5 structured iXBRL candidate union
- A6 the same candidate union with frozen RRF configuration

Selected stage: A4.
Decision: RETRIEVAL_PARTIALLY_RECOVERED.
Exact R@5: 62/120.
Exact R@10: 68/120.
Multi Any@5: 17/20.
Multi All@10: 6/20.
Calculation operand coverage@10: 5/15.

All hard-scope regression counters are zero. No generator, supervisor, calculator, validator, production index, or B3 artifact was changed. The 120-question set is CONSUMED_DEVELOPMENT_REGRESSION and must not be called fresh-blind after this run.

The dense index is external to Git under the FinancialCorpusV2 index root; this repository stores only the reproducibility script and small evaluation artifacts.
