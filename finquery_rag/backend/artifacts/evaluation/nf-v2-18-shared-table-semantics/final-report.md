# NF-V2-18A-R2 Shared Table Semantics Recovery

Base: 2c281727154b8402df226d83623de7e798e48339

The frozen SEC HTML/iXBRL parser was not changed. A physical HTML adapter converges into the existing semantic graph and preserves canonical TABLE_ROW IDs.

## Frozen S0 reference
- Historical NF-V2-18A A4: R@1 33/120, R@3 55/120, R@5 62/120, R@10 68/120, R@20 77/120; family R@5 84/120 and R@10 86/120.
- This run also records a BM25-only cached replay because CUDA was unavailable; it is not substituted for the historical S0 reference.

## Selected configuration
- Stage: S5
- Config SHA: f75b43afddd49b5d7e8ee797ef1869f93fb2d8e3e573495d6d79fb210143186b
- Exact answerable R@5: 54/105
- Exact answerable R@10: 59/105
- Exact answerable R@20: 68/105
- Family R@5/R@10/R@20/R@30: 54/59/68/78 out of 120.
- Multi Any@5/All@5/All@10: 18/6/9 out of 20.
- Calculation Operand Complete@5/@10: 3/7 out of 15.

## Shared semantic reuse
- Documents/tables/rows mapped: 60/7198/84723 canonical; physical tables/rows 7870/86624.
- Existing ID reuse folded 672 table and 1901 row occurrences; no new identity namespace was introduced.
- AtomicFacts: 19965
- Deterministic iXBRL enrichments: 1216
- Existing LogicalTable, SemanticRow, MetricPath, SemanticAxisBinding, AtomicFact and Binder contracts reused.
- No FinancialColumnContextV1 or FinancialCellEvidenceV1 created.

## Ablation summary
- S0 (BM25 cached replay; frozen A4 reference above): exact R@5 49/120; answerable R@5 49/105; answerable R@10 57/105; Multi All@10 9/20; Calc operand@10 7/15.
- S1 (adapter-only replay): exact R@5 49/120; answerable R@5 49/105; answerable R@10 57/105; Multi All@10 9/20; Calc operand@10 7/15.
- S2: exact R@5 53/120; answerable R@5 53/105; answerable R@10 67/105; Multi All@10 6/20; Calc operand@10 5/15.
- S3: exact R@5 33/120; answerable R@5 33/105; answerable R@10 53/105; Multi All@10 0/20; Calc operand@10 3/15.
- S4: exact R@5 33/120; answerable R@5 33/105; answerable R@10 53/105; Multi All@10 0/20; Calc operand@10 3/15.
- S5: exact R@5 54/120; answerable R@5 54/105; answerable R@10 59/105; Multi All@10 9/20; Calc operand@10 7/15.
- S6: exact R@5 54/120; answerable R@5 54/105; answerable R@10 59/105; Multi All@10 9/20; Calc operand@10 7/15.
- S7: exact R@5 54/120; answerable R@5 54/105; answerable R@10 59/105; Multi All@10 9/20; Calc operand@10 7/15.

## Safety and scope
- Wrong-period binding admitted: 0.
- Hard-scope violations: 0.
- False binding: 0.
- Production remains V1; no generator, validator, calculator, or production configuration changes.

## Decision
- SHARED_TABLE_SEMANTICS_PARTIALLY_RECOVERED
