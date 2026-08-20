# NF-V2-18A-R1 Fine Evidence Recovery

Development-only replay on the consumed NF-V2-17 B3 120-question regression set. B3 artifacts, generator, validator, calculator, and production V1 were not modified.

Selected configuration: **A4 coarse fallback**. Local fine/slot paths are diagnostic and were not promoted because they reduced global exact recall.
Decision: **FINE_EVIDENCE_PARTIALLY_RECOVERED**.

## Ablations

R0 A4 coarse: R@1 33/120; R@3 55/120; R@5 62/120; R@10 68/120; R@20 77/120; family R@5 84/120; Multi Any@5 17/20; Multi All@10 6/20; Calc operand@10 5/15
R1 local lexical: R@1 35/120; R@3 43/120; R@5 50/120; R@10 56/120; R@20 70/120; family R@5 64/120; Multi Any@5 17/20; Multi All@10 9/20; Calc operand@10 7/15
R2 local Qwen: UNAVAILABLE (frozen Qwen3-Reranker-4B weights/provider absent; no substitute used).
R3 slot-aware multi: R@1 30/120; R@3 38/120; R@5 48/120; R@10 56/120; R@20 70/120; family R@5 63/120; Multi Any@5 15/20; Multi All@10 9/20; Calc operand@10 7/15
R4 slot-aware calculation: R@1 27/120; R@3 35/120; R@5 43/120; R@10 53/120; R@20 69/120; family R@5 60/120; Multi Any@5 15/20; Multi All@10 9/20; Calc operand@10 5/15
R5 targeted structured audit: R@1 27/120; R@3 35/120; R@5 43/120; R@10 53/120; R@20 69/120; family R@5 60/120; Multi Any@5 15/20; Multi All@10 9/20; Calc operand@10 5/15

## Fine failure audit

Family-hit/exact-miss cases: 22
{"CHILD_RANK_FAILURE": 3, "MULTI_LEVEL_HEADER": 14, "PERIOD_COLUMN_MISMATCH": 5}

## Targeted gains

R1/R3 Multi All@10: 9/20 and 9/20 versus R0 6/20.
R1 calculation operand@10: 7/15 versus R0 5/15; R4: 5/15.
These gains are not sufficient to offset exact-recall regression, so A4 remains selected.

## Safety and cost

Wrong-period binding admitted: 0; authorization/entity/fiscal/document/version violations: 0; silent relaxation: 0.
Latency: {"cpu_fallback": true, "gpu_available": false, "notes": "Dense query vectors used the existing all-MiniLM-L6-v2 index; no generator or reranker calls were made.", "retrieval_latency": {"bm25_ms": {"count": 120, "max_ms": 43107.02785698231, "mean_ms": 1370.9520036859128, "p50_ms": 460.25224297773093, "p95_ms": 3266.069508041255}, "coarse_ms": {"count": 120, "max_ms": 44606.216983054765, "mean_ms": 2257.475030865559, "p50_ms": 1433.1610250519589, "p95_ms": 4193.822645000182}, "dense_ms": {"count": 120, "max_ms": 321.3780000805855, "mean_ms": 58.1428883849488, "p50_ms": 37.18432004097849, "p95_ms": 171.84011300560087}, "expand_ms": {"count": 120, "max_ms": 3828.941676998511, "mean_ms": 827.1938231870687, "p50_ms": 771.3546050945297, "p95_ms": 1614.3460370367393}, "index_size_bytes": 2539401084, "local_fine_ms": {"count": 120, "max_ms": 1822.748031001538, "mean_ms": 367.0961703173816, "p50_ms": 201.11241401173174, "p95_ms": 1340.4326559975743}, "slot_fanout_ms": {"count": 35, "max_ms": 1717.4351600697264, "mean_ms": 740.0003011138843, "p50_ms": 648.981882026419, "p95_ms": 1672.5481989560649}}}

Qwen local reranking was not run because frozen 4B weights/provider were unavailable. No generator call, training, or benchmark-specific tuning was performed.
