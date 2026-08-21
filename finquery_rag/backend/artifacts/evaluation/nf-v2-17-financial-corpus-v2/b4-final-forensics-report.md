# NF-V2-17B4 Fresh-Blind Failure Forensics

Read-only audit. B3 output/trace/freeze hashes were verified; no benchmark rerun, model generation, retriever/index/filter/policy change, or rescoring was performed.

B3 runtime output SHA: 9e02df6701268e83cd9dafdcc36d95736167c19cca8d244a134a69149538dd83
B3 trace SHA: 3b3b9ee227a49631dcf4f1c820172fd63cf465cbb85514d67ce9d06d2d5f6ebd
Evaluation freeze SHA: c3648925f07e878123e78e0fed21b12e0499a461d2c83e28616cfe80789c920a
Answerable questions: 105; Gold evidence objects: 155

## Findings

Gold types: {'TABLE_ROW': 147, 'TABLE': 5, 'TEXT': 3}.
Exact searchable Gold objects: 145/155; structurally searchable by table/row identity: 145/155.
Exact BM25-indexed: 145/155; exact Dense-eligible: 2/155; structured fact IDs: 0, found: 0.
Hybrid top-10 classes: parent/child 67, alternative 0, true misses 26, equivalence-incomplete candidates 10.
Filter false exclusions: {}. Eight fact/document period annotation mismatches were reviewed separately and are not hard-filter exclusions.
Query classifications: {"QUERY_OK": 105}; the runtime used the original query text. The lexical defect is BM25 query overconstraint, not query-term deletion.

## Flat curve

Frozen B3 exact-ID scoring remains unchanged. R@1=R@3=R@5=R@10 is explained by exact Gold IDs not being returned or by parent/child objects being returned under different IDs. Exact first-iteration top-10 count: 2/105; BM25 raw matches: 0/105; Dense exact Gold hits: 2/105; RRF rescued: 0; raw candidate-generation count: 2/105.

## Metric semantics

Financial generator calls: 52; grounded conditional on generation: 12; semantic unsupported conditional on generation: 40; conditional sum equals calls: True. Overall B3 metrics remain grounded 12/105 and unsupported 40/105.

## Release correctness

Released: 12; released correct: 6; released-but-incorrect: 6; runtime unsafe release: 0. The six are validator-released but fail the frozen offline reference predicate, so release safety and reference completeness are distinct.

## Causal attribution

Terminal failures (reconciled against frozen B3 scorer): {'NO_PROGRESS_FAILURE': 38, 'OTHER': 6, 'REPLAN_FAILURE': 2, 'RETRIEVAL_MISS': 26, 'MULTI_EVIDENCE_INCOMPLETE': 12, 'CALC_OPERAND_FAILURE': 15}.
Root causes (reconciled against frozen B3 scorer): {'EVIDENCE_ID_ALIGNMENT_FAILURE': 51, 'INDEX_COVERAGE_FAILURE': 12, 'RETRIEVAL_IMPLEMENTATION_FAILURE': 21, 'CALC_OPERAND_FAILURE': 15}.
Primary conclusion: EVIDENCE_ID_ALIGNMENT_FAILURE.
Secondary causes: ['RETRIEVAL_IMPLEMENTATION_FAILURE', 'BM25_QUERY_OVERCONSTRAINT', 'CALC_OPERAND_FAILURE', 'INDEX_COVERAGE_FAILURE'].

No corrected scoring is warranted; equivalence/annotation items are proposed audit candidates only. The 120-question fresh-blind benchmark is consumed and is not eligible for tuning.
