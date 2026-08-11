# NF-OPT-18 R0 鈥?Internal Reranker Representation Audit

Pure post-seal/offline audit. Model execution and retrieval rerun are false.

- Strict physical bindings: 80
- Top100 present: 68
- Qwen Top5: 43
- C1 Top100-present/Top5-miss: 25
- C2 Top100-absent: 12
- C0 high ambiguity: 0.4651
- C1 high ambiguity: 0.4000
- Structure available but not serialized (overall/C1): 1059/7

Decision: `representation_gap_supported=False`, `evidence_packet_v1_allowed=False`. Next gate: `nf_opt_18_method_reconsideration`. Production switch remains false.

Strict source identity is used for all cohorts. Semantic Fact identity is diagnostic only and never replaces physical Gold.
