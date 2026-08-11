# NF-OPT-21 R0 — Qwen/BM25 Top10 Late-Fusion Shadow Test

This is a development-shadow, Gold-blind prediction audit.  LRRF-V1 and PLRF-V1 consume only the sealed Qwen Top10 and frozen BM25 ranks; they do not run retrieval, models, training, or production changes. Both prediction files are sealed before post-seal strict/semantic diagnostics are loaded.

RRF_K = 60

Decision: `marginal`; selected variant: `lrrf_v1`; next gate: `nf_opt_21_r1_top10_listwise_selector`.
