# NF-V2-18A-R3P0 Qwen3-Reranker Recovery

This artifact restores the exact historical Qwen3-Reranker-4B revision and validates it only with synthetic, non-benchmark smoke fixtures. GPU selection is read-only and dynamic; the selected physical GPU is mapped to logical `cuda:0` in a child process. No retrieval evaluation, training, or benchmark rerun is performed.
