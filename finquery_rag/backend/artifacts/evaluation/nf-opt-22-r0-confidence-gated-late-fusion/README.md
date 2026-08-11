# NF-OPT-22 R0 — Confidence-Gated Qwen/BM25 Late Fusion

This development-shadow audit consumes only the sealed Qwen Top100 and BM25 ranks. Prediction generation is Gold-blind; strict, semantic, multi-evidence, and calculation diagnostics are loaded after the prediction seal.

Near-boundary threshold: `0.035075027495622635`

RRF_K: `60`; selected shadow method: `lrrf_v1`; effectiveness: `False`.

Production switch allowed: `false`.
