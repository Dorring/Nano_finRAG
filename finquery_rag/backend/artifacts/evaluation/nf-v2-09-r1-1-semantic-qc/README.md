# NF-V2-09 R1.1 Semantic QC

The previous R2 output was audited and preserved under `semantic_qc_rejected_pretrain`. The resealed dataset uses only deterministic TRAIN-only construction, derives calculation periods from the question, preserves requested output semantics, and emits question-directed `[C1]` targets. No training, model calls, or retrieval calls were made.
