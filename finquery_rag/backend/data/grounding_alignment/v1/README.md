# NF-V2-06 R1A Grounding Alignment Dataset V1

Base: `d871b339baaed0f15587a3ae67f156fe8632ff0c`

This is a grounding-pure, model-free behavioral alignment set. Only FinQA TRAIN and TAT-DQA TRAIN source evidence were used; no model calls, training run, internal benchmark answers, or old 39,801 SFT rows were used. The shared renderer is `FinancialGenerationViewV1`.

Accepted: 4000; train/dev/holdout: 3600/200/200.

The assistant target alone is loss-bearing. All citations are plain `[E#]`/`[C1]` IDs and every accepted row passes deterministic leakage, numeric, period, citation, negative, partial, and context-limit checks.
