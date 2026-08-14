# NF-V2-09 R1 Targeted Grounding Dataset R2

Base: `ea4d0e3cb0e8a3009ae3fbdb8a9c87cf4ae22220`. This gate is model-free: model calls, training, and retrieval calls are all zero.

## Mixture

- Targeted: 1,400 (500 direct numeric, 500 calculation no-recompute, 200 scope/period near-match, 150 extra-claim suppression, 50 partial answer without over-refusal).
- R1 replay: 350 (220 positive, 80 partial/distractor, 50 unanswerable).
- Final train mix: 1,750.

## Sources and contract

Targeted rows use only FinQA TRAIN records in this build (TAT-DQA TRAIN is an approved source but was not required; ConvFinQA TRAIN was unavailable). Replay rows come only from the existing R1 TRAIN file. No Tier-B questions, contexts, answers, or failure examples seed a row. The frozen FinancialGenerationViewV1 SHA matches: `943decf288dffb99ffa6f196abc44e0a5bdb226350cede40e0a160c4bd61f6e4`.

## Safety checks

All accepted rows use the frozen `[E#]`/`[C1]` citation namespace, have machine-auditable numeric and period support, contain no CoT/think targets, and fit the 4,096-token context limit. Every calculation row is verified by deterministic replay of its FinQA TRAIN program, has at least two distractor evidence rows, and copies only the canonical `[C1]` result without explicit arithmetic.

Next gate: `v2_09_r2_targeted_grounding_training`.
