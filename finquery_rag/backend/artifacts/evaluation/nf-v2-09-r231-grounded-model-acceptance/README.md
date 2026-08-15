# NF-V2-09 R2.3.1 Final Grounded Model Acceptance

Candidate inquery-finance-grounded-v3-r231, frozen final checkpoint model_000004.pt, was evaluated without training, retrieval, prompt changes, decoding changes, Validator changes, or checkpoint sweep. Local inference was isolated to physical GPU 3 with CUDA_VISIBLE_DEVICES=3; the process saw one logical CUDA device mapped to physical GPU 3 (RTX A6000).

## Result

The model retains the R2 numeric improvement (52/64) and R2 calculation canonical-obedience level (7/11), but the refusal boundary remains broken: 0/15 unanswerable holdout cases were correctly refused. The original frozen gate fails (47/64 grounded, 17/64 reported unsupported, 52/64 numeric). Overall behavioral tradeoff is mixed.

Semantic adjudication is diagnostic only: 16/64 reported unsupported cases are true semantic unsupported and 1/64 is evaluator/contract-only. RuntimeGenerationValidatorV1 is unchanged.

The model role is inancial_selective_generator as an evaluation candidate only. MULTI_EVIDENCE is strong on the oracle component set (5/5), but no route is enabled for production from this component result alone; production remains unchanged. No further grounding training or R2.4 dataset is authorized by this gate. Next gate: final trusted E2E evaluation / project freeze.

All predictions and runtime decisions are sealed before any post-hoc reference scoring (
eference_reads_before_prediction_seal = 0). Tier-B is oracle-verified component evaluation, not end-to-end RAG performance.
