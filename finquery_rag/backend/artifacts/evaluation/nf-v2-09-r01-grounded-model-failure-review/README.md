# NF-V2-09 R0.1 Grounded Model Failure Review

This is an offline, sealed-output attribution gate. No model, training, or retrieval calls were made. Official NF-V2-09 metrics remain unchanged; this directory separates evaluator/contract artifacts from true semantic failures.

## Frozen conclusion

Grounding Alignment R1 produced a real behavioral shift but did not pass the frozen acceptance gate. The candidate remains a selective DIRECT-only generator candidate. CALCULATION is not eligible because only 2/11 canonical calculation results were preserved; MULTI is not eligible because only 2/5 cases were grounded and citation-complete.

## R2 recommendation

A small targeted augmentation is justified: 1,400 new examples plus 350 controlled R1 replay examples (80/20), approximately a 0.5-epoch targeted alignment. The same 3,600-example second epoch is explicitly not recommended.

All runtime policies, validators, checkpoints, prompts, and frozen evaluation gates remain unchanged.
