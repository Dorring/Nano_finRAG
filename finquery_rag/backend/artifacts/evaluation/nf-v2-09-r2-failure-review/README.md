# NF-V2-09 R2 Failure Review

This is a sealed-output attribution audit. No training, retrieval, Tier-B intermediate checkpoint sweep, prompt change, validator change, or dataset rewrite was performed. A diagnostic inference pass evaluated only the 200-example alignment holdout at R2.2 steps 0-4 on physical GPU 3; it made 1,000 local holdout generations.

The refusal collapse occurs between step 1 and step 2 (14/15 -> 9/15 -> 0/15). The loader uses deterministic TaskMixture Random(42) global shuffling, so a global shuffle failure is not supported. Per-update class composition is close to the intended mix. The strongest supported cause is structural mismatch: the failed holdout cases are single-evidence, header-only Direct views with no visible metric value, while the hard-negative training bucket is dominated by multi-evidence examples (all 156 Direct negatives have at least two evidence rows) and also includes 74 calculation and 20 multi-evidence routes. Paired siblings are present, and refusal loss masking is correct.

The R2.2 checkpoint remains preserved as evaluation history. The bounded next action is one R2.3 targeted-data-repair experiment, not another identical R2.2 training run.
