# Future grounded-v3 integration

1. Merge `FinancialGenerationViewV1` from the Grounding branch.
2. Register a `local_financial` provider through `GeneratorProviderV1`.
3. Configure the grounded-v3 checkpoint without changing `TrustedRAGRuntimeV2`.
4. Run smoke and frozen V2-06 component evaluation.
5. Select route policy, then run V2-07 validator/fallback.
6. Execute the sealed V2-08 evaluation.

No runtime redesign is required.
