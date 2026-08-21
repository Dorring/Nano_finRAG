# NF-V2-15 final trusted E2E

This is a model-free replay of the sealed NF-V2-10 72-question run. The four historical generator envelopes are passed through the adopted post-generation SemanticClaimVerifierV1 and RuntimeGenerationValidatorV1. All other rows retain their frozen fail-closed evidence state.

The historical one unsafe release is retained as a post-hoc comparison; the new runtime blocks it before release. Production remains V1.
