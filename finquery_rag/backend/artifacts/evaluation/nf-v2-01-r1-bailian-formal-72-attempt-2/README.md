# NF-V2-01 R1 Formal Bailian Supervisor — Attempt 2

This is the first complete, sealed semantic evaluation after Attempt 1 was invalidated by a runner integration failure. The runner used one sequential, temperature-zero, no-retry Bailian call per frozen question. Gold/reference annotations were opened only after the prediction artifact was sealed and verified.

Attempt 2 completed the 72-call transport and serialization pipeline. All provider responses, structured parses, schemas, and Plan Validator checks passed. The frozen semantic gate was rejected because metric slot accuracy was below the configured threshold; no prompt, model, evaluator, or downstream component was changed.

No retrieval, reranking, FinancialFact, binding, calculation, generation, validation, or repair executed. Production remains V1.
