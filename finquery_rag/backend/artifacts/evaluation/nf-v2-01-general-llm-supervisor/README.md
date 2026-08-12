# NF-V2-01 — General LLM Supervisor

This is a development-shadow, question-only Supervisor replay.  Each of the
72 frozen questions received at most one temperature-zero model call.  The
result was parsed as a strict `SupervisorPlan` and passed through the
deterministic V2-00 validator.  Retrieval, reranking, binding, calculation,
generation, validation, and production routing were not executed.  Gold and
reference annotations were opened only after the prediction artifact was
sealed.
