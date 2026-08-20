# NF-V2-17B3 One-Shot Fresh-Blind Trusted Agentic RAG Execution

This directory contains the sealed execution of the frozen NF-V2-17 fresh-blind
pack. Runtime execution used the Gold-free question file in question-file
order, the frozen A5 FinancialCorpusV2 indices, the fixed RRF k=60 contract,
the frozen single-supervisor policy, the deterministic calculator gate, the
frozen Financial Specialist checkpoint and RuntimeGenerationValidatorV1 /
SemanticClaimVerifierV1. No Gold, reference answer, annotation, or benchmark
label was read until output and trace SHA seals existed.

## Frozen inputs

- Corpus freeze SHA: 63620b2183c4635f1ecff974935bc81a4d8ce678c72e72e94155d8f0a96e6929
- Questions SHA: 06b1994034a425f749a7600d168bf7e34d5e2eaba544c75f5398f71cf7d26bb3
- Gold SHA: 1185bab1aa2923388c603bcf9f15f76a38e7472c5d48c2272af4c7b6138955ff
- Reference SHA: ae75c885f2304e6ca63f63891ed7be269b73cc2d7d99835fe368b665f26bd8ad
- Evaluation freeze SHA: c3648925f07e878123e78e0fed21b12e0499a461d2c83e28616cfe80789c920a
- Candidate checkpoint SHA: 2be1d02b2129661e1bad454fbbdddd2c5c12262a6facd4369a12612cd634d794

## Sealed run

Questions: 120 (105 answerable, 15 unanswerable); companies: GOOGL and AMZN.
The effective run used CUDA physical GPU 3 through the existing verl040
environment. Raw output SHA is 9e02df6701268e83cd9dafdcc36d95736167c19cca8d244a134a69149538dd83;
trace SHA is 3b3b9ee227a49631dcf4f1c820172fd63cf465cbb85514d67ce9d06d2d5f6ebd.
Results SHA and the final execution seal are in final-results.json and
fresh-blind-execution-seal.json.

As-run result: 6/105 answerable correct, 12/105 answerable releases, 6/12
released precision, 15/15 no-answer refusals, unsafe release 0, false binding 0,
false execution 0. Decision: FRESH_BLIND_RUNTIME_PARTIAL. Production remains V1
and no production switch was made.

R@1/R@3/R@5/R@10 were 2/120, 2/120, 2/120, and 2/120 under the exact frozen
Gold-evidence-ID metric; multi Any@5 and All@5 were 0/20 and 0/20. Calculation
operand readiness and strict execution were 0/15, so all calculation cases
failed closed before execution. These are blind evaluation results, not tuning
targets.

One first execution attempt is archived as INFRA_INVALID because the temporary
evaluation driver used unsanitized FTS syntax and could not import the dense
dependency in the CUDA environment. It was never scored. The valid execution
used safe query tokenization and query vectors produced by the same cached
all-MiniLM-L6-v2 model; no index, embedding identity, RRF value, model,
validator, or benchmark input changed. See b3-infrastructure-events.json.

Historical NF-V2-15 regression is kept outside this 120-question score:
safe retained 3/3 and unsafe blocked 1/1.

No post-evaluation tuning, Gold edits, question edits, retries of valid
questions, or production changes were performed.