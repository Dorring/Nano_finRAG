# TV2-07R1 canonical readiness dataset

This directory is reserved for the formal TV2-07R1 production-readiness set. It
is intentionally empty until a fresh, frozen case set is reviewed and sealed.

The wiring fixture at ../tv2_07_production_readiness/ is not an acceptable
substitute. The previously consumed nf-v2-17-fresh-blind-eval run is also
rejected by the R1 loader.

## Query JSONL

Each row is runtime input only:

~~~json
{
  "case_id": "r1-0001",
  "question": "What was Apple's FY2024 revenue?",
  "category": "direct_fact",
  "dataset_provenance": "fresh_company_held_out",
  "input_turns": []
}
~~~

For contextual cases, retain the user-visible turn sequence. Do not put a
resolved standalone query, Gold answer, expected release, evidence IDs, or
review labels in the query row:

~~~json
{
  "case_id": "r1-mt-0001",
  "question": "How much did it grow?",
  "category": "multi_turn_context",
  "dataset_provenance": "untouched_frozen_eval",
  "input_turns": [
    {"turn_id": "t1", "role": "user", "text": "Apple FY2024 revenue?"},
    {"turn_id": "t2", "role": "user", "text": "What about FY2023?"},
    {"turn_id": "t3", "role": "user", "text": "How much did it grow?"}
  ]
}
~~~

## Label JSONL

Labels are loaded only after blind execution and are used by the scorer:

~~~json
{
  "case_id": "r1-0001",
  "category": "direct_fact",
  "answerable": true,
  "expected_release": true,
  "expected_route": "STRUCTURED_SINGLE",
  "expected_evidence_ids": ["..."],
  "expected_citation_ids": ["..."],
  "expected_calculation": null,
  "expected_reason_codes": [],
  "required_answer_terms": [],
  "forbidden_answer_terms": [],
  "annotation": {
    "gold_period": "FY2024",
    "gold_unit": "USD",
    "gold_scale": "million"
  },
  "dataset_provenance": "fresh_company_held_out"
}
~~~

Gold rows must not be passed to either runtime. dataset_provenance must be
fresh_company_held_out or untouched_frozen_eval, and every query/label
category and provenance must match.

## Required strata

The preflight requires coverage for direct fact, multi-evidence, calculation,
qualitative synthesis, negative/no-answer, period/unit/scale traps, recovery or
repair, and at least one real multi-turn input. The formal target is at least
100 cases (preferably 122éÝyø§yÓ200), with provenance recorded per case.
