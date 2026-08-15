# Interview evidence (NF-V2-11 freeze)

## What was built

A single Supervisor controls intent, route, retrieval/query rewrite, and evidence sufficiency. A route-specific trusted-evidence layer gates the deterministic Calculator and grounded Financial Specialist. The deterministic validator then releases, repairs once, falls back, or abstains.

## What the experiments established

Grounding alignment improved component grounding, numeric fidelity, and canonical calculation verbalization. The frozen R2.3.1 component results were 47/64 Grounded, 52/64 Numeric, 7/11 canonical calculations, and 5/5 explicit Multi grounded. These used oracle/trusted evidence and must be described as component evaluation.

Repeated targeted training did not make the Financial Specialist a dependable answerability judge. Evidence sufficiency and release safety therefore remain external to the model. In the final 72-question E2E run, only 4 trusted DIRECT packets reached generation; 68/72 queries failed closed, all 8 no-answer cases were refused safely, and no false execution or false binding occurred. One post-hoc semantic unit error remained, so the safety gate did not pass.

## How to state the outcome

The correct conclusion is not "the model solved the benchmark." It is: grounded generation became materially better as a component, while retrieval coverage and semantic sufficiency remain the limiting production constraints. V1 remains production; V2 is frozen research evidence.
