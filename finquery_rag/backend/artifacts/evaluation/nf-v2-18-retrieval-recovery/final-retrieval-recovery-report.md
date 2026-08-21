# NF-V2-18A Retrieval Recovery Sprint

Development-only evaluation on the consumed NF-V2-17 120-question regression set; B3 artifacts were not modified.

Selected stage: **A4**; decision: **RETRIEVAL_PARTIALLY_RECOVERED**.

## Ablations

- A0: R@1 2/120; R@3 2/120; R@5 2/120; R@10 2/120; R@20 2/120; family R@5 56/120; Multi Any@5 0/20; Multi All@10 0/20; calculation operand@10 0/15.
- A1: R@1 19/120; R@3 54/120; R@5 61/120; R@10 71/120; R@20 80/120; family R@5 86/120; Multi Any@5 18/20; Multi All@10 6/20; calculation operand@10 5/15.
- A2: R@1 30/120; R@3 52/120; R@5 59/120; R@10 69/120; R@20 77/120; family R@5 84/120; Multi Any@5 19/20; Multi All@10 6/20; calculation operand@10 5/15.
- A3: R@1 27/120; R@3 39/120; R@5 59/120; R@10 66/120; R@20 74/120; family R@5 80/120; Multi Any@5 16/20; Multi All@10 6/20; calculation operand@10 2/15.
- A4: R@1 33/120; R@3 55/120; R@5 62/120; R@10 68/120; R@20 77/120; family R@5 84/120; Multi Any@5 17/20; Multi All@10 6/20; calculation operand@10 5/15.
- A5: R@1 33/120; R@3 47/120; R@5 55/120; R@10 62/120; R@20 68/120; family R@5 81/120; Multi Any@5 16/20; Multi All@10 3/20; calculation operand@10 3/15.
- A6: R@1 33/120; R@3 47/120; R@5 55/120; R@10 62/120; R@20 68/120; family R@5 81/120; Multi Any@5 16/20; Multi All@10 3/20; calculation operand@10 3/15.

## Safety
{"authorization_leakage": 0, "created_at_misuse": 0, "document_type_violation": 0, "entity_violation": 0, "silent_relaxation": 0, "temporal_violation": 0, "version_violation": 0}

## Targets
{
  "targets": {
    "R@5": 0.7,
    "R@10": 0.8,
    "multi_any@5": 0.7,
    "multi_all@10": 0.5,
    "calculation_operand@10": 0.7
  },
  "actual": {
    "R@5": 0.5166666666666667,
    "R@10": 0.5666666666666667,
    "multi_any@5": 0.85,
    "multi_all@10": 0.3,
    "calculation_operand@10": 0.3333333333333333
  }
}

No generator, supervisor, validator, calculator, temporal policy, authorization policy, production index, or B3 output was modified.
