{
  "decision": {
    "base_commit": "621047d91fd34a7e231607993ec2915d4a03beff",
    "binder_fact_view_v2_frozen": true,
    "binder_model_frozen": true,
    "binder_semantic_policy_frozen": true,
    "current_bound": "0/56",
    "eligible_direct": "8/56",
    "false_binding": 0,
    "gate": "NF-V2-03-R7.1",
    "model_calls": 0,
    "next_gate": "v2_03_selective_admission_contract_fix",
    "production_default": "V1",
    "production_switch_allowed": false,
    "selective_admission_overconservative": true,
    "zero_coverage_justified": false
  },
  "gate": "NF-V2-03-R7.1",
  "model_calls": 0,
  "prediction_sha256": "f5b1f35875b0d4b4721cce5697f7aa654532491c748fa4a813c38dc1df98d543",
  "summary": "R7.1 found that zero released coverage is a workflow/admission-proof artifact, not evidence that every eligible fact is unsafe. The sealed R6 run made no comparative selections. Six of the eight structurally eligible direct queries were strict-correct BOUND in sealed Attempt 6; two were MISSING. A generic structural + FactView-visible-unique + one-candidate shortlist variant gives 2/56 at 100% precision and zero false binding.",
  "v2_04_not_started": true
}
