{
  "artifacts": {
    "r6_policy_sha256": "76387ba06c11f87773cb56a4cee9822d6dab2fb2f40dc8ebfb0460866f886d12",
    "r6_shortlist_seal_sha256": "412d10bb4d9498dfafdc2baf6235c2f412b1900b3c74535d47ab733404877603"
  },
  "decision": {
    "base_commit": "fbc2335555904cf0ce929b94b929ae2fe3f0f95d",
    "binder_admission": "SelectiveBindingAdmissionV1",
    "binder_fact_view": "BinderFactViewV2",
    "binder_fact_view_v2_frozen": true,
    "binder_model": "qwen3.7-plus",
    "binder_model_frozen": true,
    "binder_safety_policy": "fail_closed",
    "binder_semantic_policy_frozen": true,
    "dominant_conclusion": "selective_fail_closed_binding",
    "false_binding_under_selective_admission": 0,
    "gate": "NF-V2-03-R7",
    "model_calls": 0,
    "next_gate": "v2_04_missing_evidence_supply_repair",
    "nf_v2_03_closed": true,
    "production_default": "V1",
    "production_switch_allowed": false,
    "semantic_accuracy_claim": false
  },
  "gate": "NF-V2-03-R7",
  "interpretation": [
    "BinderFactViewV2 improved source-derived distinguishability.",
    "The larger Qwen ablation did not improve Binder accuracy.",
    "Global, slot-wise, pairwise, and shortlist formulations did not justify unsafe coverage maximization.",
    "Missing evidence is delegated to V2-04 evidence repair."
  ],
  "summary": "NF-V2-03 is closed with SelectiveBindingAdmissionV1. No model calls were made; the sealed R6 shortlist policy released no BOUND queries, so offline replay reports zero coverage and zero false binding by construction."
}
