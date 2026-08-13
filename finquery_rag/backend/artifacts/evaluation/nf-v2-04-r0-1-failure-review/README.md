{
  "decision": {
    "base_commit": "92cfb4a560f7ba56509be3316c13668a9739ef18",
    "calculation_supply_repairable": 5,
    "direct_supply_repairable": 19,
    "gate": "NF-V2-04-R0.1",
    "model_calls": 0,
    "next_gate": "v2_04_r1_targeted_supply_repair",
    "production_default": "V1",
    "production_switch_allowed": false,
    "r0_repair_policy_rejected": true,
    "r1_authorized": true,
    "r1_executed": false,
    "retrieval_calls": 0,
    "safety": {
      "admission_changed": false,
      "binder_changed": false,
      "diagnostic_multi_false_bindings": 2,
      "fabricated_financial_facts": 0,
      "false_binding_direct": 0,
      "false_operand_binding": 0,
      "financial_fact_v1_schema_modified": false,
      "gold_assisted_retrieval": 0,
      "gold_assisted_rewrite": 0,
      "question_specific_rules": 0,
      "repair_loops_over_one": 0
    },
    "v2_04_supply_repair_opportunity_insufficient": false
  },
  "gate": "NF-V2-04-R0.1",
  "summary": "Offline attribution of the sealed R0 repair. No model or retrieval calls were made. R0 added facts but produced no new safely bound Direct query; the policy is rejected and no R1 execution is authorized unless the supply-repairable threshold is met."
}
