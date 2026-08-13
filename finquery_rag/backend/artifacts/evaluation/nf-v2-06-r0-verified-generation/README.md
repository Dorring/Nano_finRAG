{
  "base_commit": "984ed8d0e3bae9c63aed593202898693e592dc87",
  "excluded_initial_rows_from_corrected_tier_partition": 6,
  "gate": "NF-V2-06-R0",
  "models": {
    "financial": "finquery-finance-v2-lr010-150",
    "general": "qwen3.7-plus"
  },
  "note": "Tier B is component-level generation-only evaluation and must not be reported as fresh-blind E2E.",
  "reference_answer_inputs": false,
  "scored_model_calls_per_model": 68,
  "semantic_retries": 0,
  "tiers": {
    "A": "runtime_trusted_v2",
    "B": "oracle_verified_generation_only"
  }
}
