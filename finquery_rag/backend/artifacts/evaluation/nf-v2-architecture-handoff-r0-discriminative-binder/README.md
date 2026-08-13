{
  "decision": {
    "base_commit": "acc08e057153768e1d3a251062041975a201f077",
    "discriminative_binder_effective": false,
    "discriminative_binder_feasible": false,
    "evidence_binding_architecture": "not_adopted",
    "gate": "NF-V2-ARCHITECTURE-HANDOFF-R0",
    "generative_binder_calls": 0,
    "generative_binder_retired": false,
    "next_gate": "v2_architecture_scope_freeze",
    "production_default": "V1",
    "production_switch_allowed": false,
    "retrieval_calls": 0,
    "revision": "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
    "scorer": "Qwen/Qwen3-Reranker-4B",
    "selected_margin": null,
    "selected_threshold": null,
    "serialization_sha256": "d4b3f52ec0fed17881199a89abac1dd470716698557274bdae20e964507a391c",
    "stage_a": {
      "calculation_rank_at_1": "6/12",
      "direct_rank_at_1": "12/21",
      "ranking_feasible": false,
      "useful_score_separation": false
    },
    "stage_b_executed": false,
    "supervisor_role": "General LLM Supervisor remains control plane; reranker is a trusted tool-plane scorer"
  },
  "gate": "NF-V2-ARCHITECTURE-HANDOFF-R0",
  "runtime": {
    "average_query_latency_ms": 7769.196458333333,
    "max_query_latency_ms": 43980.154,
    "output_tokens": 0,
    "p50_query_latency_ms": 6026.6535,
    "p95_query_latency_ms": 20916.16,
    "total_input_tokens": 1528379,
    "total_pairs": 1993,
    "wall_time_ms": 559384.613
  },
  "summary": "Qwen3-Reranker-4B was evaluated as a downstream RequiredSlot-to-BinderFactViewV2 compatibility scorer. Serialization and scores were sealed before review labels; no generative Binder or retrieval path was called."
}
