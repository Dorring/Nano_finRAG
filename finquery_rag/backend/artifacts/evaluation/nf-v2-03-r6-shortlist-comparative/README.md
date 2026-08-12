{
  "decision": {
    "base_commit": "bda77108dfaf545854041584336c8ea86767aac5",
    "binder_formulation_effective": false,
    "binder_model_frozen": "qwen3.7-plus",
    "binder_task_formulation_frozen": false,
    "diagnostic": "not_run",
    "dominant_failure": "shortlist_recall_failure",
    "formal_attempt_9": "not_run",
    "formulation": "deterministic_shortlist_comparative_v1",
    "gate": "NF-V2-03-R6",
    "gold_reads_before_prediction_seal": 0,
    "indistinguishable_safe": "not_run",
    "model": "qwen3.7-plus",
    "model_calls": 0,
    "next_gate": "v2_03_selective_binder_freeze_review",
    "production_default": "V1",
    "production_switch_allowed": false,
    "provider_calls_per_query": 0,
    "shortlist_calculation_retention": "11/12",
    "shortlist_direct_retention": "19/21",
    "shortlist_distribution": {
      "0": 0,
      "1": 8,
      "2": 4,
      "3": 4,
      "4": 7,
      "5": 24
    },
    "shortlist_gate_thresholds": {
      "calculation": "12/12",
      "direct": "20/21"
    },
    "shortlist_hard_gate": false,
    "shortlist_mean": 3.74468085106383,
    "shortlist_median": 5,
    "shortlist_p95": 5.0,
    "shortlist_unbindable_zero_eligible": "0/7",
    "structural_violations": "not_run",
    "synthetic": "not_run",
    "token_delta": "not_run",
    "unbindable_false_binding": "not_run"
  },
  "external_model_review": "cancelled_by_design",
  "gate": "NF-V2-03 R6",
  "reason": "shortlist hard gate failed before provider calls",
  "selective_freeze_policy": "selective-freeze-policy.json",
  "shortlist_audit": {
    "calculation": {
      "gold_compatible_candidate_retained": 11,
      "operand_slots": 12,
      "rows": [
        {
          "correct_candidate_rank": 1,
          "gold_handles": [
            "F01",
            "F02",
            "F03"
          ],
          "question_id": "aapl_fy2025_006",
          "retained_gold_handles": [
            "F03"
          ],
          "shortlist_size": 5,
          "slot_id": "slot_1"
        },
        {
          "correct_candidate_rank": 2,
          "gold_handles": [
            "F01",
            "F02",
            "F03"
          ],
          "question_id": "aapl_fy2025_006",
          "retained_gold_handles": [
            "F01"
          ],
          "shortlist_size": 5,
          "slot_id": "slot_2"
        },
        {
          "correct_candidate_rank": 5,
          "gold_handles": [
            "F10",
            "F11"
          ],
          "question_id": "jpm_fy2025_006",
          "retained_gold_handles": [
            "F10"
          ],
          "shortlist_size": 5,
          "slot_id": "1"
        },
        {
          "correct_candidate_rank": 5,
          "gold_handles": [
            "F10",
            "F11"
          ],
          "question_id": "jpm_fy2025_006",
          "retained_gold_handles": [
            "F11"
          ],
          "shortlist_size": 5,
          "slot_id": "2"
        },
        {
          "correct_candidate_rank": 3,
          "gold_handles": [
            "F07",
            "F08",
            "F09"
          ],
          "question_id": "ko_fy2025_006",
          "retained_gold_handles": [
            "F08"
          ],
          "shortlist_size": 5,
          "slot_id": "slot_1"
        },
        {
          "correct_candidate_rank": 3,
          "gold_handles": [
            "F07",
            "F08",
            "F09"
          ],
          "question_id": "ko_fy2025_006",
          "retained_gold_handles": [
            "F09"
          ],
          "shortlist_size": 5,
          "slot_id": "slot_2"
        },
        {
          "correct_candidate_rank": 1,
          "gold_handles": [
            "F01",
            "F02",
            "F03"
          ],
          "question_id": "pfe_fy2024_006",
          "retained_gold_handles": [
            "F01"
          ],
          "shortlist_size": 5,
          "slot_id": "1"
        },
        {
          "correct_candidate_rank": 1,
          "gold_handles": [
            "F01",
            "F02",
            "F03"
          ],
          "question_id": "pfe_fy2024_006",
          "retained_gold_handles": [
            "F03"
          ],
          "shortlist_size": 4,
          "slot_id": "2"
        },
        {
          "correct_candidate_rank": 2,
          "gold_handles": [
            "F04",
            "F05",
            "F06"
          ],
          "question_id": "tsla_fy2025_006",
          "retained_gold_handles": [
            "F04"
          ],
          "shortlist_size": 5,
          "slot_id": "rev_fy2025"
        },
        {
          "correct_candidate_rank": 2,
          "gold_handles": [
            "F04",
            "F05",
            "F06"
          ],
          "question_id": "tsla_fy2025_006",
          "retained_gold_handles": [
            "F05"
          ],
          "shortlist_size": 5,
          "slot_id": "rev_fy2024"
        },
        {
          "correct_candidate_rank": 1,
          "gold_handles": [
            "F02",
            "F03",
            "F04"
          ],
          "question_id": "v_fy2025_006",
          "retained_gold_handles": [
            "F02"
          ],
          "shortlist_size": 1,
          "slot_id": "s1"
        },
        {
          "correct_candidate_rank": null,
          "gold_handles": [],
          "question_id": "v_fy2025_006",
          "retained_gold_handles": [],
          "shortlist_size": 1,
          "slot_id": "s2"
        }
      ]
    },
    "correct_candidate_rank": {
      "Top1": 18,
      "Top2": 26,
      "Top3": 29,
      "Top5": 37
    },
    "direct": {
      "gold_compatible_candidate_retained": 19,
      "questions": 21,
      "rows": [
        {
          "correct_candidate_rank": 2,
          "gold_handles": [
            "F04",
            "F05",
            "F06"
          ],
          "question_id": "aapl_fy2025_001",
          "retained_gold_handles": [
            "F06"
          ],
          "shortlist_size": 5,
          "slot_id": "slot_1"
        },
        {
          "correct_candidate_rank": null,
          "gold_handles": [
            "F10",
            "F11",
            "F12"
          ],
          "question_id": "aapl_fy2025_002",
          "retained_gold_handles": [],
          "shortlist_size": 5,
          "slot_id": "1"
        },
        {
          "correct_candidate_rank": 4,
          "gold_handles": [
            "F01",
            "F02",
            "F03"
          ],
          "question_id": "aapl_fy2025_004",
          "retained_gold_handles": [
            "F03"
          ],
          "shortlist_size": 5,
          "slot_id": "1"
        },
        {
          "correct_candidate_rank": 4,
          "gold_handles": [
            "F01",
            "F02",
            "F03"
          ],
          "question_id": "aapl_fy2025_009",
          "retained_gold_handles": [
            "F02"
          ],
          "shortlist_size": 5,
          "slot_id": "1"
        },
        {
          "correct_candidate_rank": 5,
          "gold_handles": [
            "F19",
            "F20"
          ],
          "question_id": "jpm_fy2025_001",
          "retained_gold_handles": [
            "F19"
          ],
          "shortlist_size": 5,
          "slot_id": "1"
        },
        {
          "correct_candidate_rank": 1,
          "gold_handles": [
            "F01",
            "F02"
          ],
          "question_id": "jpm_fy2025_002",
          "retained_gold_handles": [
            "F01"
          ],
          "shortlist_size": 3,
          "slot_id": "1"
        },
        {
          "correct_candidate_rank": 4,
          "gold_handles": [
            "F13",
            "F14",
            "F15"
          ],
          "question_id": "jpm_fy2025_003",
          "retained_gold_handles": [
            "F14"
          ],
          "shortlist_size": 5,
          "slot_id": "slot_1"
        },
        {
          "correct_candidate_rank": 5,
          "gold_handles": [
            "F31",
            "F32",
            "F33"
          ],
          "question_id": "jpm_fy2025_004",
          "retained_gold_handles": [
            "F33"
          ],
          "shortlist_size": 5,
          "slot_id": "slot_1"
        },
        {
          "correct_candidate_rank": null,
          "gold_handles": [
            "F34",
            "F35",
            "F36"
          ],
          "question_id": "jpm_fy2025_009",
          "retained_gold_handles": [],
          "shortlist_size": 5,
          "slot_id": "1"
        },
        {
          "correct_candidate_rank": 1,
          "gold_handles": [
            "F01",
            "F02",
            "F03"
          ],
          "question_id": "ko_fy2025_001",
          "retained_gold_handles": [
            "F02"
          ],
          "shortlist_size": 4,
          "slot_id": "slot_1"
        },
        {
          "correct_candidate_rank": 1,
          "gold_handles": [
            "F01",
            "F02",
            "F03"
          ],
          "question_id": "ko_fy2025_002",
          "retained_gold_handles": [
            "F01"
          ],
          "shortlist_size": 4,
          "slot_id": "1"
        },
        {
          "correct_candidate_rank": 2,
          "gold_handles": [
            "F01",
            "F02",
            "F03"
          ],
          "question_id": "ko_fy2025_003",
          "retained_gold_handles": [
            "F02"
          ],
          "shortlist_size": 4,
          "slot_id": "slot_1"
        },
        {
          "correct_candidate_rank": 5,
          "gold_handles": [
            "F04",
            "F05",
            "F06"
          ],
          "question_id": "msft_fy2025_001",
          "retained_gold_handles": [
            "F05"
          ],
          "shortlist_size": 5,
          "slot_id": "1"
        },
        {
          "correct_candidate_rank": 3,
          "gold_handles": [
            "F01",
            "F02",
            "F03"
          ],
          "question_id": "msft_fy2025_003",
          "retained_gold_handles": [
            "F02"
          ],
          "shortlist_size": 3,
          "slot_id": "1"
        },
        {
          "correct_candidate_rank": 2,
          "gold_handles": [
            "F01",
            "F02",
            "F03"
          ],
          "question_id": "nvda_fy2025_002",
          "retained_gold_handles": [
            "F02"
          ],
          "shortlist_size": 2,
          "slot_id": "1"
        },
        {
          "correct_candidate_rank": 1,
          "gold_handles": [
            "F01",
            "F02"
          ],
          "question_id": "nvda_fy2025_003",
          "retained_gold_handles": [
            "F01"
          ],
          "shortlist_size": 2,
          "slot_id": "slot_1"
        },
        {
          "correct_candidate_rank": 1,
          "gold_handles": [
            "F01",
            "F02",
            "F03"
          ],
          "question_id": "nvda_fy2025_004",
          "retained_gold_handles": [
            "F02"
          ],
          "shortlist_size": 1,
          "slot_id": "data_center_revenue_fy2025"
        },
        {
          "correct_candidate_rank": 1,
          "gold_handles": [
            "F01",
            "F02",
            "F03"
          ],
          "question_id": "nvda_fy2025_005",
          "retained_gold_handles": [
            "F01"
          ],
          "shortlist_size": 1,
          "slot_id": "automotive_revenue_fy2025"
        },
        {
          "correct_candidate_rank": 1,
          "gold_handles": [
            "F01",
            "F02",
            "F03"
          ],
          "question_id": "nvda_fy2025_009",
          "retained_gold_handles": [
            "F03"
          ],
          "shortlist_size": 5,
          "slot_id": "1"
        },
        {
          "correct_candidate_rank": 2,
          "gold_handles": [
            "F04",
            "F05",
            "F06"
          ],
          "question_id": "pfe_fy2024_001",
          "retained_gold_handles": [
            "F04"
          ],
          "shortlist_size": 3,
          "slot_id": "total_revenues_fy2024"
        },
        {
          "correct_candidate_rank": 2,
          "gold_handles": [
            "F05",
            "F06",
            "F07"
          ],
          "question_id": "tsla_fy2025_001",
          "retained_gold_handles": [
            "F05"
          ],
          "shortlist_size": 5,
          "slot_id": "1"
        }
      ]
    },
    "gold_reads_after_shortlist_seal": true,
    "hard_gate": false,
    "indistinguishable": {
      "plausible_candidates_retained": 2,
      "questions": 6,
      "rows": [
        {
          "plausible_candidates_retained": true,
          "question_id": "aapl_fy2025_003",
          "shortlist_size": 5
        },
        {
          "plausible_candidates_retained": false,
          "question_id": "v_fy2025_001",
          "shortlist_size": 1
        },
        {
          "plausible_candidates_retained": false,
          "question_id": "v_fy2025_002",
          "shortlist_size": 1
        },
        {
          "plausible_candidates_retained": false,
          "question_id": "v_fy2025_003",
          "shortlist_size": 1
        },
        {
          "plausible_candidates_retained": false,
          "question_id": "v_fy2025_004",
          "shortlist_size": 1
        },
        {
          "plausible_candidates_retained": true,
          "question_id": "v_fy2025_009",
          "shortlist_size": 3
        }
      ]
    },
    "shortlist_size": {
      "distribution": {
        "0": 0,
        "1": 8,
        "2": 4,
        "3": 4,
        "4": 7,
        "5": 24
      },
      "max": 5,
      "mean": 3.74468085106383,
      "median": 5,
      "p95": 5.0
    },
    "unbindable": {
      "questions": 7,
      "rows": [
        {
          "question_id": "jpm_fy2025_005",
          "zero_eligible": false
        },
        {
          "question_id": "msft_fy2025_002",
          "zero_eligible": false
        },
        {
          "question_id": "nvda_fy2025_007",
          "zero_eligible": false
        },
        {
          "question_id": "pfe_fy2024_002",
          "zero_eligible": false
        },
        {
          "question_id": "pfe_fy2024_004",
          "zero_eligible": false
        },
        {
          "question_id": "v_fy2025_006",
          "zero_eligible": false
        },
        {
          "question_id": "v_fy2025_007",
          "zero_eligible": false
        }
      ],
      "zero_eligible": 0
    }
  }
}
