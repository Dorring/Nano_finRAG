{
  "diagnostic": {
    "base_commit": "a375175b84e3340ad6446f3d6bda8fa877fe54a6",
    "binder_model_frozen": "qwen3.7-plus",
    "binder_task_formulation_frozen": false,
    "diagnostic_all_operands": "0/6",
    "diagnostic_calculation": "1/12",
    "diagnostic_direct": "7/21",
    "dominant_failure": "pairwise_safety_failure",
    "formal_attempt_9": "not_run",
    "formulation": "batched_pairwise_compatibility_v1",
    "gate": "NF-V2-03-R5.1",
    "indistinguishable_abstention": "2/6",
    "model": "qwen3.7-plus",
    "next_gate": "v2_03_pairwise_safety_failure_review",
    "pairwise_binder_effective": false,
    "production_default": "V1",
    "production_switch_allowed": false,
    "provider_calls_per_query": 1,
    "raw_pairwise_labels": {
      "INDETERMINATE": 84,
      "MATCH": 86,
      "REJECT": 964
    },
    "structural_violations": 0,
    "synthetic": "13/14",
    "token_delta": {
      "max_input_tokens": -7,
      "mean_input_tokens": -7.0,
      "p50_input_tokens": -7,
      "p95_input_tokens": -7.0,
      "total_input_tokens": -273
    },
    "unbindable_false_binding": "6/7"
  },
  "external_model_review": "cancelled_by_design",
  "formal_attempt_9": "not_run",
  "gate": "NF-V2-03 R5.1 Pairwise Binder",
  "model": "qwen3.7-plus",
  "offline_review": {
    "failure": {
      "category_counts": {
        "SW0_correct": 17,
        "SW1_selected_nearest_metric_but_wrong_scope": 1,
        "SW2_selected_wrong_period": 0,
        "SW3_selected_wrong_statement": 13,
        "SW4_selected_parent_or_child_metric": 6,
        "SW5_selected_lexically_similar_fact": 3,
        "SW6_selected_one_of_indistinguishable_candidates": 6,
        "SW7_failed_to_select_unique_best_fact": 0,
        "SW8_missing_despite_unique_fact": 3,
        "SW9_other": 0
      },
      "cohorts": {
        "A_direct_visible_unique": 21,
        "B_calculation_visible_unique": 6,
        "C_indistinguishable": 6,
        "D_unbindable_safety": 7
      },
      "model_calls": 0,
      "rows": [
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW3_selected_wrong_statement",
          "question_id": "aapl_fy2025_001",
          "selected_handles": [
            "F03"
          ],
          "slot_id": "slot_1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW4_selected_parent_or_child_metric",
          "question_id": "aapl_fy2025_002",
          "selected_handles": [
            "F01"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW0_correct",
          "question_id": "aapl_fy2025_004",
          "selected_handles": [
            "F03",
            "F05"
          ],
          "slot_id": "1",
          "status": "AMBIGUOUS",
          "strict_correct": true
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW0_correct",
          "question_id": "aapl_fy2025_009",
          "selected_handles": [
            "F02"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": true
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW3_selected_wrong_statement",
          "question_id": "jpm_fy2025_001",
          "selected_handles": [
            "F01"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW0_correct",
          "question_id": "jpm_fy2025_002",
          "selected_handles": [
            "F01"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": true
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW3_selected_wrong_statement",
          "question_id": "jpm_fy2025_003",
          "selected_handles": [
            "F01"
          ],
          "slot_id": "slot_1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW8_missing_despite_unique_fact",
          "question_id": "jpm_fy2025_004",
          "selected_handles": [],
          "slot_id": "slot_1",
          "status": "MISSING",
          "strict_correct": false
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW8_missing_despite_unique_fact",
          "question_id": "jpm_fy2025_009",
          "selected_handles": [],
          "slot_id": "1",
          "status": "MISSING",
          "strict_correct": false
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW0_correct",
          "question_id": "ko_fy2025_001",
          "selected_handles": [
            "F02",
            "F05",
            "F07",
            "F11"
          ],
          "slot_id": "slot_1",
          "status": "AMBIGUOUS",
          "strict_correct": true
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW0_correct",
          "question_id": "ko_fy2025_002",
          "selected_handles": [
            "F01"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": true
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW0_correct",
          "question_id": "ko_fy2025_003",
          "selected_handles": [
            "F02"
          ],
          "slot_id": "slot_1",
          "status": "BOUND",
          "strict_correct": true
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW4_selected_parent_or_child_metric",
          "question_id": "msft_fy2025_001",
          "selected_handles": [
            "F03"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW8_missing_despite_unique_fact",
          "question_id": "msft_fy2025_003",
          "selected_handles": [],
          "slot_id": "1",
          "status": "MISSING",
          "strict_correct": false
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW0_correct",
          "question_id": "nvda_fy2025_002",
          "selected_handles": [
            "F02"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": true
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW0_correct",
          "question_id": "nvda_fy2025_003",
          "selected_handles": [
            "F01"
          ],
          "slot_id": "slot_1",
          "status": "BOUND",
          "strict_correct": true
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW0_correct",
          "question_id": "nvda_fy2025_004",
          "selected_handles": [
            "F02"
          ],
          "slot_id": "data_center_revenue_fy2025",
          "status": "BOUND",
          "strict_correct": true
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW0_correct",
          "question_id": "nvda_fy2025_005",
          "selected_handles": [
            "F01"
          ],
          "slot_id": "automotive_revenue_fy2025",
          "status": "BOUND",
          "strict_correct": true
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW0_correct",
          "question_id": "nvda_fy2025_009",
          "selected_handles": [
            "F03"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": true
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW3_selected_wrong_statement",
          "question_id": "pfe_fy2024_001",
          "selected_handles": [
            "F02",
            "F04"
          ],
          "slot_id": "total_revenues_fy2024",
          "status": "AMBIGUOUS",
          "strict_correct": false
        },
        {
          "cohort": "A_direct_visible_unique",
          "primary_category": "SW3_selected_wrong_statement",
          "question_id": "tsla_fy2025_001",
          "selected_handles": [
            "F03"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "B_calculation_visible_unique",
          "primary_category": "SW0_correct",
          "question_id": "aapl_fy2025_006",
          "selected_handles": [
            "F03"
          ],
          "slot_id": "slot_1",
          "status": "BOUND",
          "strict_correct": true
        },
        {
          "cohort": "B_calculation_visible_unique",
          "primary_category": "SW0_correct",
          "question_id": "aapl_fy2025_006",
          "selected_handles": [
            "F01"
          ],
          "slot_id": "slot_2",
          "status": "BOUND",
          "strict_correct": true
        },
        {
          "cohort": "B_calculation_visible_unique",
          "primary_category": "SW3_selected_wrong_statement",
          "question_id": "jpm_fy2025_006",
          "selected_handles": [
            "F01"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "B_calculation_visible_unique",
          "primary_category": "SW3_selected_wrong_statement",
          "question_id": "jpm_fy2025_006",
          "selected_handles": [
            "F02"
          ],
          "slot_id": "2",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "B_calculation_visible_unique",
          "primary_category": "SW3_selected_wrong_statement",
          "question_id": "ko_fy2025_006",
          "selected_handles": [
            "F02"
          ],
          "slot_id": "slot_1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "B_calculation_visible_unique",
          "primary_category": "SW3_selected_wrong_statement",
          "question_id": "ko_fy2025_006",
          "selected_handles": [
            "F01"
          ],
          "slot_id": "slot_2",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "B_calculation_visible_unique",
          "primary_category": "SW0_correct",
          "question_id": "pfe_fy2024_006",
          "selected_handles": [
            "F01"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": true
        },
        {
          "cohort": "B_calculation_visible_unique",
          "primary_category": "SW0_correct",
          "question_id": "pfe_fy2024_006",
          "selected_handles": [
            "F03"
          ],
          "slot_id": "2",
          "status": "BOUND",
          "strict_correct": true
        },
        {
          "cohort": "B_calculation_visible_unique",
          "primary_category": "SW3_selected_wrong_statement",
          "question_id": "tsla_fy2025_006",
          "selected_handles": [
            "F03"
          ],
          "slot_id": "rev_fy2025",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "B_calculation_visible_unique",
          "primary_category": "SW3_selected_wrong_statement",
          "question_id": "tsla_fy2025_006",
          "selected_handles": [
            "F02"
          ],
          "slot_id": "rev_fy2024",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "B_calculation_visible_unique",
          "primary_category": "SW0_correct",
          "question_id": "v_fy2025_006",
          "selected_handles": [
            "F02"
          ],
          "slot_id": "s1",
          "status": "BOUND",
          "strict_correct": true
        },
        {
          "cohort": "B_calculation_visible_unique",
          "primary_category": "SW3_selected_wrong_statement",
          "question_id": "v_fy2025_006",
          "selected_handles": [
            "F04"
          ],
          "slot_id": "s2",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "C_indistinguishable",
          "primary_category": "SW6_selected_one_of_indistinguishable_candidates",
          "question_id": "aapl_fy2025_003",
          "selected_handles": [
            "F03"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "C_indistinguishable",
          "primary_category": "SW6_selected_one_of_indistinguishable_candidates",
          "question_id": "v_fy2025_001",
          "selected_handles": [
            "F01"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "C_indistinguishable",
          "primary_category": "SW6_selected_one_of_indistinguishable_candidates",
          "question_id": "v_fy2025_002",
          "selected_handles": [
            "F01"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "C_indistinguishable",
          "primary_category": "SW6_selected_one_of_indistinguishable_candidates",
          "question_id": "v_fy2025_003",
          "selected_handles": [
            "F01"
          ],
          "slot_id": "slot_1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "C_indistinguishable",
          "primary_category": "SW6_selected_one_of_indistinguishable_candidates",
          "question_id": "v_fy2025_004",
          "selected_handles": [
            "F02"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "C_indistinguishable",
          "primary_category": "SW6_selected_one_of_indistinguishable_candidates",
          "question_id": "v_fy2025_009",
          "selected_handles": [
            "F01"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "D_unbindable_safety",
          "primary_category": "SW0_correct",
          "question_id": "jpm_fy2025_005",
          "selected_handles": [],
          "slot_id": "slot_1",
          "status": "MISSING",
          "strict_correct": false
        },
        {
          "cohort": "D_unbindable_safety",
          "primary_category": "SW4_selected_parent_or_child_metric",
          "question_id": "msft_fy2025_002",
          "selected_handles": [
            "F03"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "D_unbindable_safety",
          "primary_category": "SW4_selected_parent_or_child_metric",
          "question_id": "nvda_fy2025_007",
          "selected_handles": [
            "F02"
          ],
          "slot_id": "slot_1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "D_unbindable_safety",
          "primary_category": "SW4_selected_parent_or_child_metric",
          "question_id": "nvda_fy2025_007",
          "selected_handles": [
            "F04"
          ],
          "slot_id": "slot_2",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "D_unbindable_safety",
          "primary_category": "SW5_selected_lexically_similar_fact",
          "question_id": "pfe_fy2024_002",
          "selected_handles": [
            "F02"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "D_unbindable_safety",
          "primary_category": "SW5_selected_lexically_similar_fact",
          "question_id": "pfe_fy2024_004",
          "selected_handles": [
            "F01"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "D_unbindable_safety",
          "primary_category": "SW1_selected_nearest_metric_but_wrong_scope",
          "question_id": "v_fy2025_006",
          "selected_handles": [
            "F02"
          ],
          "slot_id": "s1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "D_unbindable_safety",
          "primary_category": "SW3_selected_wrong_statement",
          "question_id": "v_fy2025_006",
          "selected_handles": [
            "F04"
          ],
          "slot_id": "s2",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "D_unbindable_safety",
          "primary_category": "SW4_selected_parent_or_child_metric",
          "question_id": "v_fy2025_007",
          "selected_handles": [
            "F05"
          ],
          "slot_id": "1",
          "status": "BOUND",
          "strict_correct": false
        },
        {
          "cohort": "D_unbindable_safety",
          "primary_category": "SW5_selected_lexically_similar_fact",
          "question_id": "v_fy2025_007",
          "selected_handles": [
            "F01"
          ],
          "slot_id": "2",
          "status": "BOUND",
          "strict_correct": false
        }
      ]
    },
    "overbinding": {
      "bound_cases": 15,
      "category_counts": {
        "OB0_full_match_but_gold_contract_disagrees": 0,
        "OB1_partial_metric_match": 4,
        "OB2_missing_scope_evidence": 1,
        "OB3_period_conflict": 0,
        "OB4_statement_conflict": 1,
        "OB5_insufficient_context_but_model_committed": 6,
        "OB6_other": 3
      },
      "model_calls": 0,
      "rows": [
        {
          "classification": "OB5_insufficient_context_but_model_committed",
          "cohort": "C_indistinguishable",
          "competitors": [
            {
              "candidate_id": "candidate:v1:511a02069dc8d0c9c495fa04b5dfe1ad94d58da37e29e193c8580c752fa777b5",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "09d8a9afdcab52c7758357dd6ca58bff7d5fbd1253ff1950825594a337396565",
              "handle": "F01",
              "normalized_metric": "total gross margin percentage",
              "normalized_period": "FY2024",
              "pdf_page": 27,
              "physical_source_id": "60d96702c3d70da2d4d2c1a1fbeae60472c537ac7054ab2a66c3b44b179ae821:251cf6076494ea086c40b4475b866b3b22b1e91876d943db6bd575c092b78390:27",
              "raw_metric": "Total gross margin percentage",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:511a02069dc8d0c9c495fa04b5dfe1ad94d58da37e29e193c8580c752fa777b5",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "731d256f0813839c701f13b3b4e9deeb24dec4bb09f6ff268fc21c17c652e871",
              "handle": "F02",
              "normalized_metric": "total gross margin percentage",
              "normalized_period": "FY2023",
              "pdf_page": 27,
              "physical_source_id": "60d96702c3d70da2d4d2c1a1fbeae60472c537ac7054ab2a66c3b44b179ae821:251cf6076494ea086c40b4475b866b3b22b1e91876d943db6bd575c092b78390:27",
              "raw_metric": "Total gross margin percentage",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:06af0c67784c7ac6621c9fa629040925178a32f9f1fdf9e30627123f3f722e4b",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "158317caa5be4d441a4220bc6cf32037c91ca9b10f64fa2ba73a1e2bc76351da",
              "handle": "F04",
              "normalized_metric": "gross margin",
              "normalized_period": "FY2023",
              "pdf_page": 32,
              "physical_source_id": "2016faed04f03df7ef6aad0be8f232521469d6ba19428379e552d55dc23667cd:445b48c193bcf54f55a2269b48e447a3d23f7e93c8fe7c33bde493df7beb0cbf:32",
              "raw_metric": "Gross margin",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:06af0c67784c7ac6621c9fa629040925178a32f9f1fdf9e30627123f3f722e4b",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "680f4c61361d70e26365e3cc233963b9d8f2262e261fd57a3fab0960c36f3dea",
              "handle": "F05",
              "normalized_metric": "gross margin",
              "normalized_period": "FY2025",
              "pdf_page": 32,
              "physical_source_id": "2016faed04f03df7ef6aad0be8f232521469d6ba19428379e552d55dc23667cd:445b48c193bcf54f55a2269b48e447a3d23f7e93c8fe7c33bde493df7beb0cbf:32",
              "raw_metric": "Gross margin",
              "raw_period": "FY2025",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:06af0c67784c7ac6621c9fa629040925178a32f9f1fdf9e30627123f3f722e4b",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "8693b8846c462b717470f63562ecc33781577b8ead0357de9620512706f85cbf",
              "handle": "F06",
              "normalized_metric": "gross margin",
              "normalized_period": "FY2024",
              "pdf_page": 32,
              "physical_source_id": "2016faed04f03df7ef6aad0be8f232521469d6ba19428379e552d55dc23667cd:445b48c193bcf54f55a2269b48e447a3d23f7e93c8fe7c33bde493df7beb0cbf:32",
              "raw_metric": "Gross margin",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:b5e440820554a1aff3c1c829bee1d55842a2d8ce27c9ae64d488115c88622a8c",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "0812b423fe4a5400fdea95e65b61ea82c5e1b2796c3c4a5d3b2ac98afaa3b37d",
              "handle": "F07",
              "normalized_metric": "total gross margin",
              "normalized_period": "FY2024",
              "pdf_page": 27,
              "physical_source_id": "60d96702c3d70da2d4d2c1a1fbeae60472c537ac7054ab2a66c3b44b179ae821:502627d572ce139e82dfb719e861393d33fa1c5889d0b0b0801a314b2e4ad763:27",
              "raw_metric": "Total gross margin",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:b5e440820554a1aff3c1c829bee1d55842a2d8ce27c9ae64d488115c88622a8c",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "6971870c931a4b5f537387e472d1927b32a621e0cc738d3e2c6ba7dc900fdbdd",
              "handle": "F08",
              "normalized_metric": "total gross margin",
              "normalized_period": "FY2025",
              "pdf_page": 27,
              "physical_source_id": "60d96702c3d70da2d4d2c1a1fbeae60472c537ac7054ab2a66c3b44b179ae821:502627d572ce139e82dfb719e861393d33fa1c5889d0b0b0801a314b2e4ad763:27",
              "raw_metric": "Total gross margin",
              "raw_period": "FY2025",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:b5e440820554a1aff3c1c829bee1d55842a2d8ce27c9ae64d488115c88622a8c",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "e761985cb38a372515d1be54bbba3881d99143ca6fe2be495ac9960d61116812",
              "handle": "F09",
              "normalized_metric": "total gross margin",
              "normalized_period": "FY2023",
              "pdf_page": 27,
              "physical_source_id": "60d96702c3d70da2d4d2c1a1fbeae60472c537ac7054ab2a66c3b44b179ae821:502627d572ce139e82dfb719e861393d33fa1c5889d0b0b0801a314b2e4ad763:27",
              "raw_metric": "Total gross margin",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            }
          ],
          "constraints": {
            "metric": false,
            "period": true,
            "role": true,
            "scope": false,
            "statement": true
          },
          "question_id": "aapl_fy2025_003",
          "requirement": {
            "metric": "gross margin percentage",
            "period": "FY2025",
            "role": "value",
            "slot_id": "1",
            "unit": null,
            "value_type": "percentage"
          },
          "selected": {
            "candidate_id": "candidate:v1:511a02069dc8d0c9c495fa04b5dfe1ad94d58da37e29e193c8580c752fa777b5",
            "column_header": null,
            "column_header_path": null,
            "fact_id": "c020b94ff98da3c5c9039108f959cd6ceeb1cbbcd39905369dc424eca18fd2ab",
            "handle": "F03",
            "normalized_metric": "total gross margin percentage",
            "normalized_period": "FY2025",
            "pdf_page": 27,
            "physical_source_id": "60d96702c3d70da2d4d2c1a1fbeae60472c537ac7054ab2a66c3b44b179ae821:251cf6076494ea086c40b4475b866b3b22b1e91876d943db6bd575c092b78390:27",
            "raw_metric": "Total gross margin percentage",
            "raw_period": "FY2025",
            "row_hierarchy": null,
            "row_label": null,
            "section_heading": null,
            "statement_title": null,
            "table_title": null
          },
          "slot_id": "1"
        },
        {
          "classification": "OB5_insufficient_context_but_model_committed",
          "cohort": "C_indistinguishable",
          "competitors": [
            {
              "candidate_id": "candidate:v1:289c0336e376d5107ddec815a794c98a560365f265de95c0f3f45ee1e878ba83",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "b977e2afc7a4a2d9f6f2db3d4e57661cba2930fdf649c25855bf7cb0cd2be173",
              "handle": "F02",
              "normalized_metric": "net revenue",
              "normalized_period": "FY2023",
              "pdf_page": 75,
              "physical_source_id": "047c6a56c7f09d391ac993d738c97b84bae5339ad9a1873ff5d6dd98ff693439:9cea965a621d0707927d0252efb5f95117adc2b590b495799bd051d069bd2eae:75",
              "raw_metric": "Net revenue",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:289c0336e376d5107ddec815a794c98a560365f265de95c0f3f45ee1e878ba83",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "c8d61066a50fdbd0b76129b27af245ed97f4a3a38bb4cf26e81b824be4377759",
              "handle": "F03",
              "normalized_metric": "net revenue",
              "normalized_period": "FY2024",
              "pdf_page": 75,
              "physical_source_id": "047c6a56c7f09d391ac993d738c97b84bae5339ad9a1873ff5d6dd98ff693439:9cea965a621d0707927d0252efb5f95117adc2b590b495799bd051d069bd2eae:75",
              "raw_metric": "Net revenue",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:e41161a6e97b28ec2ea1117ccbf8ef0cebaa0accf8723ceaa80c7598095c1665",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "4bf7d30f77973e75a3702dfb126b676324d3df1d4d55e9a04405a215a16158ce",
              "handle": "F04",
              "normalized_metric": "net revenue",
              "normalized_period": "FY2023",
              "pdf_page": 61,
              "physical_source_id": "b298c3e02508241934593c6c7d864c63fcc2c8159753b43f5c0c077a2a24de53:012ca63ad655186409b9e0319862671bce2f21ea9a97d3094d4ceb1bd22fb53c:61",
              "raw_metric": "Net revenue",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:1360b5ebea3eec01f428ede6db02ec4a235ef9d67b03c02f93b140a70ad5a3b7",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "b48828cafd4170bd626ddf12b0a3edfd6b3a40ca6f0ed37b09701cfcb219fb83",
              "handle": "F05",
              "normalized_metric": "net revenue",
              "normalized_period": "FY2023",
              "pdf_page": 55,
              "physical_source_id": "a2b6b99ae6d9e7df75bafbe29c0b2b5dbbdfca535652b3580561fc8e72fbbd6b:6337e85f2145b8806da2191701af4a9750576550b94721c3be27a54ff2cd7082:55",
              "raw_metric": "Net revenue",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:747783bcac47c8f71b61c289a33c8b6b6b7121f23d39f8487eade7fc23fe3154",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "bb5e24ce87662cd8f8617c4eaa5d04033d62a63114d1edb69f8ee443201c9128",
              "handle": "F06",
              "normalized_metric": "net revenue",
              "normalized_period": "fy",
              "pdf_page": 4,
              "physical_source_id": "036b51da4914aa39a6c95956382ad7ab03fede68ec04f97a158f7dabf74cd2c2:04f3b8550b72399b5c7e4ade2474d1d85b0e8333fe4c24df978f254e108eb4d1:4",
              "raw_metric": "Net revenue",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:747783bcac47c8f71b61c289a33c8b6b6b7121f23d39f8487eade7fc23fe3154",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "cddd4d6326a80953034ff11f4ad06c852005914112b3a46f52b3eb5a2f48dd8e",
              "handle": "F07",
              "normalized_metric": "net revenue",
              "normalized_period": "fy",
              "pdf_page": 4,
              "physical_source_id": "036b51da4914aa39a6c95956382ad7ab03fede68ec04f97a158f7dabf74cd2c2:04f3b8550b72399b5c7e4ade2474d1d85b0e8333fe4c24df978f254e108eb4d1:4",
              "raw_metric": "Net revenue",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:747783bcac47c8f71b61c289a33c8b6b6b7121f23d39f8487eade7fc23fe3154",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "ce0fa445a1830723f871cb2e61588ac0069c9d601112e5901e95bb86b810bc41",
              "handle": "F08",
              "normalized_metric": "net revenue",
              "normalized_period": "fy",
              "pdf_page": 4,
              "physical_source_id": "036b51da4914aa39a6c95956382ad7ab03fede68ec04f97a158f7dabf74cd2c2:04f3b8550b72399b5c7e4ade2474d1d85b0e8333fe4c24df978f254e108eb4d1:4",
              "raw_metric": "Net revenue",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            }
          ],
          "constraints": {
            "metric": true,
            "period": true,
            "role": true,
            "scope": true,
            "statement": true
          },
          "question_id": "v_fy2025_001",
          "requirement": {
            "metric": "net revenue",
            "period": "FY2025",
            "role": "value",
            "slot_id": "1",
            "unit": null,
            "value_type": "numeric"
          },
          "selected": {
            "candidate_id": "candidate:v1:289c0336e376d5107ddec815a794c98a560365f265de95c0f3f45ee1e878ba83",
            "column_header": null,
            "column_header_path": null,
            "fact_id": "2a4da6647be88d9aa9a04e8c7e015e7a277ea028374f3dbe8778b5a6b97de9c3",
            "handle": "F01",
            "normalized_metric": "net revenue",
            "normalized_period": "FY2025",
            "pdf_page": 75,
            "physical_source_id": "047c6a56c7f09d391ac993d738c97b84bae5339ad9a1873ff5d6dd98ff693439:9cea965a621d0707927d0252efb5f95117adc2b590b495799bd051d069bd2eae:75",
            "raw_metric": "Net revenue",
            "raw_period": "FY2025",
            "row_hierarchy": null,
            "row_label": null,
            "section_heading": null,
            "statement_title": null,
            "table_title": null
          },
          "slot_id": "1"
        },
        {
          "classification": "OB5_insufficient_context_but_model_committed",
          "cohort": "C_indistinguishable",
          "competitors": [
            {
              "candidate_id": "candidate:v1:fd7d3044411973c1af57bf015ca3fd14efb235afbe07f01c8ff1d8d9259536db",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "4012d867da99b6066050266a59e27e67f5e3f32126a3d19c57913df5598b47a8",
              "handle": "F02",
              "normalized_metric": "total volume1",
              "normalized_period": "FY2023",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:7581db60143049afe1acbb2a768d7c5dc5ea5eb7f463a04cdbe3900c8da403a6:4",
              "raw_metric": "Total volume1",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:fd7d3044411973c1af57bf015ca3fd14efb235afbe07f01c8ff1d8d9259536db",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "76bc22dc76a49153e5903bfe42bf4fbbfff5166acfb8b28bb04dfaab708cbb08",
              "handle": "F03",
              "normalized_metric": "total volume1",
              "normalized_period": "FY2024",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:7581db60143049afe1acbb2a768d7c5dc5ea5eb7f463a04cdbe3900c8da403a6:4",
              "raw_metric": "Total volume1",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            }
          ],
          "constraints": {
            "metric": false,
            "period": true,
            "role": true,
            "scope": false,
            "statement": true
          },
          "question_id": "v_fy2025_002",
          "requirement": {
            "metric": "total volume",
            "period": "FY2025",
            "role": "value",
            "slot_id": "1",
            "unit": null,
            "value_type": "numeric"
          },
          "selected": {
            "candidate_id": "candidate:v1:fd7d3044411973c1af57bf015ca3fd14efb235afbe07f01c8ff1d8d9259536db",
            "column_header": null,
            "column_header_path": null,
            "fact_id": "1e029025fdb3d7a21173e4625d794ee6fd2e14c91147b6cfa6cad84d66e0e404",
            "handle": "F01",
            "normalized_metric": "total volume1",
            "normalized_period": "FY2025",
            "pdf_page": 4,
            "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:7581db60143049afe1acbb2a768d7c5dc5ea5eb7f463a04cdbe3900c8da403a6:4",
            "raw_metric": "Total volume1",
            "raw_period": "FY2025",
            "row_hierarchy": null,
            "row_label": null,
            "section_heading": null,
            "statement_title": null,
            "table_title": null
          },
          "slot_id": "1"
        },
        {
          "classification": "OB5_insufficient_context_but_model_committed",
          "cohort": "C_indistinguishable",
          "competitors": [
            {
              "candidate_id": "candidate:v1:38e40720cd64c77589937b3a83bf3659c1f6f7c5e1dd05d897fb593bdaf26a68",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "74820f5a381a6d55fa1e68cdc001e6e8dc0c490f272b25db121a6fdb376e710d",
              "handle": "F02",
              "normalized_metric": "transactions processed on visa's networks2",
              "normalized_period": "FY2023",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:c48ae62bcaae673a2779f950221850202adeda0cb298d10f377e450231e31e8b:4",
              "raw_metric": "Transactions processed on Visa's networks2",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:38e40720cd64c77589937b3a83bf3659c1f6f7c5e1dd05d897fb593bdaf26a68",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "7abae283bf0c21bc1bfb307f6a75943e456c93d571c8839d3fad4f52880cdc48",
              "handle": "F03",
              "normalized_metric": "transactions processed on visa's networks2",
              "normalized_period": "FY2024",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:c48ae62bcaae673a2779f950221850202adeda0cb298d10f377e450231e31e8b:4",
              "raw_metric": "Transactions processed on Visa's networks2",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:e13e280f973240a339a78156714b47b28c9f3b2acab6be52c60a9297a1ad151b",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "8fe22c706b8ae4706f23201654a15d1400475a17431335f5d55c7494fa1409b0",
              "handle": "F04",
              "normalized_metric": "network and processing",
              "normalized_period": "FY2023",
              "pdf_page": 62,
              "physical_source_id": "e291f33c854452269a270d476e9945d19b62db7039c6d0c6721777a85b1b6d48:c3f41911052e6d385064a2c10b233f2f4c1e0d7076c8095d552fad64cc7165fd:62",
              "raw_metric": "Network and processing",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            }
          ],
          "constraints": {
            "metric": false,
            "period": true,
            "role": true,
            "scope": false,
            "statement": true
          },
          "question_id": "v_fy2025_003",
          "requirement": {
            "metric": "transactions processed on Visa's networks",
            "period": "FY2025",
            "role": "value",
            "slot_id": "slot_1",
            "unit": null,
            "value_type": "numeric"
          },
          "selected": {
            "candidate_id": "candidate:v1:38e40720cd64c77589937b3a83bf3659c1f6f7c5e1dd05d897fb593bdaf26a68",
            "column_header": null,
            "column_header_path": null,
            "fact_id": "0d275e94bd8c411d5cd48e132ed3cfb6207fb60851d3f87598c52a2e822b0a44",
            "handle": "F01",
            "normalized_metric": "transactions processed on visa's networks2",
            "normalized_period": "FY2025",
            "pdf_page": 4,
            "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:c48ae62bcaae673a2779f950221850202adeda0cb298d10f377e450231e31e8b:4",
            "raw_metric": "Transactions processed on Visa's networks2",
            "raw_period": "FY2025",
            "row_hierarchy": null,
            "row_label": null,
            "section_heading": null,
            "statement_title": null,
            "table_title": null
          },
          "slot_id": "slot_1"
        },
        {
          "classification": "OB5_insufficient_context_but_model_committed",
          "cohort": "C_indistinguishable",
          "competitors": [
            {
              "candidate_id": "candidate:v1:9345b21bb1fae47731ceb136aa7846f1adca5024d7d31ded608e9f03a486e186",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "4c35227ee925f1d09de33e0714c5e05c89c4beba383342a0e254bd37fa691a61",
              "handle": "F01",
              "normalized_metric": "payments volume1",
              "normalized_period": "FY2024",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:93925e15ad34715a4b43601361d4ca1d949e380955f4fe20f934d71fd8c8a869:4",
              "raw_metric": "Payments volume1",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:9345b21bb1fae47731ceb136aa7846f1adca5024d7d31ded608e9f03a486e186",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "85a5e7bf2ad766e23ceefb0a5301d1a9eff9e6d80441eee9d1819e5450ff0794",
              "handle": "F03",
              "normalized_metric": "payments volume1",
              "normalized_period": "FY2023",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:93925e15ad34715a4b43601361d4ca1d949e380955f4fe20f934d71fd8c8a869:4",
              "raw_metric": "Payments volume1",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:3445c84bd88b5385b7c347a8d96a92bffba70d7303c0c6af4a43cdeaac06bf92",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "6ae56b60c1e298e84b185ae23e83ae47e367dc2cb583ebedd50e4bd3268fa3f5",
              "handle": "F04",
              "normalized_metric": "consumer debit(2)",
              "normalized_period": "fy",
              "pdf_page": 59,
              "physical_source_id": "b41ac6b82fee2eb8c1683631607329a1e162f8e27683309ffb742f74fd033fbd:4afb528d0d4f23e3f112b50e6294ab4dd047dccf91d192e5861251884a919dfd:59",
              "raw_metric": "Consumer debit(2)",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:3445c84bd88b5385b7c347a8d96a92bffba70d7303c0c6af4a43cdeaac06bf92",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "797058834a920501d78cb1b34cf08bab557c6c2038b5829b9fd69fdcc6917d57",
              "handle": "F05",
              "normalized_metric": "consumer debit(2)",
              "normalized_period": "fy",
              "pdf_page": 59,
              "physical_source_id": "b41ac6b82fee2eb8c1683631607329a1e162f8e27683309ffb742f74fd033fbd:4afb528d0d4f23e3f112b50e6294ab4dd047dccf91d192e5861251884a919dfd:59",
              "raw_metric": "Consumer debit(2)",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:3445c84bd88b5385b7c347a8d96a92bffba70d7303c0c6af4a43cdeaac06bf92",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "98a991cdd89d64366656c60b6b98a42493001434f55f3f08acea5203304c4b6a",
              "handle": "F06",
              "normalized_metric": "consumer debit(2)",
              "normalized_period": "fy",
              "pdf_page": 59,
              "physical_source_id": "b41ac6b82fee2eb8c1683631607329a1e162f8e27683309ffb742f74fd033fbd:4afb528d0d4f23e3f112b50e6294ab4dd047dccf91d192e5861251884a919dfd:59",
              "raw_metric": "Consumer debit(2)",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:3445c84bd88b5385b7c347a8d96a92bffba70d7303c0c6af4a43cdeaac06bf92",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "ad3b35cb95aeb1d5603899bad446c43416d4dab141db6aff6d6990d0d71bcd9d",
              "handle": "F07",
              "normalized_metric": "consumer debit(2)",
              "normalized_period": "fy",
              "pdf_page": 59,
              "physical_source_id": "b41ac6b82fee2eb8c1683631607329a1e162f8e27683309ffb742f74fd033fbd:4afb528d0d4f23e3f112b50e6294ab4dd047dccf91d192e5861251884a919dfd:59",
              "raw_metric": "Consumer debit(2)",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:3445c84bd88b5385b7c347a8d96a92bffba70d7303c0c6af4a43cdeaac06bf92",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "b72798176c49d35dc5ec245ce4e33ac7d08e7027183ec8e116c4de39a9de5875",
              "handle": "F08",
              "normalized_metric": "consumer debit(2)",
              "normalized_period": "fy",
              "pdf_page": 59,
              "physical_source_id": "b41ac6b82fee2eb8c1683631607329a1e162f8e27683309ffb742f74fd033fbd:4afb528d0d4f23e3f112b50e6294ab4dd047dccf91d192e5861251884a919dfd:59",
              "raw_metric": "Consumer debit(2)",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:3445c84bd88b5385b7c347a8d96a92bffba70d7303c0c6af4a43cdeaac06bf92",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "cd4250126cab222bc5294135767353d636e1919c3c9789e513598b3afc84a41c",
              "handle": "F09",
              "normalized_metric": "consumer debit(2)",
              "normalized_period": "fy",
              "pdf_page": 59,
              "physical_source_id": "b41ac6b82fee2eb8c1683631607329a1e162f8e27683309ffb742f74fd033fbd:4afb528d0d4f23e3f112b50e6294ab4dd047dccf91d192e5861251884a919dfd:59",
              "raw_metric": "Consumer debit(2)",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            }
          ],
          "constraints": {
            "metric": false,
            "period": true,
            "role": true,
            "scope": false,
            "statement": true
          },
          "question_id": "v_fy2025_004",
          "requirement": {
            "metric": "payments volume",
            "period": "FY2025",
            "role": "value",
            "slot_id": "1",
            "unit": null,
            "value_type": "numeric"
          },
          "selected": {
            "candidate_id": "candidate:v1:9345b21bb1fae47731ceb136aa7846f1adca5024d7d31ded608e9f03a486e186",
            "column_header": null,
            "column_header_path": null,
            "fact_id": "71bd06d15b6465ce3cc482db9381a3216507716c6a627de5f6408bab3999e909",
            "handle": "F02",
            "normalized_metric": "payments volume1",
            "normalized_period": "FY2025",
            "pdf_page": 4,
            "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:93925e15ad34715a4b43601361d4ca1d949e380955f4fe20f934d71fd8c8a869:4",
            "raw_metric": "Payments volume1",
            "raw_period": "FY2025",
            "row_hierarchy": null,
            "row_label": null,
            "section_heading": null,
            "statement_title": null,
            "table_title": null
          },
          "slot_id": "1"
        },
        {
          "classification": "OB5_insufficient_context_but_model_committed",
          "cohort": "C_indistinguishable",
          "competitors": [
            {
              "candidate_id": "candidate:v1:9345b21bb1fae47731ceb136aa7846f1adca5024d7d31ded608e9f03a486e186",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "71bd06d15b6465ce3cc482db9381a3216507716c6a627de5f6408bab3999e909",
              "handle": "F02",
              "normalized_metric": "payments volume1",
              "normalized_period": "FY2025",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:93925e15ad34715a4b43601361d4ca1d949e380955f4fe20f934d71fd8c8a869:4",
              "raw_metric": "Payments volume1",
              "raw_period": "FY2025",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:9345b21bb1fae47731ceb136aa7846f1adca5024d7d31ded608e9f03a486e186",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "85a5e7bf2ad766e23ceefb0a5301d1a9eff9e6d80441eee9d1819e5450ff0794",
              "handle": "F03",
              "normalized_metric": "payments volume1",
              "normalized_period": "FY2023",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:93925e15ad34715a4b43601361d4ca1d949e380955f4fe20f934d71fd8c8a869:4",
              "raw_metric": "Payments volume1",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:094cb4dab475e4b1f475cb76990215d113f3b486d7a578c411d817d39101b5f1",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "251af5fe82366e407265e4f0dea59a09fda5ea1eec8fe5c9dc4a56c822b520e8",
              "handle": "F04",
              "normalized_metric": "total nominal payments volume(4)",
              "normalized_period": "fy",
              "pdf_page": 59,
              "physical_source_id": "b41ac6b82fee2eb8c1683631607329a1e162f8e27683309ffb742f74fd033fbd:474b574592f42bdb5089692cf91610279076b396fd12b58af52d27e182326de2:59",
              "raw_metric": "Total nominal payments volume(4)",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:094cb4dab475e4b1f475cb76990215d113f3b486d7a578c411d817d39101b5f1",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "8655c7655ab3f750099e56dcbba81e615ab1f1029c42394460c7151ac747f36c",
              "handle": "F05",
              "normalized_metric": "total nominal payments volume(4)",
              "normalized_period": "fy",
              "pdf_page": 59,
              "physical_source_id": "b41ac6b82fee2eb8c1683631607329a1e162f8e27683309ffb742f74fd033fbd:474b574592f42bdb5089692cf91610279076b396fd12b58af52d27e182326de2:59",
              "raw_metric": "Total nominal payments volume(4)",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:094cb4dab475e4b1f475cb76990215d113f3b486d7a578c411d817d39101b5f1",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "87f369644a5a6e0ee712fd076cf6cbc84d1eea6b86db0598c98a5b300a3b1af0",
              "handle": "F06",
              "normalized_metric": "total nominal payments volume(4)",
              "normalized_period": "fy",
              "pdf_page": 59,
              "physical_source_id": "b41ac6b82fee2eb8c1683631607329a1e162f8e27683309ffb742f74fd033fbd:474b574592f42bdb5089692cf91610279076b396fd12b58af52d27e182326de2:59",
              "raw_metric": "Total nominal payments volume(4)",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:094cb4dab475e4b1f475cb76990215d113f3b486d7a578c411d817d39101b5f1",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "885dac5277dca9f84088c787a6a76eea0d725bdda856f92b4927b9896c05053f",
              "handle": "F07",
              "normalized_metric": "total nominal payments volume(4)",
              "normalized_period": "fy",
              "pdf_page": 59,
              "physical_source_id": "b41ac6b82fee2eb8c1683631607329a1e162f8e27683309ffb742f74fd033fbd:474b574592f42bdb5089692cf91610279076b396fd12b58af52d27e182326de2:59",
              "raw_metric": "Total nominal payments volume(4)",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:094cb4dab475e4b1f475cb76990215d113f3b486d7a578c411d817d39101b5f1",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "b45529a9dd8710e5a76b14bba0f557085dac999dd4d3edc96f57c9003f6fd3ae",
              "handle": "F08",
              "normalized_metric": "total nominal payments volume(4)",
              "normalized_period": "fy",
              "pdf_page": 59,
              "physical_source_id": "b41ac6b82fee2eb8c1683631607329a1e162f8e27683309ffb742f74fd033fbd:474b574592f42bdb5089692cf91610279076b396fd12b58af52d27e182326de2:59",
              "raw_metric": "Total nominal payments volume(4)",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:094cb4dab475e4b1f475cb76990215d113f3b486d7a578c411d817d39101b5f1",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "bab447b6b6df1bd6cdb9413b6a4912ee6fb86658a0e6367cbcbc08fd2162a2dd",
              "handle": "F09",
              "normalized_metric": "total nominal payments volume(4)",
              "normalized_period": "fy",
              "pdf_page": 59,
              "physical_source_id": "b41ac6b82fee2eb8c1683631607329a1e162f8e27683309ffb742f74fd033fbd:474b574592f42bdb5089692cf91610279076b396fd12b58af52d27e182326de2:59",
              "raw_metric": "Total nominal payments volume(4)",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            }
          ],
          "constraints": {
            "metric": false,
            "period": true,
            "role": true,
            "scope": false,
            "statement": true
          },
          "question_id": "v_fy2025_009",
          "requirement": {
            "metric": "payments volume",
            "period": "FY2024",
            "role": "value",
            "slot_id": "1",
            "unit": null,
            "value_type": "numeric"
          },
          "selected": {
            "candidate_id": "candidate:v1:9345b21bb1fae47731ceb136aa7846f1adca5024d7d31ded608e9f03a486e186",
            "column_header": null,
            "column_header_path": null,
            "fact_id": "4c35227ee925f1d09de33e0714c5e05c89c4beba383342a0e254bd37fa691a61",
            "handle": "F01",
            "normalized_metric": "payments volume1",
            "normalized_period": "FY2024",
            "pdf_page": 4,
            "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:93925e15ad34715a4b43601361d4ca1d949e380955f4fe20f934d71fd8c8a869:4",
            "raw_metric": "Payments volume1",
            "raw_period": "FY2024",
            "row_hierarchy": null,
            "row_label": null,
            "section_heading": null,
            "statement_title": null,
            "table_title": null
          },
          "slot_id": "1"
        },
        {
          "classification": "OB1_partial_metric_match",
          "cohort": "D_unbindable_safety",
          "competitors": [
            {
              "candidate_id": "candidate:v1:ebaad37cf41a7a63624f981911edc9ec15b15459346bb41471ca0f1a77ce2782",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "0f423057c441a75c98671d18edc5e721d5490c92a9a7fa1754e6621d648ade6b",
              "handle": "F01",
              "normalized_metric": "revenue",
              "normalized_period": "FY2025",
              "pdf_page": 26,
              "physical_source_id": "a6bd258be565d6178b60deeaa693fda2aac80dc23469fe90127cb92e6461d754:5760c88ff4ad1c874efadb6a2af5b2277f856fec19600688543c40d1d248f27b:26",
              "raw_metric": "Revenue",
              "raw_period": "FY2025",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:ebaad37cf41a7a63624f981911edc9ec15b15459346bb41471ca0f1a77ce2782",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "8fe02a6f4bd9788655a7d52d2459ec901e4a908ae386e9ba830c262a7bacc721",
              "handle": "F02",
              "normalized_metric": "revenue",
              "normalized_period": "FY2024",
              "pdf_page": 26,
              "physical_source_id": "a6bd258be565d6178b60deeaa693fda2aac80dc23469fe90127cb92e6461d754:5760c88ff4ad1c874efadb6a2af5b2277f856fec19600688543c40d1d248f27b:26",
              "raw_metric": "Revenue",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:9c9cf092d4ba42e7fc87b659442bb194683c58098625f20e6e37d06e0cf5eec7",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "b83a057ed1864a7d850659e5d835fb7c8b85d17417aff36077ddefb45c45016e",
              "handle": "F04",
              "normalized_metric": "revenue",
              "normalized_period": "FY2024",
              "pdf_page": 26,
              "physical_source_id": "a6bd258be565d6178b60deeaa693fda2aac80dc23469fe90127cb92e6461d754:af0d1dd83473f873c6601028e9eb055ab6af7e81401031551641320e07d9fe73:26",
              "raw_metric": "Revenue",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:768e3f558388d9a8ebf30e1acbab19f903afe24747e353143f0e292c4770e913",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "3203e459ca1ffa14fcac62713148f070d1dc1871a84ebcca63bf8e43e808b02a",
              "handle": "F05",
              "normalized_metric": "revenue",
              "normalized_period": "FY2024",
              "pdf_page": 26,
              "physical_source_id": "a6bd258be565d6178b60deeaa693fda2aac80dc23469fe90127cb92e6461d754:03c9f60ed8250a42897cb464484d90cb5d6745f4b66995c8cf6797585531acb3:26",
              "raw_metric": "Revenue",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:5b1b64b0fee3ad10bbf4c6e6fe849e7fbecf8f8f582d113a228a0fb52039c443",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "67d7dd1cbf8eb08a16b285fae7f12afa093cf183b6e3f9835f42c13ed00c5b3c",
              "handle": "F06",
              "normalized_metric": "revenue",
              "normalized_period": "FY2024",
              "pdf_page": 26,
              "physical_source_id": "a6bd258be565d6178b60deeaa693fda2aac80dc23469fe90127cb92e6461d754:48cf28e4a766a6caf2ac0342ff1be8fe02473c91ee98f6eebbc86a52bc5a1144:26",
              "raw_metric": "Revenue",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:67511c90a20e4e1229bdf7e37bea94ef6308701ce906cd4d75eb118abc566f31",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "5feadcc69c4405f27f0becaf514672f0aeb01c204bc4dffb97a20526f584b025",
              "handle": "F07",
              "normalized_metric": "operating income",
              "normalized_period": "FY2024",
              "pdf_page": 26,
              "physical_source_id": "a6bd258be565d6178b60deeaa693fda2aac80dc23469fe90127cb92e6461d754:0423f680a42c84b18a29036f90952aae4554bbec27dc011c462bb5b693dec4e4:26",
              "raw_metric": "Operating Income",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:67511c90a20e4e1229bdf7e37bea94ef6308701ce906cd4d75eb118abc566f31",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "fea880678df2683e3d9b004094c7259cbda02e87fa967271e0b668112c39e7e1",
              "handle": "F08",
              "normalized_metric": "operating income",
              "normalized_period": "FY2025",
              "pdf_page": 26,
              "physical_source_id": "a6bd258be565d6178b60deeaa693fda2aac80dc23469fe90127cb92e6461d754:0423f680a42c84b18a29036f90952aae4554bbec27dc011c462bb5b693dec4e4:26",
              "raw_metric": "Operating Income",
              "raw_period": "FY2025",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:30c2984bc2eed4fa736eabb40aae0c5e76e35dab77d9145eeabac16ca0665fe9",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "3b3e1f72a9c95a989ecd37eb44f74f7322d80b7243cfec29ba72d44c46d9c57b",
              "handle": "F09",
              "normalized_metric": "operating income",
              "normalized_period": "FY2025",
              "pdf_page": 26,
              "physical_source_id": "a6bd258be565d6178b60deeaa693fda2aac80dc23469fe90127cb92e6461d754:06f0f1b4c25da14faaa789fcfd9afe33cf78474656d8595a97f61faefb0c2377:26",
              "raw_metric": "Operating Income",
              "raw_period": "FY2025",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            }
          ],
          "constraints": {
            "metric": false,
            "period": true,
            "role": true,
            "scope": false,
            "statement": false
          },
          "question_id": "msft_fy2025_002",
          "requirement": {
            "metric": "Productivity and Business Processes revenue",
            "period": "FY2025",
            "role": "value",
            "slot_id": "1",
            "unit": null,
            "value_type": "numeric"
          },
          "selected": {
            "candidate_id": "candidate:v1:9c9cf092d4ba42e7fc87b659442bb194683c58098625f20e6e37d06e0cf5eec7",
            "column_header": null,
            "column_header_path": null,
            "fact_id": "b4b510f88f3297fdcb82f363f0e129f8c71dc0d8a875030502e84637cca41b7b",
            "handle": "F03",
            "normalized_metric": "revenue",
            "normalized_period": "FY2025",
            "pdf_page": 26,
            "physical_source_id": "a6bd258be565d6178b60deeaa693fda2aac80dc23469fe90127cb92e6461d754:af0d1dd83473f873c6601028e9eb055ab6af7e81401031551641320e07d9fe73:26",
            "raw_metric": "Revenue",
            "raw_period": "FY2025",
            "row_hierarchy": null,
            "row_label": null,
            "section_heading": null,
            "statement_title": null,
            "table_title": null
          },
          "slot_id": "1"
        },
        {
          "classification": "OB1_partial_metric_match",
          "cohort": "D_unbindable_safety",
          "competitors": [
            {
              "candidate_id": "candidate:v1:b8f690078d6e51c1c81063a3fe256bcfaeef4fbad4b355cc3b585fd4d275172d",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "95451b882435a15e4061b201f224b1302de906578750a22f6dcee01b3cc14b80",
              "handle": "F01",
              "normalized_metric": "data center",
              "normalized_period": "FY2023",
              "pdf_page": 172,
              "physical_source_id": "956d139d163b90f380e6cdcb5e9f14e5b72fbdb329a22512d3ccd04bbe45f736:7e731d3eb8975007be3386499df660bfd01fa67dfec355c09fb4e82dab3eb64e:172",
              "raw_metric": "Data Center",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:b8f690078d6e51c1c81063a3fe256bcfaeef4fbad4b355cc3b585fd4d275172d",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "a3798e1b9f596c81bfac2c6efd9ee5652ed8355f06ef6cc0a2e6a08df2d2e2af",
              "handle": "F03",
              "normalized_metric": "data center",
              "normalized_period": "FY2024",
              "pdf_page": 172,
              "physical_source_id": "956d139d163b90f380e6cdcb5e9f14e5b72fbdb329a22512d3ccd04bbe45f736:7e731d3eb8975007be3386499df660bfd01fa67dfec355c09fb4e82dab3eb64e:172",
              "raw_metric": "Data Center",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:b4dac3814c3a603cbdb3e1936c6d34168d4c6ec0ea69239b06a573991407a2bf",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "76ea0bd1e273e5b07b57a84390905c6784ac86c9d12bdbbf9c08f7445391359d",
              "handle": "F04",
              "normalized_metric": "gross margin",
              "normalized_period": "FY2025",
              "pdf_page": 130,
              "physical_source_id": "e267f6a2071a589e73db4751f96728efa4e5a637aeb8d619efd27324222283a9:656c5fe2833734f6a10f9157f8d01c980160403f237acba68c56be5c35bf67b6:130",
              "raw_metric": "Gross margin",
              "raw_period": "FY2025",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:b4dac3814c3a603cbdb3e1936c6d34168d4c6ec0ea69239b06a573991407a2bf",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "c85ddd3daccbd3f8cf815ed7254e83795a920c332a61a6216268204db3f76863",
              "handle": "F05",
              "normalized_metric": "gross margin",
              "normalized_period": "FY2024",
              "pdf_page": 130,
              "physical_source_id": "e267f6a2071a589e73db4751f96728efa4e5a637aeb8d619efd27324222283a9:656c5fe2833734f6a10f9157f8d01c980160403f237acba68c56be5c35bf67b6:130",
              "raw_metric": "Gross margin",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:582ac103a53a58d3f4988ea5b63e3f7fc74ececc79dfe45685a48b8bd38687bb",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "a34b9b1128d2fbdbc7ace64add7d8b07489af934233b6b529a08de02feb3d505",
              "handle": "F06",
              "normalized_metric": "total",
              "normalized_period": "FY2024",
              "pdf_page": 133,
              "physical_source_id": "35765f93385ef55879a2fc441d3309206f4c44651f4e9d57a7fc768a1b2b211a:4aa988fde8d82e0ed53ed4cd8989e1e74535f763aeed94a26db6ea26348853e4:133",
              "raw_metric": "Total",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:582ac103a53a58d3f4988ea5b63e3f7fc74ececc79dfe45685a48b8bd38687bb",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "dd32dd1ab18ba0fb4440ea5fdd20210c9634ec7d80d8e53823e2c961eeaae656",
              "handle": "F07",
              "normalized_metric": "total",
              "normalized_period": "FY2025",
              "pdf_page": 133,
              "physical_source_id": "35765f93385ef55879a2fc441d3309206f4c44651f4e9d57a7fc768a1b2b211a:4aa988fde8d82e0ed53ed4cd8989e1e74535f763aeed94a26db6ea26348853e4:133",
              "raw_metric": "Total",
              "raw_period": "FY2025",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:99bb7d1ef2beda913b93692c18ded991c72ae0986740bd8bd7b95d79caa04cd8",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "6fc21d1b016790d6d173f2d481761953a77838f638591a2dca28f5723c4c6b9a",
              "handle": "F08",
              "normalized_metric": "jan 26, 2025",
              "normalized_period": "FY2025",
              "pdf_page": 130,
              "physical_source_id": "e267f6a2071a589e73db4751f96728efa4e5a637aeb8d619efd27324222283a9:df7b744147a89ff9adf9a4af3c2052cdd5a569b10597d088f0ec879adbd03d7f:130",
              "raw_metric": "Jan 26, 2025",
              "raw_period": "FY2025",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:99bb7d1ef2beda913b93692c18ded991c72ae0986740bd8bd7b95d79caa04cd8",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "f2d8ed311b06d2d85cf2f0426a4e052191e05cbafdf160cd5efa2ce473e60b77",
              "handle": "F09",
              "normalized_metric": "jan 26, 2025",
              "normalized_period": "FY2024",
              "pdf_page": 130,
              "physical_source_id": "e267f6a2071a589e73db4751f96728efa4e5a637aeb8d619efd27324222283a9:df7b744147a89ff9adf9a4af3c2052cdd5a569b10597d088f0ec879adbd03d7f:130",
              "raw_metric": "Jan 26, 2025",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            }
          ],
          "constraints": {
            "metric": false,
            "period": true,
            "role": true,
            "scope": false,
            "statement": true
          },
          "question_id": "nvda_fy2025_007",
          "requirement": {
            "metric": "Data Center revenue",
            "period": "FY2025",
            "role": "value",
            "slot_id": "slot_1",
            "unit": null,
            "value_type": "numeric"
          },
          "selected": {
            "candidate_id": "candidate:v1:b8f690078d6e51c1c81063a3fe256bcfaeef4fbad4b355cc3b585fd4d275172d",
            "column_header": null,
            "column_header_path": null,
            "fact_id": "a35f94b795ba5eb227de96ab46915d61ad88545cde97ff71d0e5ba10d12bb273",
            "handle": "F02",
            "normalized_metric": "data center",
            "normalized_period": "FY2025",
            "pdf_page": 172,
            "physical_source_id": "956d139d163b90f380e6cdcb5e9f14e5b72fbdb329a22512d3ccd04bbe45f736:7e731d3eb8975007be3386499df660bfd01fa67dfec355c09fb4e82dab3eb64e:172",
            "raw_metric": "Data Center",
            "raw_period": "FY2025",
            "row_hierarchy": null,
            "row_label": null,
            "section_heading": null,
            "statement_title": null,
            "table_title": null
          },
          "slot_id": "slot_1"
        },
        {
          "classification": "OB1_partial_metric_match",
          "cohort": "D_unbindable_safety",
          "competitors": [
            {
              "candidate_id": "candidate:v1:b8f690078d6e51c1c81063a3fe256bcfaeef4fbad4b355cc3b585fd4d275172d",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "95451b882435a15e4061b201f224b1302de906578750a22f6dcee01b3cc14b80",
              "handle": "F01",
              "normalized_metric": "data center",
              "normalized_period": "FY2023",
              "pdf_page": 172,
              "physical_source_id": "956d139d163b90f380e6cdcb5e9f14e5b72fbdb329a22512d3ccd04bbe45f736:7e731d3eb8975007be3386499df660bfd01fa67dfec355c09fb4e82dab3eb64e:172",
              "raw_metric": "Data Center",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:b8f690078d6e51c1c81063a3fe256bcfaeef4fbad4b355cc3b585fd4d275172d",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "a35f94b795ba5eb227de96ab46915d61ad88545cde97ff71d0e5ba10d12bb273",
              "handle": "F02",
              "normalized_metric": "data center",
              "normalized_period": "FY2025",
              "pdf_page": 172,
              "physical_source_id": "956d139d163b90f380e6cdcb5e9f14e5b72fbdb329a22512d3ccd04bbe45f736:7e731d3eb8975007be3386499df660bfd01fa67dfec355c09fb4e82dab3eb64e:172",
              "raw_metric": "Data Center",
              "raw_period": "FY2025",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:b8f690078d6e51c1c81063a3fe256bcfaeef4fbad4b355cc3b585fd4d275172d",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "a3798e1b9f596c81bfac2c6efd9ee5652ed8355f06ef6cc0a2e6a08df2d2e2af",
              "handle": "F03",
              "normalized_metric": "data center",
              "normalized_period": "FY2024",
              "pdf_page": 172,
              "physical_source_id": "956d139d163b90f380e6cdcb5e9f14e5b72fbdb329a22512d3ccd04bbe45f736:7e731d3eb8975007be3386499df660bfd01fa67dfec355c09fb4e82dab3eb64e:172",
              "raw_metric": "Data Center",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:b4dac3814c3a603cbdb3e1936c6d34168d4c6ec0ea69239b06a573991407a2bf",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "c85ddd3daccbd3f8cf815ed7254e83795a920c332a61a6216268204db3f76863",
              "handle": "F05",
              "normalized_metric": "gross margin",
              "normalized_period": "FY2024",
              "pdf_page": 130,
              "physical_source_id": "e267f6a2071a589e73db4751f96728efa4e5a637aeb8d619efd27324222283a9:656c5fe2833734f6a10f9157f8d01c980160403f237acba68c56be5c35bf67b6:130",
              "raw_metric": "Gross margin",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:582ac103a53a58d3f4988ea5b63e3f7fc74ececc79dfe45685a48b8bd38687bb",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "a34b9b1128d2fbdbc7ace64add7d8b07489af934233b6b529a08de02feb3d505",
              "handle": "F06",
              "normalized_metric": "total",
              "normalized_period": "FY2024",
              "pdf_page": 133,
              "physical_source_id": "35765f93385ef55879a2fc441d3309206f4c44651f4e9d57a7fc768a1b2b211a:4aa988fde8d82e0ed53ed4cd8989e1e74535f763aeed94a26db6ea26348853e4:133",
              "raw_metric": "Total",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:582ac103a53a58d3f4988ea5b63e3f7fc74ececc79dfe45685a48b8bd38687bb",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "dd32dd1ab18ba0fb4440ea5fdd20210c9634ec7d80d8e53823e2c961eeaae656",
              "handle": "F07",
              "normalized_metric": "total",
              "normalized_period": "FY2025",
              "pdf_page": 133,
              "physical_source_id": "35765f93385ef55879a2fc441d3309206f4c44651f4e9d57a7fc768a1b2b211a:4aa988fde8d82e0ed53ed4cd8989e1e74535f763aeed94a26db6ea26348853e4:133",
              "raw_metric": "Total",
              "raw_period": "FY2025",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:99bb7d1ef2beda913b93692c18ded991c72ae0986740bd8bd7b95d79caa04cd8",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "6fc21d1b016790d6d173f2d481761953a77838f638591a2dca28f5723c4c6b9a",
              "handle": "F08",
              "normalized_metric": "jan 26, 2025",
              "normalized_period": "FY2025",
              "pdf_page": 130,
              "physical_source_id": "e267f6a2071a589e73db4751f96728efa4e5a637aeb8d619efd27324222283a9:df7b744147a89ff9adf9a4af3c2052cdd5a569b10597d088f0ec879adbd03d7f:130",
              "raw_metric": "Jan 26, 2025",
              "raw_period": "FY2025",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:99bb7d1ef2beda913b93692c18ded991c72ae0986740bd8bd7b95d79caa04cd8",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "f2d8ed311b06d2d85cf2f0426a4e052191e05cbafdf160cd5efa2ce473e60b77",
              "handle": "F09",
              "normalized_metric": "jan 26, 2025",
              "normalized_period": "FY2024",
              "pdf_page": 130,
              "physical_source_id": "e267f6a2071a589e73db4751f96728efa4e5a637aeb8d619efd27324222283a9:df7b744147a89ff9adf9a4af3c2052cdd5a569b10597d088f0ec879adbd03d7f:130",
              "raw_metric": "Jan 26, 2025",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            }
          ],
          "constraints": {
            "metric": false,
            "period": true,
            "role": true,
            "scope": false,
            "statement": true
          },
          "question_id": "nvda_fy2025_007",
          "requirement": {
            "metric": "GAAP gross margin percentage",
            "period": "FY2025",
            "role": "value",
            "slot_id": "slot_2",
            "unit": null,
            "value_type": "percentage"
          },
          "selected": {
            "candidate_id": "candidate:v1:b4dac3814c3a603cbdb3e1936c6d34168d4c6ec0ea69239b06a573991407a2bf",
            "column_header": null,
            "column_header_path": null,
            "fact_id": "76ea0bd1e273e5b07b57a84390905c6784ac86c9d12bdbbf9c08f7445391359d",
            "handle": "F04",
            "normalized_metric": "gross margin",
            "normalized_period": "FY2025",
            "pdf_page": 130,
            "physical_source_id": "e267f6a2071a589e73db4751f96728efa4e5a637aeb8d619efd27324222283a9:656c5fe2833734f6a10f9157f8d01c980160403f237acba68c56be5c35bf67b6:130",
            "raw_metric": "Gross margin",
            "raw_period": "FY2025",
            "row_hierarchy": null,
            "row_label": null,
            "section_heading": null,
            "statement_title": null,
            "table_title": null
          },
          "slot_id": "slot_2"
        },
        {
          "classification": "OB6_other",
          "cohort": "D_unbindable_safety",
          "competitors": [
            {
              "candidate_id": "candidate:v1:92e06b8ea2c3ea314cb8d73b0e64238b8f602251d8f160085146124b567570df",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "8d6c3deadbe575ea89c965107e28da5a3553ef095de59e525c92cc5872a38262",
              "handle": "F01",
              "normalized_metric": "revenues - comirnaty",
              "normalized_period": "FY2022",
              "pdf_page": 113,
              "physical_source_id": "f2bc4e98d06ae6fe6783019516fdc07965a90145503f74b1bc07a230d3c706db:15b5941503516157938adcd71ad3966d97c823659ef7607e83d19a6df90ae372:113",
              "raw_metric": "Revenues - Comirnaty",
              "raw_period": "FY2022",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:92e06b8ea2c3ea314cb8d73b0e64238b8f602251d8f160085146124b567570df",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "baac01f5ba852b75a1985f977f5b50afe11090278104eb76b55fbea5bdd003e9",
              "handle": "F03",
              "normalized_metric": "revenues - comirnaty",
              "normalized_period": "FY2023",
              "pdf_page": 113,
              "physical_source_id": "f2bc4e98d06ae6fe6783019516fdc07965a90145503f74b1bc07a230d3c706db:15b5941503516157938adcd71ad3966d97c823659ef7607e83d19a6df90ae372:113",
              "raw_metric": "Revenues - Comirnaty",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:0836c7f49a7d8c067eabc815f64ce953d2c65352f755afe2e152cf574c403db0",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "68d426431251ae9ad03ab76c3347e5408c62eae9c2c2d035138811f05347dc34",
              "handle": "F04",
              "normalized_metric": "(millions)",
              "normalized_period": "FY2024",
              "pdf_page": 113,
              "physical_source_id": "78179ebd90114e4886af38071efb20c643efd57d11b9d1a5864fe42aba876818:9fb3355e480762a29d0c766b906404784299c74b7bcf68811648d81932aedfd2:113",
              "raw_metric": "(MILLIONS)",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:87298bf18652abf2f069c5c3a213f1d130ca828350c64729eca073e764467863",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "0f07fb2955f6f80638b2e28c3b0d47985276aa1f8c849c2c13589dfd170636a3",
              "handle": "F05",
              "normalized_metric": "product revenues",
              "normalized_period": "FY2022",
              "pdf_page": 60,
              "physical_source_id": "a8055121fa63f48f32fb0fa7fd624ea6541155fef3395d279e4de69bfc94c3de:04569db197273ce7fc6d19dbec1704c3ccdf4cd08ab0f533dcc52c135ef754ed:60",
              "raw_metric": "Product revenues",
              "raw_period": "FY2022",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:87298bf18652abf2f069c5c3a213f1d130ca828350c64729eca073e764467863",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "5f6ef41497b04c48421bf7c26e8bd181e7327216e05f59e43fbd609eabee6c57",
              "handle": "F06",
              "normalized_metric": "product revenues",
              "normalized_period": "FY2023",
              "pdf_page": 60,
              "physical_source_id": "a8055121fa63f48f32fb0fa7fd624ea6541155fef3395d279e4de69bfc94c3de:04569db197273ce7fc6d19dbec1704c3ccdf4cd08ab0f533dcc52c135ef754ed:60",
              "raw_metric": "Product revenues",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:87298bf18652abf2f069c5c3a213f1d130ca828350c64729eca073e764467863",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "da6beb45c3ce3d9ce6ccd66ba36c6a3e8905a019f41b256683acb324d5592f5d",
              "handle": "F07",
              "normalized_metric": "product revenues",
              "normalized_period": "FY2024",
              "pdf_page": 60,
              "physical_source_id": "a8055121fa63f48f32fb0fa7fd624ea6541155fef3395d279e4de69bfc94c3de:04569db197273ce7fc6d19dbec1704c3ccdf4cd08ab0f533dcc52c135ef754ed:60",
              "raw_metric": "Product revenues",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:bfacd43d92ee188d279e6cf2deaa0358c66eb733119aaf1c221a791b28ee718b",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "36bcf8fa3e9f9221e3048c26d21d2ca3f8cdf191465b08201f355b964538c0c8",
              "handle": "F08",
              "normalized_metric": "biopharma",
              "normalized_period": "FY2024",
              "pdf_page": 115,
              "physical_source_id": "f8b75b829f7082abd87578ad9fb3d8f2153782e923ebd3921711674c7d0da868:08f3493f84b75beaebcab7a5469baa0e03e808396c4a729ce366e06b987a9fe6:115",
              "raw_metric": "BIOPHARMA",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:bfacd43d92ee188d279e6cf2deaa0358c66eb733119aaf1c221a791b28ee718b",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "7e895f748490fd52f2f720c5f12f0b726efdfc32f14d1009ff9fac3d5b071868",
              "handle": "F09",
              "normalized_metric": "biopharma",
              "normalized_period": "FY2022",
              "pdf_page": 115,
              "physical_source_id": "f8b75b829f7082abd87578ad9fb3d8f2153782e923ebd3921711674c7d0da868:08f3493f84b75beaebcab7a5469baa0e03e808396c4a729ce366e06b987a9fe6:115",
              "raw_metric": "BIOPHARMA",
              "raw_period": "FY2022",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            }
          ],
          "constraints": {
            "metric": false,
            "period": true,
            "role": true,
            "scope": false,
            "statement": false
          },
          "question_id": "pfe_fy2024_002",
          "requirement": {
            "metric": "Comirnaty revenue",
            "period": "FY2024",
            "role": "value",
            "slot_id": "1",
            "unit": null,
            "value_type": "numeric"
          },
          "selected": {
            "candidate_id": "candidate:v1:92e06b8ea2c3ea314cb8d73b0e64238b8f602251d8f160085146124b567570df",
            "column_header": null,
            "column_header_path": null,
            "fact_id": "9bfd7e02b272b7f734a02f9f55a50b948912d2632400f812bc6c84b751571fa1",
            "handle": "F02",
            "normalized_metric": "revenues - comirnaty",
            "normalized_period": "FY2024",
            "pdf_page": 113,
            "physical_source_id": "f2bc4e98d06ae6fe6783019516fdc07965a90145503f74b1bc07a230d3c706db:15b5941503516157938adcd71ad3966d97c823659ef7607e83d19a6df90ae372:113",
            "raw_metric": "Revenues - Comirnaty",
            "raw_period": "FY2024",
            "row_hierarchy": null,
            "row_label": null,
            "section_heading": null,
            "statement_title": null,
            "table_title": null
          },
          "slot_id": "1"
        },
        {
          "classification": "OB6_other",
          "cohort": "D_unbindable_safety",
          "competitors": [
            {
              "candidate_id": "candidate:v1:26f34fa7df39871b75c75a4e91675be3d60b935e464abe3154d35f9be97c0a84",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "600147525c3c3e3c33c5048c93a6b1ba582d4c2b4561f0305bb0373c96b3ee6a",
              "handle": "F02",
              "normalized_metric": "revenues - paxlovid",
              "normalized_period": "FY2022",
              "pdf_page": 113,
              "physical_source_id": "f2bc4e98d06ae6fe6783019516fdc07965a90145503f74b1bc07a230d3c706db:ac9a270be7eab1fe44d00c0be9321b51ccd0460cc0bba5459d0700d902cb5b78:113",
              "raw_metric": "Revenues - Paxlovid",
              "raw_period": "FY2022",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:26f34fa7df39871b75c75a4e91675be3d60b935e464abe3154d35f9be97c0a84",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "ad7f77f5a590448025a69f9bd2d2fb4f244691f206e1320e7fdbc30da42a1de5",
              "handle": "F03",
              "normalized_metric": "revenues - paxlovid",
              "normalized_period": "FY2023",
              "pdf_page": 113,
              "physical_source_id": "f2bc4e98d06ae6fe6783019516fdc07965a90145503f74b1bc07a230d3c706db:ac9a270be7eab1fe44d00c0be9321b51ccd0460cc0bba5459d0700d902cb5b78:113",
              "raw_metric": "Revenues - Paxlovid",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:0569e0f748ddc3feceaf810c75c2d667dcfc523de0f8b97fce0b779da379525c",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "1cefe1aba5365e950c6348c0a9fd324283289adb0777a511d62bfd2ead7a9174",
              "handle": "F04",
              "normalized_metric": "paxlovid",
              "normalized_period": "FY2024",
              "pdf_page": 45,
              "physical_source_id": "2c430320199793a7b760d36d6960516f6681f1c023066cdbda5affd03d55a469:893a4dded040490937a7621d45d011dfd34cd983c68f7157516335184dce0b95:45",
              "raw_metric": "Paxlovid",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:b39951ef221c459defd483f832eba25ba99d0583908c3df961d609d3855ceed2",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "5f94d04b2d5556282cf018b88159c9ab371e420b44b4e111e3e982726f75ec39",
              "handle": "F05",
              "normalized_metric": "product",
              "normalized_period": "fy",
              "pdf_page": 45,
              "physical_source_id": "2c430320199793a7b760d36d6960516f6681f1c023066cdbda5affd03d55a469:7740903622172fe8664012a2506744d77bd08b423e85a8e1b6d88c8ba70e3bcb:45",
              "raw_metric": "Product",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:b39951ef221c459defd483f832eba25ba99d0583908c3df961d609d3855ceed2",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "97f19c81a0dbcc9240032eaff7216f662d781343f8c9a8bef9db4a7d0db5555a",
              "handle": "F06",
              "normalized_metric": "product",
              "normalized_period": "fy",
              "pdf_page": 45,
              "physical_source_id": "2c430320199793a7b760d36d6960516f6681f1c023066cdbda5affd03d55a469:7740903622172fe8664012a2506744d77bd08b423e85a8e1b6d88c8ba70e3bcb:45",
              "raw_metric": "Product",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:4b926e6238e57a6dd528d78f653ed175f4d0b4dddec13b785315385b27186d05",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "af44566b83c4efb3f205acb5b582fe763d19b6af4bc3baa9821c29f418ac6241",
              "handle": "F07",
              "normalized_metric": "product revenues(a)",
              "normalized_period": "FY2023",
              "pdf_page": 77,
              "physical_source_id": "05b97cd07df99e3909db003181b2b66abbef63184f17de557469a0eaa54bcef7:13c1be1fff5f6ac92072bbf849a5278e2f244d06a2f662f2bdd2da8aa407f546:77",
              "raw_metric": "Product revenues(a)",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:4b926e6238e57a6dd528d78f653ed175f4d0b4dddec13b785315385b27186d05",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "b8a98cb3000b1f359fa154cfbe48b7724dd4db42b81cbadf58764397961419b2",
              "handle": "F08",
              "normalized_metric": "product revenues(a)",
              "normalized_period": "FY2024",
              "pdf_page": 77,
              "physical_source_id": "05b97cd07df99e3909db003181b2b66abbef63184f17de557469a0eaa54bcef7:13c1be1fff5f6ac92072bbf849a5278e2f244d06a2f662f2bdd2da8aa407f546:77",
              "raw_metric": "Product revenues(a)",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:4b926e6238e57a6dd528d78f653ed175f4d0b4dddec13b785315385b27186d05",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "ffd6dca4297c425f8eb788aa3c41a84385fda4ce289276e21e078cc51347b4b8",
              "handle": "F09",
              "normalized_metric": "product revenues(a)",
              "normalized_period": "FY2022",
              "pdf_page": 77,
              "physical_source_id": "05b97cd07df99e3909db003181b2b66abbef63184f17de557469a0eaa54bcef7:13c1be1fff5f6ac92072bbf849a5278e2f244d06a2f662f2bdd2da8aa407f546:77",
              "raw_metric": "Product revenues(a)",
              "raw_period": "FY2022",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            }
          ],
          "constraints": {
            "metric": false,
            "period": true,
            "role": true,
            "scope": false,
            "statement": false
          },
          "question_id": "pfe_fy2024_004",
          "requirement": {
            "metric": "Paxlovid revenue",
            "period": "FY2024",
            "role": "value",
            "slot_id": "1",
            "unit": null,
            "value_type": "numeric"
          },
          "selected": {
            "candidate_id": "candidate:v1:26f34fa7df39871b75c75a4e91675be3d60b935e464abe3154d35f9be97c0a84",
            "column_header": null,
            "column_header_path": null,
            "fact_id": "22d061547299571879907fd2c49c782481a16430e939621ffe07072428ee3cd5",
            "handle": "F01",
            "normalized_metric": "revenues - paxlovid",
            "normalized_period": "FY2024",
            "pdf_page": 113,
            "physical_source_id": "f2bc4e98d06ae6fe6783019516fdc07965a90145503f74b1bc07a230d3c706db:ac9a270be7eab1fe44d00c0be9321b51ccd0460cc0bba5459d0700d902cb5b78:113",
            "raw_metric": "Revenues - Paxlovid",
            "raw_period": "FY2024",
            "row_hierarchy": null,
            "row_label": null,
            "section_heading": null,
            "statement_title": null,
            "table_title": null
          },
          "slot_id": "1"
        },
        {
          "classification": "OB2_missing_scope_evidence",
          "cohort": "D_unbindable_safety",
          "competitors": [
            {
              "candidate_id": "candidate:v1:faf09e16325eadc4e0d87adcd77a9ae2f0c8188d250850a63daf113536b81832",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "4bf7d30f77973e75a3702dfb126b676324d3df1d4d55e9a04405a215a16158ce",
              "handle": "F01",
              "normalized_metric": "net revenue",
              "normalized_period": "FY2023",
              "pdf_page": 61,
              "physical_source_id": "b298c3e02508241934593c6c7d864c63fcc2c8159753b43f5c0c077a2a24de53:012ca63ad655186409b9e0319862671bce2f21ea9a97d3094d4ceb1bd22fb53c:61",
              "raw_metric": "Net revenue",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:289c0336e376d5107ddec815a794c98a560365f265de95c0f3f45ee1e878ba83",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "b977e2afc7a4a2d9f6f2db3d4e57661cba2930fdf649c25855bf7cb0cd2be173",
              "handle": "F03",
              "normalized_metric": "net revenue",
              "normalized_period": "FY2023",
              "pdf_page": 75,
              "physical_source_id": "047c6a56c7f09d391ac993d738c97b84bae5339ad9a1873ff5d6dd98ff693439:9cea965a621d0707927d0252efb5f95117adc2b590b495799bd051d069bd2eae:75",
              "raw_metric": "Net revenue",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:289c0336e376d5107ddec815a794c98a560365f265de95c0f3f45ee1e878ba83",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "c8d61066a50fdbd0b76129b27af245ed97f4a3a38bb4cf26e81b824be4377759",
              "handle": "F04",
              "normalized_metric": "net revenue",
              "normalized_period": "FY2024",
              "pdf_page": 75,
              "physical_source_id": "047c6a56c7f09d391ac993d738c97b84bae5339ad9a1873ff5d6dd98ff693439:9cea965a621d0707927d0252efb5f95117adc2b590b495799bd051d069bd2eae:75",
              "raw_metric": "Net revenue",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:1360b5ebea3eec01f428ede6db02ec4a235ef9d67b03c02f93b140a70ad5a3b7",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "b48828cafd4170bd626ddf12b0a3edfd6b3a40ca6f0ed37b09701cfcb219fb83",
              "handle": "F05",
              "normalized_metric": "net revenue",
              "normalized_period": "FY2023",
              "pdf_page": 55,
              "physical_source_id": "a2b6b99ae6d9e7df75bafbe29c0b2b5dbbdfca535652b3580561fc8e72fbbd6b:6337e85f2145b8806da2191701af4a9750576550b94721c3be27a54ff2cd7082:55",
              "raw_metric": "Net revenue",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:747783bcac47c8f71b61c289a33c8b6b6b7121f23d39f8487eade7fc23fe3154",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "bb5e24ce87662cd8f8617c4eaa5d04033d62a63114d1edb69f8ee443201c9128",
              "handle": "F06",
              "normalized_metric": "net revenue",
              "normalized_period": "fy",
              "pdf_page": 4,
              "physical_source_id": "036b51da4914aa39a6c95956382ad7ab03fede68ec04f97a158f7dabf74cd2c2:04f3b8550b72399b5c7e4ade2474d1d85b0e8333fe4c24df978f254e108eb4d1:4",
              "raw_metric": "Net revenue",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:747783bcac47c8f71b61c289a33c8b6b6b7121f23d39f8487eade7fc23fe3154",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "cddd4d6326a80953034ff11f4ad06c852005914112b3a46f52b3eb5a2f48dd8e",
              "handle": "F07",
              "normalized_metric": "net revenue",
              "normalized_period": "fy",
              "pdf_page": 4,
              "physical_source_id": "036b51da4914aa39a6c95956382ad7ab03fede68ec04f97a158f7dabf74cd2c2:04f3b8550b72399b5c7e4ade2474d1d85b0e8333fe4c24df978f254e108eb4d1:4",
              "raw_metric": "Net revenue",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:747783bcac47c8f71b61c289a33c8b6b6b7121f23d39f8487eade7fc23fe3154",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "ce0fa445a1830723f871cb2e61588ac0069c9d601112e5901e95bb86b810bc41",
              "handle": "F08",
              "normalized_metric": "net revenue",
              "normalized_period": "fy",
              "pdf_page": 4,
              "physical_source_id": "036b51da4914aa39a6c95956382ad7ab03fede68ec04f97a158f7dabf74cd2c2:04f3b8550b72399b5c7e4ade2474d1d85b0e8333fe4c24df978f254e108eb4d1:4",
              "raw_metric": "Net revenue",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:13433125d54be24c518e851795226daa0bfef93e7c737440f4b7eb284fb1245f",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "64c7d1fb6c550ac785ec3c413e288620ec2b50e8afa6bf987941b1160c71f39b",
              "handle": "F09",
              "normalized_metric": "net income",
              "normalized_period": "fy",
              "pdf_page": 55,
              "physical_source_id": "a2b6b99ae6d9e7df75bafbe29c0b2b5dbbdfca535652b3580561fc8e72fbbd6b:5ca3e413e7fab20c9dac013a9c3acaa6bf54d61d35d7a47e2513d019f61c49d0:55",
              "raw_metric": "Net income",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            }
          ],
          "constraints": {
            "metric": true,
            "period": true,
            "role": true,
            "scope": true,
            "statement": true
          },
          "question_id": "v_fy2025_006",
          "requirement": {
            "metric": "net revenue",
            "period": "FY2025",
            "role": "current_period",
            "slot_id": "s1",
            "unit": null,
            "value_type": "numeric"
          },
          "selected": {
            "candidate_id": "candidate:v1:289c0336e376d5107ddec815a794c98a560365f265de95c0f3f45ee1e878ba83",
            "column_header": null,
            "column_header_path": null,
            "fact_id": "2a4da6647be88d9aa9a04e8c7e015e7a277ea028374f3dbe8778b5a6b97de9c3",
            "handle": "F02",
            "normalized_metric": "net revenue",
            "normalized_period": "FY2025",
            "pdf_page": 75,
            "physical_source_id": "047c6a56c7f09d391ac993d738c97b84bae5339ad9a1873ff5d6dd98ff693439:9cea965a621d0707927d0252efb5f95117adc2b590b495799bd051d069bd2eae:75",
            "raw_metric": "Net revenue",
            "raw_period": "FY2025",
            "row_hierarchy": null,
            "row_label": null,
            "section_heading": null,
            "statement_title": null,
            "table_title": null
          },
          "slot_id": "s1"
        },
        {
          "classification": "OB4_statement_conflict",
          "cohort": "D_unbindable_safety",
          "competitors": [
            {
              "candidate_id": "candidate:v1:faf09e16325eadc4e0d87adcd77a9ae2f0c8188d250850a63daf113536b81832",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "4bf7d30f77973e75a3702dfb126b676324d3df1d4d55e9a04405a215a16158ce",
              "handle": "F01",
              "normalized_metric": "net revenue",
              "normalized_period": "FY2023",
              "pdf_page": 61,
              "physical_source_id": "b298c3e02508241934593c6c7d864c63fcc2c8159753b43f5c0c077a2a24de53:012ca63ad655186409b9e0319862671bce2f21ea9a97d3094d4ceb1bd22fb53c:61",
              "raw_metric": "Net revenue",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:289c0336e376d5107ddec815a794c98a560365f265de95c0f3f45ee1e878ba83",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "2a4da6647be88d9aa9a04e8c7e015e7a277ea028374f3dbe8778b5a6b97de9c3",
              "handle": "F02",
              "normalized_metric": "net revenue",
              "normalized_period": "FY2025",
              "pdf_page": 75,
              "physical_source_id": "047c6a56c7f09d391ac993d738c97b84bae5339ad9a1873ff5d6dd98ff693439:9cea965a621d0707927d0252efb5f95117adc2b590b495799bd051d069bd2eae:75",
              "raw_metric": "Net revenue",
              "raw_period": "FY2025",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:289c0336e376d5107ddec815a794c98a560365f265de95c0f3f45ee1e878ba83",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "b977e2afc7a4a2d9f6f2db3d4e57661cba2930fdf649c25855bf7cb0cd2be173",
              "handle": "F03",
              "normalized_metric": "net revenue",
              "normalized_period": "FY2023",
              "pdf_page": 75,
              "physical_source_id": "047c6a56c7f09d391ac993d738c97b84bae5339ad9a1873ff5d6dd98ff693439:9cea965a621d0707927d0252efb5f95117adc2b590b495799bd051d069bd2eae:75",
              "raw_metric": "Net revenue",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:1360b5ebea3eec01f428ede6db02ec4a235ef9d67b03c02f93b140a70ad5a3b7",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "b48828cafd4170bd626ddf12b0a3edfd6b3a40ca6f0ed37b09701cfcb219fb83",
              "handle": "F05",
              "normalized_metric": "net revenue",
              "normalized_period": "FY2023",
              "pdf_page": 55,
              "physical_source_id": "a2b6b99ae6d9e7df75bafbe29c0b2b5dbbdfca535652b3580561fc8e72fbbd6b:6337e85f2145b8806da2191701af4a9750576550b94721c3be27a54ff2cd7082:55",
              "raw_metric": "Net revenue",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:747783bcac47c8f71b61c289a33c8b6b6b7121f23d39f8487eade7fc23fe3154",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "bb5e24ce87662cd8f8617c4eaa5d04033d62a63114d1edb69f8ee443201c9128",
              "handle": "F06",
              "normalized_metric": "net revenue",
              "normalized_period": "fy",
              "pdf_page": 4,
              "physical_source_id": "036b51da4914aa39a6c95956382ad7ab03fede68ec04f97a158f7dabf74cd2c2:04f3b8550b72399b5c7e4ade2474d1d85b0e8333fe4c24df978f254e108eb4d1:4",
              "raw_metric": "Net revenue",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:747783bcac47c8f71b61c289a33c8b6b6b7121f23d39f8487eade7fc23fe3154",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "cddd4d6326a80953034ff11f4ad06c852005914112b3a46f52b3eb5a2f48dd8e",
              "handle": "F07",
              "normalized_metric": "net revenue",
              "normalized_period": "fy",
              "pdf_page": 4,
              "physical_source_id": "036b51da4914aa39a6c95956382ad7ab03fede68ec04f97a158f7dabf74cd2c2:04f3b8550b72399b5c7e4ade2474d1d85b0e8333fe4c24df978f254e108eb4d1:4",
              "raw_metric": "Net revenue",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:747783bcac47c8f71b61c289a33c8b6b6b7121f23d39f8487eade7fc23fe3154",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "ce0fa445a1830723f871cb2e61588ac0069c9d601112e5901e95bb86b810bc41",
              "handle": "F08",
              "normalized_metric": "net revenue",
              "normalized_period": "fy",
              "pdf_page": 4,
              "physical_source_id": "036b51da4914aa39a6c95956382ad7ab03fede68ec04f97a158f7dabf74cd2c2:04f3b8550b72399b5c7e4ade2474d1d85b0e8333fe4c24df978f254e108eb4d1:4",
              "raw_metric": "Net revenue",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:13433125d54be24c518e851795226daa0bfef93e7c737440f4b7eb284fb1245f",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "64c7d1fb6c550ac785ec3c413e288620ec2b50e8afa6bf987941b1160c71f39b",
              "handle": "F09",
              "normalized_metric": "net income",
              "normalized_period": "fy",
              "pdf_page": 55,
              "physical_source_id": "a2b6b99ae6d9e7df75bafbe29c0b2b5dbbdfca535652b3580561fc8e72fbbd6b:5ca3e413e7fab20c9dac013a9c3acaa6bf54d61d35d7a47e2513d019f61c49d0:55",
              "raw_metric": "Net income",
              "raw_period": "fy",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            }
          ],
          "constraints": {
            "metric": true,
            "period": true,
            "role": true,
            "scope": true,
            "statement": false
          },
          "question_id": "v_fy2025_006",
          "requirement": {
            "metric": "net revenue",
            "period": "FY2024",
            "role": "base_period",
            "slot_id": "s2",
            "unit": null,
            "value_type": "numeric"
          },
          "selected": {
            "candidate_id": "candidate:v1:289c0336e376d5107ddec815a794c98a560365f265de95c0f3f45ee1e878ba83",
            "column_header": null,
            "column_header_path": null,
            "fact_id": "c8d61066a50fdbd0b76129b27af245ed97f4a3a38bb4cf26e81b824be4377759",
            "handle": "F04",
            "normalized_metric": "net revenue",
            "normalized_period": "FY2024",
            "pdf_page": 75,
            "physical_source_id": "047c6a56c7f09d391ac993d738c97b84bae5339ad9a1873ff5d6dd98ff693439:9cea965a621d0707927d0252efb5f95117adc2b590b495799bd051d069bd2eae:75",
            "raw_metric": "Net revenue",
            "raw_period": "FY2024",
            "row_hierarchy": null,
            "row_label": null,
            "section_heading": null,
            "statement_title": null,
            "table_title": null
          },
          "slot_id": "s2"
        },
        {
          "classification": "OB1_partial_metric_match",
          "cohort": "D_unbindable_safety",
          "competitors": [
            {
              "candidate_id": "candidate:v1:38e40720cd64c77589937b3a83bf3659c1f6f7c5e1dd05d897fb593bdaf26a68",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "0d275e94bd8c411d5cd48e132ed3cfb6207fb60851d3f87598c52a2e822b0a44",
              "handle": "F01",
              "normalized_metric": "transactions processed on visa's networks2",
              "normalized_period": "FY2025",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:c48ae62bcaae673a2779f950221850202adeda0cb298d10f377e450231e31e8b:4",
              "raw_metric": "Transactions processed on Visa's networks2",
              "raw_period": "FY2025",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:38e40720cd64c77589937b3a83bf3659c1f6f7c5e1dd05d897fb593bdaf26a68",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "74820f5a381a6d55fa1e68cdc001e6e8dc0c490f272b25db121a6fdb376e710d",
              "handle": "F02",
              "normalized_metric": "transactions processed on visa's networks2",
              "normalized_period": "FY2023",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:c48ae62bcaae673a2779f950221850202adeda0cb298d10f377e450231e31e8b:4",
              "raw_metric": "Transactions processed on Visa's networks2",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:38e40720cd64c77589937b3a83bf3659c1f6f7c5e1dd05d897fb593bdaf26a68",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "7abae283bf0c21bc1bfb307f6a75943e456c93d571c8839d3fad4f52880cdc48",
              "handle": "F03",
              "normalized_metric": "transactions processed on visa's networks2",
              "normalized_period": "FY2024",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:c48ae62bcaae673a2779f950221850202adeda0cb298d10f377e450231e31e8b:4",
              "raw_metric": "Transactions processed on Visa's networks2",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:9345b21bb1fae47731ceb136aa7846f1adca5024d7d31ded608e9f03a486e186",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "4c35227ee925f1d09de33e0714c5e05c89c4beba383342a0e254bd37fa691a61",
              "handle": "F04",
              "normalized_metric": "payments volume1",
              "normalized_period": "FY2024",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:93925e15ad34715a4b43601361d4ca1d949e380955f4fe20f934d71fd8c8a869:4",
              "raw_metric": "Payments volume1",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:9345b21bb1fae47731ceb136aa7846f1adca5024d7d31ded608e9f03a486e186",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "85a5e7bf2ad766e23ceefb0a5301d1a9eff9e6d80441eee9d1819e5450ff0794",
              "handle": "F06",
              "normalized_metric": "payments volume1",
              "normalized_period": "FY2023",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:93925e15ad34715a4b43601361d4ca1d949e380955f4fe20f934d71fd8c8a869:4",
              "raw_metric": "Payments volume1",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            }
          ],
          "constraints": {
            "metric": false,
            "period": true,
            "role": true,
            "scope": false,
            "statement": true
          },
          "question_id": "v_fy2025_007",
          "requirement": {
            "metric": "payments volume",
            "period": "FY2025",
            "role": "value",
            "slot_id": "1",
            "unit": null,
            "value_type": "numeric"
          },
          "selected": {
            "candidate_id": "candidate:v1:9345b21bb1fae47731ceb136aa7846f1adca5024d7d31ded608e9f03a486e186",
            "column_header": null,
            "column_header_path": null,
            "fact_id": "71bd06d15b6465ce3cc482db9381a3216507716c6a627de5f6408bab3999e909",
            "handle": "F05",
            "normalized_metric": "payments volume1",
            "normalized_period": "FY2025",
            "pdf_page": 4,
            "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:93925e15ad34715a4b43601361d4ca1d949e380955f4fe20f934d71fd8c8a869:4",
            "raw_metric": "Payments volume1",
            "raw_period": "FY2025",
            "row_hierarchy": null,
            "row_label": null,
            "section_heading": null,
            "statement_title": null,
            "table_title": null
          },
          "slot_id": "1"
        },
        {
          "classification": "OB6_other",
          "cohort": "D_unbindable_safety",
          "competitors": [
            {
              "candidate_id": "candidate:v1:38e40720cd64c77589937b3a83bf3659c1f6f7c5e1dd05d897fb593bdaf26a68",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "74820f5a381a6d55fa1e68cdc001e6e8dc0c490f272b25db121a6fdb376e710d",
              "handle": "F02",
              "normalized_metric": "transactions processed on visa's networks2",
              "normalized_period": "FY2023",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:c48ae62bcaae673a2779f950221850202adeda0cb298d10f377e450231e31e8b:4",
              "raw_metric": "Transactions processed on Visa's networks2",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:38e40720cd64c77589937b3a83bf3659c1f6f7c5e1dd05d897fb593bdaf26a68",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "7abae283bf0c21bc1bfb307f6a75943e456c93d571c8839d3fad4f52880cdc48",
              "handle": "F03",
              "normalized_metric": "transactions processed on visa's networks2",
              "normalized_period": "FY2024",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:c48ae62bcaae673a2779f950221850202adeda0cb298d10f377e450231e31e8b:4",
              "raw_metric": "Transactions processed on Visa's networks2",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:9345b21bb1fae47731ceb136aa7846f1adca5024d7d31ded608e9f03a486e186",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "4c35227ee925f1d09de33e0714c5e05c89c4beba383342a0e254bd37fa691a61",
              "handle": "F04",
              "normalized_metric": "payments volume1",
              "normalized_period": "FY2024",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:93925e15ad34715a4b43601361d4ca1d949e380955f4fe20f934d71fd8c8a869:4",
              "raw_metric": "Payments volume1",
              "raw_period": "FY2024",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:9345b21bb1fae47731ceb136aa7846f1adca5024d7d31ded608e9f03a486e186",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "71bd06d15b6465ce3cc482db9381a3216507716c6a627de5f6408bab3999e909",
              "handle": "F05",
              "normalized_metric": "payments volume1",
              "normalized_period": "FY2025",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:93925e15ad34715a4b43601361d4ca1d949e380955f4fe20f934d71fd8c8a869:4",
              "raw_metric": "Payments volume1",
              "raw_period": "FY2025",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            },
            {
              "candidate_id": "candidate:v1:9345b21bb1fae47731ceb136aa7846f1adca5024d7d31ded608e9f03a486e186",
              "column_header": null,
              "column_header_path": null,
              "fact_id": "85a5e7bf2ad766e23ceefb0a5301d1a9eff9e6d80441eee9d1819e5450ff0794",
              "handle": "F06",
              "normalized_metric": "payments volume1",
              "normalized_period": "FY2023",
              "pdf_page": 4,
              "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:93925e15ad34715a4b43601361d4ca1d949e380955f4fe20f934d71fd8c8a869:4",
              "raw_metric": "Payments volume1",
              "raw_period": "FY2023",
              "row_hierarchy": null,
              "row_label": null,
              "section_heading": null,
              "statement_title": null,
              "table_title": null
            }
          ],
          "constraints": {
            "metric": false,
            "period": true,
            "role": true,
            "scope": false,
            "statement": true
          },
          "question_id": "v_fy2025_007",
          "requirement": {
            "metric": "number of transactions processed on Visa's networks",
            "period": "FY2025",
            "role": "value",
            "slot_id": "2",
            "unit": null,
            "value_type": "numeric"
          },
          "selected": {
            "candidate_id": "candidate:v1:38e40720cd64c77589937b3a83bf3659c1f6f7c5e1dd05d897fb593bdaf26a68",
            "column_header": null,
            "column_header_path": null,
            "fact_id": "0d275e94bd8c411d5cd48e132ed3cfb6207fb60851d3f87598c52a2e822b0a44",
            "handle": "F01",
            "normalized_metric": "transactions processed on visa's networks2",
            "normalized_period": "FY2025",
            "pdf_page": 4,
            "physical_source_id": "de5f2135a73332bc4b56938ced9dbde61886567defe3496a720b7f03cc40b4ab:c48ae62bcaae673a2779f950221850202adeda0cb298d10f377e450231e31e8b:4",
            "raw_metric": "Transactions processed on Visa's networks2",
            "raw_period": "FY2025",
            "row_hierarchy": null,
            "row_label": null,
            "section_heading": null,
            "statement_title": null,
            "table_title": null
          },
          "slot_id": "2"
        }
      ]
    }
  },
  "reason": "cost_and_project_scope",
  "synthetic": {
    "adapter_valid": 14,
    "benchmark_questions_used": 0,
    "binding_validator": 14,
    "calculation_correct_slots": 8,
    "calculation_groups_correct": 4,
    "calculation_groups_total": 6,
    "calculation_total_slots": 12,
    "dto_valid": 14,
    "formulation": "batched_pairwise_compatibility_v1",
    "gate": "NF-V2-03-R5.1",
    "indistinguishable_safe": 2,
    "indistinguishable_total": 2,
    "model": "qwen3.7-plus",
    "pass": true,
    "provider_calls": 14,
    "provider_success": 14,
    "rows": [
      {
        "actual": {
          "s1": [
            "F01"
          ]
        },
        "adapter_valid": true,
        "ambiguous_slots": [],
        "binder_model_status": "BOUND",
        "binding": {
          "ambiguous_slots": [],
          "invalid_reasons": [],
          "missing_slots": [],
          "slot_bindings": {
            "s1": [
              "f01"
            ]
          },
          "status": "BOUND"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
        "calculation_correct_slots": 0,
        "calculation_total_slots": 0,
        "dto_valid": true,
        "expected": {
          "s1": [
            "F01"
          ]
        },
        "fact_count": 2,
        "false_binding": false,
        "final_binding_status": "BOUND",
        "groups": [
          "synthetic"
        ],
        "intent": "DIRECT_FACT",
        "invalid_reasons": [],
        "metadata": {
          "errno": null,
          "error": null,
          "exception_cause_message": null,
          "exception_cause_type": null,
          "exception_chain": [],
          "exception_type": null,
          "finish_reason": "stop",
          "http_status": null,
          "input_tokens": 729,
          "latency_ms": 2163.474,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 39,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 71,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-2702bb6d-6c25-9d21-9d22-aaf17a7e9615",
          "structured_output_success": true,
          "total_tokens": 768
        },
        "missing_slots": [],
        "operation": null,
        "pairwise_outcomes": {
          "s1": {
            "F01": "MATCH",
            "F02": "REJECT"
          }
        },
        "provider_response_success": true,
        "question": "Select the supplied revenue fact.",
        "question_id": "v2_syn_01",
        "raw_response": "{\n  \"tasks\": {\n   \"s1\": {\n   \"F01\": \"MATCH\",\n   \"F02\": \"REJECT\"\n  }\n}\n}",
        "required_slots": [
          {
            "metric": "revenue",
            "period": "FY2026",
            "role": "value",
            "slot_id": "s1",
            "unit": null,
            "value_type": "numeric"
          }
        ],
        "safe_indistinguishable": false,
        "selected_fact_ids": [
          "f01"
        ],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 0,
        "slots_bound": 1,
        "slots_missing": 0,
        "slots_requested": 1,
        "structured_output_success": true,
        "tags": [],
        "validation_reasons": []
      },
      {
        "actual": {
          "s1": [
            "F01"
          ]
        },
        "adapter_valid": true,
        "ambiguous_slots": [],
        "binder_model_status": "BOUND",
        "binding": {
          "ambiguous_slots": [],
          "invalid_reasons": [],
          "missing_slots": [],
          "slot_bindings": {
            "s1": [
              "f01"
            ]
          },
          "status": "BOUND"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
        "calculation_correct_slots": 0,
        "calculation_total_slots": 0,
        "dto_valid": true,
        "expected": {
          "s1": [
            "F01"
          ]
        },
        "fact_count": 2,
        "false_binding": false,
        "final_binding_status": "BOUND",
        "groups": [
          "synthetic"
        ],
        "intent": "DIRECT_FACT",
        "invalid_reasons": [],
        "metadata": {
          "errno": null,
          "error": null,
          "exception_cause_message": null,
          "exception_cause_type": null,
          "exception_chain": [],
          "exception_type": null,
          "finish_reason": "stop",
          "http_status": null,
          "input_tokens": 743,
          "latency_ms": 1530.473,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 39,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 71,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-eaad070e-aeaf-91b1-8d1f-0cdd4fed880f",
          "structured_output_success": true,
          "total_tokens": 782
        },
        "missing_slots": [],
        "operation": null,
        "pairwise_outcomes": {
          "s1": {
            "F01": "MATCH",
            "F02": "REJECT"
          }
        },
        "provider_response_success": true,
        "question": "Select the operating income from the correct statement.",
        "question_id": "v2_syn_02",
        "raw_response": "{\n  \"tasks\": {\n   \"s1\": {\n   \"F01\": \"MATCH\",\n   \"F02\": \"REJECT\"\n  }\n}\n}",
        "required_slots": [
          {
            "metric": "operating income",
            "period": "FY2026",
            "role": "value",
            "slot_id": "s1",
            "unit": null,
            "value_type": "numeric"
          }
        ],
        "safe_indistinguishable": false,
        "selected_fact_ids": [
          "f01"
        ],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 0,
        "slots_bound": 1,
        "slots_missing": 0,
        "slots_requested": 1,
        "structured_output_success": true,
        "tags": [],
        "validation_reasons": []
      },
      {
        "actual": {
          "s1": [
            "F01",
            "F02"
          ]
        },
        "adapter_valid": true,
        "ambiguous_slots": [
          "s1"
        ],
        "binder_model_status": "AMBIGUOUS",
        "binding": {
          "ambiguous_slots": [
            "s1"
          ],
          "invalid_reasons": [],
          "missing_slots": [],
          "slot_bindings": {
            "s1": [
              "f01",
              "f02"
            ]
          },
          "status": "AMBIGUOUS"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
        "calculation_correct_slots": 0,
        "calculation_total_slots": 0,
        "dto_valid": true,
        "expected": {
          "s1": [
            "F01"
          ]
        },
        "fact_count": 2,
        "false_binding": false,
        "final_binding_status": "AMBIGUOUS",
        "groups": [
          "synthetic"
        ],
        "intent": "DIRECT_FACT",
        "invalid_reasons": [],
        "metadata": {
          "errno": null,
          "error": null,
          "exception_cause_message": null,
          "exception_cause_type": null,
          "exception_chain": [],
          "exception_type": null,
          "finish_reason": "stop",
          "http_status": null,
          "input_tokens": 739,
          "latency_ms": 1378.285,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 39,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 73,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-377ed121-1cf8-9d8e-b74d-e2fb643a21d8",
          "structured_output_success": true,
          "total_tokens": 778
        },
        "missing_slots": [],
        "operation": null,
        "pairwise_outcomes": {
          "s1": {
            "F01": "MATCH",
            "F02": "MATCH"
          }
        },
        "provider_response_success": true,
        "question": "Select the regional row.",
        "question_id": "v2_syn_03",
        "raw_response": "{\n  \"tasks\": {\n   \"s1\": {\n   \"F01\": \"MATCH\",\n   \"F02\": \"MATCH\"\n   }\n  }\n}",
        "required_slots": [
          {
            "metric": "revenue",
            "period": "FY2026",
            "role": "value",
            "slot_id": "s1",
            "unit": null,
            "value_type": "numeric"
          }
        ],
        "safe_indistinguishable": false,
        "selected_fact_ids": [
          "f01",
          "f02"
        ],
        "semantic_correct": false,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 1,
        "slots_bound": 1,
        "slots_missing": 0,
        "slots_requested": 1,
        "structured_output_success": true,
        "tags": [],
        "validation_reasons": []
      },
      {
        "actual": {
          "s1": [
            "F01"
          ]
        },
        "adapter_valid": true,
        "ambiguous_slots": [],
        "binder_model_status": "BOUND",
        "binding": {
          "ambiguous_slots": [],
          "invalid_reasons": [],
          "missing_slots": [],
          "slot_bindings": {
            "s1": [
              "f01"
            ]
          },
          "status": "BOUND"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
        "calculation_correct_slots": 0,
        "calculation_total_slots": 0,
        "dto_valid": true,
        "expected": {
          "s1": [
            "F01"
          ]
        },
        "fact_count": 2,
        "false_binding": false,
        "final_binding_status": "BOUND",
        "groups": [
          "synthetic"
        ],
        "intent": "DIRECT_FACT",
        "invalid_reasons": [],
        "metadata": {
          "errno": null,
          "error": null,
          "exception_cause_message": null,
          "exception_cause_type": null,
          "exception_chain": [],
          "exception_type": null,
          "finish_reason": "stop",
          "http_status": null,
          "input_tokens": 737,
          "latency_ms": 1898.167,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 39,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 71,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-316d38a8-7df2-9cd0-93cf-a34256992091",
          "structured_output_success": true,
          "total_tokens": 776
        },
        "missing_slots": [],
        "operation": null,
        "pairwise_outcomes": {
          "s1": {
            "F01": "MATCH",
            "F02": "REJECT"
          }
        },
        "provider_response_success": true,
        "question": "Select the fact under the current header.",
        "question_id": "v2_syn_04",
        "raw_response": "{\n  \"tasks\": {\n   \"s1\": {\n   \"F01\": \"MATCH\",\n   \"F02\": \"REJECT\"\n  }\n}\n}",
        "required_slots": [
          {
            "metric": "units",
            "period": "FY2026",
            "role": "value",
            "slot_id": "s1",
            "unit": null,
            "value_type": "numeric"
          }
        ],
        "safe_indistinguishable": false,
        "selected_fact_ids": [
          "f01"
        ],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 0,
        "slots_bound": 1,
        "slots_missing": 0,
        "slots_requested": 1,
        "structured_output_success": true,
        "tags": [],
        "validation_reasons": []
      },
      {
        "actual": {
          "current": [
            "F01"
          ],
          "prior": [
            "F02"
          ]
        },
        "adapter_valid": true,
        "ambiguous_slots": [],
        "binder_model_status": "BOUND",
        "binding": {
          "ambiguous_slots": [],
          "invalid_reasons": [],
          "missing_slots": [],
          "slot_bindings": {
            "current": [
              "f01"
            ],
            "prior": [
              "f02"
            ]
          },
          "status": "BOUND"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
        "calculation_correct_slots": 0,
        "calculation_total_slots": 0,
        "dto_valid": true,
        "expected": {
          "current": [
            "F01"
          ],
          "prior": [
            "F02"
          ]
        },
        "fact_count": 2,
        "false_binding": false,
        "final_binding_status": "BOUND",
        "groups": [
          "synthetic"
        ],
        "intent": "CALCULATION",
        "invalid_reasons": [],
        "metadata": {
          "errno": null,
          "error": null,
          "exception_cause_message": null,
          "exception_cause_type": null,
          "exception_chain": [],
          "exception_type": null,
          "finish_reason": "stop",
          "http_status": null,
          "input_tokens": 775,
          "latency_ms": 2133.31,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 69,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 124,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-08a6c3d1-f8dd-9a63-8610-d91bf819b04b",
          "structured_output_success": true,
          "total_tokens": 844
        },
        "missing_slots": [],
        "operation": "growth_rate",
        "pairwise_outcomes": {
          "current": {
            "F01": "MATCH",
            "F02": "REJECT"
          },
          "prior": {
            "F01": "REJECT",
            "F02": "MATCH"
          }
        },
        "provider_response_success": true,
        "question": "Select current and prior independently.",
        "question_id": "v2_syn_05",
        "raw_response": "{\n  \"tasks\": {\n   \"current\": {\n  \"F01\": \"MATCH\",\n  \"F02\": \"REJECT\"\n},\n\"prior\": {\n  \"F01\": \"REJECT\",\n  \"F02\": \"MATCH\"\n}\n  }\n}",
        "required_slots": [
          {
            "metric": "sales",
            "period": "FY2026",
            "role": "current",
            "slot_id": "current",
            "unit": null,
            "value_type": "numeric"
          },
          {
            "metric": "sales",
            "period": "FY2025",
            "role": "prior",
            "slot_id": "prior",
            "unit": null,
            "value_type": "numeric"
          }
        ],
        "safe_indistinguishable": false,
        "selected_fact_ids": [
          "f01",
          "f02"
        ],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 0,
        "slots_bound": 2,
        "slots_missing": 0,
        "slots_requested": 2,
        "structured_output_success": true,
        "tags": [],
        "validation_reasons": []
      },
      {
        "actual": {
          "denominator": [
            "F02"
          ],
          "numerator": [
            "F01"
          ]
        },
        "adapter_valid": true,
        "ambiguous_slots": [],
        "binder_model_status": "BOUND",
        "binding": {
          "ambiguous_slots": [],
          "invalid_reasons": [],
          "missing_slots": [],
          "slot_bindings": {
            "denominator": [
              "f02"
            ],
            "numerator": [
              "f01"
            ]
          },
          "status": "BOUND"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
        "calculation_correct_slots": 0,
        "calculation_total_slots": 0,
        "dto_valid": true,
        "expected": {
          "denominator": [
            "F02"
          ],
          "numerator": [
            "F01"
          ]
        },
        "fact_count": 2,
        "false_binding": false,
        "final_binding_status": "BOUND",
        "groups": [
          "synthetic"
        ],
        "intent": "CALCULATION",
        "invalid_reasons": [],
        "metadata": {
          "errno": null,
          "error": null,
          "exception_cause_message": null,
          "exception_cause_type": null,
          "exception_chain": [],
          "exception_type": null,
          "finish_reason": "stop",
          "http_status": null,
          "input_tokens": 791,
          "latency_ms": 1839.553,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 63,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 124,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-61515579-ae30-9f0c-a611-53c2d868f301",
          "structured_output_success": true,
          "total_tokens": 854
        },
        "missing_slots": [],
        "operation": "percentage_share",
        "pairwise_outcomes": {
          "denominator": {
            "F01": "REJECT",
            "F02": "MATCH"
          },
          "numerator": {
            "F01": "MATCH",
            "F02": "REJECT"
          }
        },
        "provider_response_success": true,
        "question": "Select numerator and denominator.",
        "question_id": "v2_syn_06",
        "raw_response": "{\n  \"tasks\": {\n   \"numerator\": {\n\"F01\": \"MATCH\",\n\"F02\": \"REJECT\"\n},\n\"denominator\": {\n\"F01\": \"REJECT\",\n\"F02\": \"MATCH\"\n}\n  }\n}",
        "required_slots": [
          {
            "metric": "gross profit",
            "period": "FY2026",
            "role": "numerator",
            "slot_id": "numerator",
            "unit": null,
            "value_type": "numeric"
          },
          {
            "metric": "sales",
            "period": "FY2026",
            "role": "denominator",
            "slot_id": "denominator",
            "unit": null,
            "value_type": "numeric"
          }
        ],
        "safe_indistinguishable": false,
        "selected_fact_ids": [
          "f01",
          "f02"
        ],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 0,
        "slots_bound": 2,
        "slots_missing": 0,
        "slots_requested": 2,
        "structured_output_success": true,
        "tags": [],
        "validation_reasons": []
      },
      {
        "actual": {},
        "adapter_valid": true,
        "ambiguous_slots": [],
        "binder_model_status": "MISSING",
        "binding": {
          "ambiguous_slots": [],
          "invalid_reasons": [],
          "missing_slots": [
            "s1"
          ],
          "slot_bindings": {},
          "status": "MISSING"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
        "calculation_correct_slots": 0,
        "calculation_total_slots": 0,
        "dto_valid": true,
        "expected": {
          "s1": []
        },
        "fact_count": 2,
        "false_binding": false,
        "final_binding_status": "MISSING",
        "groups": [
          "synthetic"
        ],
        "intent": "DIRECT_FACT",
        "invalid_reasons": [],
        "metadata": {
          "errno": null,
          "error": null,
          "exception_cause_message": null,
          "exception_cause_type": null,
          "exception_chain": [],
          "exception_type": null,
          "finish_reason": "stop",
          "http_status": null,
          "input_tokens": 729,
          "latency_ms": 1382.644,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 40,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 72,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-1d5ddb32-f7af-9c57-8dc2-9f084761c8d1",
          "structured_output_success": true,
          "total_tokens": 769
        },
        "missing_slots": [
          "s1"
        ],
        "operation": null,
        "pairwise_outcomes": {
          "s1": {
            "F01": "REJECT",
            "F02": "REJECT"
          }
        },
        "provider_response_success": true,
        "question": "No supplied fact has the requested scope.",
        "question_id": "v2_syn_07",
        "raw_response": "{\n  \"tasks\": {\n   \"s1\": {\n   \"F01\": \"REJECT\",\n   \"F02\": \"REJECT\"\n  }\n}\n}",
        "required_slots": [
          {
            "metric": "regional margin",
            "period": "FY2026",
            "role": "value",
            "slot_id": "s1",
            "unit": null,
            "value_type": "numeric"
          }
        ],
        "safe_indistinguishable": false,
        "selected_fact_ids": [],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 0,
        "slots_bound": 0,
        "slots_missing": 1,
        "slots_requested": 1,
        "structured_output_success": true,
        "tags": [
          "unbindable"
        ],
        "validation_reasons": []
      },
      {
        "actual": {
          "s1": [
            "F01",
            "F02"
          ]
        },
        "adapter_valid": true,
        "ambiguous_slots": [
          "s1"
        ],
        "binder_model_status": "AMBIGUOUS",
        "binding": {
          "ambiguous_slots": [
            "s1"
          ],
          "invalid_reasons": [],
          "missing_slots": [],
          "slot_bindings": {
            "s1": [
              "f01",
              "f02"
            ]
          },
          "status": "AMBIGUOUS"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
        "calculation_correct_slots": 0,
        "calculation_total_slots": 0,
        "dto_valid": true,
        "expected": {
          "s1": [
            "F01",
            "F02"
          ]
        },
        "fact_count": 2,
        "false_binding": false,
        "final_binding_status": "AMBIGUOUS",
        "groups": [
          "synthetic"
        ],
        "intent": "DIRECT_FACT",
        "invalid_reasons": [],
        "metadata": {
          "errno": null,
          "error": null,
          "exception_cause_message": null,
          "exception_cause_type": null,
          "exception_chain": [],
          "exception_type": null,
          "finish_reason": "stop",
          "http_status": null,
          "input_tokens": 719,
          "latency_ms": 1511.623,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 39,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 73,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-645adfd7-4b44-9010-a941-3fe7e27cbcf9",
          "structured_output_success": true,
          "total_tokens": 758
        },
        "missing_slots": [],
        "operation": null,
        "pairwise_outcomes": {
          "s1": {
            "F01": "MATCH",
            "F02": "MATCH"
          }
        },
        "provider_response_success": true,
        "question": "Two identical statements remain ambiguous.",
        "question_id": "v2_syn_08",
        "raw_response": "{\n  \"tasks\": {\n   \"s1\": {\n   \"F01\": \"MATCH\",\n   \"F02\": \"MATCH\"\n   }\n  }\n}",
        "required_slots": [
          {
            "metric": "cash",
            "period": "FY2026",
            "role": "value",
            "slot_id": "s1",
            "unit": null,
            "value_type": "numeric"
          }
        ],
        "safe_indistinguishable": true,
        "selected_fact_ids": [
          "f01",
          "f02"
        ],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 1,
        "slots_bound": 1,
        "slots_missing": 0,
        "slots_requested": 1,
        "structured_output_success": true,
        "tags": [
          "indistinguishable"
        ],
        "validation_reasons": []
      },
      {
        "actual": {
          "s1": [
            "F01",
            "F02"
          ]
        },
        "adapter_valid": true,
        "ambiguous_slots": [
          "s1"
        ],
        "binder_model_status": "AMBIGUOUS",
        "binding": {
          "ambiguous_slots": [
            "s1"
          ],
          "invalid_reasons": [],
          "missing_slots": [],
          "slot_bindings": {
            "s1": [
              "pw09a",
              "pw09b"
            ]
          },
          "status": "AMBIGUOUS"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
        "calculation_correct_slots": 0,
        "calculation_total_slots": 0,
        "dto_valid": true,
        "expected": {
          "s1": "AMBIGUOUS"
        },
        "fact_count": 2,
        "false_binding": false,
        "final_binding_status": "AMBIGUOUS",
        "groups": [
          "synthetic"
        ],
        "intent": "DIRECT_FACT",
        "invalid_reasons": [],
        "metadata": {
          "errno": null,
          "error": null,
          "exception_cause_message": null,
          "exception_cause_type": null,
          "exception_chain": [],
          "exception_type": null,
          "finish_reason": "stop",
          "http_status": null,
          "input_tokens": 737,
          "latency_ms": 1663.977,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 39,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 73,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-d3e7707c-f7a3-90cc-8ffd-e4455c6712e8",
          "structured_output_success": true,
          "total_tokens": 776
        },
        "missing_slots": [],
        "operation": null,
        "pairwise_outcomes": {
          "s1": {
            "F01": "MATCH",
            "F02": "MATCH"
          }
        },
        "provider_response_success": true,
        "question": "Two supplied records lack enough structural context to distinguish the requested revenue.",
        "question_id": "pair_syn_09",
        "raw_response": "{\n  \"tasks\": {\n   \"s1\": {\n   \"F01\": \"MATCH\",\n   \"F02\": \"MATCH\"\n   }\n  }\n}",
        "required_slots": [
          {
            "metric": "revenue",
            "period": "FY2026",
            "role": "value",
            "slot_id": "s1",
            "unit": null,
            "value_type": "numeric"
          }
        ],
        "safe_indistinguishable": true,
        "selected_fact_ids": [
          "pw09a",
          "pw09b"
        ],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 1,
        "slots_bound": 1,
        "slots_missing": 0,
        "slots_requested": 1,
        "structured_output_success": true,
        "tags": [
          "indistinguishable"
        ],
        "validation_reasons": []
      },
      {
        "actual": {},
        "adapter_valid": true,
        "ambiguous_slots": [],
        "binder_model_status": "MISSING",
        "binding": {
          "ambiguous_slots": [],
          "invalid_reasons": [],
          "missing_slots": [
            "s1"
          ],
          "slot_bindings": {},
          "status": "MISSING"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
        "calculation_correct_slots": 0,
        "calculation_total_slots": 0,
        "dto_valid": true,
        "expected": {
          "s1": "MISSING"
        },
        "fact_count": 2,
        "false_binding": false,
        "final_binding_status": "MISSING",
        "groups": [
          "synthetic"
        ],
        "intent": "DIRECT_FACT",
        "invalid_reasons": [],
        "metadata": {
          "errno": null,
          "error": null,
          "exception_cause_message": null,
          "exception_cause_type": null,
          "exception_chain": [],
          "exception_type": null,
          "finish_reason": "stop",
          "http_status": null,
          "input_tokens": 742,
          "latency_ms": 1552.618,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 40,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 72,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-311a889e-a407-9df3-bf30-13be6b3c098b",
          "structured_output_success": true,
          "total_tokens": 782
        },
        "missing_slots": [
          "s1"
        ],
        "operation": null,
        "pairwise_outcomes": {
          "s1": {
            "F01": "REJECT",
            "F02": "REJECT"
          }
        },
        "provider_response_success": true,
        "question": "No supplied record proves the requested regional margin.",
        "question_id": "pair_syn_10",
        "raw_response": "{\n  \"tasks\": {\n   \"s1\": {\n   \"F01\": \"REJECT\",\n   \"F02\": \"REJECT\"\n  }\n}\n}",
        "required_slots": [
          {
            "metric": "regional margin",
            "period": "FY2026",
            "role": "value",
            "slot_id": "s1",
            "unit": null,
            "value_type": "numeric"
          }
        ],
        "safe_indistinguishable": false,
        "selected_fact_ids": [],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 0,
        "slots_bound": 0,
        "slots_missing": 1,
        "slots_requested": 1,
        "structured_output_success": true,
        "tags": [
          "unbindable"
        ],
        "validation_reasons": []
      },
      {
        "actual": {
          "component": [
            "F01"
          ],
          "total": [
            "F02"
          ]
        },
        "adapter_valid": true,
        "ambiguous_slots": [],
        "binder_model_status": "BOUND",
        "binding": {
          "ambiguous_slots": [],
          "invalid_reasons": [],
          "missing_slots": [],
          "slot_bindings": {
            "component": [
              "pw11a"
            ],
            "total": [
              "pw11b"
            ]
          },
          "status": "BOUND"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
        "calculation_correct_slots": 2,
        "calculation_total_slots": 2,
        "dto_valid": true,
        "expected": {
          "component": [
            "F01"
          ],
          "total": [
            "F02"
          ]
        },
        "fact_count": 2,
        "false_binding": false,
        "final_binding_status": "BOUND",
        "groups": [
          "synthetic"
        ],
        "intent": "CALCULATION",
        "invalid_reasons": [],
        "metadata": {
          "errno": null,
          "error": null,
          "exception_cause_message": null,
          "exception_cause_type": null,
          "exception_chain": [],
          "exception_type": null,
          "finish_reason": "stop",
          "http_status": null,
          "input_tokens": 796,
          "latency_ms": 1933.915,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 69,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 126,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-ce68e79e-1e2b-998a-a616-137ca850bb4b",
          "structured_output_success": true,
          "total_tokens": 865
        },
        "missing_slots": [],
        "operation": "percentage_share",
        "pairwise_outcomes": {
          "component": {
            "F01": "MATCH",
            "F02": "REJECT"
          },
          "total": {
            "F01": "REJECT",
            "F02": "MATCH"
          }
        },
        "provider_response_success": true,
        "question": "Select component and total independently.",
        "question_id": "pair_syn_11",
        "raw_response": "{\n  \"tasks\": {\n   \"component\": {\n  \"F01\": \"MATCH\",\n  \"F02\": \"REJECT\"\n},\n\"total\": {\n  \"F01\": \"REJECT\",\n  \"F02\": \"MATCH\"\n}\n  }\n}",
        "required_slots": [
          {
            "metric": "gross profit",
            "period": "FY2026",
            "role": "component",
            "slot_id": "component",
            "unit": null,
            "value_type": "numeric"
          },
          {
            "metric": "sales",
            "period": "FY2026",
            "role": "total",
            "slot_id": "total",
            "unit": null,
            "value_type": "numeric"
          }
        ],
        "safe_indistinguishable": false,
        "selected_fact_ids": [
          "pw11a",
          "pw11b"
        ],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 0,
        "slots_bound": 2,
        "slots_missing": 0,
        "slots_requested": 2,
        "structured_output_success": true,
        "tags": [
          "calculation"
        ],
        "validation_reasons": []
      },
      {
        "actual": {
          "denominator": [
            "F02"
          ],
          "numerator": [
            "F01"
          ]
        },
        "adapter_valid": true,
        "ambiguous_slots": [],
        "binder_model_status": "BOUND",
        "binding": {
          "ambiguous_slots": [],
          "invalid_reasons": [],
          "missing_slots": [],
          "slot_bindings": {
            "denominator": [
              "pw12b"
            ],
            "numerator": [
              "pw12a"
            ]
          },
          "status": "BOUND"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
        "calculation_correct_slots": 2,
        "calculation_total_slots": 2,
        "dto_valid": true,
        "expected": {
          "denominator": [
            "F02"
          ],
          "numerator": [
            "F01"
          ]
        },
        "fact_count": 2,
        "false_binding": false,
        "final_binding_status": "BOUND",
        "groups": [
          "synthetic"
        ],
        "intent": "CALCULATION",
        "invalid_reasons": [],
        "metadata": {
          "errno": null,
          "error": null,
          "exception_cause_message": null,
          "exception_cause_type": null,
          "exception_chain": [],
          "exception_type": null,
          "finish_reason": "stop",
          "http_status": null,
          "input_tokens": 801,
          "latency_ms": 2056.04,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 71,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 132,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-277a9e92-4d43-9fe9-aacc-b628be39d945",
          "structured_output_success": true,
          "total_tokens": 872
        },
        "missing_slots": [],
        "operation": "percentage_share",
        "pairwise_outcomes": {
          "denominator": {
            "F01": "REJECT",
            "F02": "MATCH"
          },
          "numerator": {
            "F01": "MATCH",
            "F02": "REJECT"
          }
        },
        "provider_response_success": true,
        "question": "Select numerator and denominator independently.",
        "question_id": "pair_syn_12",
        "raw_response": "{\n  \"tasks\": {\n   \"numerator\": {\n  \"F01\": \"MATCH\",\n  \"F02\": \"REJECT\"\n},\n\"denominator\": {\n  \"F01\": \"REJECT\",\n  \"F02\": \"MATCH\"\n}\n  }\n}",
        "required_slots": [
          {
            "metric": "net income",
            "period": "FY2026",
            "role": "numerator",
            "slot_id": "numerator",
            "unit": null,
            "value_type": "numeric"
          },
          {
            "metric": "sales",
            "period": "FY2026",
            "role": "denominator",
            "slot_id": "denominator",
            "unit": null,
            "value_type": "numeric"
          }
        ],
        "safe_indistinguishable": false,
        "selected_fact_ids": [
          "pw12a",
          "pw12b"
        ],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 0,
        "slots_bound": 2,
        "slots_missing": 0,
        "slots_requested": 2,
        "structured_output_success": true,
        "tags": [
          "calculation"
        ],
        "validation_reasons": []
      },
      {
        "actual": {
          "minuend": [
            "F01"
          ],
          "subtrahend": [
            "F02"
          ]
        },
        "adapter_valid": true,
        "ambiguous_slots": [],
        "binder_model_status": "BOUND",
        "binding": {
          "ambiguous_slots": [],
          "invalid_reasons": [],
          "missing_slots": [],
          "slot_bindings": {
            "minuend": [
              "pw13a"
            ],
            "subtrahend": [
              "pw13b"
            ]
          },
          "status": "BOUND"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
        "calculation_correct_slots": 2,
        "calculation_total_slots": 2,
        "dto_valid": true,
        "expected": {
          "minuend": [
            "F01"
          ],
          "subtrahend": [
            "F02"
          ]
        },
        "fact_count": 2,
        "false_binding": false,
        "final_binding_status": "BOUND",
        "groups": [
          "synthetic"
        ],
        "intent": "CALCULATION",
        "invalid_reasons": [],
        "metadata": {
          "errno": null,
          "error": null,
          "exception_cause_message": null,
          "exception_cause_type": null,
          "exception_chain": [],
          "exception_type": null,
          "finish_reason": "stop",
          "http_status": null,
          "input_tokens": 808,
          "latency_ms": 2024.821,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 74,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 129,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-2fe7e157-be55-93f8-a74c-48479c7b10f8",
          "structured_output_success": true,
          "total_tokens": 882
        },
        "missing_slots": [],
        "operation": "difference",
        "pairwise_outcomes": {
          "minuend": {
            "F01": "MATCH",
            "F02": "REJECT"
          },
          "subtrahend": {
            "F01": "REJECT",
            "F02": "MATCH"
          }
        },
        "provider_response_success": true,
        "question": "Select minuend and subtrahend independently.",
        "question_id": "pair_syn_13",
        "raw_response": "{\n  \"tasks\": {\n   \"minuend\": {\n  \"F01\": \"MATCH\",\n  \"F02\": \"REJECT\"\n},\n\"subtrahend\": {\n  \"F01\": \"REJECT\",\n  \"F02\": \"MATCH\"\n}\n  }\n}",
        "required_slots": [
          {
            "metric": "sales",
            "period": "FY2026",
            "role": "minuend",
            "slot_id": "minuend",
            "unit": null,
            "value_type": "numeric"
          },
          {
            "metric": "sales",
            "period": "FY2025",
            "role": "subtrahend",
            "slot_id": "subtrahend",
            "unit": null,
            "value_type": "numeric"
          }
        ],
        "safe_indistinguishable": false,
        "selected_fact_ids": [
          "pw13a",
          "pw13b"
        ],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 0,
        "slots_bound": 2,
        "slots_missing": 0,
        "slots_requested": 2,
        "structured_output_success": true,
        "tags": [
          "calculation"
        ],
        "validation_reasons": []
      },
      {
        "actual": {
          "current": [
            "F01"
          ],
          "prior": [
            "F02"
          ]
        },
        "adapter_valid": true,
        "ambiguous_slots": [],
        "binder_model_status": "BOUND",
        "binding": {
          "ambiguous_slots": [],
          "invalid_reasons": [],
          "missing_slots": [],
          "slot_bindings": {
            "current": [
              "pw14a"
            ],
            "prior": [
              "pw14b"
            ]
          },
          "status": "BOUND"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
        "calculation_correct_slots": 2,
        "calculation_total_slots": 2,
        "dto_valid": true,
        "expected": {
          "current": [
            "F01"
          ],
          "prior": [
            "F02"
          ]
        },
        "fact_count": 2,
        "false_binding": false,
        "final_binding_status": "BOUND",
        "groups": [
          "synthetic"
        ],
        "intent": "CALCULATION",
        "invalid_reasons": [],
        "metadata": {
          "errno": null,
          "error": null,
          "exception_cause_message": null,
          "exception_cause_type": null,
          "exception_chain": [],
          "exception_type": null,
          "finish_reason": "stop",
          "http_status": null,
          "input_tokens": 791,
          "latency_ms": 2166.004,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 69,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 124,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-0ec70702-ef48-9ae2-9deb-360a99de7521",
          "structured_output_success": true,
          "total_tokens": 860
        },
        "missing_slots": [],
        "operation": "difference",
        "pairwise_outcomes": {
          "current": {
            "F01": "MATCH",
            "F02": "REJECT"
          },
          "prior": {
            "F01": "REJECT",
            "F02": "MATCH"
          }
        },
        "provider_response_success": true,
        "question": "Select current and prior balances independently.",
        "question_id": "pair_syn_14",
        "raw_response": "{\n  \"tasks\": {\n   \"current\": {\n  \"F01\": \"MATCH\",\n  \"F02\": \"REJECT\"\n},\n\"prior\": {\n  \"F01\": \"REJECT\",\n  \"F02\": \"MATCH\"\n}\n  }\n}",
        "required_slots": [
          {
            "metric": "cash",
            "period": "FY2026",
            "role": "current",
            "slot_id": "current",
            "unit": null,
            "value_type": "numeric"
          },
          {
            "metric": "cash",
            "period": "FY2025",
            "role": "prior",
            "slot_id": "prior",
            "unit": null,
            "value_type": "numeric"
          }
        ],
        "safe_indistinguishable": false,
        "selected_fact_ids": [
          "pw14a",
          "pw14b"
        ],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 0,
        "slots_bound": 2,
        "slots_missing": 0,
        "slots_requested": 2,
        "structured_output_success": true,
        "tags": [
          "calculation"
        ],
        "validation_reasons": []
      }
    ],
    "semantic_correct": 13,
    "semantic_total": 14,
    "structural_healthy": true,
    "structured_output": 14,
    "unbindable_false_binding": 0,
    "unbindable_total": 2
  }
}
