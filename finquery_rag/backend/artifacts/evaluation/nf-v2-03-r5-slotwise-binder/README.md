{
  "diagnostic": {
    "binder_model_frozen": "qwen3.7-plus",
    "binder_task_formulation_frozen": false,
    "diagnostic_all_operands": "2/6",
    "diagnostic_calculation": "5/12",
    "diagnostic_direct": "9/21",
    "diagnostic_provider_calls": 39,
    "formal_attempt_9": "not_run",
    "formulation": "slotwise_discriminative_v1",
    "gate": "NF-V2-03-R5",
    "indistinguishable_abstention": "0/6",
    "model": "qwen3.7-plus",
    "next_gate": "v2_03_slotwise_failure_review",
    "production_default": "V1",
    "production_switch_allowed": false,
    "provider_calls_per_query": 1,
    "slotwise_binder_effective": "partial",
    "structural_violations": 0,
    "synthetic": "11/12",
    "token_delta": {
      "max_input_tokens": 366,
      "mean_input_tokens": 41.5897435897441,
      "p50_input_tokens": -26,
      "p95_input_tokens": 278.0,
      "total_input_tokens": 1622
    },
    "unbindable_false_binding": "6/7"
  },
  "external_model_review": "cancelled_by_design",
  "formal_attempt_9": "not_run",
  "gate": "NF-V2-03 R5 Slot-wise Binder",
  "model": "qwen3.7-plus",
  "reason": "cost_and_project_scope",
  "synthetic": {
    "adapter_valid": 12,
    "benchmark_questions_used": 0,
    "binding_validator": 12,
    "calculation_groups_correct": 6,
    "calculation_groups_total": 6,
    "dto_valid": 12,
    "false_binding": 0,
    "formulation": "slotwise_discriminative_v1",
    "gate": "NF-V2-03-R5",
    "model": "qwen3.7-plus",
    "pass": true,
    "provider_calls": 12,
    "provider_success": 12,
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
          "input_tokens": 736,
          "latency_ms": 1953.355,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 25,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 39,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-caae18f5-3c32-9006-af16-ce9ba6da0b0e",
          "structured_output_success": true,
          "total_tokens": 761
        },
        "missing_slots": [],
        "operation": null,
        "provider_response_success": true,
        "question": "Select the supplied revenue fact.",
        "question_id": "v2_syn_01",
        "raw_response": "{\n  \"tasks\": {\n   \"s1\": [\n\"F01\"\n]\n  }\n}",
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
          "input_tokens": 750,
          "latency_ms": 1191.361,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 25,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 39,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-9ab47949-750b-936a-b3e8-aa4f0884aab8",
          "structured_output_success": true,
          "total_tokens": 775
        },
        "missing_slots": [],
        "operation": null,
        "provider_response_success": true,
        "question": "Select the operating income from the correct statement.",
        "question_id": "v2_syn_02",
        "raw_response": "{\n  \"tasks\": {\n   \"s1\": [\n\"F01\"\n]\n  }\n}",
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
        "validation_reasons": []
      },
      {
        "actual": {
          "s1": []
        },
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
        "dto_valid": true,
        "expected": {
          "s1": [
            "F01"
          ]
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
          "input_tokens": 746,
          "latency_ms": 1435.236,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 21,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 36,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-2afd96d5-8e80-9290-ade2-3b99f3b55c61",
          "structured_output_success": true,
          "total_tokens": 767
        },
        "missing_slots": [
          "s1"
        ],
        "operation": null,
        "provider_response_success": true,
        "question": "Select the regional row.",
        "question_id": "v2_syn_03",
        "raw_response": "{\n  \"tasks\": {\n   \"s1\": [\n   ]\n  }\n}",
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
        "selected_fact_ids": [],
        "semantic_correct": false,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 0,
        "slots_bound": 0,
        "slots_missing": 1,
        "slots_requested": 1,
        "structured_output_success": true,
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
          "input_tokens": 744,
          "latency_ms": 1153.264,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 25,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 39,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-3ebb224e-01a7-9b8f-8008-458dba02799e",
          "structured_output_success": true,
          "total_tokens": 769
        },
        "missing_slots": [],
        "operation": null,
        "provider_response_success": true,
        "question": "Select the fact under the current header.",
        "question_id": "v2_syn_04",
        "raw_response": "{\n  \"tasks\": {\n   \"s1\": [\n\"F01\"\n]\n  }\n}",
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
          "input_tokens": 782,
          "latency_ms": 1468.43,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 37,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 64,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-1f37e898-912d-9275-b6f0-6d08e5765c59",
          "structured_output_success": true,
          "total_tokens": 819
        },
        "missing_slots": [],
        "operation": "growth_rate",
        "provider_response_success": true,
        "question": "Select current and prior independently.",
        "question_id": "v2_syn_05",
        "raw_response": "{\n  \"tasks\": {\n   \"current\": [\n\"F01\"\n],\n\"prior\": [\n\"F02\"\n]\n  }\n}",
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
          "input_tokens": 798,
          "latency_ms": 1509.598,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 39,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 72,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-c83f5fec-4575-9d62-98c2-18b35ca2ef97",
          "structured_output_success": true,
          "total_tokens": 837
        },
        "missing_slots": [],
        "operation": "percentage_share",
        "provider_response_success": true,
        "question": "Select numerator and denominator.",
        "question_id": "v2_syn_06",
        "raw_response": "{\n  \"tasks\": {\n   \"numerator\": [\n\"F01\"\n],\n\"denominator\": [\n\"F02\"\n]\n  }\n}",
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
        "validation_reasons": []
      },
      {
        "actual": {
          "s1": []
        },
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
          "input_tokens": 736,
          "latency_ms": 1138.924,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 18,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 32,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-655dae7c-d25c-9354-821c-131716e7204d",
          "structured_output_success": true,
          "total_tokens": 754
        },
        "missing_slots": [
          "s1"
        ],
        "operation": null,
        "provider_response_success": true,
        "question": "No supplied fact has the requested scope.",
        "question_id": "v2_syn_07",
        "raw_response": "{\n  \"tasks\": {\n   \"s1\": []\n  }\n}",
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
        "selected_fact_ids": [],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 0,
        "slots_bound": 0,
        "slots_missing": 1,
        "slots_requested": 1,
        "structured_output_success": true,
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
          "input_tokens": 726,
          "latency_ms": 1547.376,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 33,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 55,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-019fb0c0-b825-9644-8e34-998d8977b314",
          "structured_output_success": true,
          "total_tokens": 759
        },
        "missing_slots": [],
        "operation": null,
        "provider_response_success": true,
        "question": "Two identical statements remain ambiguous.",
        "question_id": "v2_syn_08",
        "raw_response": "{\n  \"tasks\": {\n   \"s1\": [\n   \"F01\",\n   \"F02\"\n   ]\n  }\n}",
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
              "sw_f13"
            ],
            "total": [
              "sw_f14"
            ]
          },
          "status": "BOUND"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
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
          "input_tokens": 803,
          "latency_ms": 1368.401,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 37,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 66,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-6129d957-8088-91bb-b7d4-d5a943e20405",
          "structured_output_success": true,
          "total_tokens": 840
        },
        "missing_slots": [],
        "operation": "percentage_share",
        "provider_response_success": true,
        "question": "Select component and total independently.",
        "question_id": "slotwise_syn_13",
        "raw_response": "{\n  \"tasks\": {\n   \"component\": [\n\"F01\"\n],\n\"total\": [\n\"F02\"\n]\n  }\n}",
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
        "selected_fact_ids": [
          "sw_f13",
          "sw_f14"
        ],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 0,
        "slots_bound": 2,
        "slots_missing": 0,
        "slots_requested": 2,
        "structured_output_success": true,
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
              "sw_f15"
            ],
            "subtrahend": [
              "sw_f16"
            ]
          },
          "status": "BOUND"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
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
          "input_tokens": 815,
          "latency_ms": 1784.526,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 42,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 69,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-5e47b79a-5286-9e00-895f-081ff7e9f32b",
          "structured_output_success": true,
          "total_tokens": 857
        },
        "missing_slots": [],
        "operation": "difference",
        "provider_response_success": true,
        "question": "Select minuend and subtrahend independently.",
        "question_id": "slotwise_syn_14",
        "raw_response": "{\n  \"tasks\": {\n   \"minuend\": [\n\"F01\"\n],\n\"subtrahend\": [\n\"F02\"\n]\n  }\n}",
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
        "selected_fact_ids": [
          "sw_f15",
          "sw_f16"
        ],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 0,
        "slots_bound": 2,
        "slots_missing": 0,
        "slots_requested": 2,
        "structured_output_success": true,
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
              "sw_f18"
            ],
            "numerator": [
              "sw_f17"
            ]
          },
          "status": "BOUND"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
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
          "input_tokens": 808,
          "latency_ms": 1559.947,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 39,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 72,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-4dd755b0-76b5-910b-a4f3-620328400f4f",
          "structured_output_success": true,
          "total_tokens": 847
        },
        "missing_slots": [],
        "operation": "percentage_share",
        "provider_response_success": true,
        "question": "Select numerator and denominator independently.",
        "question_id": "slotwise_syn_15",
        "raw_response": "{\n  \"tasks\": {\n   \"numerator\": [\n\"F01\"\n],\n\"denominator\": [\n\"F02\"\n]\n  }\n}",
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
        "selected_fact_ids": [
          "sw_f17",
          "sw_f18"
        ],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 0,
        "slots_bound": 2,
        "slots_missing": 0,
        "slots_requested": 2,
        "structured_output_success": true,
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
              "sw_f19"
            ],
            "prior": [
              "sw_f20"
            ]
          },
          "status": "BOUND"
        },
        "binding_schema_valid": true,
        "binding_validator_pass": true,
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
          "input_tokens": 798,
          "latency_ms": 1564.239,
          "model": "qwen3.7-plus",
          "model_role": "strong_general_llm",
          "output_tokens": 37,
          "provider": "bailian",
          "provider_response_success": true,
          "provider_role": "evidence_binder",
          "raw_content_length": 64,
          "reasoning_tokens": null,
          "request_id": "chatcmpl-bc3e66bb-19e2-9fa8-b56a-b146f3c875f0",
          "structured_output_success": true,
          "total_tokens": 835
        },
        "missing_slots": [],
        "operation": "difference",
        "provider_response_success": true,
        "question": "Select current and prior balances independently.",
        "question_id": "slotwise_syn_16",
        "raw_response": "{\n  \"tasks\": {\n   \"current\": [\n\"F01\"\n],\n\"prior\": [\n\"F02\"\n]\n  }\n}",
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
        "selected_fact_ids": [
          "sw_f19",
          "sw_f20"
        ],
        "semantic_correct": true,
        "skipped_no_fact_supply": false,
        "slots_ambiguous": 0,
        "slots_bound": 2,
        "slots_missing": 0,
        "slots_requested": 2,
        "structured_output_success": true,
        "validation_reasons": []
      }
    ],
    "semantic_correct": 11,
    "semantic_total": 12,
    "structured_output": 12
  }
}
