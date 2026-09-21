"""H2A-3B2: the pre-F10 B3 legacy context baseline, kept as a change record.

This is *history, not a target.*  It records what the specialist boundary
produced at the H2A-3B0 commit -- post-F5, **pre-F10** -- and it is retained
only so that the defect it documents stays checkable.  Nothing migrates against
it: ``b3_legacy_context_baseline.BASELINE_V3`` is the migration baseline, and
H2A-3B3's differential compares the compiled pack against that.

Why keep a superseded record at all.  F10 was five fabricated defaults in the
renderer, and the way to show they were real -- rather than a tidy-up someone
decided to do -- is to keep the rendering that contained them.  Three of the
prompts below assert values nobody established: ``Scope: Revenue`` where the
evidence had no scope, ``Scale: 1`` where it had no scale, and ``Currency: not
specified`` beside them, which is what an honest absence marker looks like and
is the reason the other two are distinguishable from one.

Its inputs are the same authored ``SCENARIOS`` the live baseline still uses, so
the two records differ only in what came out.  They differ in ``prompt`` and
``prompt_sha256`` for four of the five scenarios -- every one whose evidence was
missing a field a fabricated default was standing in for -- and ``multi_fact``,
whose evidence carries every field, is byte-identical across both.  That is
asserted in both directions beside the live baseline: this file is what the
defect looked like, and that comparison is what makes "F10 was real" checkable
rather than asserted.
"""

from __future__ import annotations

from typing import Any

BASELINE_V1: dict[str, dict[str, Any]] = {'multi_fact': {'route': 'MULTI',
                'route_reason': 'Multi-fact synthesis: Local Specialist combines distinct verified '
                                'facts.',
                'route_target': 'LOCAL_SPECIALIST',
                'bound_evidence_ids': ['e1', 'e2'],
                'citation_ids': ['citation:a', 'citation:b'],
                'calculation_ids': [],
                'disclosed_fields': ['evidence.citation_id',
                                     'evidence.currency',
                                     'evidence.document_id',
                                     'evidence.evidence_id',
                                     'evidence.metric',
                                     'evidence.page',
                                     'evidence.period',
                                     'evidence.scale',
                                     'evidence.scope',
                                     'evidence.unit',
                                     'evidence.value'],
                'projected_evidence': [{'evidence_id': 'e1',
                                        'citation_id': 'citation:a',
                                        'metric': 'Revenue',
                                        'period': 'FY2024',
                                        'value': '391',
                                        'unit': 'USD',
                                        'currency': 'USD',
                                        'scale': 'million',
                                        'scope': 'consolidated',
                                        'document_id': 'doc-1',
                                        'page': 7},
                                       {'evidence_id': 'e2',
                                        'citation_id': 'citation:b',
                                        'metric': 'Revenue',
                                        'period': 'FY2023',
                                        'value': '383',
                                        'unit': 'USD',
                                        'currency': 'USD',
                                        'scale': 'million',
                                        'scope': 'consolidated',
                                        'document_id': 'doc-1',
                                        'page': 0}],
                'projected_calculation': None,
                'prompt': '[QUESTION]\n'
                          'What was revenue?\n'
                          '\n'
                          '[VERIFIED EVIDENCE]\n'
                          '\n'
                          '[E1]\n'
                          'Metric: Revenue\n'
                          'Period: FY2024\n'
                          'Scope: consolidated\n'
                          'Value: 391\n'
                          'Unit: USD\n'
                          'Currency: USD\n'
                          'Scale: million\n'
                          'Source: doc-1:7\n'
                          '\n'
                          '[E2]\n'
                          'Metric: Revenue\n'
                          'Period: FY2023\n'
                          'Scope: consolidated\n'
                          'Value: 383\n'
                          'Unit: USD\n'
                          'Currency: USD\n'
                          'Scale: million\n'
                          'Source: doc-1:0\n'
                          '\n'
                          '[ANSWER RULES]\n'
                          '1. Use only the verified evidence and calculation above.\n'
                          '2. Do not introduce outside financial knowledge.\n'
                          '3. Preserve supplied numbers, periods, units, currencies and scales '
                          'exactly.\n'
                          '4. Do not recalculate canonical calculation results.\n'
                          '5. Cite factual claims using the supplied [E#] / [C#] IDs.\n'
                          '6. If required evidence is missing, explicitly state that the provided '
                          'evidence is insufficient.\n'
                          '7. Answer concisely.',
                'prompt_sha256': '3b9f6c4ecd5989117bfcfd485f5ec28c6d60db36b67c30e80d11b832437d1214',
                'input_tokens': 'NOT_DETERMINED',
                'candidate_answer': 'The reported revenue was 391 million USD [E1].',
                'candidate_status': 'CANDIDATE_READY_FOR_VALIDATION',
                'candidate_generation_id': 'G1-3eb2adbea9aae99a'},
 'temporal': {'route': 'TEMPORAL_SYNTHESIS',
              'route_reason': 'Temporal/multi-period synthesis: Local Specialist synthesizes '
                              'timeline.',
              'route_target': 'LOCAL_SPECIALIST',
              'bound_evidence_ids': ['e1', 'e2'],
              'citation_ids': [],
              'calculation_ids': [],
              'disclosed_fields': ['evidence.document_id',
                                   'evidence.evidence_id',
                                   'evidence.metric',
                                   'evidence.page',
                                   'evidence.period',
                                   'evidence.scope',
                                   'evidence.unit',
                                   'evidence.value'],
              'projected_evidence': [{'evidence_id': 'e1',
                                      'metric': 'Revenue',
                                      'period': 'FY2024',
                                      'value': '391',
                                      'unit': 'USD',
                                      'scope': 'consolidated',
                                      'document_id': 'doc-2',
                                      'page': 12},
                                     {'evidence_id': 'e2',
                                      'metric': 'Revenue',
                                      'period': 'FY2023',
                                      'value': '383',
                                      'unit': 'USD',
                                      'scope': 'consolidated',
                                      'document_id': 'doc-2'}],
              'projected_calculation': None,
              'prompt': '[QUESTION]\n'
                        'Compare revenue year-over-year.\n'
                        '\n'
                        '[VERIFIED EVIDENCE]\n'
                        '\n'
                        '[E1]\n'
                        'Metric: Revenue\n'
                        'Period: FY2024\n'
                        'Scope: consolidated\n'
                        'Value: 391\n'
                        'Unit: USD\n'
                        'Currency: not specified\n'
                        'Scale: 1\n'
                        'Source: doc-2:12\n'
                        '\n'
                        '[E2]\n'
                        'Metric: Revenue\n'
                        'Period: FY2023\n'
                        'Scope: consolidated\n'
                        'Value: 383\n'
                        'Unit: USD\n'
                        'Currency: not specified\n'
                        'Scale: 1\n'
                        'Source: doc-2:not specified\n'
                        '\n'
                        '[ANSWER RULES]\n'
                        '1. Use only the verified evidence and calculation above.\n'
                        '2. Do not introduce outside financial knowledge.\n'
                        '3. Preserve supplied numbers, periods, units, currencies and scales '
                        'exactly.\n'
                        '4. Do not recalculate canonical calculation results.\n'
                        '5. Cite factual claims using the supplied [E#] / [C#] IDs.\n'
                        '6. If required evidence is missing, explicitly state that the provided '
                        'evidence is insufficient.\n'
                        '7. Answer concisely.',
              'prompt_sha256': '89b7b2c4f4581ae1f254a894c4469a55fd4055f1751adbb0411084066231494c',
              'input_tokens': 'NOT_DETERMINED',
              'candidate_answer': 'The reported revenue was 391 million USD [E1].',
              'candidate_status': 'CANDIDATE_READY_FOR_VALIDATION',
              'candidate_generation_id': 'G1-c3ea4ba3b6313dd3'},
 'qualitative_single': {'route': 'QUALITATIVE',
                        'route_reason': 'Qualitative grounded QA: Local Specialist generates '
                                        'verified text answer.',
                        'route_target': 'LOCAL_SPECIALIST',
                        'bound_evidence_ids': ['e1'],
                        'citation_ids': [],
                        'calculation_ids': [],
                        'disclosed_fields': ['evidence.document_id',
                                             'evidence.evidence_id',
                                             'evidence.metric',
                                             'evidence.page',
                                             'evidence.period',
                                             'evidence.scope',
                                             'evidence.value'],
                        'projected_evidence': [{'evidence_id': 'e1',
                                                'metric': 'Risk',
                                                'period': 'FY2024',
                                                'value': 'supply chain',
                                                'scope': 'consolidated',
                                                'document_id': 'doc-3',
                                                'page': 41}],
                        'projected_calculation': None,
                        'prompt': '[QUESTION]\n'
                                  'What are the principal risk factors?\n'
                                  '\n'
                                  '[VERIFIED EVIDENCE]\n'
                                  '\n'
                                  '[E1]\n'
                                  'Metric: Risk\n'
                                  'Period: FY2024\n'
                                  'Scope: consolidated\n'
                                  'Value: supply chain\n'
                                  'Unit: not specified\n'
                                  'Currency: not specified\n'
                                  'Scale: 1\n'
                                  'Source: doc-3:41\n'
                                  '\n'
                                  '[ANSWER RULES]\n'
                                  '1. Use only the verified evidence and calculation above.\n'
                                  '2. Do not introduce outside financial knowledge.\n'
                                  '3. Preserve supplied numbers, periods, units, currencies and '
                                  'scales exactly.\n'
                                  '4. Do not recalculate canonical calculation results.\n'
                                  '5. Cite factual claims using the supplied [E#] / [C#] IDs.\n'
                                  '6. If required evidence is missing, explicitly state that the '
                                  'provided evidence is insufficient.\n'
                                  '7. Answer concisely.',
                        'prompt_sha256': 'c334f92c8c4b8bd27ea4cc05f0951d8ad3666216a3da33cacaeaef48f1becae9',
                        'input_tokens': 'NOT_DETERMINED',
                        'candidate_answer': 'The reported revenue was 391 million USD [E1].',
                        'candidate_status': 'CANDIDATE_READY_FOR_VALIDATION',
                        'candidate_generation_id': 'G1-33bc1706f5827798'},
 'calculation_with_explanation': {'route': 'CALCULATION_WITH_EXPLANATION',
                                  'route_reason': 'Calculation with synthesis: Local Specialist '
                                                  'consumes pre-computed C1 and cites evidence.',
                                  'route_target': 'LOCAL_SPECIALIST',
                                  'bound_evidence_ids': ['e1', 'e2'],
                                  'citation_ids': [],
                                  'calculation_ids': ['C1-1965a2ad00ce9277'],
                                  'disclosed_fields': ['evidence.document_id',
                                                       'evidence.evidence_id',
                                                       'evidence.metric',
                                                       'evidence.page',
                                                       'evidence.period',
                                                       'evidence.scope',
                                                       'evidence.unit',
                                                       'evidence.value',
                                                       'calculation.operation',
                                                       'calculation.unit',
                                                       'calculation.value'],
                                  'projected_evidence': [{'evidence_id': 'e1',
                                                          'metric': 'Revenue',
                                                          'period': 'FY2024',
                                                          'value': '391',
                                                          'unit': 'USD',
                                                          'scope': 'consolidated',
                                                          'document_id': 'doc-1',
                                                          'page': 7},
                                                         {'evidence_id': 'e2',
                                                          'metric': 'Revenue',
                                                          'period': 'FY2023',
                                                          'value': '383',
                                                          'unit': 'USD',
                                                          'scope': 'consolidated',
                                                          'document_id': 'doc-1',
                                                          'page': 7}],
                                  'projected_calculation': {'operation': 'difference',
                                                            'unit': 'USD',
                                                            'value': '8'},
                                  'prompt': '[QUESTION]\n'
                                            'Why did revenue change?\n'
                                            '\n'
                                            '[VERIFIED EVIDENCE]\n'
                                            '\n'
                                            '[E1]\n'
                                            'Metric: Revenue\n'
                                            'Period: FY2024\n'
                                            'Scope: consolidated\n'
                                            'Value: 391\n'
                                            'Unit: USD\n'
                                            'Currency: not specified\n'
                                            'Scale: 1\n'
                                            'Source: doc-1:7\n'
                                            '\n'
                                            '[E2]\n'
                                            'Metric: Revenue\n'
                                            'Period: FY2023\n'
                                            'Scope: consolidated\n'
                                            'Value: 383\n'
                                            'Unit: USD\n'
                                            'Currency: not specified\n'
                                            'Scale: 1\n'
                                            'Source: doc-1:7\n'
                                            '\n'
                                            '[VERIFIED CALCULATION]\n'
                                            '\n'
                                            '[C1]\n'
                                            'Operation: difference\n'
                                            'Value: 8 USD\n'
                                            '\n'
                                            '[ANSWER RULES]\n'
                                            '1. Use only the verified evidence and calculation '
                                            'above.\n'
                                            '2. Do not introduce outside financial knowledge.\n'
                                            '3. Preserve supplied numbers, periods, units, '
                                            'currencies and scales exactly.\n'
                                            '4. Do not recalculate canonical calculation results.\n'
                                            '5. Cite factual claims using the supplied [E#] / [C#] '
                                            'IDs.\n'
                                            '6. If required evidence is missing, explicitly state '
                                            'that the provided evidence is insufficient.\n'
                                            '7. Answer concisely.',
                                  'prompt_sha256': '97227c5103388a12252c62126180143d654f0defe5ffd536d772f2071be5b316',
                                  'input_tokens': 'NOT_DETERMINED',
                                  'candidate_answer': 'The reported revenue was 391 million USD '
                                                      '[E1].',
                                  'candidate_status': 'CANDIDATE_READY_FOR_VALIDATION',
                                  'candidate_generation_id': 'G1-4fb459f0e6f7144f'},
 'page_variants': {'route': 'MULTI',
                   'route_reason': 'Multi-fact synthesis: Local Specialist combines distinct '
                                   'verified facts.',
                   'route_target': 'LOCAL_SPECIALIST',
                   'bound_evidence_ids': ['e1', 'e2', 'e3'],
                   'citation_ids': [],
                   'calculation_ids': [],
                   'disclosed_fields': ['evidence.document_id',
                                        'evidence.evidence_id',
                                        'evidence.metric',
                                        'evidence.page',
                                        'evidence.period',
                                        'evidence.value'],
                   'projected_evidence': [{'evidence_id': 'e1',
                                           'metric': 'Revenue',
                                           'period': 'FY2024',
                                           'value': '391',
                                           'document_id': 'doc-1',
                                           'page': 7},
                                          {'evidence_id': 'e2',
                                           'metric': 'Cost of revenue',
                                           'period': 'FY2024',
                                           'value': '214',
                                           'document_id': 'doc-1',
                                           'page': 0},
                                          {'evidence_id': 'e3',
                                           'metric': 'Operating income',
                                           'period': 'FY2024',
                                           'value': '123',
                                           'document_id': 'doc-1'}],
                   'projected_calculation': None,
                   'prompt': '[QUESTION]\n'
                             'Summarise the reported figures.\n'
                             '\n'
                             '[VERIFIED EVIDENCE]\n'
                             '\n'
                             '[E1]\n'
                             'Metric: Revenue\n'
                             'Period: FY2024\n'
                             'Scope: Revenue\n'
                             'Value: 391\n'
                             'Unit: not specified\n'
                             'Currency: not specified\n'
                             'Scale: 1\n'
                             'Source: doc-1:7\n'
                             '\n'
                             '[E2]\n'
                             'Metric: Cost of revenue\n'
                             'Period: FY2024\n'
                             'Scope: Cost of revenue\n'
                             'Value: 214\n'
                             'Unit: not specified\n'
                             'Currency: not specified\n'
                             'Scale: 1\n'
                             'Source: doc-1:0\n'
                             '\n'
                             '[E3]\n'
                             'Metric: Operating income\n'
                             'Period: FY2024\n'
                             'Scope: Operating income\n'
                             'Value: 123\n'
                             'Unit: not specified\n'
                             'Currency: not specified\n'
                             'Scale: 1\n'
                             'Source: doc-1:not specified\n'
                             '\n'
                             '[ANSWER RULES]\n'
                             '1. Use only the verified evidence and calculation above.\n'
                             '2. Do not introduce outside financial knowledge.\n'
                             '3. Preserve supplied numbers, periods, units, currencies and scales '
                             'exactly.\n'
                             '4. Do not recalculate canonical calculation results.\n'
                             '5. Cite factual claims using the supplied [E#] / [C#] IDs.\n'
                             '6. If required evidence is missing, explicitly state that the '
                             'provided evidence is insufficient.\n'
                             '7. Answer concisely.',
                   'prompt_sha256': '92b0aa7f7b5f597b0ee731e3c1f09fab0edeb644b73d996f86bb8aaa2755ab23',
                   'input_tokens': 'NOT_DETERMINED',
                   'candidate_answer': 'The reported revenue was 391 million USD [E1].',
                   'candidate_status': 'CANDIDATE_READY_FOR_VALIDATION',
                   'candidate_generation_id': 'G1-bb689d5479800896'}}

