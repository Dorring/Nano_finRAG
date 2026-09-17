"""H2A-3B0: the B3 legacy context baseline, frozen before the compiler exists.

What this is, and what it is not
--------------------------------

This module records **what the specialist boundary produced at the commit that
froze it** -- the route decision, the admitted evidence it selected, the
post-disclosure field set, the exact model input, and the candidate it returned.

It is a **change detector, not an oracle.**  Every expected value below was
captured from the implementation, so it cannot testify that the implementation is
*correct*.  What it can testify is that the implementation has not *moved* --
which is precisely the question H2A-3B2 has to answer when it replaces the
hand-written `select -> project -> assemble` sequence with
`ContextCompiler.compile(...)`.

That distinction is the whole reason this file exists rather than a set of
hand-written expectations.  H2A-2's independent-oracle matrix applies to claims
about *correctness*; a migration differential is a claim about *equivalence*, and
equivalence to a predecessor can only be established against the predecessor's
recorded behaviour.  The two are complements, and 3B2 needs both:

* this baseline, for "the compiler changed nothing";
* independently authored semantic assertions, for "and the result was right".

Neither substitutes for the other, and a 3B2 differential that ships only this
file would be the tautology the anti-circularity rule forbids.

The inputs are authored, the outputs are captured
--------------------------------------------------

The states below are hand-written; the expectations are recorded.  Keeping the
inputs authored is what stops the baseline from degenerating into "whatever the
code does, frozen" -- the scenarios are chosen to cover the shapes B3 is
actually reached with, and a scenario that stopped reaching the specialist would
fail here rather than silently record a new route.

Token counts are NOT DETERMINED
--------------------------------

B3 is the one boundary with a real tokenizer, and it is the checkpoint's -- which
lives behind `torch` and a SHA256-pinned checkpoint.  Neither is available to the
suite, so `input_tokens` is recorded as `NOT_DETERMINED` rather than approximated
with a character or word count.  Reciting a number here would be exactly the
fabrication H2A-3A's tokenizer matrix was written to prevent.  The canonical
environment can fill it; nothing else may.
"""

from __future__ import annotations

import hashlib
from typing import Any

#: Sentinel for a measurement this environment cannot take honestly.
NOT_DETERMINED = "NOT_DETERMINED"

#: Authored inputs.  Chosen to cover every shape that actually reaches B3:
#: multi-fact, temporal, qualitative-single, calculation-with-explanation, and
#: the page shapes H2A-3B0 fixed.
SCENARIOS: dict[str, dict[str, Any]] = {
    "multi_fact": {
        "question": "What was revenue?",
        "intent": "DIRECT_FACT",
        "evidence": [
            {
                "evidence_id": "e1", "metric": "Revenue", "value": "391",
                "period": "FY2024", "scope": "consolidated", "unit": "USD",
                "currency": "USD", "scale": "million", "document_id": "doc-1",
                "citation_id": "citation:a", "page": 7,
            },
            {
                "evidence_id": "e2", "metric": "Revenue", "value": "383",
                "period": "FY2023", "scope": "consolidated", "unit": "USD",
                "currency": "USD", "scale": "million", "document_id": "doc-1",
                "citation_id": "citation:b", "page": 0,
            },
        ],
        "calculation": None,
    },
    "temporal": {
        "question": "Compare revenue year-over-year.",
        "intent": "DIRECT_FACT",
        "evidence": [
            {
                "evidence_id": "e1", "metric": "Revenue", "value": "391",
                "period": "FY2024", "scope": "consolidated", "unit": "USD",
                "document_id": "doc-2", "page": 12,
            },
            {
                "evidence_id": "e2", "metric": "Revenue", "value": "383",
                "period": "FY2023", "scope": "consolidated", "unit": "USD",
                "document_id": "doc-2",
            },
        ],
        "calculation": None,
    },
    "qualitative_single": {
        "question": "What are the principal risk factors?",
        "intent": "DIRECT_FACT",
        "evidence": [
            {
                "evidence_id": "e1", "metric": "Risk", "value": "supply chain",
                "period": "FY2024", "scope": "consolidated", "document_id": "doc-3",
                "page": 41,
            },
        ],
        "calculation": None,
    },
    "calculation_with_explanation": {
        "question": "Why did revenue change?",
        "intent": "MULTI_EVIDENCE",
        "evidence": [
            {
                "evidence_id": "e1", "metric": "Revenue", "value": "391",
                "period": "FY2024", "scope": "consolidated", "unit": "USD",
                "document_id": "doc-1", "page": 7,
            },
            {
                "evidence_id": "e2", "metric": "Revenue", "value": "383",
                "period": "FY2023", "scope": "consolidated", "unit": "USD",
                "document_id": "doc-1", "page": 7,
            },
        ],
        "calculation": {"operation": "difference", "value": "8", "unit": "USD"},
    },
    "page_variants": {
        "question": "Summarise the reported figures.",
        "intent": "DIRECT_FACT",
        "evidence": [
            {
                "evidence_id": "e1", "metric": "Revenue", "value": "391",
                "period": "FY2024", "document_id": "doc-1", "page": 7,
            },
            {
                "evidence_id": "e2", "metric": "Cost of revenue", "value": "214",
                "period": "FY2024", "document_id": "doc-1", "page": 0,
            },
            {
                "evidence_id": "e3", "metric": "Operating income", "value": "123",
                "period": "FY2024", "document_id": "doc-1",
            },
        ],
        "calculation": None,
    },
}


class RecordingSpecialist:
    """Records exactly what the boundary handed it, and renders what a model reads.

    The prompt is rendered with the production renderer rather than reimplemented
    here, so the recorded "model input" is the same text the specialist generator
    would encode.
    """

    def __init__(self) -> None:
        from src.generation.specialist_prompt import render_specialist_prompt

        self._render = render_specialist_prompt
        self.payloads: list[list[dict[str, Any]]] = []
        self.calculations: list[Any] = []
        self.prompts: list[str] = []

    def generate(
        self,
        question: str,
        evidence_items: list[dict[str, Any]],
        calculation_result: Any = None,
    ) -> str:
        self.payloads.append([dict(item) for item in evidence_items])
        self.calculations.append(calculation_result)
        self.prompts.append(
            self._render(question, evidence_items, calculation_result)
        )
        return "The reported revenue was 391 million USD [E1]."


def _calculation(spec: dict[str, Any]):
    from decimal import Decimal

    from src.domain.calculation import (
        CalculationOperation,
        CalculationResult,
        CalculationStatus,
    )

    return CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=CalculationOperation(spec["operation"]),
        value=Decimal(spec["value"]),
        unit=spec.get("unit"),
    )


def build_state(scenario: str):
    """Build the authoritative state for an authored scenario."""

    from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1

    spec = SCENARIOS[scenario]
    state = AdaptiveRAGStateV1.new(
        f"b3-baseline-{scenario}",
        spec["question"],
        intent=spec["intent"],
    )
    state.add_evidence(
        [EvidencePacketV1.from_mapping(item) for item in spec["evidence"]]
    )
    state.bound_evidence_ids = [item["evidence_id"] for item in spec["evidence"]]
    if spec["calculation"] is not None:
        state._calculation_result_obj = _calculation(spec["calculation"])
    return state


def observe(scenario: str) -> dict[str, Any]:
    """Run the legacy B3 context path and record everything it decided.

    One implementation, used by both the capture and the assertion, so the
    frozen values and the checked values cannot drift apart.
    """

    from src.runtime.trusted_v2_generation import TrustedV2GenerationCapability

    specialist = RecordingSpecialist()
    capability = TrustedV2GenerationCapability(specialist=specialist)
    result = capability.generate(build_state(scenario))

    return {
        # what was selected
        "route": result.route,
        "route_reason": result.route_reason,
        "route_target": capability.last_decision.target.value,
        "bound_evidence_ids": list(result.bound_evidence_ids),
        "citation_ids": list(result.citation_ids),
        "calculation_ids": list(result.calculation_ids),
        # what the disclosure authority let across
        "disclosed_fields": list(capability.trace_snapshot()["disclosed_fields"]),
        "projected_evidence": specialist.payloads[0],
        "projected_calculation": specialist.calculations[0],
        # the model input
        "prompt": specialist.prompts[0],
        # Carried alongside the text so a regression reports a changed digest
        # rather than a sixty-line string diff.  Derived from the prompt in the
        # same call, so it cannot describe a different one.
        "prompt_sha256": hashlib.sha256(
            specialist.prompts[0].encode("utf-8")
        ).hexdigest(),
        "input_tokens": NOT_DETERMINED,
        # the candidate
        "candidate_answer": result.candidate_answer,
        "candidate_status": result.candidate_status,
        "candidate_generation_id": result.candidate_generation_id,
    }


#: Captured from the implementation at the H2A-3B0 commit, with F5 fixed and the
#: compiler not yet introduced.  See the module docstring for what this may and
#: may not be used to claim.
BASELINE: dict[str, dict[str, Any]] = {'multi_fact': {'route': 'MULTI',
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
