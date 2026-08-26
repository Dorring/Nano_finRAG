"""Unit tests for conversation contracts using standard unittest."""

import unittest
from src.conversation.contracts import (
    AssistantProvenance,
    ConversationResolution,
    ConversationTurnOutcome,
    DialogueState,
    DialogueTurn,
    PendingClarification,
    ReasonCode,
)


class TestConversationContracts(unittest.TestCase):
    def test_dialogue_turn_instantiation(self):
        turn = DialogueTurn(
            turn_id="turn_1",
            user_query="What was Apple FY2024 revenue?",
            standalone_query="What was Apple FY2024 revenue?",
            referenced_evidence_ids=["chunk_123"],
            topic="AAPL_REVENUE",
        )
        self.assertEqual(turn.turn_id, "turn_1")
        self.assertEqual(turn.referenced_evidence_ids, ["chunk_123"])
        d = turn.to_dict()
        self.assertEqual(d["turn_id"], "turn_1")

    def test_dialogue_state_instantiation(self):
        state = DialogueState(
            conversation_id="conv_100",
            active_entity="Apple",
            active_metric="Revenue",
            active_period="FY2024",
            active_topic="AAPL_REVENUE",
            turn_count=1,
        )
        self.assertEqual(state.active_entity, "Apple")
        self.assertEqual(state.turn_count, 1)
        d = state.to_dict()
        self.assertEqual(d["conversation_id"], "conv_100")
        self.assertEqual(d["active_metric"], "Revenue")

    def test_lifecycle_fields_round_trip_without_financial_facts(self):
        state = DialogueState(
            conversation_id="conv_lifecycle",
            active_entity="Apple",
            active_metric=None,
            active_period="FY2023",
            pending_clarification=PendingClarification(
                reason_codes=[ReasonCode.AMBIGUOUS_METRIC],
                candidates=["Revenue", "Operating Margin"],
                unresolved_fields=["metric"],
                source_turn_id="turn_2",
                entity="Apple",
                period="FY2023",
            ),
            last_assistant_provenance=AssistantProvenance(
                assistant_turn_id="request-1",
                evidence_ids=["chunk-1"],
                citation_ids=["citation-1"],
                calculation_ids=[],
                release_status="RELEASED",
                outcome=ConversationTurnOutcome.FINANCIAL_ANSWER,
            ),
            last_processed_request_id="request-1",
            last_processed_original_query="Apple FY2024 Revenue?",
            last_turn_outcome=ConversationTurnOutcome.FINANCIAL_ANSWER,
        )
        restored = DialogueState.from_dict(state.to_dict())
        self.assertEqual(restored.pending_clarification.candidates, ["Revenue", "Operating Margin"])
        self.assertEqual(restored.last_assistant_provenance.evidence_ids, ["chunk-1"])
        self.assertEqual(restored.last_processed_request_id, "request-1")
        self.assertEqual(restored.last_turn_outcome, ConversationTurnOutcome.FINANCIAL_ANSWER)
        payload = restored.to_dict()
        self.assertNotIn("answer_numeric_value", payload)
        self.assertNotIn("trusted_answer", payload)

    def test_conversation_resolution_standalone(self):
        res = ConversationResolution(
            supported=True,
            requires_context=False,
            standalone_query="What was Apple FY2024 revenue?",
            explicit_fields=["entity", "metric", "period"],
            reason_codes=[ReasonCode.NO_CONTEXT_REQUIRED],
        )
        self.assertTrue(res.supported)
        self.assertFalse(res.requires_context)
        self.assertIn(ReasonCode.NO_CONTEXT_REQUIRED, res.reason_codes)
        self.assertFalse(res.clarification_required)

    def test_conversation_resolution_clarification(self):
        res = ConversationResolution(
            supported=True,
            requires_context=True,
            ambiguity_detected=True,
            clarification_required=True,
            clarification_question="Did you mean Apple's Revenue or Operating Margin for FY2023?",
            clarification_options=["Revenue", "Operating Margin", "Both"],
            reason_codes=[ReasonCode.AMBIGUOUS_METRIC],
        )
        self.assertTrue(res.clarification_required)
        self.assertTrue(res.ambiguity_detected)
        self.assertIn(ReasonCode.AMBIGUOUS_METRIC, res.reason_codes)


if __name__ == "__main__":
    unittest.main()
