"""Unit tests for conversation contracts using standard unittest."""

import unittest
from src.conversation.contracts import (
    ConversationResolution,
    DialogueState,
    DialogueTurn,
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
