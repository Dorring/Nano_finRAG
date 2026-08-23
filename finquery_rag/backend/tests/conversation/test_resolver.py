"""Unit tests for ContextualQueryResolver."""

import unittest
from src.conversation.contracts import (
    DialogueState,
    DialogueTurn,
    ReasonCode,
)
from src.conversation.resolver import ContextualQueryResolver


class TestContextualQueryResolver(unittest.TestCase):
    def setUp(self):
        self.resolver = ContextualQueryResolver()

    def test_fast_path_first_turn(self):
        query = "What was Apple FY2024 revenue?"
        res = self.resolver.resolve(query, dialogue_state=None)
        self.assertTrue(res.supported)
        self.assertFalse(res.requires_context)
        self.assertEqual(res.standalone_query, query)
        self.assertIn(ReasonCode.NO_CONTEXT_REQUIRED, res.reason_codes)

    def test_fast_path_self_contained_query(self):
        query = "What do Billings represent in financial reporting?"
        state = DialogueState(conversation_id="c1", active_entity="Apple", active_metric="Revenue", turn_count=1)
        res = self.resolver.resolve(query, dialogue_state=state)
        self.assertTrue(res.supported)
        self.assertFalse(res.requires_context)
        self.assertEqual(res.standalone_query, query)

    def test_entity_inheritance(self):
        state = DialogueState(
            conversation_id="c1",
            active_entity="Apple",
            active_metric="Revenue",
            active_period="FY2024",
            turn_count=1,
            recent_turns=[
                DialogueTurn(turn_id="t1", user_query="What was Apple FY2024 revenue?", standalone_query="What was Apple FY2024 revenue?")
            ]
        )
        query = "What about Microsoft?"
        res = self.resolver.resolve(query, dialogue_state=state)
        self.assertTrue(res.supported)
        self.assertTrue(res.requires_context)
        self.assertIn("MICROSOFT", res.standalone_query.upper())
        self.assertIn("FY2024", res.standalone_query)
        self.assertIn("Revenue", res.standalone_query)
        self.assertTrue(res.topic_switch)

    def test_relative_period_resolution(self):
        state = DialogueState(
            conversation_id="c1",
            active_entity="Apple",
            active_metric="Revenue",
            active_period="FY2024",
            turn_count=1,
            recent_turns=[
                DialogueTurn(turn_id="t1", user_query="What was Apple FY2024 revenue?", standalone_query="What was Apple FY2024 revenue?")
            ]
        )
        query = "What about the previous year?"
        res = self.resolver.resolve(query, dialogue_state=state)
        self.assertTrue(res.supported)
        self.assertTrue(res.requires_context)
        self.assertIn("FY2023", res.standalone_query)
        self.assertIn(ReasonCode.RELATIVE_PERIOD_RESOLVED, res.reason_codes)

    def test_cross_turn_calculation(self):
        state = DialogueState(
            conversation_id="c1",
            active_entity="Apple",
            active_metric="Revenue",
            active_period="FY2024",
            comparison_period="FY2023",
            turn_count=2,
            recent_turns=[
                DialogueTurn(turn_id="t1", user_query="What was Apple FY2024 revenue?", standalone_query="What was Apple FY2024 revenue?"),
                DialogueTurn(turn_id="t2", user_query="What about FY2023?", standalone_query="What was Apple FY2023 revenue?")
            ]
        )
        query = "How much did it grow?"
        res = self.resolver.resolve(query, dialogue_state=state)
        self.assertTrue(res.supported)
        self.assertTrue(res.requires_context)
        self.assertIn("Calculate the change in Apple Revenue", res.standalone_query)
        self.assertIn(ReasonCode.CROSS_TURN_CALCULATION_RESOLVED, res.reason_codes)

    def test_ambiguity_clarification(self):
        state = DialogueState(
            conversation_id="c1",
            active_entity="Apple",
            active_topic="MULTIPLE_METRICS_REVENUE_MARGIN",
            turn_count=1,
            recent_turns=[
                DialogueTurn(turn_id="t1", user_query="Give me Apple Revenue and Operating Margin for FY2024", standalone_query="...")
            ]
        )
        query = "What about 2023?"
        res = self.resolver.resolve(query, dialogue_state=state)
        self.assertTrue(res.clarification_required)
        self.assertTrue(res.ambiguity_detected)
        self.assertIn(ReasonCode.AMBIGUOUS_METRIC, res.reason_codes)

    def test_out_of_scope(self):
        query = "Recommend a movie to watch tonight."
        res = self.resolver.resolve(query, dialogue_state=None)
        self.assertFalse(res.supported)
        self.assertIn(ReasonCode.OUT_OF_SCOPE, res.reason_codes)


if __name__ == "__main__":
    unittest.main()
