"""Unit tests for ContextRelevanceFilter and ContextBudgetManager."""

import unittest
from src.conversation.contracts import DialogueState, DialogueTurn
from src.conversation.context_budget import ContextBudgetManager
from src.conversation.relevance_filter import ContextRelevanceFilter


class TestRelevanceAndBudget(unittest.TestCase):
    def setUp(self):
        self.filter = ContextRelevanceFilter()
        self.budget = ContextBudgetManager(recent_turns_limit=4, target_tokens=1000)

    def test_relevance_filter_topic_switch_penalty(self):
        state = DialogueState(conversation_id="c1", active_entity="Apple", active_metric="Revenue", turn_count=3)
        turns = [
            DialogueTurn(turn_id="t1", user_query="What was Apple FY2024 revenue?", standalone_query="Apple FY2024 revenue"),
            DialogueTurn(turn_id="t2", user_query="What about 2023?", standalone_query="Apple FY2023 revenue"),
            DialogueTurn(turn_id="t3", user_query="Thanks!", standalone_query="Thanks!"),
        ]
        # Query switches to Tesla
        filtered = self.filter.filter_turns("What was Tesla operating income?", state, turns)
        # Noisy turn 'Thanks!' should be filtered out
        self.assertNotIn("t3", [t.turn_id for t in filtered])

    def test_budget_manager_protects_referenced_turn(self):
        state = DialogueState(
            conversation_id="c1",
            active_entity="Apple",
            active_metric="Revenue",
            referenced_turn_ids=["t1"],  # t1 is explicitly referenced
            turn_count=10,
        )
        # Create 10 turns
        turns = [
            DialogueTurn(turn_id=f"t{i}", user_query=f"Turn {i} query", standalone_query=f"Turn {i} standalone")
            for i in range(1, 11)
        ]
        selected, _, total_tok = self.budget.prepare_context("Current query", state, turns)
        selected_ids = [t.turn_id for t in selected]
        # t1 must be retained because it is protected
        self.assertIn("t1", selected_ids)
        # Recent turns should also be retained
        self.assertIn("t10", selected_ids)
        self.assertLessEqual(len(selected), 5)

    def test_token_growth_stabilization_500_turns(self):
        state = DialogueState(conversation_id="c1", active_entity="Apple", active_metric="Revenue", turn_count=500)
        turns = [
            DialogueTurn(turn_id=f"t{i}", user_query=f"What was Apple FY20{i%20:02d} revenue?", standalone_query=f"Apple FY20{i%20:02d} revenue")
            for i in range(1, 501)
        ]
        
        # Test 5, 20, 50, 100, 500 turns
        tokens_log = []
        for n in [5, 20, 50, 100, 500]:
            sub_turns = turns[:n]
            filtered = self.filter.filter_turns("What was the previous year?", state, sub_turns)
            selected, compressed, tok = self.budget.prepare_context("What was the previous year?", state, filtered)
            tokens_log.append((n, tok, len(selected)))

        # Verify token count does not grow linearly with n
        tok_5 = tokens_log[0][1]
        tok_500 = tokens_log[-1][1]
        # For 500 turns, tokens should remain bounded within budget (< target_tokens + margin)
        self.assertLess(tok_500, self.budget.max_tokens)
        self.assertLessEqual(tokens_log[-1][2], 6)  # at most recent 4 + protected turns


if __name__ == "__main__":
    unittest.main()
