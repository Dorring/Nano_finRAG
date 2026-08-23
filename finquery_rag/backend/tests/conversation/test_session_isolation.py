"""Unit tests for Multi-Session State Isolation."""

import unittest
from src.conversation.service import ConversationContextManager


class TestSessionIsolation(unittest.TestCase):
    def setUp(self):
        self.mgr = ConversationContextManager()

    def test_session_isolation_between_conversations(self):
        conv_a = "session_A_apple"
        conv_b = "session_B_tesla"
        
        # Session A asks about Apple
        self.mgr.process_user_turn(conv_a, "What was Apple FY2024 revenue?")
        
        # Session B asks about Tesla
        self.mgr.process_user_turn(conv_b, "What was Tesla FY2024 automotive gross margin?")
        
        # Session A follow-up: What about previous year? -> Should resolve to Apple
        res_a = self.mgr.process_user_turn(conv_a, "What about the previous year?")
        self.assertIn("Apple", res_a.standalone_query)
        self.assertNotIn("Tesla", res_a.standalone_query)
        
        # Session B follow-up: What about previous year? -> Should resolve to Tesla
        res_b = self.mgr.process_user_turn(conv_b, "What about the previous year?")
        self.assertIn("Tesla", res_b.standalone_query)
        self.assertNotIn("Apple", res_b.standalone_query)


if __name__ == "__main__":
    unittest.main()
