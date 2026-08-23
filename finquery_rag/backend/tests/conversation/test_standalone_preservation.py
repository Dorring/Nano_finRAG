"""Unit tests for Standalone Preservation and Context Anti-Pollution."""

import unittest
from src.conversation.contracts import ReasonCode
from src.conversation.service import ConversationContextManager


class TestStandalonePreservation(unittest.TestCase):
    def setUp(self):
        self.mgr = ConversationContextManager()

    def test_explicit_query_after_long_unrelated_history(self):
        conv_id = "test_pollution_1"
        
        # 5 turns discussing Apple Revenue
        self.mgr.process_user_turn(conv_id, "What was Apple FY2024 revenue?")
        self.mgr.record_assistant_turn(conv_id, "Apple revenue was $391B.")
        self.mgr.process_user_turn(conv_id, "What about FY2023?")
        self.mgr.record_assistant_turn(conv_id, "Apple FY2023 revenue was $383B.")
        self.mgr.process_user_turn(conv_id, "And FY2022?")
        self.mgr.record_assistant_turn(conv_id, "Apple FY2022 revenue was $394B.")
        
        # Turn 4: User suddenly asks explicit standalone query about Microsoft
        new_query = "What was Microsoft FY2023 operating income?"
        res = self.mgr.process_user_turn(conv_id, new_query)
        
        self.assertTrue(res.supported)
        # Standalone query must preserve Microsoft and Operating Income with 0 Apple pollution
        self.assertIn("MICROSOFT", res.standalone_query.upper())
        self.assertIn("OPERATING INCOME", res.standalone_query.upper())
        self.assertNotIn("Apple", res.standalone_query)
        self.assertNotIn("Revenue", res.standalone_query)


if __name__ == "__main__":
    unittest.main()
