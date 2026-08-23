"""Adversarial Trust Boundary Tests.

Verifies the strict invariant: CONVERSATION_CONTEXT_NOT_EVIDENCE.
Historical assistant text (even if hallucinated or wrong) must NEVER become
VerifiedEvidence or be directly used as Calculator operands.
"""

import unittest
from src.conversation.service import ConversationContextManager


class TestTrustBoundary(unittest.TestCase):
    def setUp(self):
        self.mgr = ConversationContextManager()

    def test_hallucinated_assistant_number_not_propagated_to_query(self):
        conv_id = "test_trust_1"
        
        # Turn 1: User asks for Apple FY2024 revenue
        res1 = self.mgr.process_user_turn(conv_id, "What was Apple FY2024 revenue?")
        self.assertTrue(res1.supported)
        
        # Turn 1 Assistant response hallucinated a crazy number: $999B
        self.mgr.record_assistant_turn(
            conv_id,
            assistant_response="Apple FY2024 revenue was $999.00 Billion.",
            referenced_evidence_ids=["chunk_hallucinated_999b"]
        )
        
        # Turn 2: User asks for previous year
        res2 = self.mgr.process_user_turn(conv_id, "What about FY2023?")
        self.assertTrue(res2.supported)
        
        # Turn 2 Assistant response: $900B
        self.mgr.record_assistant_turn(
            conv_id,
            assistant_response="Apple FY2023 revenue was $900.00 Billion.",
            referenced_evidence_ids=["chunk_hallucinated_900b"]
        )
        
        # Turn 3: User asks for calculation: How much did it grow?
        res3 = self.mgr.process_user_turn(conv_id, "How much did it grow?")
        self.assertTrue(res3.supported)
        
        # Invariant Check:
        # The standalone query must be a semantic calculation request,
        # and must NOT inject the fake $999B or $900B numbers as operands!
        self.assertNotIn("999", res3.standalone_query)
        self.assertNotIn("900", res3.standalone_query)
        self.assertIn("Calculate", res3.standalone_query)
        self.assertIn("Apple", res3.standalone_query)
        self.assertIn("Revenue", res3.standalone_query)


if __name__ == "__main__":
    unittest.main()
