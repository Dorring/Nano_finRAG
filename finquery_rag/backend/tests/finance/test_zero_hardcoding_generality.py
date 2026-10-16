"""Verification test suite for zero-hardcoding and dynamic registration generalization."""

import pytest
from src.conversation.resolver import KNOWN_ENTITIES, ContextualQueryResolver, register_known_entity
from src.conversation.relevance_filter import ContextRelevanceFilter
from src.conversation.contracts import DialogueState, DialogueTurn
from src.retrieval.metadata_scope import MetadataFilterPlannerV1
from src.retrieval_v3.query_features import register_query_entity, extract_metric_phrases
from src.pdf_retrieval_v4.operand_planner import register_operand_issuer, _clean_phrase
from src.pdf_retrieval_v4.document_tree import DocumentHierarchyTree
from rag_v2.sources.market_api import register_market_metric, normalize_market_metric, BaseMarketProvider
from rag_v2.sources.sql_fact_store import normalize_period, SqlFactStore


class TestZeroHardcodingGenerality:
    def test_dynamic_entity_registration_in_conversation(self):
        # Register an unseen custom company
        custom_entity = "syntheticbio"
        register_known_entity(custom_entity)
        assert custom_entity in KNOWN_ENTITIES

        # Test topic switch in ContextRelevanceFilter
        filt = ContextRelevanceFilter()
        state = DialogueState(
            conversation_id="s1",
            active_entity="Apple",
            active_metric="revenue",
            active_period="FY2024",
            active_topic="operating_performance",
        )
        turns = [
            DialogueTurn(turn_id="1", user_query="What was Apple revenue?", standalone_query="What was Apple revenue?", assistant_response="383B"),
            DialogueTurn(turn_id="2", user_query="What about SyntheticBio?", standalone_query="What about SyntheticBio?", assistant_response=""),
        ]
        # Current query introduces syntheticbio -> Topic switch should be recognized
        filtered = filt.filter_turns("What was SyntheticBio net income?", state, turns)
        assert isinstance(filtered, list)

    def test_dynamic_ticker_registration_in_metadata_planner(self):
        # Register a new ticker
        new_ticker = "FOOBAR"
        MetadataFilterPlannerV1.register_known_ticker(new_ticker)
        planner = MetadataFilterPlannerV1()
        scope = planner.plan(f"What was FY2023 revenue for {new_ticker}?")
        ticker_cond = next((c for c in scope.conditions if c.field == "ticker"), None)
        assert ticker_cond is not None
        assert new_ticker in ticker_cond.values

    def test_dynamic_query_entity_and_operand_issuer_stripping(self):
        # Test query features entity stripping
        register_query_entity("AcmeCorp")
        phrases = extract_metric_phrases("What was AcmeCorp total revenues?", ())
        assert all("acmecorp" not in p.raw_text.lower() for p in phrases)

        # Test operand planner issuer stripping
        register_operand_issuer("AcmeCorp")
        cleaned = _clean_phrase("AcmeCorp total revenues for FY2023")
        assert "acmecorp" not in cleaned.lower()

    def test_dynamic_tree_topic_term_registration(self):
        DocumentHierarchyTree.register_topic_term("restructuring")
        assert "restructuring" in DocumentHierarchyTree._CUSTOM_TOPIC_TERMS

        tree = DocumentHierarchyTree(doc_id="TEST_10K")
        # Pruning with the custom topic hint should execute cleanly without error
        pruned = tree.prune(section_hint="restructuring activities")
        assert isinstance(pruned, list)

    def test_dynamic_market_metric_registration(self):
        register_market_metric("ev to ebitda", "ev_ebitda")
        assert normalize_market_metric("EV to EBITDA") == "ev_ebitda"

        # Test ISO currency normalization
        provider = BaseMarketProvider()
        pkt = provider.as_evidence_packet(
            quote={"entity": "TokyoElectron", "ticker": "TOELY", "metric": "stock_price", "value": "12000", "unit": "JPY"},
            slot_id="slot_1",
        )
        assert pkt["currency"] == "JPY"

    def test_sql_fact_store_multi_decade_period_normalization(self):
        assert normalize_period("1998") == "FY1998"
        assert normalize_period("FY1995") == "FY1995"
        assert normalize_period("2024") == "FY2024"

        # Currency test
        store = SqlFactStore(":memory:")
        pkt = store.as_evidence_packet(
            fact={"entity": "Siemens", "entity_id": "sie", "metric": "revenue", "period": "FY2023", "value": "77B", "unit": "EUR"},
            slot_id="slot_2",
        )
        assert pkt["currency"] == "EUR"
