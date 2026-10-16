"""Comprehensive verification suite for the 12 advanced financial capabilities.

Covers:
1. Fiscal year temporal drift & TTM alignment
2. Cross-currency arithmetic & dynamic FX conversion
3. Restatement resolution & comparative column preference
4. Stock splits & per-share metric adjustment
5. GAAP vs Non-GAAP disambiguation & Critic routing
6. Source authority hierarchy & multi-source arbitration
7. Point-in-Time (PIT) integrity & look-ahead bias prevention
8. Matrix dialogue state tracking
9. Auditable proof tree generation
10. Market API TTL cache & circuit breaker resilience
11. Async multi-source parallel fan-out
12. Calibrated fail-closed diagnostic explanation
"""

import asyncio
from decimal import Decimal
import pytest

from src.domain.calculation import (
    CalculationOperand,
    CalculationOperation,
    CalculationPlan,
    CalculationStatus,
)
from src.finance.fiscal_alignment import (
    check_fiscal_alignment,
    get_fiscal_year_end_month,
    register_fiscal_year_end,
)
from src.finance.stock_splits import (
    adjust_per_share_metric,
    get_split_adjustment_factor,
    register_stock_split,
)
from src.finance.calibrated_abstention import format_fail_closed_explanation
from src.finance.calculation_executor import execute_plan, _harmonize_operand_currencies
from src.conversation.contracts import DialogueState
from src.conversation.resolver import update_dialogue_matrix, get_dialogue_matrix
from rag_v2.routing.source_router import DataSourceType, SourceRouter
from rag_v2.adaptive.diagnostic_critic import DiagnosticCritic, CriticDiagnosisCategory
from rag_v2.sources.market_api import (
    CachedMarketProvider,
    CircuitBreakerMarketProvider,
    SandboxMarketProvider,
    get_fx_rate,
    register_fx_rate,
)
from rag_v2.sources.sql_fact_store import SqlFactStore
from rag_v2.sources.multi_source_adapter import (
    MultiSourceRetrievalAdapter,
    SourceAuthority,
    arbitrate_candidate_packets,
    determine_source_authority,
)
from rag_v2.adaptive.adaptive_contracts import (
    AdaptiveRAGStateV1,
    ReasonCode,
    ReplanActionV1,
    ToolCapability,
)


class Test12AdvancedFinancialCapabilities:

    # -------------------------------------------------------------
    # 1. 异构财年偏差与 TTM 对齐
    # -------------------------------------------------------------
    def test_point_1_fiscal_calendar_alignment(self):
        # Apple ends in Sept (9), Microsoft ends in June (6) -> 3 months drift
        rep = check_fiscal_alignment("Apple", "FY2024", "Microsoft", "FY2024")
        assert not rep.is_aligned
        assert rep.drift_months == 3
        assert rep.warning is not None
        assert "TTM" in rep.recommendation

        # Test dynamic registration
        register_fiscal_year_end("CustomCorp", 9)
        assert get_fiscal_year_end_month("CustomCorp") == 9
        rep2 = check_fiscal_alignment("Apple", "FY2024", "CustomCorp", "FY2024")
        assert rep2.is_aligned
        assert rep2.drift_months == 0

        # Test 2-digit year generality (FY24 vs FY24)
        rep_2digit = check_fiscal_alignment("Apple", "FY24", "Microsoft", "FY24")
        assert not rep_2digit.is_aligned
        assert rep_2digit.drift_months == 3

    # -------------------------------------------------------------
    # 2. 跨币种运算与动态 FX 汇率折算
    # -------------------------------------------------------------
    def test_point_2_cross_currency_and_fx(self):
        # Built-in rates: EUR/USD = 1.08
        rate = get_fx_rate("EUR", "USD")
        assert rate == Decimal("1.08")

        # Custom rate registration
        register_fx_rate("SGD", "USD", Decimal("0.75"))
        assert get_fx_rate("SGD", "USD") == Decimal("0.75")
        assert get_fx_rate("USD", "SGD") == Decimal("1") / Decimal("0.75")

        # Standard triangular cross-rate conversion via USD (EUR -> JPY)
        rate_eur_jpy = get_fx_rate("EUR", "JPY")
        assert rate_eur_jpy == Decimal("1.08") * Decimal("149.25")

        # Test operand harmonization
        op_usd = CalculationOperand(name="AAPL_Rev", value=Decimal("100"), unit="USD")
        op_eur = CalculationOperand(name="SAP_Rev", value=Decimal("100"), unit="EUR")
        harmonized = _harmonize_operand_currencies((op_usd, op_eur), target_currency="USD")
        assert harmonized[0].value == Decimal("100")
        assert harmonized[0].unit == "USD"
        assert harmonized[1].value == Decimal("108.00")  # 100 * 1.08
        assert harmonized[1].unit == "USD"

    # -------------------------------------------------------------
    # 3. 历史重述与追溯调整识别
    # -------------------------------------------------------------
    def test_point_3_restatements_resolution(self):
        store = SqlFactStore(":memory:")
        # Original reporting in FY2022 10-K
        store.upsert_fact(
            entity="Apple", metric="revenue", period="FY2022",
            value="394328", filing_date="2022-10-28", is_restated=False,
        )
        # Restated reporting in FY2023 10-K
        store.upsert_fact(
            entity="Apple", metric="revenue", period="FY2022",
            value="394320", filing_date="2023-10-27", is_restated=True,
        )

        # Default query prefers restated audited column
        fact_restated = store.query_fact("Apple", "revenue", "FY2022", prefer_restated=True)
        assert fact_restated is not None
        assert fact_restated["value"] == "394320"
        assert fact_restated["is_restated"] == 1

        pkt = store.as_evidence_packet(fact_restated, slot_id="slot_1")
        assert pkt["is_restated"] is True
        assert "[Restated]" in pkt["citation_id"]

        # Option to fetch original as-reported column
        fact_orig = store.query_fact("Apple", "revenue", "FY2022", prefer_restated=False)
        assert fact_orig is not None
        assert fact_orig["value"] == "394328"

    # -------------------------------------------------------------
    # 4. 拆股送转与每股指标复权调整
    # -------------------------------------------------------------
    def test_point_4_stock_splits_adjustment(self):
        # NVDA 10-for-1 split occurred on 2024-06-07
        factor = get_split_adjustment_factor("NVDA", "2024-01-01", "2024-07-01")
        assert factor == Decimal("10")

        # Entity name resolution (NVIDIA -> NVDA)
        factor_name = get_split_adjustment_factor("NVIDIA", "2024-01-01", "2024-07-01")
        assert factor_name == Decimal("10")

        # Pre-split EPS of $12.00 is adjusted to $1.20 post-split equivalent
        adj_val, split_fac, has_split = adjust_per_share_metric(
            "NVDA", "diluted_eps", Decimal("12.00"), "2024-01-01", "2024-07-01",
        )
        assert has_split is True
        assert split_fac == Decimal("10")
        assert adj_val == Decimal("1.20")

        # Bilingual per-share metric detection
        adj_cn, _, has_cn = adjust_per_share_metric(
            "NVIDIA", "每股收益 (EPS)", Decimal("12.00"), "2024-01-01", "2024-07-01",
        )
        assert has_cn is True
        assert adj_cn == Decimal("1.20")

        # Custom split registration
        register_stock_split("TEST", "2025-01-01", Decimal("2"))
        f2 = get_split_adjustment_factor("TEST", "2024-12-01", "2025-02-01")
        assert f2 == Decimal("2")

    # -------------------------------------------------------------
    # 5. GAAP 与 Non-GAAP（Adjusted）口径消歧
    # -------------------------------------------------------------
    def test_point_5_gaap_vs_non_gaap(self):
        # 1. Routing test: Non-GAAP queries route to MD&A Non-GAAP reconciliation section
        decision = SourceRouter.route_query("What was Tesla's adjusted EBITDA in 2023?")
        assert decision.target_source == DataSourceType.NARRATIVE_DISCLOSURES
        assert "Non-GAAP" in (decision.section_hint or "")

        # 2. Critic diagnosis: detects GAAP vs Non-GAAP mismatch
        slot = {"slot_id": "slot_adj", "metric": "adjusted ebitda", "period": "FY2023", "entity": "Tesla"}
        candidates = [{
            "source_text": "Tesla Consolidated Statements of Operations (GAAP) operating income was 8891.",
            "value": "8891",
            "metric": "operating income",
            "period": "FY2023",
        }]
        diag = DiagnosticCritic.diagnose_slot(slot, candidates, base_query="Tesla adjusted EBITDA")
        assert diag is not None
        assert diag.category == CriticDiagnosisCategory.GAAP_VS_NON_GAAP_MISMATCH
        assert "NON_GAAP" in diag.structural_constraints.get("accounting_standard", "")

    # -------------------------------------------------------------
    # 6. 数据源权威等级仲裁
    # -------------------------------------------------------------
    def test_point_6_source_authority_arbitration(self):
        pkt_statement = {
            "slot_ids": ["slot_rev"], "entity_id": "aapl", "metric": "revenue", "period": "FY2024",
            "value": "391035", "source": "PDF_RETRIEVAL", "citation_id": "Consolidated Statements of Operations",
        }
        pkt_sql = {
            "slot_ids": ["slot_rev"], "entity_id": "aapl", "metric": "revenue", "period": "FY2024",
            "value": "391030", "source": "SQL_FACT_STORE", "citation_id": "[10-K iXBRL] Apple FY2024 revenue",
        }
        pkt_market = {
            "slot_ids": ["slot_rev"], "entity_id": "aapl", "metric": "revenue", "period": "FY2024",
            "value": "390000", "source": "MARKET_API", "citation_id": "[Market API] Real-time",
        }

        assert determine_source_authority(pkt_statement) == SourceAuthority.AUDITED_PRIMARY_STATEMENT
        assert determine_source_authority(pkt_sql) == SourceAuthority.STRUCTURED_IXBRL_SQL
        assert determine_source_authority(pkt_market) == SourceAuthority.MARKET_API_REALTIME

        # When candidates conflict on slot_rev, Statement face wins
        arbitrated = arbitrate_candidate_packets([pkt_sql, pkt_market, pkt_statement])
        assert len(arbitrated) == 1
        assert arbitrated[0]["source"] == "PDF_RETRIEVAL"
        assert arbitrated[0]["value"] == "391035"
        assert "arbitration_record" in arbitrated[0]
        assert arbitrated[0]["arbitration_record"]["suppressed_count"] == 2

        # Test bilingual Chinese citation identification
        pkt_cn = {
            "slot_ids": ["slot_rev"], "entity_id": "aapl", "metric": "revenue", "period": "FY2024",
            "value": "391035", "source": "PDF_RETRIEVAL", "citation_id": "合并利润表",
        }
        assert determine_source_authority(pkt_cn) == SourceAuthority.AUDITED_PRIMARY_STATEMENT

    # -------------------------------------------------------------
    # 7. PIT 时点数据库与防未来函数
    # -------------------------------------------------------------
    def test_point_7_pit_lookahead_bias(self):
        store = SqlFactStore(":memory:")
        # Fact filed on 2023-10-27
        store.upsert_fact(
            entity="Microsoft", metric="net_income", period="FY2023",
            value="72361", filing_date="2023-10-27",
        )
        # Future fact filed on 2024-10-24
        store.upsert_fact(
            entity="Microsoft", metric="net_income", period="FY2024",
            value="88136", filing_date="2024-10-24",
        )

        # Query as of 2024-01-01 (point in time) -> FY2024 should NOT be accessible
        fact_future = store.query_fact("Microsoft", "net_income", "FY2024", as_of_date="2024-01-01")
        assert fact_future is None

        # Query as of 2024-12-31 -> FY2024 is accessible
        fact_present = store.query_fact("Microsoft", "net_income", "FY2024", as_of_date="2024-12-31")
        assert fact_present is not None
        assert fact_present["value"] == "88136"

        # Test 2-digit fiscal year querying (FY23 -> FY2023)
        fact_2digit = store.query_fact("Microsoft", "net_income", "FY23", as_of_date="2024-12-31")
        assert fact_2digit is not None
        assert fact_2digit["value"] == "72361"

    # -------------------------------------------------------------
    # 8. 矩阵式多实体-多时段会话状态机
    # -------------------------------------------------------------
    def test_point_8_matrix_dialogue_tracking(self):
        state = DialogueState(conversation_id="conv_matrix")

        # Turn 1: user introduces Apple and Microsoft for 2024
        update_dialogue_matrix(state, entities=["Apple", "Microsoft"], periods=["FY2024"], metrics=["revenue"])
        m1 = get_dialogue_matrix(state)
        assert set(m1["entities"]) == {"apple", "microsoft"}
        assert m1["periods"] == ["FY2024"]

        # Turn 2: user asks "And in 2023?" -> entities should be preserved, period appended
        update_dialogue_matrix(state, periods=["FY2023"])
        m2 = get_dialogue_matrix(state)
        assert set(m2["entities"]) == {"apple", "microsoft"}
        assert set(m2["periods"]) == {"FY2024", "FY2023"}

        # Verify serialization / deserialization fidelity
        s_dict = state.to_dict()
        state_restored = DialogueState.from_dict(s_dict)
        assert state_restored.active_matrix == state.active_matrix

    # -------------------------------------------------------------
    # 9. 投研级公式底稿与审计证明树 (Proof Tree)
    # -------------------------------------------------------------
    def test_point_9_auditable_proof_tree(self):
        plan = CalculationPlan(
            status=CalculationStatus.READY,
            operation=CalculationOperation.SPREAD,
            operands=(
                CalculationOperand(name="AAPL_PE", value=Decimal("34.5"), unit="ratio", document_name="Market API", page=1),
                CalculationOperand(name="MSFT_PE", value=Decimal("31.2"), unit="ratio", document_name="Market API", page=1),
            ),
            formula_version="spread.v1",
            precision=2,
            target_metric="pe_spread",
        )
        res = execute_plan(plan)
        assert res.status == CalculationStatus.EXECUTED
        assert res.value == Decimal("3.30")
        assert res.proof_tree is not None
        assert "Spread" in res.proof_tree["latex_formula"]
        assert len(res.proof_tree["operands"]) == 2
        assert res.proof_tree["deterministic_verification"] == "PASS_FAIL_CLOSED_VALIDATED"

    # -------------------------------------------------------------
    # 10. 外部行情 API 本地 TTL 缓存与熔断降级
    # -------------------------------------------------------------
    def test_point_10_market_api_cache_and_circuit_breaker(self):
        sandbox = SandboxMarketProvider()
        # 1. TTL Cache Test
        cached = CachedMarketProvider(sandbox, ttl_seconds=60.0)
        q1 = cached.get_quote("Apple", "stock_price")
        assert q1 is not None
        assert q1.get("is_cached") is not True

        # Second call within TTL hits cache
        q2 = cached.get_quote("Apple", "stock_price")
        assert q2 is not None
        assert q2.get("is_cached") is True

        # 2. Circuit Breaker Test
        # Mock a failing provider
        class FlakyProvider:
            def get_quote(self, entity, metric):
                raise ConnectionError("Upstream timeout")

        breaker = CircuitBreakerMarketProvider(
            provider=FlakyProvider(),
            failure_threshold=3,
            recovery_timeout=30.0,
            fallback_provider=sandbox,
        )

        # 3 calls fail and trip the circuit breaker
        for _ in range(3):
            breaker.get_quote("Apple", "stock_price")

        assert breaker.state == "OPEN"
        # 4th call immediately routes to fallback without raising exception
        fb = breaker.get_quote("Apple", "stock_price")
        assert fb is not None
        assert fb.get("circuit_breaker") == "OPEN_FALLBACK"

    # -------------------------------------------------------------
    # 11. 多数据源异步并发检索
    # -------------------------------------------------------------
    def test_point_11_async_multi_source_fan_out(self):
        store = SqlFactStore(":memory:")
        store.upsert_fact(entity="Apple", metric="revenue", period="FY2024", value="391035")
        market = SandboxMarketProvider()

        adapter = MultiSourceRetrievalAdapter(sql_store=store, market_provider=market)

        action = ReplanActionV1(
            capability=ToolCapability.STRUCTURED_FINANCIAL_LOOKUP,
            query="Apple FY2024 revenue",
            reason_code=ReasonCode.MISSING_SLOT,
            target_slots=("slot_rev",),
        )
        state = AdaptiveRAGStateV1.new(
            request_id="req_async_01",
            query="Apple FY2024 revenue",
            required_slots=[{"slot_id": "slot_rev", "entity": "Apple", "metric": "revenue", "period": "FY2024"}],
        )

        # Execute async retrieval
        results = asyncio.run(adapter.retrieve_async(action, state))
        assert len(results) == 1
        assert results[0]["value"] == "391035"

    # -------------------------------------------------------------
    # 12. 细粒度 Fail-Closed 诊断性拒答 UX
    # -------------------------------------------------------------
    def test_point_12_calibrated_fail_closed_ux(self):
        # 1. Negative CAGR base
        msg_cagr = format_fail_closed_explanation(
            "CAGR base value must be positive",
            operation="cagr",
            diagnostic_context={"base_value": "-$1,200M"},
        )
        assert "实数域无解" in msg_cagr
        assert "净利润绝对增减额" in msg_cagr

        # 2. Zero denominator in multiple
        msg_div = format_fail_closed_explanation(
            "denominator must be non-zero",
            operation="valuation_multiple",
            diagnostic_context={"denominator_metric": "每股收益 (EPS)"},
        )
        assert "分母为零在算术上无定义" in msg_div

        # 3. Missing slots
        msg_missing = format_fail_closed_explanation(
            "missing slots",
            missing_slots=[{"entity": "Apple", "metric": "capex", "period": "FY2024"}],
        )
        assert "缺少计算必需的权威事实" in msg_missing
        assert "Apple capex FY2024" in msg_missing
