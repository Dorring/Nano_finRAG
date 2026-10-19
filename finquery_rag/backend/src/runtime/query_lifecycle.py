"""Shared Conversation-aware execution for the query HTTP transports."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..conversation.contracts import ConversationTurnOutcome
from ..conversation.shadow_service import ConversationShadowService
from ..conversation.sqlite_store import ConversationStateConflictError
from .query_execution_service import QueryExecutionService
from .response_mapper import to_legacy_query_dict
from .runtime_adapters import LegacyFinancialRuntimeAdapter
from .runtime_contract import FinancialQARuntime, FinancialQueryRequest, FinancialQueryResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UserTurnExecutionRequest:
    """Transport-neutral input for one user turn."""

    request_id: str
    user_id: int
    original_query: str
    session_id: str | None = None
    document_names: list[str] = field(default_factory=list)
    n_results: int = 3
    conversation_mode: str = "off"


@dataclass
class UserTurnExecutionResult:
    """Final business result shared by JSON and validated-final SSE."""

    status: str
    answer: str | None
    clarification: dict[str, Any] | None
    citations: list[dict[str, Any]]
    evidence_ids: list[str]
    citation_ids: list[str]
    calculation_ids: list[str]
    reason_codes: list[str]
    release_status: str
    runtime_version: str
    conversation_mode: str
    original_query: str
    standalone_query: str
    query_as_resolved: bool
    request_id: str
    session_id: str | None
    legacy_result: dict[str, Any]
    runtime_result: FinancialQueryResult | None = None
    idempotent_replay: bool = False


class QueryLifecycleService:
    """Single Conversation -> Runtime -> final commit business path."""

    def __init__(
        self,
        *,
        session_manager: Any,
        memory_store: Any,
        get_rag_engine: Callable[[], Any],
        get_conversation_service: Callable[[], ConversationShadowService],
        financial_runtime_adapter_enabled: Callable[[], bool],
        active_query_requires_context: Callable[[str], bool],
        active_query_is_out_of_scope: Callable[[str], bool],
        assistant_session_metadata: Callable[..., dict[str, Any]],
        execution_service_factory: Callable[[Any], QueryExecutionService] = QueryExecutionService,
        financial_runtime_factory: Callable[
            [Any, FinancialQueryRequest], FinancialQARuntime
        ] | None = None,
        financial_runtime_requires_engine: Callable[[], bool] | None = None,
    ) -> None:
        self.session_manager = session_manager
        self.memory_store = memory_store
        self.get_rag_engine = get_rag_engine
        self.get_conversation_service = get_conversation_service
        self.financial_runtime_adapter_enabled = financial_runtime_adapter_enabled
        self.active_query_requires_context = active_query_requires_context
        self.active_query_is_out_of_scope = active_query_is_out_of_scope
        self.assistant_session_metadata = assistant_session_metadata
        self.execution_service_factory = execution_service_factory
        self.financial_runtime_factory = financial_runtime_factory
        self.financial_runtime_requires_engine = financial_runtime_requires_engine

    @staticmethod
    def _value(value: Any) -> str:
        return str(getattr(value, "value", value))

    @staticmethod
    def _empty_legacy(answer: str, docs: Sequence[str]) -> dict[str, Any]:
        return {
            "answer": answer,
            "sources": [],
            "searched_docs": list(docs),
            "retrieved_chunks": [],
            "retrieval_debug": {},
            "calculations": [],
        }

    @staticmethod
    def _control_outcome(status: str) -> str:
        return {
            "CLARIFICATION_REQUIRED": ConversationTurnOutcome.CLARIFICATION,
            "OUT_OF_SCOPE": ConversationTurnOutcome.OUT_OF_SCOPE,
        }.get(status, ConversationTurnOutcome.ERROR)

    @classmethod
    def _runtime_outcome(cls, runtime: FinancialQueryResult | None) -> str:
        if runtime is None:
            return ConversationTurnOutcome.FINANCIAL_ANSWER
        return {
            "FAIL_CLOSED": ConversationTurnOutcome.FAIL_CLOSED,
            "OUT_OF_SCOPE": ConversationTurnOutcome.OUT_OF_SCOPE,
            "ERROR": ConversationTurnOutcome.ERROR,
        }.get(cls._value(runtime.status), ConversationTurnOutcome.FINANCIAL_ANSWER)

    def _history(
        self, request: UserTurnExecutionRequest
    ) -> list[dict[str, Any]] | None:
        if not request.session_id:
            return None
        return self.session_manager.get_recent_messages(
            request.session_id, request.user_id
        )

    def _profile(self, user_id: int) -> dict[str, Any] | None:
        profile = self.memory_store.get_profile(user_id)
        return profile if isinstance(profile, dict) else None

    def _commit_control(
        self,
        *,
        request: UserTurnExecutionRequest,
        answer: str,
        status: str,
        reason_codes: Sequence[str],
        service: ConversationShadowService | None,
        replay: bool,
    ) -> None:
        if not request.session_id or replay:
            return
        self.session_manager.add_message(
            request.session_id, request.user_id, "user", request.original_query
        )
        self.session_manager.add_message(
            request.session_id,
            request.user_id,
            "assistant",
            answer,
            metadata={
                "control_status": status,
                "reason_codes": list(reason_codes),
                "evidence_ids": [],
                "calculation_ids": [],
            },
        )
        if service is None:
            return
        try:
            service.record_control_turn(
                user_id=request.user_id,
                session_id=request.session_id,
                request_id=request.request_id,
                original_query=request.original_query,
                outcome=self._control_outcome(status),
                reason_codes=list(reason_codes),
            )
        except ConversationStateConflictError:
            raise
        except Exception:
            logger.exception("conversation control metadata failed")

    def _control_result(
        self,
        *,
        request: UserTurnExecutionRequest,
        status: str,
        answer: str,
        reason_codes: Sequence[str],
        service: ConversationShadowService | None,
        replay: bool,
        options: Sequence[str] | None = None,
    ) -> UserTurnExecutionResult:
        reason_codes = list(reason_codes)
        clarification = (
            {
                "question": answer,
                "reason_codes": reason_codes,
                "options": list(options or []),
            }
            if status == "CLARIFICATION_REQUIRED"
            else None
        )
        self._commit_control(
            request=request,
            answer=answer,
            status=status,
            reason_codes=reason_codes,
            service=service,
            replay=replay,
        )
        legacy_result = self._empty_legacy(answer, request.document_names)
        # Control outcomes (clarification/out-of-scope) stop before a
        # FinancialQARuntime is invoked, so there is no runtime trace object
        # to provide an id.  The validated logical request id remains a safe,
        # deterministic correlation id for these user-visible outcomes.
        legacy_result["trace_id"] = request.request_id
        return UserTurnExecutionResult(
            status=status,
            answer=answer,
            clarification=clarification,
            citations=[],
            evidence_ids=[],
            citation_ids=[],
            calculation_ids=[],
            reason_codes=reason_codes,
            release_status="NOT_APPLICABLE",
            runtime_version="V1",
            conversation_mode=request.conversation_mode,
            original_query=request.original_query,
            standalone_query=request.original_query,
            query_as_resolved=False,
            request_id=request.request_id,
            session_id=request.session_id,
            legacy_result=legacy_result,
            idempotent_replay=replay,
        )

    async def _run_runtime(
        self,
        *,
        request: UserTurnExecutionRequest,
        query: str,
        history: list[dict[str, Any]] | None,
        profile: dict[str, Any] | None,
        query_as_resolved: bool,
        service: ConversationShadowService | None,
        replay: bool,
    ) -> UserTurnExecutionResult:
        runtime_request = FinancialQueryRequest(
            request_id=request.request_id,
            user_id=str(request.user_id),
            session_id=request.session_id or f"__stateless__:{request.request_id}",
            original_query=request.original_query,
            standalone_query=query,
            query_as_resolved=query_as_resolved,
            conversation_metadata={},
            request_metadata={
                "document_names": list(request.document_names),
                "n_results": request.n_results,
                "conversation_history": history,
                "memory_profile": profile,
            },
        )
        # V2 owns its R4 index and structured fact store.  Do not eagerly
        # construct the legacy RAG engine for an official V2 request: doing so
        # would make V2 depend on V1 retrieval dependencies before its own
        # production builder is reached.  Keep the historical eager behavior
        # for callers that do not provide an explicit runtime dependency
        # policy, and for V1/shadow paths which still need the legacy engine.
        requires_engine = self.financial_runtime_requires_engine
        engine = (
            self.get_rag_engine()
            if bool(request.document_names) or requires_engine is None or requires_engine()
            else None
        )
        runtime: FinancialQueryResult | None = None
        if self.financial_runtime_adapter_enabled():
            runtime_impl = (
                self.financial_runtime_factory(engine, runtime_request)
                if self.financial_runtime_factory is not None
                else LegacyFinancialRuntimeAdapter(engine)
            )
            runtime = await self.execution_service_factory(runtime_impl).execute(
                runtime_request
            )
            logger.info(
                "financial runtime completed runtime=%s route=%s status=%s release=%s reason_codes=%s",
                self._value(runtime.runtime_version),
                (
                    runtime.runtime_metadata.attributes.get("route")
                    if runtime.runtime_metadata is not None
                    else None
                ),
                self._value(runtime.status),
                self._value(runtime.release_status),
                list(runtime.reason_codes),
            )
            legacy = to_legacy_query_dict(runtime)
            # V2 intentionally does not construct the legacy RAG engine, so
            # early coordinator failures may not have a trace execution id.
            # Keep the public correlation contract total by using the already
            # validated logical request id as a deterministic fallback.
            legacy.setdefault("trace_id", request.request_id)
        else:
            if engine is None:
                raise RuntimeError(
                    "legacy query execution requires the RAG engine",
                )
            kwargs = {
                "question": query,
                "doc_names": list(request.document_names),
                "n_results": request.n_results,
                "user_id": request.user_id,
                "conversation_history": history,
                "memory_profile": profile,
            }
            if query_as_resolved:
                kwargs["query_as_resolved"] = True
            legacy = dict(await engine.query(**kwargs))
        if query_as_resolved and query != request.original_query:
            legacy["rewritten_question"] = query

        if request.session_id and not replay:
            self.session_manager.add_message(
                request.session_id,
                request.user_id,
                "user",
                request.original_query,
            )
            self.session_manager.add_message(
                request.session_id,
                request.user_id,
                "assistant",
                legacy["answer"],
                metadata=self.assistant_session_metadata(result=legacy),
            )
        clarification_payload = (
            legacy.get("clarification")
            or (
                runtime.runtime_metadata.attributes.get("clarification")
                if runtime and runtime.runtime_metadata and isinstance(runtime.runtime_metadata.attributes, Mapping)
                else None
            )
        )
        if service is not None and request.session_id and not replay:
            try:
                evidence_ids = list(runtime.evidence_ids) if runtime else []
                citation_ids = list(runtime.citation_ids) if runtime else []
                calculation_ids = list(runtime.calculation_ids) if runtime else []
                release_status = (
                    self._value(runtime.release_status) if runtime else "NOT_APPLICABLE"
                )
                service_outcome = (
                    "CLARIFICATION"
                    if clarification_payload
                    else self._runtime_outcome(runtime)
                )
                service.record_assistant_turn(
                    user_id=request.user_id,
                    session_id=request.session_id,
                    request_id=request.request_id,
                    original_query=request.original_query,
                    referenced_evidence_ids=evidence_ids,
                    citation_ids=citation_ids,
                    calculation_ids=calculation_ids,
                    release_status=release_status,
                    outcome=service_outcome,
                    clarification=clarification_payload,
                )
            except ConversationStateConflictError:
                raise
            except Exception:
                logger.exception("conversation assistant metadata failed")
        status_value = (
            "CLARIFICATION_REQUIRED"
            if clarification_payload
            else (self._value(runtime.status) if runtime else "ANSWER")
        )
        return UserTurnExecutionResult(
            status=status_value,
            answer=legacy.get("answer"),
            clarification=clarification_payload,
            citations=list(legacy.get("sources", [])),
            evidence_ids=list(runtime.evidence_ids) if runtime else [],
            citation_ids=list(runtime.citation_ids) if runtime else [],
            calculation_ids=list(runtime.calculation_ids) if runtime else [],
            reason_codes=list(runtime.reason_codes) if runtime else [],
            release_status=(
                self._value(runtime.release_status) if runtime else "NOT_APPLICABLE"
            ),
            runtime_version=self._value(runtime.runtime_version) if runtime else "V1",
            conversation_mode=request.conversation_mode,
            original_query=request.original_query,
            standalone_query=query,
            query_as_resolved=query_as_resolved,
            request_id=request.request_id,
            session_id=request.session_id,
            legacy_result=legacy,
            runtime_result=runtime,
            idempotent_replay=replay,
        )

    @staticmethod
    def _get_conversational_reply(question: str) -> str | None:
        raw = (question or "").strip()
        text = raw.lower()

        greetings = ["hi", "hello", "hey", "nihao", "hi there", "hey there", "good morning", "good afternoon", "good evening", "你好", "您好", "嗨", "哈喽", "早上好", "下午好", "晚上好"]
        if any(text == g or (text.startswith(g) and len(text) <= len(g) + 2) for g in greetings):
            return (
                "您好！我是 Nano_finRAG 机构级智能财报问答分析助手。\n"
                "您可以向我提问关于苹果、微软、英伟达、特斯拉等上市公司的权威财报数据或进行财务计算，例如：\n"
                "  * According to Apple's FY2025 10-K filing, what was the reported Deferred?\n"
                "  * What was Apple's Total net sales in FY2025?\n"
                "  * Compare the gross margin of Apple and Microsoft in FY2024\n"
                "请问您想查询哪家公司的财务数据？"
            )

        thanks = ["thank you", "thanks", "thx", "谢谢", "感谢", "多谢", "非常感谢"]
        if any(text == t or (text.startswith(t) and len(text) <= len(t) + 2) for t in thanks):
            return "不客气！如果您还有其他关于财报科目、附注披露或确定性财务计算的疑问，欢迎随时提问。"

        byes = ["bye", "goodbye", "see you", "再见", "拜拜"]
        if any(text == b or (text.startswith(b) and len(text) <= len(b) + 2) for b in byes):
            return "再见！祝您投研工作顺利，有财报核算需求随时提问。"

        chat_inquiry = [
            "不支持普通聊天吗", "支持普通聊天吗", "不能聊天吗", "可以聊天吗", "能聊天吗",
            "不支持闲聊吗", "支持闲聊吗", "能闲聊吗", "可以闲聊吗", "闲聊", "普通聊天",
            "为什么不能聊天", "只支持财报吗", "只能问财报吗", "不能闲聊吗", "聊天吗",
        ]
        if any(k in text for k in chat_inquiry):
            return (
                "是的。Nano_finRAG 是专为「上市公司官方经审计定期财报（SEC 10-K / 10-Q）」深度核验与形式化分析设计的机构级金融问答系统，并非开放域通用闲聊机器人。\n\n"
                "系统核心定位与设计原则：\n"
                "1. 0.00% 形式化防幻觉阻断（Fail-Closed）：严守金融研报合规红线，对未披露或证据不足的数据绝不凭空编造；\n"
                "2. 审计级证据链溯源：每个披露数据点均精确锚定到财报具体 PDF 页码与表格单元格坐标；\n"
                "3. 报表重名多口径消歧：遇到重名或不同口径会计科目（如 Deferred）主动发起反思追问；\n"
                "4. 确定性财务沙盒计算：杜邦分析、毛利率对比、CAGR 与财务预测均由独立安全沙盒执行，杜绝大模型算术幻觉。\n\n"
                "推荐您可以尝试提问已收录的美股标的（如 Apple、Microsoft、Tesla、NVIDIA 等）：\n"
                "  * According to Apple's FY2025 10-K filing, what was the reported Deferred?\n"
                "  * What was Microsoft's Net Income in FY2024?\n"
                "  * Compare the gross margin of Apple and Microsoft in FY2024"
            )

        stock_price_keywords = [
            "股价", "股票价格", "实时行情", "走势", "涨跌", "现价", "收盘价", "开盘价",
            "盘前", "盘后", "k线", "市值多少", "能买吗", "买入", "卖出", "会涨吗",
            "stock price", "share price", "trading at", "stock quote", "market price", "price target"
        ]
        if any(k in text for k in stock_price_keywords):
            return (
                "【关于二级市场股价/行情提示】\n"
                "本系统专精于上市公司已披露的「经审计历史定期财务报告」（SEC 10-K / 10-Q，如营业收入、净利润、资产负债表、递延收益等确定性财务披露），暂不支持二级市场「实时股票行情/即时股价/盘中交易」数据。\n\n"
                "此外，当前知识库收录的权威标的主要为已完成底稿索引的美股科技与权重企业（如 Apple、Microsoft、NVIDIA、Tesla、JPMorgan 等），尚未收录比亚迪（BYD / 002594 / 1211.HK）等 A 股/港股标的的定期财报底稿。\n\n"
                "建议您查询已收录美股标的的官方经审计财务指标，例如：\n"
                "  * According to Apple's FY2025 10-K filing, what was the reported Deferred?\n"
                "  * What was Tesla's Total Revenues in FY2024?\n"
                "  * Compare the gross margin of Apple and Microsoft in FY2024"
            )

        api_inquiry = [
            "调用api", "调用外部api", "调用接口", "调用外部接口", "联网检索", "联网搜索",
            "不能联网吗", "可以联网吗", "支持联网吗", "能联网吗", "上网搜", "网上搜",
            "从网上找", "外部检索", "实时检索", "联网", "调用api进行检索", "调用api检索",
            "web search", "online search", "call api", "search internet",
        ]
        if any(k in text for k in api_inquiry):
            return (
                "【关于外部 API 联网与数据源架构说明】\n"
                "目前系统采用「本地审计底稿 + 严格闭环验证（Fail-Closed）」架构，未开放直接调用通用公网搜索 API，主要出于以下金融严谨性考量：\n\n"
                "1. 投研级数据合规与审计溯源（Auditability）：\n"
                "   机构级财务分析对数据真实性要求极高。公网开放搜索存在大量非官方解读、自媒体二手数据或统计口径不一的问题。本系统强制要求每一笔披露数据必须能精准溯源至 SEC 官方原文的【具体 PDF 页码与表格单元格坐标】。\n"
                "2. 0.00% 形式化防幻觉阻断（Fail-Closed）：\n"
                "   如果允许大模型从公网开放文本中自由提炼数据，极易引发大模型拼接幻觉。系统坚持数据放行前必须经过形式化计算沙盒与坐标校验，未经验证的外部数据一律严格阻断。\n"
                "3. 权威垂直数据源扩展机制：\n"
                "   系统设计了合规数据接入接口（如 SEC EDGAR 官方实时披露 Feed、专业金融终端 Wind/Bloomberg API），外部数据经结构化清洗、坐标对齐并建立审计底稿后，即可无缝纳入权威问答库。\n\n"
                "建议您体验当前已收录标的（如 Apple、Microsoft、NVIDIA、Tesla 等）的高精度问答与多轮消歧能力！"
            )

        identity = ["你是谁", "你能做什么", "怎么用", "如何使用", "你能干嘛", "你是干什么的", "who are you", "what can you do"]
        if any(k in text for k in identity):
            return (
                "我是 Nano_finRAG 机构级智能财报问答与多轮消歧交互助手。\n"
                "系统核心能力：\n"
                "1. 权威证据链溯源：所有披露数据精准锚定至上市公司 10-K/10-Q 底稿页码与表格单元格坐标。\n"
                "2. 报表重名多口径消歧：当科目存在多处披露或会计口径差异时，主动发起反思追问并支持多轮撤销重选。\n"
                "3. 确定性财务沙盒计算：支持杜邦三因子拆解、毛利率对比、CAGR 复合增长与多情景预测，0.00% 形式化防幻觉保障。\n"
                "4. 覆盖主流标的：已全面索引 Apple、Microsoft、NVIDIA、Tesla、JPMorgan 等上市公司财报。"
            )

        return None

    def _conversational_result(
        self,
        *,
        request: UserTurnExecutionRequest,
        answer: str,
    ) -> UserTurnExecutionResult:
        if request.session_id:
            self.session_manager.add_message(
                request.session_id, request.user_id, "user", request.original_query
            )
            self.session_manager.add_message(
                request.session_id,
                request.user_id,
                "assistant",
                answer,
                metadata={
                    "control_status": "CONVERSATION",
                    "reason_codes": ["CONVERSATIONAL_GREETING"],
                    "evidence_ids": [],
                    "calculation_ids": [],
                },
            )
        legacy_result = self._empty_legacy(answer, request.document_names)
        legacy_result["trace_id"] = request.request_id
        legacy_result["intent"] = "conversation"
        return UserTurnExecutionResult(
            status="ANSWER",
            answer=answer,
            clarification=None,
            citations=[],
            evidence_ids=[],
            citation_ids=[],
            calculation_ids=[],
            reason_codes=["CONVERSATIONAL_GREETING"],
            release_status="RELEASED",
            runtime_version="V1",
            conversation_mode=request.conversation_mode,
            original_query=request.original_query,
            standalone_query=request.original_query,
            query_as_resolved=False,
            request_id=request.request_id,
            session_id=request.session_id,
            legacy_result=legacy_result,
            idempotent_replay=False,
        )

    @staticmethod
    def _get_market_quote_reply(question: str) -> dict[str, Any] | None:
        raw = (question or "").strip()
        text = raw.lower()

        metric_patterns = {
            "stock_price": [
                "股价", "股票价格", "现价", "最新价", "收盘价", "开盘价", "行情", "走势", "涨跌",
                "stock price", "share price", "trading at", "stock quote", "current price", "last price"
            ],
            "pe_ratio": [
                "市盈率", "动态市盈率", "静态市盈率", "pe ratio", "p/e", " pe ", "trailing pe", "forward pe"
            ],
            "market_cap": [
                "市值", "总市值", "market cap", "market capitalization"
            ],
            "exchange_rate": [
                "汇率", "exchange rate", "fx rate", "spot rate"
            ],
        }

        detected_metric = None
        for m, patterns in metric_patterns.items():
            if any(p in text for p in patterns):
                detected_metric = m
                break

        if not detected_metric:
            return None

        from rag_v2.sources.market_api import _ENTITY_TICKER_MAP, LiveMarketProvider, SandboxMarketProvider

        sorted_entities = sorted(_ENTITY_TICKER_MAP.keys(), key=len, reverse=True)
        matched_entity = None
        for ent in sorted_entities:
            if re.search(r"[\u4e00-\u9fff]", ent):
                if ent in text:
                    matched_entity = ent
                    break
            else:
                if re.search(rf"(?<!\w){re.escape(ent)}(?!\w)", text):
                    matched_entity = ent
                    break

        if not matched_entity:
            return None

        try:
            provider = LiveMarketProvider(timeout_seconds=2.0, fallback_provider=SandboxMarketProvider())
            quote = provider.get_quote(matched_entity, detected_metric)
        except Exception:
            try:
                quote = SandboxMarketProvider().get_quote(matched_entity, detected_metric)
            except Exception:
                quote = None

        if not quote:
            return None

        metric_display_names = {
            "stock_price": "实时股票价格 (Stock Price)",
            "pe_ratio": "市盈率估值 (P/E Ratio)",
            "market_cap": "企业总市值 (Market Capitalization)",
            "exchange_rate": "即期汇率 (Exchange Rate)",
        }
        metric_label = metric_display_names.get(detected_metric, detected_metric)
        source_channel = "【真实公网即时行情接口】" if quote.get("source") == "MARKET_API_LIVE" else "【沙盒基准行情库】"

        answer = (
            f"【📈 市场行情与估值指标放行 (Market API)】\n\n"
            f"  • 标的名称: {quote['entity']} (证券代码: {quote['ticker']})\n"
            f"  • 指标科目: {metric_label}\n"
            f"  • 最新数据: {quote['value']} {quote['unit']}\n"
            f"  • 数据源接入通道: {quote.get('source', 'MARKET_API')} {source_channel}\n"
            f"  • 披露时效状态: {quote.get('timestamp', 'REALTIME')}\n\n"
            f"说明: 该数据由 Nano_finRAG 市场行情 API 接口直接调取并放行。如需查询经审计的历史定期财报底稿（SEC 10-K/10-Q 资产负债、利润表、现金流或附注科目），欢迎随时提问。"
        )

        return {
            "answer": answer,
            "quote": quote,
            "entity": matched_entity,
            "metric": detected_metric,
        }

    def _market_quote_result(
        self,
        *,
        request: UserTurnExecutionRequest,
        market_payload: dict[str, Any],
    ) -> UserTurnExecutionResult:
        answer = market_payload["answer"]
        quote = market_payload["quote"]
        if request.session_id:
            self.session_manager.add_message(
                request.session_id, request.user_id, "user", request.original_query
            )
            self.session_manager.add_message(
                request.session_id,
                request.user_id,
                "assistant",
                answer,
                metadata={
                    "control_status": "MARKET_QUOTE",
                    "reason_codes": ["MARKET_API_RELEASE"],
                    "evidence_ids": [quote.get("ticker", "")],
                    "calculation_ids": [],
                },
            )
        legacy_result = self._empty_legacy(answer, request.document_names)
        legacy_result["trace_id"] = request.request_id
        legacy_result["intent"] = "market_quote"
        legacy_result["calculations"] = [
            {
                "operation": quote.get("metric"),
                "value": f"{quote.get('value')} {quote.get('unit')}",
                "status": "SUCCESS",
                "formula": f"MarketData({quote.get('ticker')}) -> {quote.get('source')}",
            }
        ]
        return UserTurnExecutionResult(
            status="ANSWER",
            answer=answer,
            clarification=None,
            citations=[],
            evidence_ids=[quote.get("ticker", "")],
            citation_ids=[],
            calculation_ids=[],
            reason_codes=["MARKET_API_RELEASE"],
            release_status="RELEASED",
            runtime_version="V1",
            conversation_mode=request.conversation_mode,
            original_query=request.original_query,
            standalone_query=request.original_query,
            query_as_resolved=False,
            request_id=request.request_id,
            session_id=request.session_id,
            legacy_result=legacy_result,
            idempotent_replay=False,
        )

    async def execute_user_turn(
        self, request: UserTurnExecutionRequest
    ) -> UserTurnExecutionResult:
        """Resolve, gate, execute V1, commit, and return one final result."""
        history = self._history(request)
        profile = self._profile(request.user_id)
        service: ConversationShadowService | None = None
        replay = False
        if request.conversation_mode == "shadow":
            try:
                service = self.get_conversation_service()
                replay = bool(
                    service.is_request_processed(
                        user_id=request.user_id,
                        session_id=request.session_id,
                        request_id=request.request_id,
                        original_query=request.original_query,
                    )
                )
                service.observe(
                    request_id=request.request_id,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    original_query=request.original_query,
                    prior_history=history,
                )
            except ConversationStateConflictError:
                raise
            except Exception:
                logger.exception("conversation shadow invocation failed")

        market_payload = self._get_market_quote_reply(request.original_query)
        if market_payload:
            return self._market_quote_result(
                request=request,
                market_payload=market_payload,
            )

        conv_reply = self._get_conversational_reply(request.original_query)
        if conv_reply:
            return self._conversational_result(
                request=request,
                answer=conv_reply,
            )

        query = request.original_query
        execution_history = history
        query_as_resolved = False
        if request.conversation_mode == "on":
            if request.session_id is None:
                if self.active_query_is_out_of_scope(request.original_query):
                    return self._control_result(
                        request=request,
                        status="OUT_OF_SCOPE",
                        answer="This question is outside the supported financial document scope.",
                        reason_codes=["OUT_OF_SCOPE"],
                        service=None,
                        replay=False,
                    )
                if self.active_query_requires_context(request.original_query):
                    answer = "I need the company, metric, and period context before I can answer this safely. (当前提问缺少必要的财务上下文信息，请提供包含目标公司与指标的完整问题。)"
                    return self._control_result(
                        request=request,
                        status="CLARIFICATION_REQUIRED",
                        answer=answer,
                        reason_codes=["CONTEXT_UNAVAILABLE"],
                        service=None,
                        replay=False,
                    )
                execution_history = None
            else:
                service = self.get_conversation_service()
                replay = bool(
                    service.is_request_processed(
                        user_id=request.user_id,
                        session_id=request.session_id,
                        request_id=request.request_id,
                        original_query=request.original_query,
                    )
                )
                if not history and self.active_query_requires_context(request.original_query):
                    answer = "I need the earlier company, metric, and period context before I can answer this safely. (当前提问缺少必要的财务上下文信息，请提供包含目标公司与指标的完整问题。)"
                    return self._control_result(
                        request=request,
                        status="CLARIFICATION_REQUIRED",
                        answer=answer,
                        reason_codes=["CONTEXT_UNAVAILABLE"],
                        service=service,
                        replay=replay,
                    )
                try:
                    resolution = service.resolve_active(
                        request_id=request.request_id,
                        user_id=request.user_id,
                        session_id=request.session_id,
                        original_query=request.original_query,
                        prior_history=history,
                    )
                except ConversationStateConflictError:
                    raise
                except Exception as exc:
                    logger.exception("active conversation resolution failed")
                    if self.active_query_requires_context(request.original_query):
                        reason_codes = (
                            ["CONTEXT_STATE_UNAVAILABLE"]
                            if "STATE" in type(exc).__name__.upper()
                            or "SQLITE" in type(exc).__name__.upper()
                            else ["CONTEXT_RESOLUTION_FAILED"]
                        )
                        answer = "I need the company, metric, and period stated explicitly before I can answer this safely."
                        return self._control_result(
                            request=request,
                            status="CLARIFICATION_REQUIRED",
                            answer=answer,
                            reason_codes=reason_codes,
                            service=service,
                            replay=replay,
                        )
                    service = None
                    execution_history = None
                else:
                    codes = [
                        str(getattr(code, "value", code))
                        for code in (resolution.reason_codes or [])
                    ]
                    replay = replay or "IDEMPOTENT_REPLAY" in codes
                    if not resolution.supported:
                        return self._control_result(
                            request=request,
                            status="OUT_OF_SCOPE",
                            answer="This question is outside the supported financial document scope.",
                            reason_codes=codes or ["OUT_OF_SCOPE"],
                            service=service,
                            replay=replay,
                        )
                    if resolution.clarification_required:
                        answer = resolution.clarification_question or "Which company, metric, or period should I use?"
                        return self._control_result(
                            request=request,
                            status="CLARIFICATION_REQUIRED",
                            answer=answer,
                            reason_codes=codes or ["AMBIGUOUS_REFERENCE"],
                            options=resolution.clarification_options,
                            service=service,
                            replay=replay,
                        )
                    query = (resolution.standalone_query or "").strip()
                    if not query:
                        answer = "Please restate the company, metric, and period so I can answer this safely."
                        return self._control_result(
                            request=request,
                            status="CLARIFICATION_REQUIRED",
                            answer=answer,
                            reason_codes=["INVALID_RESOLUTION"],
                            service=service,
                            replay=replay,
                        )
                    query_as_resolved = bool(
                        resolution.requires_context or query != request.original_query
                    )
                    execution_history = None
        return await self._run_runtime(
            request=request,
            query=query,
            history=execution_history,
            profile=profile,
            query_as_resolved=query_as_resolved,
            service=service,
            replay=replay,
        )
