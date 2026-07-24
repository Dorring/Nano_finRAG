from .trace import TraceLogger
import os
import re

from .vector_store import query_collection, list_all_documents, get_front_matter_chunks
from .retrieval import SqliteBM25Retriever
from .reranker import build_reranker
from .intent import classify_query_intent
from src.domain.query import QueryRequest
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.retrieval.candidate_fusion import (
    normalize_scores as _normalize_scores,
    dedupe_chunks as _dedupe_chunks,
    chunk_doc_name as _chunk_doc_name,
    ensure_multi_doc_coverage as _ensure_multi_doc_coverage,
    boost_front_matter_chunks as _boost_front_matter_chunks,
    summarize_retrieved_chunks as _summarize_retrieved_chunks,
    source_from_chunk as _source_from_chunk,
)
from src.retrieval.context_builder import ContextBuilder, EvidenceSufficiencyEvaluator
from src.generation.prompt_builder import get_system_prompt
from src.generation.llm_gateway import LLMGateway
from src.generation.response_renderer import validate_answer
from src.generation.deterministic_answers import DeterministicAnswerExtractor
from src.application.rag_orchestrator import RAGOrchestrator
from src.finance.calculation_pipeline import CalculationPipeline
from src.validation.validation_pipeline import GroundedValidationPipeline
from src.services.memory_profile import build_memory_profile_context

# 尝试导入 tiktoken，如果未安装则降级为字符估算
try:
    import tiktoken
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False


class RAGEngine:
    """
    多文档检索增强生成系统（适配 NanoChat 2B 金融垂类模型）。
    支持查询单个文档或跨多个文档进行查询。

    完整的 RAG 流水线：
    1. 混合检索（稠密向量 + 稀疏BM25）
    2. 使用倒数秩融合（RRF）进行重排
    3. 上下文构建（带 Token 动态截断与部分保留）
    4. 大语言模型（LLM）异步生成

    关键约束：
    - NanoChat 2B 模型上下文长度仅 2048 token
    - 不支持独立的 system 角色（适配层会自动合并到 user 消息）
    - System prompt 需精简，为检索上下文和生成留足空间
    """

    # 2048 上下文的紧凑分配：
    # system_prompt(~150) + 检索上下文(~1000) + 用户问题(~100) + 生成回答(~700) + 特殊token(~50) ≈ 2000 < 2048
    DEFAULT_MAX_CONTEXT_TOKENS = 1100
    DEFAULT_MAX_NEW_TOKENS = 512
    DEFAULT_TOP_K_CHUNKS = 3

    def __init__(self, llm_client, model_name: str = "nanochat",
                 use_hybrid: bool = True,
                 max_context_tokens: int = None,
                 max_new_tokens: int = None,
                 bm25_db_path: str | None = None,
                 trace_db_path: str | None = None,
                 reranker_name: str | None = None,
                 reranker_model: str | None = None,
                 retrieval_candidate_multiplier: int = 4,
                 enable_calculation_pipeline: bool = True,
                 enable_validation_pipeline: bool = True):
        """
        RAGEngine 类的初始化方法。

        Args:
            llm_client: OpenAI API 客户端实例，指向 nanochat OpenAI 兼容适配层。
            model_name (str): 模型名称，对应 chat_openai_compat.py 暴露的模型名，默认 "nanochat"。
            use_hybrid (bool): 是否启用 BM25 + 向量搜索的混合检索模式，默认为 True。
            max_context_tokens (int): 上下文最大 Token 限制，默认 1100（适配 2048 上下文窗口）。
            max_new_tokens (int): 模型单次最大生成 Token 数，默认 512。
            bm25_db_path (str | None): SQLite FTS5 稀疏检索数据库路径。未传入时读取 BM25_DB_PATH。
            trace_db_path (str | None): TraceLogger SQLite 路径。未传入时读取 TRACE_DB_PATH。
            reranker_name (str | None): Optional reranker name. None disables reranking.
            reranker_model (str | None): Optional reranker model name/path for model-backed rerankers.
            retrieval_candidate_multiplier (int): Candidate expansion factor for hybrid retrieval.
            enable_calculation_pipeline (bool): Whether to enable the Phase 3 deterministic
                calculation pipeline. Defaults to True so production callers get calculation
                support without explicit wiring. Set to False for testing or rollback.
            enable_validation_pipeline (bool): Whether to enable the Phase 4 grounded
                validation pipeline (answerability + response validation + repair).
                Defaults to True so production callers get validation without explicit
                wiring. Set to False for testing or rollback.
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.use_hybrid = use_hybrid
        self.max_context_tokens = max_context_tokens or self.DEFAULT_MAX_CONTEXT_TOKENS
        self.max_new_tokens = max_new_tokens or self.DEFAULT_MAX_NEW_TOKENS

        if bm25_db_path is None:
            bm25_db_path = os.getenv("BM25_DB_PATH", "rag_bm25.db")
        if trace_db_path is None:
            trace_db_path = os.getenv("TRACE_DB_PATH", "trace_log.db")

        self.bm25_db_path = bm25_db_path
        self.trace_db_path = trace_db_path
        self.bm25_retriever = SqliteBM25Retriever(db_path=bm25_db_path)
        self.trace_logger = TraceLogger(db_path=trace_db_path, sample_rate=1.0, redact_content=True)
        self.min_score_threshold = 0.0  # chunks below this score are discarded
        self.rrf_sufficiency_threshold = float(os.getenv("RAG_RRF_SUFFICIENCY_THRESHOLD", "0.008"))
        self.dense_sufficiency_threshold = float(os.getenv("RAG_DENSE_SUFFICIENCY_THRESHOLD", "0.15"))
        self.numeric_rrf_floor = float(os.getenv("RAG_NUMERIC_RRF_FLOOR", "0.008"))
        self.numeric_dense_floor = float(os.getenv("RAG_NUMERIC_DENSE_FLOOR", "0.08"))
        self.reranker = build_reranker(reranker_name, model_name_or_path=reranker_model)
        self.retrieval_candidate_multiplier = max(1, int(retrieval_candidate_multiplier or 1))
        self._last_retrieval_debug = self._make_retrieval_debug(0, 0)
        self._query_processor = QueryProcessor(memory_profile_context_fn=build_memory_profile_context)

        # 初始化 Token 计算器
        if TOKENIZER_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = None

        self._retrieval_pipeline = RetrievalPipeline(
            dense_query_fn=query_collection,
            bm25_retriever=self.bm25_retriever,
            reranker=self.reranker,
            query_processor=self._query_processor,
            candidate_multiplier=self.retrieval_candidate_multiplier,
            use_hybrid=self.use_hybrid,
        )
        self._context_builder = ContextBuilder(
            max_context_tokens=self.max_context_tokens,
            min_score_threshold=self.min_score_threshold,
            tokenizer=self.tokenizer,
        )
        self._sufficiency_evaluator = EvidenceSufficiencyEvaluator(
            rrf_sufficiency_threshold=self.rrf_sufficiency_threshold,
            dense_sufficiency_threshold=self.dense_sufficiency_threshold,
        )
        self._llm_gateway = LLMGateway(
            llm_client=self.llm_client,
            model_name=self.model_name,
            max_new_tokens=self.max_new_tokens,
        )
        self._deterministic_extractor = DeterministicAnswerExtractor(
            query_processor=self._query_processor,
        )
        # Phase 3: Deterministic calculation pipeline. Enabled by default so
        # production callers get calculation support without explicit wiring.
        # Can be disabled via enable_calculation_pipeline=False for testing
        # or rollback.
        self._calculation_pipeline = (
            CalculationPipeline()
            if enable_calculation_pipeline
            else None
        )
        # Phase 4: Grounded validation pipeline. Enabled by default so
        # production callers get answerability evaluation + response
        # validation + repair without explicit wiring. Can be disabled
        # via enable_validation_pipeline=False for testing or rollback.
        self._validation_pipeline = (
            GroundedValidationPipeline()
            if enable_validation_pipeline
            else None
        )
        self._orchestrator = RAGOrchestrator(
            query_processor=self._query_processor,
            retrieval_pipeline=self._retrieval_pipeline,
            context_builder=self._context_builder,
            sufficiency_evaluator=self._sufficiency_evaluator,
            llm_gateway=self._llm_gateway,
            deterministic_extractor=self._deterministic_extractor,
            trace_logger=self.trace_logger,
            intent_classifier=classify_query_intent,
            list_all_documents_fn=list_all_documents,
            get_front_matter_chunks_fn=get_front_matter_chunks,
            numeric_rrf_floor=self.numeric_rrf_floor,
            numeric_dense_floor=self.numeric_dense_floor,
            model_name=self.model_name,
            calculation_pipeline=self._calculation_pipeline,
            validation_pipeline=self._validation_pipeline,
        )

    def _get_bm25_retriever(self, doc_name=str, user_id: int = None):
        """获取 SQLite FTS5 稀疏检索器。如果未启用混合检索则返回 None。"""
        if not self.use_hybrid:
            return None
        return self.bm25_retriever

    def _normalize_scores(self, chunks: list) -> list:
        return _normalize_scores(chunks)

    def _make_retrieval_debug(self, candidate_count: int, returned_count: int) -> dict:
        return {
            "reranker": self.reranker.name if self.reranker else None,
            "reranker_enabled": self.reranker is not None,
            "candidate_count": candidate_count,
            "returned_count": returned_count,
            "candidate_multiplier": self.retrieval_candidate_multiplier,
        }

    def _apply_reranker(self, query: str, chunks: list, top_k: int) -> list:
        candidate_count = len(chunks)
        if not self.reranker:
            selected = chunks[:top_k]
        else:
            selected = self.reranker.rerank(query, chunks, top_k=top_k)
        self._last_retrieval_debug = self._make_retrieval_debug(
            candidate_count,
            len(selected),
        )
        return selected

    @staticmethod
    def _dedupe_chunks(chunks: list) -> list:
        return _dedupe_chunks(chunks)

    @staticmethod
    def _chunk_doc_name(chunk: dict) -> str | None:
        return _chunk_doc_name(chunk)

    def _ensure_multi_doc_coverage(self, candidates: list, selected: list, doc_names: list[str], top_k: int | None) -> list:
        return _ensure_multi_doc_coverage(candidates, selected, doc_names, top_k)

    def _has_cjk(self, text: str) -> bool:
        return self._query_processor._has_cjk(text)

    def _is_document_front_matter_query(self, query: str) -> bool:
        return self._query_processor.is_front_matter_query(query)

    def _expand_retrieval_query(self, query: str) -> str:
        return self._query_processor.expand(query)

    def _boost_front_matter_chunks(self, query: str, chunks: list) -> list:
        return _boost_front_matter_chunks(
            query, chunks,
            is_front_matter_query_fn=self._query_processor.is_front_matter_query,
        )

    @staticmethod
    def _summarize_retrieved_chunks(chunks: list) -> list:
        return _summarize_retrieved_chunks(chunks)

    @staticmethod
    def _source_from_chunk(chunk: dict) -> dict:
        return _source_from_chunk(chunk)

    def retrieve_single_document(self, doc_name: str, query: str, user_id: int = None, n_results: int = 3) -> list:
        result = self._retrieval_pipeline.retrieve_single(
            document_name=doc_name, query=query, user_id=user_id, top_k=n_results,
        )
        self._last_retrieval_debug = self._retrieval_pipeline._last_retrieval_debug
        return result

    async def retrieve_multiple_documents(self, doc_names: list[str], query: str, user_id: int = None, n_results: int = 3) -> list:
        result = await self._retrieval_pipeline.retrieve_multiple(
            document_names=doc_names, query=query, user_id=user_id, top_k=n_results,
        )
        self._last_retrieval_debug = self._retrieval_pipeline._last_retrieval_debug
        return result

    def _is_title_query(self, query: str) -> bool:
        return self._query_processor.is_title_query(query)

    def _is_numeric_financial_query(self, query: str) -> bool:
        return self._query_processor.is_numeric_query(query)

    def _should_try_deterministic_numeric_answer(self, query: str, chunks: list) -> bool:
        return self._query_processor.should_try_deterministic_numeric_answer(query, chunks)

    def _should_generate_with_low_confidence(self, query: str, chunks: list) -> bool:
        return self._query_processor.should_generate_with_low_confidence(
            query, chunks,
            numeric_rrf_floor=self.numeric_rrf_floor,
            numeric_dense_floor=self.numeric_dense_floor,
        )

    def answer_front_matter_query(self, query: str, chunks: list) -> dict | None:
        return self._deterministic_extractor.answer_front_matter_query(query, chunks)

    @staticmethod
    def _clean_deterministic_title(title: str) -> str:
        return DeterministicAnswerExtractor._clean_deterministic_title(title)

    @staticmethod
    def _is_valid_deterministic_title(title: str) -> bool:
        return DeterministicAnswerExtractor._is_valid_deterministic_title(title)

    def retrieve_front_matter_chunks(self, doc_names: list[str], query: str, user_id: int | None = None) -> list:
        """Direct metadata lookup for deterministic front-matter questions."""
        if not self._is_title_query(query) or not doc_names or user_id is None:
            return []
        if "reporting period" in (query or "").lower():
            return []
        chunks = []
        for doc_name in doc_names:
            for chunk in get_front_matter_chunks(doc_name=doc_name, user_id=user_id, subtype="title"):
                title = re.sub(r"^title\s*:\s*", "", chunk.get("content", ""), flags=re.IGNORECASE).strip()
                if self._is_valid_deterministic_title(self._clean_deterministic_title(title)):
                    chunks.append(chunk)
        return chunks

    @staticmethod
    def _parent_context_key(chunk: dict) -> str | None:
        return ContextBuilder._parent_context_key(chunk)

    def _merge_parent_context_chunks(self, chunks: list) -> list:
        return self._context_builder._merge_parent_context_chunks(chunks)

    @staticmethod
    def _compact_child_snippet(content: str, *, max_chars: int = 500) -> str:
        return ContextBuilder._compact_child_snippet(content, max_chars=max_chars)

    @staticmethod
    def _compose_parent_context(parent_excerpt: str, child_snippets: list[str]) -> str:
        return ContextBuilder._compose_parent_context(parent_excerpt, child_snippets)

    def build_context(self, chunks: list) -> tuple:
        # Sync threshold and token budget in case they were changed after init
        self._context_builder._min_score_threshold = self.min_score_threshold
        self._context_builder._max_context_tokens = self.max_context_tokens
        return self._context_builder.build(chunks)

    def _get_system_prompt(self) -> str:
        return get_system_prompt()

    def _validate_answer(self, answer: str, sources: list) -> str:
        return validate_answer(answer, sources, max_new_tokens=self.max_new_tokens)

    def _check_context_sufficiency(self, chunks: list) -> tuple:
        result = self._sufficiency_evaluator.evaluate(chunks)
        return result.is_sufficient, result.best_score, result.average_score

    def _compute_confidence(self, chunks: list) -> float:
        return self._sufficiency_evaluator.confidence(chunks)

    def _looks_like_followup_question(self, question: str) -> bool:
        return self._query_processor.looks_like_followup_question(question)

    def _is_valid_rewritten_query(self, original: str, rewritten: str) -> bool:
        return self._query_processor.is_valid_rewritten_query(original, rewritten)

    async def _rewrite_query_with_context(
        self,
        question: str,
        conversation_history: list,
        memory_profile: dict | None = None,
    ) -> str:
        return await self._query_processor.rewrite(
            question, conversation_history, memory_profile,
            llm_client=self.llm_client, model_name=self.model_name,
        )

    async def generate_answer(self, context: str, query: str) -> str:
        return await self._llm_gateway.generate(context, query)

    def answer_numeric_query_from_context(self, query: str, context: str, sources: list) -> dict | None:
        return self._deterministic_extractor.answer_numeric_query_from_context(query, context, sources)

    def answer_factual_query_from_context(self, query: str, context: str, sources: list) -> dict | None:
        return self._deterministic_extractor.answer_factual_query_from_context(query, context, sources)

    def answer_deterministic_query_from_context(self, query: str, context: str, sources: list) -> dict | None:
        return self._deterministic_extractor.answer_deterministic_query_from_context(query, context, sources)


    @staticmethod
    def _parse_context_lines(context: str) -> list[dict]:
        return DeterministicAnswerExtractor._parse_context_lines(context)

    def _rank_context_evidence(self, query: str, context: str, *, require_number: bool, window_radius: int = 1) -> list[dict]:
        return self._deterministic_extractor._rank_context_evidence(query, context, require_number=require_number, window_radius=window_radius)

    @staticmethod
    def _evidence_window(parsed: list[dict], index: int, *, radius: int, require_number: bool, query_terms: set[str]) -> str:
        return DeterministicAnswerExtractor._evidence_window(parsed, index, radius=radius, require_number=require_number, query_terms=query_terms)

    @staticmethod
    def _select_distinct_evidence(evidence: list[dict], *, limit: int) -> list[dict]:
        return DeterministicAnswerExtractor._select_distinct_evidence(evidence, limit=limit)

    @staticmethod
    def _normalize_numeric_phrase(value: str, query: str, evidence_text: str) -> str:
        return DeterministicAnswerExtractor._normalize_numeric_phrase(value, query, evidence_text)

    @classmethod
    def _extract_numeric_phrases(cls, query: str, text: str) -> list[str]:
        return DeterministicAnswerExtractor._extract_numeric_phrases(query, text)

    @classmethod
    def _summarize_numeric_values(cls, query: str, selected: list[dict], *, context: str | None = None) -> str | None:
        return DeterministicAnswerExtractor._summarize_numeric_values(query, selected, context=context)

    @classmethod
    def _targeted_numeric_summary(cls, query: str, selected: list[dict], *, context: str | None = None) -> str | None:
        return DeterministicAnswerExtractor._targeted_numeric_summary(query, selected, context=context)

    @staticmethod
    def _parse_comma_number(value: str) -> int:
        return DeterministicAnswerExtractor._parse_comma_number(value)

    @staticmethod
    def _summarize_factual_evidence(query: str, selected: list[dict], *, context: str | None = None) -> str | None:
        return DeterministicAnswerExtractor._summarize_factual_evidence(query, selected, context=context)

    @staticmethod
    def _best_sentence_with_terms(text: str, terms: tuple[str, ...]) -> str | None:
        return DeterministicAnswerExtractor._best_sentence_with_terms(text, terms)

    @staticmethod
    def _first_evidence_sentence(text: str) -> str | None:
        return DeterministicAnswerExtractor._first_evidence_sentence(text)

    @staticmethod
    def _should_try_deterministic_factual_answer(query: str) -> bool:
        return QueryProcessor().should_try_deterministic_factual_answer(query)

    @staticmethod
    def _important_query_terms(query: str) -> set[str]:
        return DeterministicAnswerExtractor._important_query_terms(query)

    @staticmethod
    def _numeric_evidence_score(line: str, query_terms: set[str]) -> float:
        return DeterministicAnswerExtractor._numeric_evidence_score(line, query_terms)

    @staticmethod
    def _factual_evidence_score(line: str, query_terms: set[str]) -> float:
        return DeterministicAnswerExtractor._factual_evidence_score(line, query_terms)

    @staticmethod
    def _is_numeric_financial_query(query: str) -> bool:
        return QueryProcessor().is_numeric_query(query)

    def generate_answer_stream(self, context: str, query: str):
        return self._llm_gateway.generate_stream(context, query)

    async def query(
        self,
        question: str,
        doc_names: list[str] | None = None,
        user_id: int = None,
        n_results: int = 3,
        conversation_history: list = None,
        memory_profile: dict | None = None,
    ) -> dict:
        """Facade entry point: build a ``QueryRequest``, delegate to the
        orchestrator, and unwrap the legacy dict for API compatibility.

        Dependencies are injected once at construction time. This method
        must NOT reassign ``self._orchestrator._*`` fields to keep the
        facade in sync with test mocks; tests should mock the orchestrator
        boundary (``engine._orchestrator.answer``) instead.
        """
        request = QueryRequest(
            question=question,
            document_names=tuple(doc_names or ()),
            user_id=user_id,
            conversation_history=tuple(conversation_history or ()),
            memory_profile=memory_profile,
        )
        result = await self._orchestrator.answer(request, n_results=n_results)
        return result.to_legacy_dict()

    def _handle_conversational_query(self, query: str) -> str | None:
        return RAGOrchestrator._handle_conversational_query(query)
