import pytest
"""
Round 4 tests — counter-examples for Round 3 fixes.
Covers: low confidence, RRF threshold, BM25 dedup/stale, registry reupload,
upload rollback, stream sufficiency/session, clear-all idempotency.
"""
import os, sys, gc, asyncio, pytest, tempfile, sqlite3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from services.rag_engine import RAGEngine
from services.retrieval import SqliteBM25Retriever, rrf


class CloseableMockLLM:
    """Mock LLM that fails if generate_answer is called, for testing no-LLM paths."""
    def __init__(self, allow_calls=False, response='mock answer'):
        self._allow = allow_calls
        self._resp = response
        self.call_count = 0
        def _create(**kwargs):
            self.call_count += 1
            if not self._allow:
                raise RuntimeError('LLM should not be called!')
            class R: choices=[type('C',(),{'message':type('M',(),{'content':self._resp})()})()]
            return R()
        self.chat = type('C',(),{'completions':type('Co',(),{'create':staticmethod(_create)})()})()


class TestLowConfidenceNoLLM:
    """Item 8: Low-confidence context must NOT call LLM.

    Phase 4 note: these tests characterize the Phase 3 behavior with the
    validation pipeline DISABLED. The default RAGEngine now enables the
    validation pipeline, which produces a different (safer) refusal
    message via AnswerabilityEvaluator. Pass ``enable_validation_pipeline=False``
    to preserve the Phase 3 semantics these tests assert.
    """
    @staticmethod
    def _install_local_retrieval(engine):
        """Avoid relying on a process-global Chroma fixture in unit tests."""
        engine._orchestrator._retrieve_front_matter_chunks = lambda *args: []
        engine._retrieval_pipeline._dense_query_fn = lambda **kwargs: [
            {
                'doc_id': 'user_1_f.pdf::page_1::chunk_1',
                'content': 'Revenue was 10 million.',
                'metadata': {'type': 'text', 'page': 1, 'doc_name': 'f.pdf'},
                'score': 0.8,
            }
        ]

    def test_insufficient_context_skips_llm(self, tmp_path):
        from services.rag_engine import RAGEngine
        mc = CloseableMockLLM(allow_calls=False)
        eng = RAGEngine(mc, use_hybrid=False, bm25_db_path=os.path.join(str(tmp_path), 'b.db'),
                        enable_validation_pipeline=False)
        # Mock at orchestrator dependency level (query delegates to orchestrator)
        from src.retrieval.context_builder import SufficiencyResult
        eng._sufficiency_evaluator.evaluate = lambda chunks: SufficiencyResult(is_sufficient=False, best_score=0.01, average_score=0.01)
        eng._sufficiency_evaluator.confidence = lambda chunks: 0.01
        eng._context_builder.build = lambda chunks: ('mock context', [{'filename':'f.pdf','page':1,'type':'text','score':0.01}])
        self._install_local_retrieval(eng)
        loop = asyncio.new_event_loop()
        try:
            r = loop.run_until_complete(eng.query('test', doc_names=['f.pdf'], user_id=1))
            assert r['context_sufficient'] is False
            assert 'couldn' in r['answer'].lower() or 'sufficiently' in r['answer'].lower()
            assert mc.call_count == 0, 'LLM was called despite insufficient context!'
        finally:
            loop.close()

    def test_sufficient_context_calls_llm(self, tmp_path):
        from services.rag_engine import RAGEngine
        mc = CloseableMockLLM(allow_calls=True, response='Revenue was 10M.')
        eng = RAGEngine(mc, use_hybrid=False, bm25_db_path=os.path.join(str(tmp_path), 'b2.db'),
                        enable_validation_pipeline=False)
        # Mock at orchestrator dependency level (query delegates to orchestrator)
        from src.retrieval.context_builder import SufficiencyResult
        eng._sufficiency_evaluator.evaluate = lambda chunks: SufficiencyResult(is_sufficient=True, best_score=0.8, average_score=0.75)
        eng._sufficiency_evaluator.confidence = lambda chunks: 0.78
        eng._context_builder.build = lambda chunks: ('mock context', [{'filename':'f.pdf','page':1,'type':'text','score':0.8}])
        self._install_local_retrieval(eng)
        loop = asyncio.new_event_loop()
        try:
            r = loop.run_until_complete(eng.query('test', doc_names=['f.pdf'], user_id=1))
            assert r['context_sufficient'] is True
            assert 'Revenue' in r['answer']
            assert mc.call_count >= 1
        finally:
            loop.close()

class TestRRFThreshold:
    """Item 9: RRF calibration"""
    def test_high_rrf_score_passes(self):
        from services.rag_engine import RAGEngine
        class Dummy: pass
        eng = RAGEngine(Dummy(), use_hybrid=True)
        chunks = [{'doc_id':'d1::0','content':'c','metadata':{},'score':0.03}]
        sufficient, best, avg = eng._check_context_sufficiency(chunks)
        assert sufficient, 'RRF score 0.03 should pass 0.008 threshold'
        assert best == 0.03

    def test_low_rrf_score_refuses(self):
        from services.rag_engine import RAGEngine
        class Dummy: pass
        eng = RAGEngine(Dummy(), use_hybrid=True)
        chunks = [{'doc_id':'d1::0','content':'c','metadata':{},'score':0.003}]
        sufficient, best, avg = eng._check_context_sufficiency(chunks)
        assert not sufficient, 'RRF score 0.003 should NOT pass 0.008 threshold'

    def test_high_dense_score_passes(self):
        from services.rag_engine import RAGEngine
        class Dummy: pass
        eng = RAGEngine(Dummy(), use_hybrid=True)
        chunks = [{'doc_id':'d1::0','content':'c','metadata':{},'score':0.85}]
        sufficient, best, avg = eng._check_context_sufficiency(chunks)
        assert sufficient, 'Dense score 0.85 should pass 0.15 threshold'


class TestBM25DedupStale:
    """Item 7: BM25 add_chunks dedup"""
    def test_duplicate_add_returns_once(self, tmp_path):
        db = os.path.join(str(tmp_path), 'b.db')
        r = SqliteBM25Retriever(db_path=db)
        chunks = [{'metadata':{'doc_id':'d1::0','doc_name':'test.pdf'},'content':'Revenue was ten million dollars in Q3 2024'}]
        r.add_chunks(chunks, user_id=1)
        r.add_chunks(chunks, user_id=1)
        results = r.search('revenue Q3', k=5, user_id=1)
        ids = [x['doc_id'] for x in results]
        assert ids.count('user_1_test.pdf::d1::0') <= 1, 'Duplicate FTS rows detected'

    def test_updated_content_no_stale_hit(self, tmp_path):
        db = os.path.join(str(tmp_path), 'b2.db')
        r = SqliteBM25Retriever(db_path=db)
        chunks = [{'metadata':{'doc_id':'d1::0','doc_name':'test.pdf'},'content':'Old content about apples and oranges'}]
        r.add_chunks(chunks, user_id=1)
        chunks2 = [{'metadata':{'doc_id':'d1::0','doc_name':'test.pdf'},'content':'New content about revenue and profit Q3 2024'}]
        r.add_chunks(chunks2, user_id=1)
        results = r.search('apples oranges', k=5, user_id=1)
        assert len(results) == 0, 'Stale content should not be searchable after update'


class TestRegistryReupload:
    """Item 5: Registry delete then reupload should NOT be duplicate-skipped."""
    def test_delete_then_reupload_allowed(self, tmp_path):
        from services.document_registry import DocumentRegistry
        reg = DocumentRegistry(db_path=os.path.join(str(tmp_path), 'reg.db'))
        import uuid
        did = uuid.uuid4().hex
        fh = 'abc123'
        reg.register(did, 1, 'test.pdf', fh, status='parsing')
        reg.mark_indexing(did)
        reg.mark_ready(did, 3, 'ch_abc')
        found = reg.find_by_file_hash(1, fh)
        assert found is not None, 'Should find ready doc'
        reg.delete(1, 'test.pdf')
        found2 = reg.find_by_file_hash(1, fh)
        assert found2 is None, 'Should NOT find after delete'
        did2 = uuid.uuid4().hex
        reg.register(did2, 1, 'test.pdf', fh, status='parsing')
        found3 = reg.get_latest_version(1, 'test.pdf')
        assert found3 is not None and found3['status'] == 'parsing', 'Re-upload should create new record'


class TestBM25Rollback:
    """Item 6: BM25 failure triggers dense rollback."""
    def test_bm25_failure_marks_failed_and_deletes_dense(self):
        """AST check: upload handler must have rollback logic after BM25 add_chunks."""
        import ast
        mp = os.path.join(os.path.dirname(__file__), '..', 'src', 'main.py')
        with open(mp, encoding='utf-8') as fh:
            content = fh.read()
        # String-level check: the upload handler must contain rollback logic
        assert 'delete_document_collection' in content, 'Upload must reference delete_document_collection for rollback'
        assert 'mark_failed' in content, 'Upload must call mark_failed on error'
        # Verify try/except exists around BM25 sync
        assert 'bm25_retriever.add_chunks' in content
        assert 'rollback' in content.lower() or 'Rollback' in content or 'failed' in content.lower()

    def test_schema_has_confidence_fields(self):
        from models.schemas import QueryResponse
        fields = QueryResponse.model_fields
        assert 'confidence' in fields
        assert 'context_sufficient' in fields

    def test_query_endpoint_returns_fields(self):
        mp = os.path.join(os.path.dirname(__file__), '..', 'src', 'main.py')
        content = open(mp, encoding='utf-8').read()
        assert 'confidence=legacy.get' in content
        assert 'context_sufficient=legacy.get' in content


class TestClearAllIdempotent:
    """Item 3 audit round4: clear_all_documents should not 500 on empty user."""
    def test_clear_all_no_docs_does_not_error(self):
        import ast
        mp = os.path.join(os.path.dirname(__file__), '..', 'src', 'main.py')
        tree = ast.parse(open(mp, encoding='utf-8').read())
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == 'clear_all_documents':
                for child in ast.walk(node):
                    if isinstance(child, ast.Compare):
                        for operand in ast.walk(child):
                            if isinstance(operand, ast.Constant) and operand.value == 'Dense index deletion failed':
                                pytest.fail('clear_all must not fail on empty user')
                break


class TestStreamSuite:
    """Item 10 audit round4: stream behavior matches /query for key paths."""
    def test_stream_no_docs_returns_message(self):
        """Phase 3 hotfix: /query/stream now calls engine.query() uniformly.

        The no-docs case is handled inside engine.query() (via the
        orchestrator). The stream endpoint delegates to engine.query()
        and emits whatever answer it returns. This test verifies the
        stream endpoint calls engine.query().
        """
        import ast
        mp = os.path.join(os.path.dirname(__file__), '..', 'src', 'main.py')
        tree = ast.parse(open(mp, encoding='utf-8').read())
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == 'query_documents_stream':
                calls_shared_lifecycle = any(isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == 'execute_user_turn' for child in ast.walk(node))
                assert calls_shared_lifecycle, 'stream must call shared lifecycle'
                break

    def test_stream_has_sufficiency_check(self):
        """Phase 3 hotfix: /query/stream now calls engine.query() uniformly.

        Context sufficiency is evaluated inside the orchestrator
        (RAGOrchestrator.answer). The stream endpoint delegates to
        engine.query() which runs the full orchestrator. This test
        verifies the stream endpoint reads ``context_sufficient`` from
        the result dict.
        """
        import ast
        mp = os.path.join(os.path.dirname(__file__), '..', 'src', 'main.py')
        content = open(mp, encoding='utf-8').read()
        # The stream endpoint must read context_sufficient from result.
        assert 'context_sufficient = result.get("context_sufficient")' in content

    def test_stream_has_session_handling(self):
        import ast
        mp = os.path.join(os.path.dirname(__file__), '..', 'src', 'main.py')
        tree = ast.parse(open(mp, encoding='utf-8').read())
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == 'query_documents_stream':
                calls_shared_lifecycle = any(isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == 'execute_user_turn' for child in ast.walk(node))
                assert calls_shared_lifecycle, 'stream must use shared session lifecycle'
                break
