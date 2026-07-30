"""
Phase 4 tests — session memory, query rewriting, and session isolation.
All tests are CPU-safe, no GPU, no external LLM.
"""
import os, sys, gc, asyncio, tempfile, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from services.session_manager import SessionManager

def _close_and_gc(mgr):
    try: mgr.close()
    except Exception: pass
    gc.collect()

class MockLLMClient:
    def __init__(self, rewrite_response=None, answer_response='Revenue was 10M.'):
        _rewrite, _answer = rewrite_response, answer_response
        def _create(**kwargs):
            is_rewrite = kwargs.get('max_tokens', 512) == 100
            text = _rewrite if (is_rewrite and _rewrite) else _answer
            class R: choices=[type('C',(),{'message':type('M',(),{'content':text})()})()]
            return R()
        self.chat = type('C',(),{'completions':type('Co',(),{'create':staticmethod(_create)})()})()

class TestSessionManager:
    def test_add_and_retrieve(self, tmp_path):
        db = os.path.join(str(tmp_path), 's.db')
        mgr = SessionManager(db_path=db)
        try:
            mgr.add_message('s1', 1, 'user', 'What is revenue?')
            mgr.add_message('s1', 1, 'assistant', 'Revenue was 10M.')
            msgs = mgr.get_recent_messages('s1', 1)
            assert len(msgs) == 2
            assert msgs[0]['role'] == 'user'
            assert msgs[1]['role'] == 'assistant'
        finally:
            _close_and_gc(mgr)

    def test_chronological_order(self, tmp_path):
        db = os.path.join(str(tmp_path), 's.db')
        mgr = SessionManager(db_path=db)
        try:
            for i in range(6):
                role = 'user' if i % 2 == 0 else 'assistant'
                mgr.add_message('s1', 1, role, 'Msg %d' % i)
            msgs = mgr.get_recent_messages('s1', 1, n_pairs=3)
            assert msgs[0]['content'] == 'Msg 0'
            assert msgs[-1]['content'] == 'Msg 5'
        finally:
            _close_and_gc(mgr)

    def test_max_history_limit(self, tmp_path):
        db = os.path.join(str(tmp_path), 's.db')
        mgr = SessionManager(db_path=db, max_history=4)
        try:
            for i in range(10):
                role = 'user' if i % 2 == 0 else 'assistant'
                mgr.add_message('s1', 1, role, 'Msg %d' % i)
            msgs = mgr.get_recent_messages('s1', 1)
            assert len(msgs) == 8
            assert msgs[0]['content'] == 'Msg 2'
            assert msgs[-1]['content'] == 'Msg 9'
        finally:
            _close_and_gc(mgr)

    def test_clear_session(self, tmp_path):
        db = os.path.join(str(tmp_path), 's.db')
        mgr = SessionManager(db_path=db)
        try:
            mgr.add_message('s1', 1, 'user', 'Hello')
            mgr.add_message('s1', 1, 'assistant', 'Hi')
            assert mgr.get_session_count('s1', 1) == 2
            assert mgr.clear_session('s1', 1) is True
            assert mgr.get_session_count('s1', 1) == 0
            assert mgr.clear_session('s1', 1) is False
        finally:
            _close_and_gc(mgr)

    def test_empty_returns_empty(self, tmp_path):
        db = os.path.join(str(tmp_path), 's.db')
        mgr = SessionManager(db_path=db)
        try:
            assert mgr.get_recent_messages('nonexistent', 1) == []
        finally:
            _close_and_gc(mgr)

class TestSessionTenantIsolation:
    def test_different_users_same_session(self, tmp_path):
        db = os.path.join(str(tmp_path), 's.db')
        mgr = SessionManager(db_path=db)
        try:
            mgr.add_message('s1', 1, 'user', 'U1 question')
            mgr.add_message('s1', 1, 'assistant', 'U1 answer')
            mgr.add_message('s1', 2, 'user', 'U2 question')
            assert len(mgr.get_recent_messages('s1', 1)) == 2
            assert len(mgr.get_recent_messages('s1', 2)) == 1
        finally:
            _close_and_gc(mgr)

    def test_clear_only_target_user(self, tmp_path):
        db = os.path.join(str(tmp_path), 's.db')
        mgr = SessionManager(db_path=db)
        try:
            mgr.add_message('s1', 1, 'user', 'Q1')
            mgr.add_message('s1', 2, 'user', 'Q2')
            mgr.clear_session('s1', 1)
            assert mgr.get_session_count('s1', 1) == 0
            assert mgr.get_session_count('s1', 2) == 1
        finally:
            _close_and_gc(mgr)

    def test_fail_closed_none_user_id(self, tmp_path):
        db = os.path.join(str(tmp_path), 's.db')
        mgr = SessionManager(db_path=db)
        try:
            mgr.add_message('s1', None, 'user', 'no')
            assert mgr.get_session_count('s1', None) == 0
        finally:
            _close_and_gc(mgr)

    def test_fail_closed_empty_session(self, tmp_path):
        db = os.path.join(str(tmp_path), 's.db')
        mgr = SessionManager(db_path=db)
        try:
            mgr.add_message('', 1, 'user', 'no')
            assert mgr.get_session_count('', 1) == 0
        finally:
            _close_and_gc(mgr)

    def test_reject_unknown_role(self, tmp_path):
        db = os.path.join(str(tmp_path), 's.db')
        mgr = SessionManager(db_path=db)
        try:
            mgr.add_message('s1', 1, 'system', 'no')
            assert mgr.get_session_count('s1', 1) == 0
        finally:
            _close_and_gc(mgr)

class TestSessionSchema:
    def test_schema_version(self, tmp_path):
        db = os.path.join(str(tmp_path), 's.db')
        mgr = SessionManager(db_path=db)
        try:
            import sqlite3
            c = sqlite3.connect(db)
            row = c.execute("SELECT version FROM schema_version WHERE component='session_manager'").fetchone()
            c.close()
            assert row is not None
            assert row[0] == SessionManager.SCHEMA_VERSION
        finally:
            _close_and_gc(mgr)

    def test_idempotent_init(self, tmp_path):
        db = os.path.join(str(tmp_path), 's.db')
        mgr = SessionManager(db_path=db)
        try:
            mgr._init_db()
            mgr.add_message('s1', 1, 'user', 'test')
            assert mgr.get_session_count('s1', 1) == 1
        finally:
            _close_and_gc(mgr)

class TestQueryRewriting:
    def test_rewrite_with_history(self, tmp_path):
        from services.rag_engine import RAGEngine
        mc = MockLLMClient(rewrite_response='What was the revenue in Q3 2024?')
        eng = RAGEngine(mc, bm25_db_path=os.path.join(str(tmp_path), 'b.db'))
        history = [
            {'role':'user','content':'Tell me about Q3 report'},
            {'role':'assistant','content':'Q3 shows strong growth.'},
        ]
        loop = asyncio.new_event_loop()
        try:
            r = loop.run_until_complete(eng._rewrite_query_with_context('What about revenue?', history))
            assert r == 'What was the revenue in Q3 2024?'
        finally:
            loop.close()

    def test_no_history_returns_original(self, tmp_path):
        from services.rag_engine import RAGEngine
        mc = MockLLMClient()
        eng = RAGEngine(mc, bm25_db_path=os.path.join(str(tmp_path), 'b.db'))
        loop = asyncio.new_event_loop()
        try:
            r = loop.run_until_complete(eng._rewrite_query_with_context('What is revenue?', []))
            assert r == 'What is revenue?'
        finally:
            loop.close()

    def test_short_history_returns_original(self, tmp_path):
        from services.rag_engine import RAGEngine
        mc = MockLLMClient()
        eng = RAGEngine(mc, bm25_db_path=os.path.join(str(tmp_path), 'b.db'))
        loop = asyncio.new_event_loop()
        try:
            r = loop.run_until_complete(eng._rewrite_query_with_context('What is revenue?', [{'role':'user','content':'Hello'}]))
            assert r == 'What is revenue?'
        finally:
            loop.close()

    def test_failure_falls_back(self, tmp_path):
        from services.rag_engine import RAGEngine
        class F:
            def __init__(s):
                def c(**kw): raise RuntimeError('down')
                s.chat = type('C',(),{'completions':type('Co',(),{'create':staticmethod(c)})()})()
        eng = RAGEngine(F(), bm25_db_path=os.path.join(str(tmp_path), 'f.db'))
        history = [{'role':'user','content':'Hello'},{'role':'assistant','content':'Hi'}]
        loop = asyncio.new_event_loop()
        try:
            r = loop.run_until_complete(eng._rewrite_query_with_context('What is revenue?', history))
            assert r == 'What is revenue?'
        finally:
            loop.close()

class TestQueryReturnsRewrittenQuestion:
    @staticmethod
    def _without_documents(engine):
        """Keep query-rewrite tests independent from vector-store state."""
        engine._orchestrator._list_all_documents = lambda _user_id: []

    def test_with_history_returns_rewritten(self, tmp_path):
        from services.rag_engine import RAGEngine
        mc = MockLLMClient(rewrite_response='What was the revenue in Q3?')
        eng = RAGEngine(mc, use_hybrid=False, bm25_db_path=os.path.join(str(tmp_path), 'b.db'))
        self._without_documents(eng)
        history = [{'role':'user','content':'Tell me about Q3'},{'role':'assistant','content':'Q3 was great'}]
        loop = asyncio.new_event_loop()
        try:
            r = loop.run_until_complete(eng.query('What about revenue?', doc_names=[], user_id=1, conversation_history=history))
            assert 'rewritten_question' in r
        finally:
            loop.close()

    def test_without_history_no_rewritten(self, tmp_path):
        from services.rag_engine import RAGEngine
        mc = MockLLMClient()
        eng = RAGEngine(mc, use_hybrid=False, bm25_db_path=os.path.join(str(tmp_path), 'b.db'))
        self._without_documents(eng)
        loop = asyncio.new_event_loop()
        try:
            r = loop.run_until_complete(eng.query('What is revenue?', doc_names=[], user_id=1))
            assert r.get('rewritten_question') is None
        finally:
            loop.close()

class TestHistoryNotInRetrievalContext:
    def test_build_context_only_takes_chunks(self):
        import ast
        p = os.path.join(os.path.dirname(__file__), '..', 'src', 'services', 'rag_engine.py')
        tree = ast.parse(open(p, encoding='utf-8').read())
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == 'query':
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        f = child.func
                        if isinstance(f, ast.Attribute) and f.attr == 'build_context':
                            assert len(child.args) == 1, 'build_context should only take chunks'
                break

    def test_no_add_documents_in_query(self):
        import ast
        p = os.path.join(os.path.dirname(__file__), '..', 'src', 'services', 'rag_engine.py')
        tree = ast.parse(open(p, encoding='utf-8').read())
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == 'query':
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        f = child.func
                        if isinstance(f, ast.Attribute) and f.attr == 'add_documents':
                            pytest.fail('query() must not call add_documents')
                        if isinstance(f, ast.Name) and f.id == 'add_documents':
                            pytest.fail('query() must not call add_documents')
                break
