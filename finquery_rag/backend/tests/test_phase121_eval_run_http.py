import json

from src.evaluation import eval_runner
from src.evaluation.evaluation import EvaluationCase
from src.evaluation.eval_cli import main as eval_cli_main


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_run_http_case_posts_query_payload_and_normalizes_prediction(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({
            "answer": "Revenue was $219 million.",
            "sources": [{"filename": "report.pdf", "page": 3}],
            "retrieved_chunks": [
                {
                    "doc_id": "user_1_report.pdf::page_3::chunk_7",
                    "filename": "report.pdf",
                    "page": 3,
                }
            ],
            "retrieval_debug": {"candidate_count": 16, "returned_count": 8},
            "confidence": 0.8,
            "context_sufficient": True,
            "intent": "document_qa",
            "trace_id": "abc123",
            "answerability": {"status": "answerable"},
            "validation": {"status": "passed"},
            "repair": {"fallback_used": False},
        })

    monkeypatch.setattr(eval_runner.urllib.request, "urlopen", fake_urlopen)
    case = EvaluationCase.from_dict({
        "id": "c1",
        "question": "What was revenue?",
        "document_names": ["report.pdf"],
    })

    prediction = eval_runner.run_http_case(
        case,
        api_base="http://127.0.0.1:8000/",
        token="token-123",
        n_results=8,
        timeout=12,
    )

    assert captured["url"] == "http://127.0.0.1:8000/query"
    assert captured["payload"] == {
        "question": "What was revenue?",
        "document_names": ["report.pdf"],
        "n_results": 8,
    }
    assert captured["headers"]["Authorization"] == "Bearer token-123"
    assert captured["timeout"] == 12
    assert prediction["id"] == "c1"
    assert prediction["answer"] == "Revenue was $219 million."
    assert prediction["retrieved_chunks"] == [
        {
            "doc_id": "user_1_report.pdf::page_3::chunk_7",
            "filename": "report.pdf",
            "page": 3,
        }
    ]
    assert prediction["retrieval_debug"] == {"candidate_count": 16, "returned_count": 8}
    assert prediction["trace_id"] == "abc123"
    assert prediction["answerability"] == {"status": "answerable"}
    assert prediction["validation"] == {"status": "passed"}
    assert prediction["repair"] == {"fallback_used": False}


def test_run_jsonl_cases_http_writes_predictions(tmp_path, monkeypatch):
    cases = tmp_path / "cases.jsonl"
    out = tmp_path / "predictions.jsonl"
    _write_jsonl(cases, [{"id": "c1", "question": "Q?"}])

    monkeypatch.setattr(
        eval_runner,
        "_post_json",
        lambda url, payload, token, timeout: {"answer": "A", "sources": []},
    )
    monkeypatch.setattr(
        eval_runner,
        "_get_json",
        lambda url, token, timeout: {"email": "qh@bb.com"},
    )

    predictions = eval_runner.run_jsonl_cases_http(
        str(cases),
        str(out),
        api_base="http://backend",
        token="token",
    )

    assert len(predictions) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["answer"] == "A"


def test_run_jsonl_cases_http_preflights_auth_before_queries(tmp_path, monkeypatch):
    cases = tmp_path / "cases.jsonl"
    out = tmp_path / "predictions.jsonl"
    _write_jsonl(cases, [{"id": "c1", "question": "Q?"}])
    calls = []

    monkeypatch.setattr(
        eval_runner,
        "_get_json",
        lambda url, token, timeout: calls.append(("GET", url)) or {"email": "qh@bb.com"},
    )
    monkeypatch.setattr(
        eval_runner,
        "_post_json",
        lambda url, payload, token, timeout: calls.append(("POST", url)) or {"answer": "A", "sources": []},
    )

    eval_runner.run_jsonl_cases_http(
        str(cases),
        str(out),
        api_base="http://backend",
        token="token",
    )

    assert calls == [
        ("GET", "http://backend/me"),
        ("POST", "http://backend/query"),
    ]


def test_run_jsonl_cases_http_fails_fast_on_invalid_token(tmp_path, monkeypatch):
    cases = tmp_path / "cases.jsonl"
    out = tmp_path / "predictions.jsonl"
    _write_jsonl(cases, [{"id": "c1", "question": "Q?"}])

    monkeypatch.setattr(
        eval_runner,
        "_get_json",
        lambda url, token, timeout: {"error": "HTTP 401", "detail": '{"detail":"Could not validate credentials"}'},
    )
    monkeypatch.setattr(
        eval_runner,
        "_post_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("query should not run")),
    )

    try:
        eval_runner.run_jsonl_cases_http(
            str(cases),
            str(out),
            api_base="http://backend",
            token="expired",
        )
    except ValueError as exc:
        assert "auth preflight failed" in str(exc)
        assert "HTTP 401" in str(exc)
    else:
        raise AssertionError("expected auth preflight failure")
    assert not out.exists()


def test_eval_cli_run_http_requires_token(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("FINQUERY_TOKEN", raising=False)
    cases = tmp_path / "cases.jsonl"
    out = tmp_path / "predictions.jsonl"
    _write_jsonl(cases, [{"id": "c1", "question": "Q?"}])

    code = eval_cli_main([
        "run-http",
        "--cases", str(cases),
        "--out", str(out),
        "--api-base", "http://backend",
    ])

    assert code == 2
    assert "token is required" in capsys.readouterr().err
