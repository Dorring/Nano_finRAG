import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

from src.evaluation.eval_cli import main as eval_cli_main
from src.services.health import collect_config_snapshot
from src.services.retrieval_config import (
    DEFAULT_EMBEDDING_MODEL,
    build_retrieval_model_config,
    get_embedding_model_name,
)
from src.pdf_retrieval_v4.candidate_view_index import CandidateViewIndexReader


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_embedding_model_name_defaults_and_respects_env(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL_NAME", raising=False)
    assert get_embedding_model_name() == DEFAULT_EMBEDDING_MODEL

    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")
    assert get_embedding_model_name() == "intfloat/multilingual-e5-small"


def test_retrieval_model_config_validates_cross_encoder_local_path(tmp_path, monkeypatch):
    model_dir = tmp_path / "reranker"
    model_dir.mkdir()
    monkeypatch.setenv("RAG_RERANKER", "cross-encoder")
    monkeypatch.setenv("RAG_RERANKER_MODEL", str(model_dir))
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", str(tmp_path / "missing-embedding"))

    cfg = build_retrieval_model_config()

    assert cfg["ok"] is False
    assert cfg["reranker_model_path_exists"] is True
    assert "EMBEDDING_MODEL_NAME points to a missing local path" in cfg["errors"]


def test_health_config_reports_embedding_and_reranker_diagnostics(tmp_path, monkeypatch):
    embedding_dir = tmp_path / "embedding"
    embedding_dir.mkdir()
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", str(embedding_dir))
    monkeypatch.setenv("RAG_RERANKER", "cross-encoder")
    monkeypatch.delenv("RAG_RERANKER_MODEL", raising=False)

    cfg = collect_config_snapshot()

    assert cfg["ok"] is False
    assert cfg["retrieval"]["embedding_model"] == str(embedding_dir)
    assert cfg["retrieval"]["embedding_model_path_exists"] is True
    assert "RAG_RERANKER_MODEL is required when RAG_RERANKER=cross-encoder" in cfg["errors"]


def test_vector_store_uses_configured_embedding_model(monkeypatch):
    mock_embed_fn = MagicMock()
    mock_st_ef = MagicMock()
    mock_st_ef.SentenceTransformerEmbeddingFunction.return_value = mock_embed_fn
    for name in ["chromadb", "chromadb.utils", "chromadb.utils.embedding_functions"]:
        monkeypatch.setitem(sys.modules, name, MagicMock())
    monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", mock_st_ef)
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "local-embedding-model")
    sys.modules.pop("src.services.vector_store", None)
    sys.modules.pop("services.vector_store", None)

    __import__("src.services.vector_store")

    mock_st_ef.SentenceTransformerEmbeddingFunction.assert_called_with(
        model_name="local-embedding-model"
    )


def test_candidate_reader_uses_configured_local_embedding_model(tmp_path, monkeypatch):
    metadata = tmp_path / "candidate-metadata.sqlite"
    with sqlite3.connect(metadata) as conn:
        conn.execute(
            "CREATE TABLE view_metadata ("
            "lane TEXT, view_id TEXT, candidate_key TEXT, view_type TEXT, "
            "retrieval_text TEXT, document_id TEXT, metadata_json TEXT)"
        )

    mock_sentence_transformers = MagicMock()
    mock_sentence_transformers.SentenceTransformer.return_value = MagicMock()
    monkeypatch.setitem(sys.modules, "sentence_transformers", mock_sentence_transformers)
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "/models/all-MiniLM-L6-v2")

    reader = CandidateViewIndexReader(metadata.parent)
    try:
        reader._encoder_instance()
    finally:
        reader.close()

    mock_sentence_transformers.SentenceTransformer.assert_called_once_with(
        "/models/all-MiniLM-L6-v2", device="cpu"
    )


def test_eval_cli_retrieval_eval_bundle_writes_reports(tmp_path, capsys):
    cases = tmp_path / "cases.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    out_dir = tmp_path / "bundle"
    _write_jsonl(cases, [{
        "id": "c1",
        "question": "What was revenue?",
        "expected_answer_contains": ["$10M"],
        "expected_numbers": ["10"],
        "expected_sources": [{"filename": "q3.pdf", "page": 2}],
    }])
    _write_jsonl(predictions, [{
        "id": "c1",
        "answer": "Revenue was $10M.",
        "sources": [{"filename": "q3.pdf", "page": 2}],
        "retrieved_chunks": [{"filename": "q3.pdf", "page": 2}],
    }])

    code = eval_cli_main([
        "retrieval-eval-bundle",
        "--cases",
        str(cases),
        "--predictions",
        str(predictions),
        "--k",
        "1",
        "--out-dir",
        str(out_dir),
    ])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert Path(payload["outputs"]["score"]).exists()
    assert Path(payload["outputs"]["retrieval_diagnostics"]).exists()
    assert Path(payload["outputs"]["interview_report"]).exists()
    assert json.loads((out_dir / "interview_report.json").read_text(encoding="utf-8"))["summary"]["answer_pass_rate"] == 1.0
