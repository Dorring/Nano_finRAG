"""Backend stream error event regression checks."""
import json
from pathlib import Path

from src.services.streaming import make_stream_error_event


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _decode_sse_data(event: str) -> dict:
    assert event.startswith("data: ")
    assert event.endswith("\n\n")
    return json.loads(event.removeprefix("data: ").strip())


def test_stream_error_event_uses_api_error_envelope_shape():
    event = make_stream_error_event(
        "stream_error",
        "Streaming query failed. Please retry.",
        retryable=True,
        trace_id="trace-1",
    )
    payload = _decode_sse_data(event)

    assert payload["type"] == "error"
    assert payload["detail"] == {
        "error_code": "stream_error",
        "message": "Streaming query failed. Please retry.",
    }
    assert payload["message"] == "Streaming query failed. Please retry."
    assert payload["retryable"] is True
    assert payload["trace_id"] == "trace-1"


def test_query_stream_generator_has_error_event_fallback():
    content = (SRC / "main.py").read_text(encoding="utf-8")

    assert "make_stream_error_event" in content
    assert "except Exception as exc:" in content
    assert "safe_log_query_trace(engine, trace_payload)" in content
    assert "yield make_stream_error_event(" in content
    assert "trace_id=trace_id" in content
    # Phase 4 hotfix: diagnostics spans multiple lines with error_code and exception_type.
    assert '"stream_error": True' in content
    assert '"error_code": "STREAM_INTERNAL_ERROR"' in content
