"""Run FinQuery RAG against evaluation cases and persist predictions."""
from __future__ import annotations

import inspect
import json
import time
import urllib.error
import urllib.request
from typing import Any, Iterable

from .evaluation import EvaluationCase, load_jsonl_cases, write_jsonl


EVAL_RUN_N_RESULTS_MIN = 1
EVAL_RUN_N_RESULTS_MAX = 20


async def run_case(
    case: EvaluationCase,
    rag_engine: Any,
    user_id: int,
    n_results: int = 5,
) -> dict[str, Any]:
    """Run one EvaluationCase through a RAGEngine-compatible object."""
    user_id = _validate_user_id(user_id)
    n_results = validate_n_results(n_results)
    start = time.time()
    result = rag_engine.query(
        question=case.question,
        doc_names=list(case.document_names) or None,
        user_id=user_id,
        n_results=n_results,
    )
    if inspect.isawaitable(result):
        result = await result
    latency_ms = (time.time() - start) * 1000

    sources = result.get("sources", []) or []
    prediction = {
        "id": case.case_id,
        "question": case.question,
        "answer": result.get("answer", ""),
        "sources": sources,
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "searched_docs": result.get("searched_docs", []),
        "confidence": result.get("confidence"),
        "context_sufficient": result.get("context_sufficient"),
        "intent": result.get("intent"),
        "intent_confidence": result.get("intent_confidence"),
        "retrieval_debug": result.get("retrieval_debug", {}),
        "trace_id": result.get("trace_id"),
        "latency_ms": latency_ms,
    }
    # Keep the public validation verdict in offline predictions.  This makes
    # safe refusals distinguishable from retrieval/model regressions without
    # exposing trace-only internals or changing score semantics.
    for field in ("answerability", "validation", "repair"):
        if field in result:
            prediction[field] = result.get(field)
    if "rewritten_question" in result:
        prediction["rewritten_question"] = result.get("rewritten_question")
    return prediction


async def run_cases(
    cases: Iterable[EvaluationCase],
    rag_engine: Any,
    user_id: int,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    user_id = _validate_user_id(user_id)
    n_results = validate_n_results(n_results)
    predictions = []
    for case in cases:
        predictions.append(
            await run_case(
                case=case,
                rag_engine=rag_engine,
                user_id=user_id,
                n_results=n_results,
            )
        )
    return predictions


async def run_jsonl_cases(
    cases_path: str,
    output_path: str,
    rag_engine: Any,
    user_id: int,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """Load JSONL cases, run the RAG engine, and write predictions JSONL."""
    user_id = _validate_user_id(user_id)
    n_results = validate_n_results(n_results)
    cases = load_jsonl_cases(cases_path)
    predictions = await run_cases(cases, rag_engine, user_id=user_id, n_results=n_results)
    write_jsonl(output_path, predictions)
    return predictions


def run_http_case(
    case: EvaluationCase,
    api_base: str,
    token: str,
    n_results: int = 5,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Run one EvaluationCase through a running FinQuery HTTP backend."""
    n_results = validate_n_results(n_results)
    api_base = _validate_api_base(api_base)
    token = _validate_token(token)
    timeout = _validate_timeout(timeout)

    payload = {
        "question": case.question,
        "document_names": list(case.document_names) or None,
        "n_results": n_results,
    }
    start = time.time()
    result = _post_json(f"{api_base}/query", payload, token=token, timeout=timeout)
    latency_ms = (time.time() - start) * 1000
    return _prediction_from_result(case, result, latency_ms)


def run_http_cases(
    cases: Iterable[EvaluationCase],
    api_base: str,
    token: str,
    n_results: int = 5,
    timeout: float = 180.0,
) -> list[dict[str, Any]]:
    """Run EvaluationCases through a running FinQuery HTTP backend."""
    n_results = validate_n_results(n_results)
    api_base = _validate_api_base(api_base)
    token = _validate_token(token)
    timeout = _validate_timeout(timeout)
    _preflight_http_auth(api_base, token, timeout)
    predictions = []
    for case in cases:
        predictions.append(
            run_http_case(
                case=case,
                api_base=api_base,
                token=token,
                n_results=n_results,
                timeout=timeout,
            )
        )
    return predictions


def _preflight_http_auth(api_base: str, token: str, timeout: float) -> None:
    """Fail fast when the backend token is invalid.

    Without this preflight, run-http records one HTTP 401 prediction per case and
    the downstream score looks like a retrieval/model regression. Auth failures
    are environment failures, not eval samples.
    """
    result = _get_json(f"{api_base}/me", token=token, timeout=timeout)
    if result.get("error"):
        detail = result.get("detail", "")
        raise ValueError(f"run-http auth preflight failed: {result['error']} {detail}".strip())


def run_jsonl_cases_http(
    cases_path: str,
    output_path: str,
    api_base: str,
    token: str,
    n_results: int = 5,
    timeout: float = 180.0,
) -> list[dict[str, Any]]:
    """Load JSONL cases, run the HTTP backend, and write predictions JSONL."""
    cases = load_jsonl_cases(cases_path)
    predictions = run_http_cases(
        cases,
        api_base=api_base,
        token=token,
        n_results=n_results,
        timeout=timeout,
    )
    write_jsonl(output_path, predictions)
    return predictions


def _post_json(url: str, payload: dict[str, Any], token: str, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {exc.code}", "detail": detail}
    except Exception as exc:  # noqa: BLE001 - eval should persist per-case failures.
        return {"error": type(exc).__name__, "detail": str(exc)}


def _get_json(url: str, token: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {exc.code}", "detail": detail}
    except Exception as exc:  # noqa: BLE001 - eval should fail fast on auth preflight.
        return {"error": type(exc).__name__, "detail": str(exc)}


def _prediction_from_result(
    case: EvaluationCase,
    result: dict[str, Any],
    latency_ms: float,
) -> dict[str, Any]:
    sources = result.get("sources", []) or []
    prediction = {
        "id": case.case_id,
        "question": case.question,
        "answer": result.get("answer", ""),
        "sources": sources,
        "retrieved_chunks": result.get("retrieved_chunks", sources),
        "searched_docs": result.get("searched_docs", []),
        "confidence": result.get("confidence"),
        "context_sufficient": result.get("context_sufficient"),
        "intent": result.get("intent"),
        "intent_confidence": result.get("intent_confidence"),
        "retrieval_debug": result.get("retrieval_debug", {}),
        "trace_id": result.get("trace_id"),
        "latency_ms": latency_ms,
    }
    # Keep the public validation verdict in HTTP-backed predictions too.  The
    # in-process and online runners must expose equivalent diagnostics.
    for field in ("answerability", "validation", "repair"):
        if field in result:
            prediction[field] = result.get(field)
    if "rewritten_question" in result:
        prediction["rewritten_question"] = result.get("rewritten_question")
    if "error" in result:
        prediction["error"] = result.get("error")
        prediction["error_detail"] = result.get("detail", "")
    return prediction


def _validate_api_base(value: str) -> str:
    if not value or not str(value).strip():
        raise ValueError("api_base is required")
    return str(value).strip().rstrip("/")


def _validate_token(value: str) -> str:
    if not value or not str(value).strip():
        raise ValueError("token is required; pass --token or set FINQUERY_TOKEN")
    return str(value).strip()


def _validate_timeout(value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be a number") from exc
    if parsed <= 0:
        raise ValueError("timeout must be > 0")
    return parsed

def validate_n_results(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_results must be an integer") from exc
    if parsed < EVAL_RUN_N_RESULTS_MIN:
        raise ValueError(f"n_results must be >= {EVAL_RUN_N_RESULTS_MIN}")
    if parsed > EVAL_RUN_N_RESULTS_MAX:
        raise ValueError(f"n_results must be <= {EVAL_RUN_N_RESULTS_MAX}")
    return parsed


def _validate_user_id(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("user_id must be an integer") from exc
    if parsed < 1:
        raise ValueError("user_id must be >= 1")
    return parsed
