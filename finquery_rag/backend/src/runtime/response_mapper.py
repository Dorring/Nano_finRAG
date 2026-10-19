"""Compatibility mapping from I3 runtime results to the legacy API dict."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping

from .runtime_contract import (
    FinancialQueryResult,
    RuntimeStatus,
)


BENCHMARK_ENTITIES = {
    "apple", "aapl", "microsoft", "msft", "nvidia", "nvda", "tesla", "tsla",
    "visa", "v", "pfizer", "pfe", "coca-cola", "ko", "jpmorgan", "jpm",
    "amazon", "amzn", "google", "googl", "goog", "meta", "alphabet",
}


def _build_fail_closed_guidance(result: FinancialQueryResult) -> str:
    """Generate structured, calibrated Chinese guidance when execution fails closed."""
    trace = result.debug_metadata.get("trace") or {}
    plan = result.debug_metadata.get("plan") or {}
    query = str(result.debug_metadata.get("query") or "")

    entity = ""
    metric = ""
    period = ""

    slots = plan.get("required_slots") or []
    if slots and isinstance(slots[0], Mapping):
        first_slot = slots[0]
        entity = str(first_slot.get("entity") or "")
        metric = str(first_slot.get("metric") or "")
        period = str(first_slot.get("period") or "")

    if not entity and query:
        m_entity = re.search(r"\b(BYD|NIO|XPEV|BABA|JD|Tencent)\b", query, re.I)
        if m_entity:
            entity = m_entity.group(1)
        else:
            m_cn = re.search(r"(比亚迪|宁德时代|腾讯|阿里巴巴|京东|美团|百度)", query)
            if m_cn:
                entity = m_cn.group(1)

    reason_codes = list(result.reason_codes or trace.get("reason_codes") or ["FAIL_CLOSED"])
    reason_str = ", ".join(reason_codes)

    is_outside_benchmark = False
    norm_entity = entity.strip().lower()
    if norm_entity:
        is_outside_benchmark = norm_entity not in BENCHMARK_ENTITIES
    elif query:
        query_lower = query.lower()
        if any(term in query_lower for term in ["byd", "比亚迪", "宁德时代", "tencent", "腾讯", "baba", "阿里"]):
            is_outside_benchmark = True
            if not entity:
                entity = "比亚迪 (BYD)" if ("byd" in query_lower or "比亚迪" in query_lower) else "目标非美股标杆主体"

    target_desc = f"{entity} {period} {metric}".strip()
    if not target_desc:
        target_desc = "所查询的财务标的及科目"

    lines = [
        "### 🛡️ 【Nano-FinRAG 形式化审计门禁拒答】",
        "",
        "> 依据严格的金融审计安全规范（Fail-Closed Invariant），为防止产生推测性数值或会计幻觉，系统已安全拦截未获权威底稿核验证明的回答生成。",
        "",
        "**📌 审计拦截诊断**：",
        f"- **检索标的**：`{target_desc}`",
    ]

    if is_outside_benchmark:
        lines.extend([
            f"- **拦截原因诊断**：目标主体 `{entity or '该企业'}` 不在系统当前预载的 **SEC 10-K 权威年报基准库**（覆盖 Apple、Microsoft、NVIDIA、Tesla、Visa、Coca-Cola 等美股标杆），且未在当前会话的私有就绪文档中找到匹配的权威审计底稿。",
            "",
            "**💡 解决方案与后续指引**：",
            "1. **上传私有财报**：请点击左上角「上传财报文档」上传该公司（如比亚迪年报/半年报、A股/港股披露文件）PDF，系统将自动进行物理切片并提取表格单元格与坐标并纳入索引。",
            "2. **调用行情接口**：如需获取该公司的实时股票行情、交易量或市盈率估值，建议通过行情系统/Market API 获取最新即时行情。",
            "3. **基准库查询**：如需体验权威年报检索与多步复杂财务计算（如复合年均增长率 CAGR、多公司息税前利润差额对比等），欢迎查询 Apple、NVIDIA、Microsoft、Tesla 等预载标杆财报。",
        ])
    else:
        lines.extend([
            f"- **拦截原因诊断**：已锁定公司底稿，但在指定报告期（`{period or '目标年份'}`）内未定位到经过严格形式化审计核验的 `{metric or '目标财务指标'}` 确切数值（拦截状态码：`{reason_str}`）。",
            "",
            "**💡 解决方案与后续指引**：",
            "1. **核对指标披露科目**：SEC 10-K 年报中科目名称可能为特定会计口径，建议尝试标准科目词，如 `Total net sales`（总净销售额）、`Operating income`（营业利润）、`Cash and cash equivalents`（现金及现金等价物）。",
            "2. **检查报告期范围**：请核实所查指标是否仅在附注章节披露，或尝试指定特定财年（如 FY2024 / FY2025）。",
            "3. **上传补充材料**：若该科目位于非核心财务报表（如特定附注附表、8-K 补充公告），可上传对应补充披露 PDF 重新发起审计查询。",
        ])

    return "\n".join(lines)


class LegacyResponseMappingError(RuntimeError):
    """Raised when a runtime result cannot preserve the legacy API shape."""


def _structured_trace_id(result: FinancialQueryResult) -> str | None:
    """Return a trace identifier from structured runtime metadata only.

    V1 carries its trace id in the historical compatibility payload.  V2 does
    not construct the legacy RAG engine, so its bounded coordinator exposes a
    stable ``execution_id`` inside the sanitized structured trace instead.
    Keeping this fallback here preserves the public observability contract
    without parsing answer text or forcing V2 to instantiate V1 dependencies.
    """

    candidates: list[object] = [result.debug_metadata.get("trace_id")]
    trace = result.debug_metadata.get("trace")
    if isinstance(trace, Mapping):
        candidates.append(trace.get("execution_id"))
    runtime_metadata = result.runtime_metadata
    if runtime_metadata is not None:
        candidates.append(runtime_metadata.attributes.get("execution_id"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def to_legacy_query_dict(result: FinancialQueryResult) -> dict:
    """Return the exact public V1 payload expected by the existing endpoint.

    The adapter stores non-answer/source fields in a small compatibility
    payload containing fields that already existed in the V1 response. This
    function restores that payload and replaces only the answer and sources
    with their typed contract equivalents. Runtime status and release status
    remain internal in I3; the public QueryResponse schema is unchanged.
    """

    if not isinstance(result, FinancialQueryResult):
        raise TypeError("result must be FinancialQueryResult")
    if result.status is RuntimeStatus.ERROR:
        if result.runtime_version.value == "V2":
            err_msg = result.answer or (
                "Upstream planning service unavailable (e.g. LLM API balance exhausted or provider error). Safe-refusal engaged."
            )
            return {
                "answer": err_msg,
                "sources": [],
                "searched_docs": [],
                "retrieved_chunks": [],
                "retrieval_debug": {},
                "calculations": [],
                "status": "FAIL_CLOSED",
            }
        raise LegacyResponseMappingError(
            "Legacy V1 runtime execution failed before a response was produced.",
        )
    v2_fail_closed = (
        result.runtime_version.value == "V2"
        and result.status is RuntimeStatus.FAIL_CLOSED
    )
    if result.answer is None and not v2_fail_closed:
        raise LegacyResponseMappingError(
            "Financial runtime returned no answer.",
        )

    legacy_payload = result.debug_metadata.get("legacy_response")
    if not isinstance(legacy_payload, Mapping):
        if result.runtime_version.value != "V2":
            raise LegacyResponseMappingError(
                "Legacy V1 response compatibility payload is missing.",
            )
        # V2 owns a structured runtime contract rather than the historical
        # RAGEngine payload. Keep the public transport shape stable without
        # inferring provenance from answer text.
        fallback_answer = result.answer
        if not fallback_answer and v2_fail_closed:
            fallback_answer = _build_fail_closed_guidance(result)
        if not fallback_answer:
            raise LegacyResponseMappingError(
                "Trusted V2 runtime returned no user-visible response.",
            )
        legacy_payload = {
            "answer": fallback_answer,
            "sources": copy.deepcopy(result.citations),
            "searched_docs": [],
            "retrieved_chunks": [],
            "retrieval_debug": {},
            "calculations": copy.deepcopy(result.calculations),
        }
    payload = copy.deepcopy(dict(legacy_payload))
    if "searched_docs" not in payload:
        raise LegacyResponseMappingError(
            "Legacy V1 response is missing searched_docs.",
        )
    payload["answer"] = (
        result.answer
        if result.answer is not None
        else (
            _build_fail_closed_guidance(result)
            if v2_fail_closed
            else None
        )
    )
    payload["sources"] = copy.deepcopy(result.citations)
    if result.runtime_version.value == "V2":
        payload["calculations"] = copy.deepcopy(result.calculations)
        clarification = None
        if result.runtime_metadata and isinstance(result.runtime_metadata.attributes, Mapping):
            clarification = result.runtime_metadata.attributes.get("clarification")
        if clarification:
            payload["clarification"] = copy.deepcopy(clarification)
            payload["status"] = "CLARIFICATION_REQUIRED"
        if result.reason_codes:
            payload["reason_codes"] = list(result.reason_codes)
    if result.runtime_version.value == "V2" and not payload.get("trace_id"):
        trace_id = _structured_trace_id(result)
        if trace_id is not None:
            payload["trace_id"] = trace_id
    return payload
