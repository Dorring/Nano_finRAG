"""Calibrated Fail-Closed Diagnostic Explanation for Financial RAG.

Transforms raw mathematical and validation block reasons into informative,
professional, and actionable financial explanations for users and investment committees.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


def format_fail_closed_explanation(
    block_reason: str,
    *,
    missing_slots: list[Mapping[str, Any]] | None = None,
    diagnostic_context: Mapping[str, Any] | None = None,
    operation: str | None = None,
) -> str:
    """Generate human-friendly, calibrated financial explanation when execution is BLOCKED.

    Avoids generic or opaque errors like 'RELEASE_VALIDATOR_DECLINED' or 'PRIMITIVE_DECLINED'.
    Provides exact financial root cause and actionable guidance.
    """
    ctx = dict(diagnostic_context or {})
    reason_norm = block_reason.lower()

    # Case 1: Negative Base in CAGR
    if "cagr base value must be positive" in reason_norm or "cagr" in str(operation).lower() and "negative" in reason_norm:
        base_val = ctx.get("base_value", "负数")
        return (
            f"【无法计算复合年增长率 (CAGR)】：基期指标数值为负 ({base_val})。"
            "在严格金融数学与会计准则中，负基数开分数方在实数域无解（无确定实数解）；"
            "建议切换为分析【净利润绝对增减额 (Difference)】或【营业利润趋势】。"
        )

    # Case 2: Zero Denominator in Valuation Multiple / Ratio
    if "denominator must be non-zero" in reason_norm or "division by zero" in reason_norm or "zero" in reason_norm and "multiple" in str(operation).lower():
        denom_metric = ctx.get("denominator_metric", "分母指标")
        return (
            f"【无法计算估值乘数/财务比率】：{denom_metric}数值为 0。"
            "分母为零在算术上无定义；建议核实该报告期指标是否处于盈亏平衡点，或选择市净率 (P/B)、市销率 (P/S) 等替代估值指标。"
        )

    # Case 3: Unit / Currency Mismatch
    if "unit_mismatch" in reason_norm or "currency" in reason_norm or "cross-currency" in reason_norm:
        unit_a = ctx.get("unit_a", "货币A")
        unit_b = ctx.get("unit_b", "货币B")
        return (
            f"【无法直接执行算术运算】：参与运算的操作数存在异构量纲冲突 ({unit_a} 与 {unit_b})。"
            "为确保财务严谨性，禁止将不同币种或不可约减物理单位直接加减；建议引入即期汇率 (FX Rate) 折算后再行计算。"
        )

    # Case 4: Scale Mismatch
    if "scale_mismatch" in reason_norm or "unsupported scale" in reason_norm:
        scale_val = ctx.get("scale", "未知量级")
        return (
            f"【量纲尺度无法对齐】：检测到未经标准化的数值尺度 ({scale_val})。"
            "系统已自动拦截以防止产生百万倍量级计算偏差，建议补充标准量纲单位 (如 million, thousand, units)。"
        )

    # Case 5: Missing or Unadmitted Slots
    if missing_slots:
        slot_names = [f"{s.get('entity', '')} {s.get('metric', '')} {s.get('period', '')}".strip() for s in missing_slots]
        slot_str = "、".join(f"[{s}]" for s in slot_names if s)
        return (
            f"【缺少计算必需的权威事实】：尚未从报告中锁定完整操作数 {slot_str}。"
            "已启动安全失败闭合 (Fail-Closed) 机制以杜绝虚构数值幻觉；建议提供更详尽的财报文档或细化提问科目。"
        )

    # Case 6: Fallback Default
    return (
        f"【计算安全拦截】：{block_reason}。"
        "系统遵循严格 Fail-Closed 安全审计门禁，未生成推测性数值；建议核对指标出处与计算前提。"
    )
