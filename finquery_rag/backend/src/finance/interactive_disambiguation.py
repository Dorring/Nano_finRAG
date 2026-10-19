"""Active reflective follow-up clarification for ambiguous financial coordinates.

When a query denotes a row label that resolves to multiple conflicting values
in the authoritative store (such as Apple's FY2025 'Deferred' having 3 distinct
values: 604, (1,804), and (139)), this component transforms what would
otherwise be a silent Fail-Closed refusal into an informative, structured,
reflective follow-up prompt with explicit candidate options and page contexts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any


@dataclass(frozen=True)
class DisambiguationOption:
    """One candidate value/row extracted from the conflicting store facts."""

    index: int
    value: str
    page: int | str | None
    context: str
    fact_id: str | None


@dataclass(frozen=True)
class DisambiguationResult:
    """Human-readable prompt and machine-readable options for clarification."""

    prompt: str
    options: list[str]
    structured_options: list[dict[str, Any]]
    entity: str
    metric: str
    period: str
    reason_codes: list[str]

    def to_clarification_dict(self) -> dict[str, Any]:
        return {
            "question": self.prompt,
            "options": list(self.options),
            "reason_codes": list(self.reason_codes),
            "entity": self.entity,
            "metric": self.metric,
            "period": self.period,
            "structured_options": list(self.structured_options),
        }


class InteractiveDisambiguator:
    """Diagnose coordinate-level fact ambiguities and build clarification dialogs."""

    @staticmethod
    def _clean_text(text: Any) -> str:
        if not text:
            return ""
        return " ".join(str(text).split())

    @classmethod
    def extract_options(
        cls,
        conflicting_facts: Sequence[Mapping[str, Any]],
    ) -> list[DisambiguationOption]:
        """Extract unique values and their physical locations from facts."""
        seen_values: set[str] = set()
        options: list[DisambiguationOption] = []

        for idx, fact in enumerate(conflicting_facts, 1):
            val = str(fact.get("value") or fact.get("raw_value") or "").strip()
            if not val or val in seen_values:
                continue
            seen_values.add(val)

            page = fact.get("page") or fact.get("pdf_page")
            context = cls._clean_text(
                fact.get("raw_content") or fact.get("source_text") or fact.get("content") or ""
            )
            # Truncate overly long table rows for readable UX
            if len(context) > 80:
                context = context[:77] + "..."

            options.append(
                DisambiguationOption(
                    index=len(options) + 1,
                    value=val,
                    page=page,
                    context=context,
                    fact_id=str(fact.get("candidate_id") or fact.get("fact_id") or ""),
                )
            )

        return options

    @classmethod
    def build_clarification(
        cls,
        query: str,
        ambiguities: Sequence[Mapping[str, Any]],
    ) -> DisambiguationResult | None:
        """Construct a professional reflective clarification prompt and option list.

        Args:
            query: The user's input query.
            ambiguities: List of detected coordinate conflicts, each containing
                         'entity', 'metric', 'period', and 'conflicting_facts'.
        """
        if not ambiguities:
            return None

        if len(ambiguities) == 1:
            primary = ambiguities[0]
            entity = str(primary.get("entity") or "目标公司")
            metric = str(primary.get("metric") or "所查指标")
            period = str(primary.get("period") or "报告期")
            conflicting_facts = primary.get("conflicting_facts") or ()

            options = cls.extract_options(conflicting_facts)
            if not options:
                return None

            # Build clean option descriptions
            formatted_options: list[str] = []
            structured: list[dict[str, Any]] = []
            for opt in options:
                page_desc = f" (第 {opt.page} 页)" if opt.page is not None else ""
                ctx_desc = f" [上下文: {opt.context}]" if opt.context else ""
                opt_str = f"数值 {opt.value}{page_desc}{ctx_desc}"
                formatted_options.append(opt_str)
                structured.append({
                    "index": opt.index,
                    "value": opt.value,
                    "page": opt.page,
                    "context": opt.context,
                    "fact_id": opt.fact_id,
                    "metric": metric,
                })

            # Compose professional financial clarification prompt
            lines = [
                f"在 {entity} 的 {period} 财报中，会计科目 '{metric}' 检索到多项具有不同财务内涵或处于不同报表位置的披露数据：",
                "",
            ]
            for opt in options:
                page_str = f"第 {opt.page} 页" if opt.page is not None else "未知页码"
                ctx_str = f"；报表行: '{opt.context}'" if opt.context else ""
                lines.append(f"{opt.index}. 金额/数值: {opt.value}（来自 {page_str}{ctx_str}）")

            lines.extend([
                "",
                f"由于上述披露项的会计口径或所处报表/附注位置存在差异，为确保财务数据绝对严谨，请问您具体指的是哪一项？",
                "（您可以直接回复对应序号、具体金额或补充科目范围，系统将据此精准放行）",
            ])

            prompt = "\n".join(lines)

            return DisambiguationResult(
                prompt=prompt,
                options=formatted_options,
                structured_options=structured,
                entity=entity,
                metric=metric,
                period=period,
                reason_codes=["COORDINATE_CONFLICT", "CLARIFICATION_REQUIRED"],
            )

        # Multi-slot cascade: multiple coordinates are simultaneously ambiguous
        entity = str(ambiguities[0].get("entity") or "目标公司")
        period = str(ambiguities[0].get("period") or "报告期")
        metrics_list = [str(a.get("metric") or "所查指标") for a in ambiguities]
        combined_metric = " 与 ".join(metrics_list)

        formatted_options: list[str] = []
        structured: list[dict[str, Any]] = []
        lines = [
            f"在 {entity} 的 {period} 财报中，检测到以下多个会计科目存在重名或多处披露歧义：",
            "",
        ]

        global_idx = 1
        for a_idx, amb in enumerate(ambiguities, 1):
            m_name = str(amb.get("metric") or f"科目{a_idx}")
            facts = amb.get("conflicting_facts") or ()
            slot_options = cls.extract_options(facts)
            if not slot_options:
                continue

            lines.append(f"【科目 {a_idx}: {m_name}】")
            for opt in slot_options:
                page_desc = f" (第 {opt.page} 页)" if opt.page is not None else ""
                ctx_desc = f" [上下文: {opt.context}]" if opt.context else ""
                opt_str = f"[{m_name}] 选项 {global_idx}: 数值 {opt.value}{page_desc}{ctx_desc}"
                formatted_options.append(opt_str)

                page_str = f"第 {opt.page} 页" if opt.page is not None else "未知页码"
                ctx_str = f"；报表行: '{opt.context}'" if opt.context else ""
                lines.append(f"  {global_idx}. 金额/数值: {opt.value}（来自 {page_str}{ctx_str}）")

                structured.append({
                    "index": global_idx,
                    "value": opt.value,
                    "page": opt.page,
                    "context": opt.context,
                    "fact_id": opt.fact_id,
                    "metric": m_name,
                    "slot_id": amb.get("slot_id"),
                })
                global_idx += 1
            lines.append("")

        if not formatted_options:
            return None

        lines.extend([
            f"由于上述各科目在不同报表行或附注中的会计内涵存在差异，请问您分别需要采用哪一项披露？",
            "（您可以回复对应序号或指定各科目目标金额，例如 'Deferred: 604'）",
        ])

        prompt = "\n".join(lines)

        return DisambiguationResult(
            prompt=prompt,
            options=formatted_options,
            structured_options=structured,
            entity=entity,
            metric=combined_metric,
            period=period,
            reason_codes=["COORDINATE_CONFLICT", "CLARIFICATION_REQUIRED"],
        )


__all__ = [
    "DisambiguationOption",
    "DisambiguationResult",
    "InteractiveDisambiguator",
]
