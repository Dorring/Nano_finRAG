"""Fine-grained Critic diagnosis and query reformulation for adaptive RAG."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidenceEvaluationV1


class CriticDiagnosisCategory(str, Enum):
    """Categorized root causes for evidence admission failures."""

    QUARTERLY_VS_ANNUAL = "QUARTERLY_VS_ANNUAL"         # Candidate is Q1-Q4 interim, query needs annual 10-K
    SEGMENT_VS_CONSOLIDATED = "SEGMENT_VS_CONSOLIDATED"   # Candidate is segment/product line, query needs company total
    RESTATED_VS_ORIGINAL = "RESTATED_VS_ORIGINAL"       # Comparative restatement column mismatch
    FOOTNOTE_MISSING = "FOOTNOTE_MISSING"               # Metric is in footnote disclosures, not statement face
    METRIC_ALIAS_GAP = "METRIC_ALIAS_GAP"               # Terminology mismatch (e.g. COGS vs Cost of sales)
    UNKNOWN_DEFECT = "UNKNOWN_DEFECT"


@dataclass(frozen=True)
class CriticDiagnosis:
    """Structured diagnostic evaluation for a single failed slot."""

    category: CriticDiagnosisCategory
    target_slot_id: str
    explanation: str
    suggested_query: str
    target_section: str | None = None
    structural_constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "target_slot_id": self.target_slot_id,
            "explanation": self.explanation,
            "suggested_query": self.suggested_query,
            "target_section": self.target_section,
            "structural_constraints": self.structural_constraints,
        }


# Pattern matchers for failure detection
_INTERIM_PATTERNS = re.compile(
    r"\b(three months ended|nine months ended|six months ended|q[1-4]|first quarter|second quarter|third quarter|fourth quarter)\b",
    re.IGNORECASE,
)

_ANNUAL_PATTERNS = re.compile(
    r"\b(twelve months ended|year ended|annual|years ended)\b",
    re.IGNORECASE,
)

_SEGMENT_TERMS = re.compile(
    r"\b(segment|division|product line|geographic|reporting unit|business unit|operating segment|disaggregated|commercial|consumer|enterprise|automotive|cloud|hardware|software|services|advertising|retail|subscription|iphone|mac|ipad|wearables|intelligent cloud|productivity and business|more personal computing)\b",
    re.IGNORECASE,
)

_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "cogs": ("cost of sales", "cost of revenue", "cost of products"),
    "cost of goods sold": ("cost of sales", "cost of revenue", "cost of products"),
    "revenue": ("total net sales", "net sales", "total revenues"),
    "operating expenses": ("total operating expenses", "operating costs"),
    "net income": ("net earnings", "net loss", "net income (loss)"),
    "operating income": ("operating profit", "operating earnings", "income from operations"),
    "gross profit": ("gross margin", "gross income"),
    "ebit": ("operating income", "income from operations"),
    "long-term debt": ("term loans", "senior notes", "non-current portion of long-term debt"),
    "short-term debt": ("commercial paper", "current portion of long-term debt", "notes payable"),
    "cash and cash equivalents": ("cash", "cash and equivalents", "cash and short-term investments"),
    "sg&a": ("selling, general and administrative", "selling and marketing"),
    "r&d": ("research and development",),
    "eps": ("diluted earnings per share", "diluted eps", "basic earnings per share", "basic eps"),
}


class DiagnosticCritic:
    """Evaluates why candidate packets failed to satisfy a slot and plans targeted recovery."""

    @classmethod
    def diagnose_slot(
        cls,
        slot: Mapping[str, Any],
        candidates: Iterable[Mapping[str, Any]],
        base_query: str = "",
    ) -> CriticDiagnosis | None:
        """Inspect candidate packets against a missing or unadmitted slot."""
        slot_id = str(slot.get("slot_id", "unknown"))
        metric = str(slot.get("metric", "")).strip()
        period = str(slot.get("period", "")).strip()
        entity = str(slot.get("entity", "")).strip()
        scope = str(slot.get("scope", "")).strip().lower()

        candidate_list = list(candidates)
        all_text = " ".join(
            f"{c.get('source_text', '')} {c.get('value', '')} {c.get('metric', '')} {c.get('period', '')}"
            for c in candidate_list
        ).lower()

        # Check 1: Quarterly vs Annual Mismatch
        is_annual_target = bool(re.search(r"\b(fy\d{4}|\d{4})\b", period, re.IGNORECASE))
        if is_annual_target and _INTERIM_PATTERNS.search(all_text) and not _ANNUAL_PATTERNS.search(all_text):
            query = f"{entity} {metric} {period} annual consolidated twelve months ended"
            # Target statement section dynamically depending on slot metric line item
            m_lower = metric.lower()
            if any(k in m_lower for k in ("asset", "liabilit", "equity", "debt", "cash and cash", "marketable securities", "payable", "receivable", "inventory")):
                target_section = "Item 8. Consolidated Balance Sheets"
            elif any(k in m_lower for k in ("cash flow", "operating activities", "investing activities", "financing activities")):
                target_section = "Item 8. Consolidated Statements of Cash Flows"
            else:
                target_section = "Item 8. Consolidated Statements of Operations"

            return CriticDiagnosis(
                category=CriticDiagnosisCategory.QUARTERLY_VS_ANNUAL,
                target_slot_id=slot_id,
                explanation=f"Retrieved candidate packets contain interim/quarterly figures rather than full-year {period} audited total",
                suggested_query=query.strip(),
                target_section=target_section,
                structural_constraints={"period_semantics": "annual", "require_annual_table": True},
            )

        # Check 2: Segment vs Consolidated Total
        is_consolidated_target = scope in ("", "consolidated", "total")
        if is_consolidated_target and _SEGMENT_TERMS.search(all_text) and not ("consolidated" in all_text or "total" in all_text):
            query = f"{entity} total consolidated {metric} {period} consolidated statements"
            return CriticDiagnosis(
                category=CriticDiagnosisCategory.SEGMENT_VS_CONSOLIDATED,
                target_slot_id=slot_id,
                explanation=f"Retrieved candidate packets describe disaggregated segment/division numbers instead of consolidated {metric}",
                suggested_query=query.strip(),
                target_section="Item 8. Consolidated Financial Statements",
                structural_constraints={"scope": "consolidated", "exclude_segments": True},
            )

        # Check 3: Footnote vs Primary Statement Face
        footnote_indicators = ("restricted cash", "debt maturity", "lease liability", "commitments", "contingencies", "share-based", "segment revenue", "fair value", "income taxes")
        if any(term in metric.lower() for term in footnote_indicators):
            query = f"{entity} {period} Note disclosures {metric}"
            return CriticDiagnosis(
                category=CriticDiagnosisCategory.FOOTNOTE_MISSING,
                target_slot_id=slot_id,
                explanation=f"Metric '{metric}' is typically reported in footnote disclosures rather than the primary statement face",
                suggested_query=query.strip(),
                target_section="Item 8. Note Disclosures",
                structural_constraints={"target_disclosure_type": "footnote"},
            )

        # Check 4: Metric Alias Gap
        metric_lower = metric.lower()
        matched_aliases = None
        if metric_lower in _ALIAS_MAP:
            matched_aliases = _ALIAS_MAP[metric_lower]
        else:
            for k, v in _ALIAS_MAP.items():
                if k in metric_lower:
                    matched_aliases = v
                    break

        if matched_aliases is not None:
            query = f"{entity} {period} {' OR '.join(matched_aliases)}"
            return CriticDiagnosis(
                category=CriticDiagnosisCategory.METRIC_ALIAS_GAP,
                target_slot_id=slot_id,
                explanation=f"Metric '{metric}' has common accounting aliases ({', '.join(matched_aliases)})",
                suggested_query=query.strip(),
                target_section=None,
                structural_constraints={"aliases": list(matched_aliases)},
            )

        # Default fallback diagnosis if candidates exist but didn't match
        if candidate_list:
            query = f"{entity} {metric} {period} official 10-K filing"
            return CriticDiagnosis(
                category=CriticDiagnosisCategory.UNKNOWN_DEFECT,
                target_slot_id=slot_id,
                explanation=f"Evidence packets retrieved for slot '{slot_id}' lacked exact coordinate alignment",
                suggested_query=query.strip(),
                target_section=None,
                structural_constraints={"strict_coordinate": True},
            )

        return None

    @classmethod
    def diagnose(
        cls,
        state: AdaptiveRAGStateV1,
        evaluation: EvidenceEvaluationV1,
    ) -> list[CriticDiagnosis]:
        """Diagnose all missing slots in an evaluation."""
        diagnoses: list[CriticDiagnosis] = []
        slot_map = {str(s.get("slot_id")): s for s in state.required_slots}
        candidates = state.evidence_packets or []

        for slot_id in evaluation.missing_slots:
            slot = slot_map.get(str(slot_id))
            if slot is None:
                continue
            diag = cls.diagnose_slot(slot, candidates, base_query=state.normalized_query)
            if diag is not None:
                diagnoses.append(diag)

        return diagnoses
