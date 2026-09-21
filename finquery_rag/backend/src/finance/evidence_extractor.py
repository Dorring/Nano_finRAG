"""Evidence extractor: find numeric operands in retrieved evidence.

Given a tuple of ``EvidenceItem`` and a ``RoutingDecision`` (which carries
the expected operand roles), the extractor searches evidence content for
keyword matches and extracts the nearest parseable number for each role.

The extractor is deliberately *best-effort*: if a role cannot be found or
is ambiguous, the caller (plan builder) will create a BLOCKED plan and the
orchestrator will fall back to the LLM. This keeps the pipeline safe — no
guesswork, just deterministic extraction with explicit blocking.

Extraction strategy (per operand role):
1. Look up the role's keyword set (EN + ZH) in ``ROLE_KEYWORDS``.
2. Search each ``EvidenceItem.content`` for the keywords (case-insensitive
   for EN, substring for ZH).
3. When a keyword is found, extract the sentence containing it.
4. Parse numbers from that sentence using ``parse_financial_number``.
5. Return the first parseable number as a ``CalculationOperand`` with
   ``source_text`` and ``evidence_chunk_id`` bound.

Layer dependency: ``domain -> finance -> application -> services``. This
module imports from ``src.domain`` and ``src.finance`` (both allowed) and
stdlib only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from src.domain.calculation import CalculationOperand
from src.domain.evidence import EvidenceItem
from src.finance.operation_router import RoutingDecision
from src.domain.calculation import CalculationOperation
from src.finance.primitive_tools import _infer_scale, parse_financial_number


# Keyword mapping for each operand role. Keywords are checked in order;
# the first match wins. EN keywords are matched case-insensitively.
ROLE_KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "revenue": {
        "en": ("total revenue", "net revenue", "revenue", "sales", "turnover"),
        "zh": ("营业收入", "营收", "收入"),
    },
    "cogs": {
        "en": ("cost of goods sold", "cost of revenue", "cogs", "cost of sales"),
        "zh": ("营业成本", "销售成本"),
    },
    "net_income": {
        "en": ("net income", "net profit", "net earnings", "earnings"),
        "zh": ("净利润", "净利"),
    },
    "total_liabilities": {
        "en": ("total liabilities", "liabilities"),
        "zh": ("总负债", "负债合计", "负债总计"),
    },
    "total_assets": {
        "en": ("total assets", "total asset"),
        "zh": ("总资产", "资产合计", "资产总计"),
    },
    "current": {
        "en": ("current", "fy2025", "2025", "this year", "this period"),
        "zh": ("本期", "当期", "本年"),
    },
    "previous": {
        "en": ("previous", "prior", "fy2024", "2024", "last year", "prior year"),
        "zh": ("上期", "去年同期", "上年"),
    },
    "part": {
        "en": ("part", "segment", "portion", "division"),
        "zh": ("部分", "分部", "组成部分"),
    },
    "total": {
        "en": ("total", "overall", "grand total", "all segments"),
        "zh": ("总计", "合计", "总额"),
    },
}


# Regex to find numeric tokens (with optional commas, percent, parentheses)
# within a sentence. This is intentionally greedy on the number part but
# stops at non-numeric suffixes.
_NUMBER_TOKEN_RE = re.compile(
    r"(?P<negative>\()?"
    r"(?P<number>[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)"
    r"(?P<percent>%)?"
    r"(?P<close>\))?"
)

# Sentence splitter: splits on periods, newlines, semicolons. Keeps the
# delimiter attached to the preceding sentence for context.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;\n])\s+")


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of extracting operands from evidence.

    - ``operands``: the operands that were successfully extracted, in the
      order of ``expected_roles``.
    - ``found_roles``: the subset of ``expected_roles`` that were found.
    - ``missing_roles``: the roles that could not be found.
    - ``warnings``: non-fatal ambiguity warnings (e.g. multiple candidates).
    """

    operands: tuple[CalculationOperand, ...]
    found_roles: tuple[str, ...]
    missing_roles: tuple[str, ...]
    warnings: tuple[str, ...]
    expected_roles: tuple[str, ...]
    source_scale: str | None = None

    @property
    def all_found(self) -> bool:
        return len(self.missing_roles) == 0


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on periods, semicolons, and newlines."""
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _extract_numbers_from_sentence(sentence: str) -> list[tuple[Decimal, str]]:
    """Extract all parseable numbers from a sentence.

    Returns a list of ``(value, raw_text)`` tuples in order of appearance.
    Each ``raw_text`` is the substring that was parsed (for source_text
    binding).
    """
    results: list[tuple[Decimal, str]] = []
    for match in _NUMBER_TOKEN_RE.finditer(sentence):
        # Skip numbers that are part of identifiers like FY2025, Q1, v2
        start = match.start()
        if (
            start > 0
            and sentence[start - 1].isascii()
            and sentence[start - 1].isalpha()
        ):
            continue
        raw = match.group(0)
        # Expand the raw text to include trailing scale words
        end = match.end()
        tail = sentence[end : end + 20].strip()
        scale_word = _extract_scale_word(tail)
        if scale_word:
            raw = sentence[match.start() : end + len(scale_word)]
        parsed = parse_financial_number(raw)
        # The number as the sentence states it.  A filing that writes `5.49 %`
        # in a rate row is stating 5.49, and that number is what binds to the
        # fact; the ratio form is a reading taken later, by whoever needs a
        # multiplicative factor.  This used to come out 100x apart depending on
        # whether the sign was written against the number or after a space.
        if parsed.ok and parsed.points_value is not None:
            results.append((parsed.points_value, raw))
    return results


def _extract_scale_word(text: str) -> str:
    """Extract a leading scale word from the tail of a number match."""
    scale_words = (
        "million",
        "billion",
        "thousand",
        "trillion",
        "m",
        "bn",
        "k",
        "千万",
        "百万",
        "万",
        "亿",
    )
    lowered = text.lower()
    for word in sorted(scale_words, key=len, reverse=True):
        if lowered.startswith(word) or text.startswith(word):
            # Ensure the character after the scale word is not alphanumeric
            # (avoids matching "millionaire" for "million").
            end_idx = len(word)
            if end_idx >= len(text) or not (
                text[end_idx].isascii() and text[end_idx].isalnum()
            ):
                return text[:end_idx]
    return ""


def _find_keyword_in_text(
    text: str, keywords_en: tuple[str, ...], keywords_zh: tuple[str, ...]
) -> str | None:
    """Return the first keyword found in text, or None."""
    lowered = text.lower()
    for kw in keywords_en:
        if kw in lowered:
            return kw
    for kw in keywords_zh:
        if kw in text:
            return kw
    return None


def _extract_operand_for_role(
    role: str,
    evidence: tuple[EvidenceItem, ...],
) -> tuple[CalculationOperand | None, list[str]]:
    """Extract a single operand for the given role from evidence.

    Returns ``(operand_or_none, warnings)``.
    """
    kw_map = ROLE_KEYWORDS.get(role)
    if kw_map is None:
        return None, [f"role '{role}' has no keyword mapping"]

    keywords_en = kw_map.get("en", ())
    keywords_zh = kw_map.get("zh", ())
    warnings: list[str] = []
    candidates: list[tuple[Decimal, str, EvidenceItem]] = []

    for item in evidence:
        sentences = _split_sentences(item.content)
        for sentence in sentences:
            found_kw = _find_keyword_in_text(sentence, keywords_en, keywords_zh)
            if found_kw is None:
                continue
            numbers = _extract_numbers_from_sentence(sentence)
            if not numbers:
                continue
            # Take the first number in the sentence as the candidate.
            value, raw = numbers[0]
            candidates.append((value, raw, item))
            if len(numbers) > 1:
                warnings.append(
                    f"role '{role}': sentence has {len(numbers)} numbers, "
                    f"taking the first"
                )

    if not candidates:
        return None, warnings

    if len(candidates) > 1:
        warnings.append(
            f"role '{role}': {len(candidates)} candidate values found, using the first"
        )

    value, raw, item = candidates[0]
    operand = CalculationOperand(
        name=role,
        value=value,
        source_text=raw,
        evidence_chunk_id=item.chunk_id,
        document_name=item.document_name,
        page=item.page,
    )
    return operand, warnings


def _extract_scale_conversion_operands(
    evidence: tuple[EvidenceItem, ...],
) -> ExtractionResult:
    """Extract a single numeric operand and its source scale for SCALE_CONVERSION."""

    for item in evidence:
        sentences = _split_sentences(item.content)
        for sentence in sentences:
            numbers = _extract_numbers_from_sentence(sentence)
            if not numbers:
                continue
            value, raw = numbers[0]
            source_scale = _infer_scale(sentence) or ""
            operand = CalculationOperand(
                name="value",
                value=value,
                scale=source_scale or None,
                source_text=raw,
                evidence_chunk_id=item.chunk_id,
                document_name=item.document_name,
                page=item.page,
            )
            return ExtractionResult(
                operands=(operand,),
                found_roles=("value",),
                missing_roles=(),
                warnings=(),
                expected_roles=("value",),
                source_scale=source_scale,
            )

    return ExtractionResult(
        operands=(),
        found_roles=(),
        missing_roles=("value",),
        warnings=("no numeric operand found in evidence",),
        expected_roles=("value",),
        source_scale=None,
    )


def extract_operands(
    evidence: tuple[EvidenceItem, ...],
    routing_decision: RoutingDecision,
) -> ExtractionResult:
    """Extract calculation operands from retrieved evidence.

    Args:
        evidence: The retrieved evidence items from the RAG pipeline.
        routing_decision: The routing decision carrying expected operand
            roles and the target metric/operation.

    Returns:
        An ``ExtractionResult`` with found operands, missing roles, and
        any ambiguity warnings.
    """
    expected_roles = routing_decision.operand_roles

    if (
        routing_decision.operation is CalculationOperation.SCALE_CONVERSION
        and not expected_roles
    ):
        return _extract_scale_conversion_operands(evidence)

    if not expected_roles:
        # Generic operations (sum, difference, average) have no fixed roles.
        # Return an empty result; the plan builder will handle this.
        return ExtractionResult(
            operands=(),
            found_roles=(),
            missing_roles=(),
            warnings=("no fixed operand roles for this operation",),
            expected_roles=(),
        )

    found: list[CalculationOperand] = []
    found_roles: list[str] = []
    missing_roles: list[str] = []
    all_warnings: list[str] = []

    for role in expected_roles:
        operand, warnings = _extract_operand_for_role(role, evidence)
        if operand is not None:
            found.append(operand)
            found_roles.append(role)
        else:
            missing_roles.append(role)
        all_warnings.extend(warnings)

    return ExtractionResult(
        operands=tuple(found),
        found_roles=tuple(found_roles),
        missing_roles=tuple(missing_roles),
        warnings=tuple(all_warnings),
        expected_roles=expected_roles,
    )
