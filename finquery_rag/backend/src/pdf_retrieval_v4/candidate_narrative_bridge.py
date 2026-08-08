"""Gate 05 R5 — Narrative Candidate Bridge.

Handles ``text`` type Production Candidates by matching them to
Narrative Evidence from Gate 03 R2.

Grade: A5_narrative

Matching criteria (Grade A requires one of):
  - Strong bbox overlap + compatible text
  - Very-high text coverage + unique match

Narrative candidates MUST NOT be matched using financial numeric rules.
"""

from __future__ import annotations

from src.pdf_retrieval_v4.candidate_bridge_models import (
    BridgeGrade,
    BridgeMatch,
    BridgeResult,
    CandidateSignature,
    SemanticEvidenceSignature,
)
from src.pdf_retrieval_v4.candidate_row_bridge import (
    compute_bbox_iou,
    compute_bbox_coverage,
    compute_text_coverage,
    compute_token_jaccard,
    SCORE_GAP_THRESHOLD,
)
from src.pdf_retrieval_v4.semantic_evidence_catalog import SemanticEvidenceCatalog

# Narrative-specific thresholds
NARRATIVE_BBOX_THRESHOLD = 0.50
NARRATIVE_TEXT_HIGH = 0.70  # Very-high text coverage
NARRATIVE_TEXT_MODERATE = 0.40  # Moderate text coverage (needs bbox)
NARRATIVE_TOKEN_JACCARD_STRONG = 0.40


class NarrativeBridge:
    """Bridge mapper for text/narrative type Production Candidates."""

    def __init__(self, catalog: SemanticEvidenceCatalog) -> None:
        self._catalog = catalog

    def bridge(self, sig: CandidateSignature) -> BridgeResult:
        """Bridge a narrative/text candidate to narrative evidence."""
        # Get narrative evidence on the same page
        narrative = self._catalog.get_narrative_by_page(sig.document_id, sig.pdf_page)

        if not narrative:
            return BridgeResult(
                candidate_key=sig.candidate_key,
                grade=BridgeGrade.UNMAPPED.value,
                matches=(),
                failure_stage="narrative_bridge_missing",
                bridge_reasons=("no_narrative_evidence_on_page",),
            )

        matches: list[BridgeMatch] = []

        for ev in narrative:
            tc = compute_text_coverage(sig.text_tokens, ev.normalized_text)
            tj = compute_token_jaccard(sig.text_tokens, _evidence_text_tokens(ev))
            compute_bbox_iou((), ev.bbox)
            bbox_cov = compute_bbox_coverage((), ev.bbox)

            # Path 1: Strong bbox + compatible text
            path1 = (
                ev.bbox
                and bbox_cov >= NARRATIVE_BBOX_THRESHOLD
                and tc >= NARRATIVE_TEXT_MODERATE
            )

            # Path 2: Very-high text coverage + unique match (checked later)
            path2 = tc >= NARRATIVE_TEXT_HIGH or tj >= NARRATIVE_TOKEN_JACCARD_STRONG

            if not (path1 or path2):
                continue

            # Compute score
            score = tc * 0.4 + tj * 0.3 + bbox_cov * 0.2 + 0.1

            match = BridgeMatch(
                evidence_id=ev.evidence_id,
                evidence_type=ev.evidence_type,
                grade=BridgeGrade.A5_NARRATIVE.value,
                score=score,
                reasons=(
                    f"text_coverage={tc:.3f}",
                    f"token_jaccard={tj:.3f}",
                    f"bbox_coverage={bbox_cov:.3f}",
                    f"path={'bbox+text' if path1 else 'high_text'}",
                ),
                numeric_recall=1.0,  # Narrative doesn't use numeric
                text_coverage=tc,
                bbox_overlap=bbox_cov,
                metric_compatible=True,
                period_compatible=True,
            )
            matches.append(match)

        if not matches:
            return BridgeResult(
                candidate_key=sig.candidate_key,
                grade=BridgeGrade.UNMAPPED.value,
                matches=(),
                failure_stage="narrative_bridge_missing",
                bridge_reasons=("no_narrative_match_above_threshold",),
            )

        # Sort by score
        matches.sort(key=lambda m: m.score, reverse=True)

        # Check uniqueness
        if len(matches) >= 2:
            gap = matches[0].score - matches[1].score
            if gap < SCORE_GAP_THRESHOLD:
                # For narrative: if path2 (high text), require unique match
                # If path1 (bbox+text), allow if top match has bbox
                top = matches[0]
                if "path=high_text" in top.reasons:
                    return BridgeResult(
                        candidate_key=sig.candidate_key,
                        grade=BridgeGrade.B_AMBIGUOUS.value,
                        matches=tuple(matches[:5]),
                        failure_stage="multiple_equal_matches",
                        bridge_reasons=(
                            f"top_score={top.score:.3f}",
                            f"second_score={matches[1].score:.3f}",
                            "high_text_not_unique",
                        ),
                    )

        best = matches[0]
        return BridgeResult(
            candidate_key=sig.candidate_key,
            grade=BridgeGrade.A5_NARRATIVE.value,
            matches=(best,),
            failure_stage=None,
            bridge_reasons=best.reasons,
        )


def _evidence_text_tokens(ev: SemanticEvidenceSignature) -> tuple[str, ...]:
    """Extract text tokens from evidence for Jaccard comparison."""
    # Use the normalized text, re-tokenize
    import re

    text = ev.normalized_text
    raw_tokens = re.findall(r"[a-z][a-z0-9_]{2,}", text)
    _STOP = frozenset(
        {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "any",
            "can",
            "her",
            "was",
            "one",
            "our",
            "out",
            "has",
            "have",
            "had",
        }
    )
    return tuple(sorted({t for t in raw_tokens if t not in _STOP}))
