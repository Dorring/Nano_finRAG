"""Gate 05 R5 — Table-row Candidate Bridge.

Matches ``table_row`` type Production Candidates to Semantic Evidence
(rows, facts, matrices) using three grade levels:

  A1_direct       — existing stable identity (candidate already has row_ids)
  A2_bbox_signature — same doc + same page + strong bbox overlap + numeric + text
  A3_row_signature — same doc + same page + numeric_recall=1.0 + text/metric + unique

Shared matching utilities are exported for use by other bridge mappers.
"""

from __future__ import annotations


from src.pdf_retrieval_v4.candidate_bridge_models import (
    BRIDGE_ELIGIBLE_ROW_TYPES,
    BridgeGrade,
    BridgeMatch,
    BridgeResult,
    CandidateSignature,
    SemanticEvidenceSignature,
)
from src.pdf_retrieval_v4.semantic_evidence_catalog import SemanticEvidenceCatalog


# ---------------------------------------------------------------------------
# Shared Matching Utilities
# ---------------------------------------------------------------------------


def compute_numeric_recall(
    candidate_nums: tuple[str, ...],
    evidence_nums: tuple[str, ...],
) -> float:
    """Fraction of candidate numbers found in evidence.

    If candidate has no numbers, returns 1.0 (vacuously true).
    """
    if not candidate_nums:
        return 1.0
    cand_set = set(candidate_nums)
    evid_set = set(evidence_nums)
    if not evid_set:
        return 0.0
    return len(cand_set & evid_set) / len(cand_set)


def compute_numeric_precision(
    candidate_nums: tuple[str, ...],
    evidence_nums: tuple[str, ...],
) -> float:
    """Fraction of evidence numbers found in candidate.

    If evidence has no numbers, returns 1.0 (vacuously true).
    """
    if not evidence_nums:
        return 1.0
    cand_set = set(candidate_nums)
    evid_set = set(evidence_nums)
    if not cand_set:
        return 0.0
    return len(cand_set & evid_set) / len(evid_set)


def compute_text_coverage(
    candidate_tokens: tuple[str, ...],
    evidence_text: str,
) -> float:
    """Fraction of candidate text tokens found in evidence text.

    If candidate has no tokens, returns 0.0.
    """
    if not candidate_tokens:
        return 0.0
    evid_lower = evidence_text.lower()
    matched = sum(1 for t in candidate_tokens if t in evid_lower)
    return matched / len(candidate_tokens)


def compute_token_jaccard(
    tokens_a: tuple[str, ...],
    tokens_b: tuple[str, ...],
) -> float:
    """Jaccard similarity between two token sets."""
    if not tokens_a and not tokens_b:
        return 1.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def compute_bbox_iou(
    bbox_a: tuple[float, ...],
    bbox_b: tuple[float, ...],
) -> float:
    """Intersection-over-Union of two bounding boxes.

    Bbox format: (x0, y0, x1, y1).
    Returns 0.0 if either bbox is empty or no overlap.
    """
    if len(bbox_a) != 4 or len(bbox_b) != 4:
        return 0.0

    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b

    # Intersection
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0

    inter_area = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compute_bbox_coverage(
    candidate_bbox: tuple[float, ...],
    evidence_bbox: tuple[float, ...],
) -> float:
    """Fraction of candidate bbox covered by evidence bbox.

    Returns 0.0 if either bbox is empty.
    """
    if len(candidate_bbox) != 4 or len(evidence_bbox) != 4:
        return 0.0

    cx0, cy0, cx1, cy1 = candidate_bbox
    ex0, ey0, ex1, ey1 = evidence_bbox

    # Intersection
    ix0 = max(cx0, ex0)
    iy0 = max(cy0, ey0)
    ix1 = min(cx1, ex1)
    iy1 = min(cy1, ey1)

    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0

    inter_area = (ix1 - ix0) * (iy1 - iy0)
    cand_area = (cx1 - cx0) * (cy1 - cy0)

    if cand_area <= 0:
        return 0.0

    return inter_area / cand_area


def metric_compatible(
    candidate_metric: str | None,
    evidence_metric: str | None,
) -> bool:
    """Check if candidate and evidence metric paths are compatible.

    Compatible if:
    - Either is empty/None (no constraint)
    - They are equal (case-insensitive)
    - One is a substring of the other
    """
    if not candidate_metric or not evidence_metric:
        return True
    cm = candidate_metric.lower().strip()
    em = evidence_metric.lower().strip()
    if cm == em:
        return True
    # Check substring (handles multi-level paths)
    if cm in em or em in cm:
        return True
    # Check token overlap
    cm_tokens = set(cm.split())
    em_tokens = set(em.split())
    if cm_tokens and em_tokens:
        overlap = len(cm_tokens & em_tokens) / max(len(cm_tokens), len(em_tokens))
        return overlap >= 0.5
    return False


def period_compatible(
    candidate_periods: tuple[str, ...],
    evidence_periods: tuple[str, ...],
) -> bool:
    """Check if candidate and evidence periods are compatible.

    Compatible if:
    - Either is empty (no constraint)
    - They share at least one period
    """
    if not candidate_periods or not evidence_periods:
        return True
    cand_set = set(p.upper() for p in candidate_periods)
    evid_set = set(p.upper() for p in evidence_periods)
    return bool(cand_set & evid_set)


# ---------------------------------------------------------------------------
# Row Bridge Mapper
# ---------------------------------------------------------------------------

# Thresholds
BBOX_COVERAGE_THRESHOLD = 0.70
TEXT_COVERAGE_STRONG = 0.60
TEXT_COVERAGE_MODERATE = 0.40
NUMERIC_RECALL_PERFECT = 1.0
SCORE_GAP_THRESHOLD = 0.05  # Minimum gap to declare unique best match


class RowBridge:
    """Bridge mapper for table_row type Production Candidates.

    Matching priority:
      A1 — existing stable direct identity
      A2 — bbox + signature
      A3 — row-text + numeric (unique best match)
      B  — ambiguous (multiple equal matches)
      unmapped — no match
    """

    def __init__(self, catalog: SemanticEvidenceCatalog) -> None:
        self._catalog = catalog

    def bridge(self, sig: CandidateSignature) -> BridgeResult:
        """Bridge a table_row candidate to semantic evidence."""
        # Get candidate evidence on same page
        page_evidence = self._catalog.get_rows_by_page(sig.document_id, sig.pdf_page)

        # Filter to eligible row types (metric_row, subtotal, total)
        eligible = [e for e in page_evidence if e.row_type in BRIDGE_ELIGIBLE_ROW_TYPES]

        if not eligible:
            return BridgeResult(
                candidate_key=sig.candidate_key,
                grade=BridgeGrade.UNMAPPED.value,
                matches=(),
                failure_stage="candidate_text_signature_mismatch",
                bridge_reasons=("no_eligible_rows_on_page",),
            )

        # A1: Check for existing direct identity
        a1_result = self._try_a1_direct(sig, eligible)
        if a1_result is not None:
            return a1_result

        # A2: Try bbox + signature matching
        a2_result = self._try_a2_bbox(sig, eligible)
        if a2_result is not None:
            return a2_result

        # A3: Try row-text + numeric matching
        a3_result = self._try_a3_row_signature(sig, eligible)
        if a3_result is not None:
            return a3_result

        # Check for ambiguity (multiple equal matches)
        all_matches = self._score_all(sig, eligible)
        if len(all_matches) >= 2:
            best = all_matches[0]
            second = all_matches[1]
            if abs(best.score - second.score) < SCORE_GAP_THRESHOLD:
                # Ambiguous — fail closed
                return BridgeResult(
                    candidate_key=sig.candidate_key,
                    grade=BridgeGrade.B_AMBIGUOUS.value,
                    matches=tuple(all_matches[:5]),
                    failure_stage="multiple_equal_matches",
                    bridge_reasons=(
                        f"top_score={best.score:.3f}",
                        f"second_score={second.score:.3f}",
                        "score_gap_below_threshold",
                    ),
                )

        # Unmapped
        if all_matches:
            best = all_matches[0]
            # Determine failure stage
            failure = self._classify_failure(sig, best)
            return BridgeResult(
                candidate_key=sig.candidate_key,
                grade=BridgeGrade.UNMAPPED.value,
                matches=(),
                failure_stage=failure,
                bridge_reasons=(
                    f"best_score={best.score:.3f}",
                    f"numeric_recall={best.numeric_recall:.3f}",
                    f"text_coverage={best.text_coverage:.3f}",
                ),
            )

        return BridgeResult(
            candidate_key=sig.candidate_key,
            grade=BridgeGrade.UNMAPPED.value,
            matches=(),
            failure_stage="candidate_text_signature_mismatch",
            bridge_reasons=("no_scored_matches",),
        )

    def _try_a1_direct(
        self,
        sig: CandidateSignature,
        eligible: list[SemanticEvidenceSignature],
    ) -> BridgeResult | None:
        """A1: Direct identity via existing row_ids from prior gate."""
        if not sig.existing_row_ids:
            return None

        # Check if any existing row_id matches an eligible evidence row
        eligible_by_id = {e.evidence_id: e for e in eligible}
        matched: list[BridgeMatch] = []
        for row_id in sig.existing_row_ids:
            if row_id in eligible_by_id:
                ev = eligible_by_id[row_id]
                matched.append(
                    BridgeMatch(
                        evidence_id=ev.evidence_id,
                        evidence_type=ev.evidence_type,
                        grade=BridgeGrade.A1_DIRECT.value,
                        score=1.0,
                        reasons=("existing_row_id_match",),
                        numeric_recall=1.0,
                        text_coverage=1.0,
                        bbox_overlap=1.0,
                        metric_compatible=True,
                        period_compatible=True,
                    )
                )

        if matched:
            return BridgeResult(
                candidate_key=sig.candidate_key,
                grade=BridgeGrade.A1_DIRECT.value,
                matches=tuple(matched),
                failure_stage=None,
                bridge_reasons=("direct_identity_from_existing_row_ids",),
            )
        return None

    def _try_a2_bbox(
        self,
        sig: CandidateSignature,
        eligible: list[SemanticEvidenceSignature],
    ) -> BridgeResult | None:
        """A2: BBox + signature matching."""
        # Candidate needs to have numbers for A2
        if not sig.numeric_multiset:
            return None

        matches: list[BridgeMatch] = []
        for ev in eligible:
            if not ev.bbox:
                continue

            bbox_cov = compute_bbox_coverage(sig._candidate_bbox(), ev.bbox)
            if bbox_cov < BBOX_COVERAGE_THRESHOLD:
                continue

            nr = compute_numeric_recall(sig.numeric_multiset, ev.numeric_multiset)
            if nr < NUMERIC_RECALL_PERFECT:
                continue

            tc = compute_text_coverage(sig.text_tokens, ev.normalized_text)
            mp = self._catalog.get_metric_path_for_row(ev.row_id or "")
            mc = metric_compatible(
                sig.existing_metric_paths[0] if sig.existing_metric_paths else None,
                mp,
            )

            # Require either strong text coverage or metric compatibility
            if tc < TEXT_COVERAGE_MODERATE and not mc:
                continue

            score = bbox_cov * 0.4 + nr * 0.3 + tc * 0.2 + (0.1 if mc else 0.0)
            matches.append(
                BridgeMatch(
                    evidence_id=ev.evidence_id,
                    evidence_type=ev.evidence_type,
                    grade=BridgeGrade.A2_BBOX_SIGNATURE.value,
                    score=score,
                    reasons=(
                        f"bbox_coverage={bbox_cov:.3f}",
                        f"numeric_recall={nr:.3f}",
                        f"text_coverage={tc:.3f}",
                        f"metric_compatible={mc}",
                    ),
                    numeric_recall=nr,
                    text_coverage=tc,
                    bbox_overlap=bbox_cov,
                    metric_compatible=mc,
                    period_compatible=True,
                )
            )

        if not matches:
            return None

        # Sort by score descending
        matches.sort(key=lambda m: m.score, reverse=True)

        # Check uniqueness
        if len(matches) >= 2:
            gap = matches[0].score - matches[1].score
            if gap < SCORE_GAP_THRESHOLD:
                return None  # Fall through to A3 or B

        best = matches[0]
        return BridgeResult(
            candidate_key=sig.candidate_key,
            grade=BridgeGrade.A2_BBOX_SIGNATURE.value,
            matches=(best,),
            failure_stage=None,
            bridge_reasons=best.reasons,
        )

    def _try_a3_row_signature(
        self,
        sig: CandidateSignature,
        eligible: list[SemanticEvidenceSignature],
    ) -> BridgeResult | None:
        """A3: Row-text + numeric matching (no bbox requirement)."""
        if not sig.numeric_multiset:
            return None

        matches: list[BridgeMatch] = []
        for ev in eligible:
            nr = compute_numeric_recall(sig.numeric_multiset, ev.numeric_multiset)
            if nr < NUMERIC_RECALL_PERFECT:
                continue

            tc = compute_text_coverage(sig.text_tokens, ev.normalized_text)
            mp = self._catalog.get_metric_path_for_row(ev.row_id or "")
            mc = metric_compatible(
                sig.existing_metric_paths[0] if sig.existing_metric_paths else None,
                mp,
            )

            # Require either strong text match or (metric compatible AND moderate text)
            # OR perfect numeric recall with at least minimal text overlap
            if tc < TEXT_COVERAGE_STRONG:
                if not mc or tc < TEXT_COVERAGE_MODERATE:
                    if nr < NUMERIC_RECALL_PERFECT or tc < 0.20:
                        continue

            bbox_ov = compute_bbox_iou(sig._candidate_bbox(), ev.bbox)
            score = nr * 0.4 + tc * 0.3 + (0.2 if mc else 0.0) + bbox_ov * 0.1

            matches.append(
                BridgeMatch(
                    evidence_id=ev.evidence_id,
                    evidence_type=ev.evidence_type,
                    grade=BridgeGrade.A3_ROW_SIGNATURE.value,
                    score=score,
                    reasons=(
                        f"numeric_recall={nr:.3f}",
                        f"text_coverage={tc:.3f}",
                        f"metric_compatible={mc}",
                    ),
                    numeric_recall=nr,
                    text_coverage=tc,
                    bbox_overlap=bbox_ov,
                    metric_compatible=mc,
                    period_compatible=True,
                )
            )

        if not matches:
            return None

        matches.sort(key=lambda m: m.score, reverse=True)

        # A3 requires unique best match
        if len(matches) >= 2:
            gap = matches[0].score - matches[1].score
            if gap < SCORE_GAP_THRESHOLD:
                return None  # Fall through to B_ambiguous

        best = matches[0]
        return BridgeResult(
            candidate_key=sig.candidate_key,
            grade=BridgeGrade.A3_ROW_SIGNATURE.value,
            matches=(best,),
            failure_stage=None,
            bridge_reasons=best.reasons,
        )

    def _score_all(
        self,
        sig: CandidateSignature,
        eligible: list[SemanticEvidenceSignature],
    ) -> list[BridgeMatch]:
        """Score all eligible evidence for a candidate (for ambiguity check)."""
        matches: list[BridgeMatch] = []
        for ev in eligible:
            nr = compute_numeric_recall(sig.numeric_multiset, ev.numeric_multiset)
            tc = compute_text_coverage(sig.text_tokens, ev.normalized_text)
            mp = self._catalog.get_metric_path_for_row(ev.row_id or "")
            mc = metric_compatible(
                sig.existing_metric_paths[0] if sig.existing_metric_paths else None,
                mp,
            )
            bbox_ov = compute_bbox_coverage(sig._candidate_bbox(), ev.bbox)
            score = (
                nr * 0.35
                + tc * 0.25
                + bbox_ov * 0.2
                + (0.15 if mc else 0.0)
                + (0.05 if period_compatible(sig.period_tokens, ev.periods) else 0.0)
            )
            matches.append(
                BridgeMatch(
                    evidence_id=ev.evidence_id,
                    evidence_type=ev.evidence_type,
                    grade="scored",
                    score=score,
                    reasons=(
                        f"numeric_recall={nr:.3f}",
                        f"text_coverage={tc:.3f}",
                        f"bbox_overlap={bbox_ov:.3f}",
                        f"metric_compatible={mc}",
                    ),
                    numeric_recall=nr,
                    text_coverage=tc,
                    bbox_overlap=bbox_ov,
                    metric_compatible=mc,
                    period_compatible=period_compatible(sig.period_tokens, ev.periods),
                )
            )
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    def _classify_failure(
        self,
        sig: CandidateSignature,
        best: BridgeMatch,
    ) -> str:
        """Classify the failure stage for an unmapped candidate."""
        if best.numeric_recall < NUMERIC_RECALL_PERFECT:
            return "numeric_signature_mismatch"
        if best.text_coverage < TEXT_COVERAGE_MODERATE:
            return "candidate_text_signature_mismatch"
        if not best.metric_compatible:
            return "metric_signature_mismatch"
        if not best.period_compatible:
            return "period_signature_mismatch"
        return "candidate_text_signature_mismatch"


# Add helper method to CandidateSignature for bbox access
# (candidates don't have bbox in the current data, so return empty)
def _candidate_bbox(self: CandidateSignature) -> tuple[float, ...]:
    """Return candidate bbox (empty — candidates don't carry bbox)."""
    return ()


# Monkey-patch the method onto CandidateSignature
CandidateSignature._candidate_bbox = _candidate_bbox  # type: ignore[attr-defined]
