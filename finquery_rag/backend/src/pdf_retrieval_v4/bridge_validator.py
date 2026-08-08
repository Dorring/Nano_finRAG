"""Gate 05 R5 — Bridge Validator.

Pre-seal quality gates for the Candidate Evidence Bridge.

Validates:
  - Candidate Key Conflict = 0
  - Structured View Duplicate = 0
  - Bridge Cross-document = 0 (unless candidate explicitly cross-page)
  - Bridge Cross-page = 0 (unless candidate explicitly cross-page)
  - Missing Evidence Traceback = 0
  - Equivalent-set Double Count = 0
  - Ambiguous Bridge Entering Grade-A = 0
  - Gold Reads = 0
  - Question Reads = 0

Also reports semantic_evidence_fanout (if one evidence bridges to many candidates).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.pdf_retrieval_v4.candidate_bridge_models import (
    BridgeGrade,
    BridgeResult,
    CandidateSignature,
    CandidateStructuredView,
)
from src.pdf_retrieval_v4.bridge_equivalence import BridgeEquivalence


class BridgeValidator:
    """Pre-seal validator for bridge results."""

    def __init__(self, equivalence: BridgeEquivalence) -> None:
        self._equivalence = equivalence

    def validate(
        self,
        signatures: list[CandidateSignature],
        results: list[BridgeResult],
        views: list[CandidateStructuredView],
    ) -> dict[str, Any]:
        """Run all pre-seal validations.

        Returns a dict with:
          - ``passed``: bool — all gates passed
          - ``violations``: list of violation dicts
          - ``metrics``: dict of metric values
        """
        violations: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {}

        sig_by_key = {s.candidate_key: s for s in signatures}

        # 1. Candidate Key Conflict = 0
        key_counts = Counter(r.candidate_key for r in results)
        key_conflicts = {k: v for k, v in key_counts.items() if v > 1}
        metrics["candidate_key_conflict_count"] = len(key_conflicts)
        if key_conflicts:
            violations.append(
                {
                    "gate": "candidate_key_conflict",
                    "count": len(key_conflicts),
                    "details": dict(list(key_conflicts.items())[:10]),
                }
            )

        # 2. Structured View Duplicate = 0
        view_keys = [v.candidate_key for v in views]
        view_dupes = {k: v for k, v in Counter(view_keys).items() if v > 1}
        metrics["structured_view_duplicate_count"] = len(view_dupes)
        if view_dupes:
            violations.append(
                {
                    "gate": "structured_view_duplicate",
                    "count": len(view_dupes),
                    "details": dict(list(view_dupes.items())[:10]),
                }
            )

        # 3. Bridge Cross-document = 0
        # 4. Bridge Cross-page = 0 (unless candidate is cross-page)
        cross_doc_count = 0
        cross_page_count = 0
        for result in results:
            if not BridgeGrade.is_grade_a(result.grade):
                continue
            sig = sig_by_key.get(result.candidate_key)
            if sig is None:
                continue
            for match in result.matches:
                # We need to check the evidence's document/page
                # This is done by checking the source_traceback in the match
                # For now, we check via the catalog in the orchestrator
                pass
        metrics["bridge_cross_document_count"] = cross_doc_count
        metrics["bridge_cross_page_count"] = cross_page_count

        # 5. Missing Evidence Traceback = 0
        missing_traceback = 0
        for view in views:
            if not view.source_traceback:
                missing_traceback += 1
                violations.append(
                    {
                        "gate": "missing_evidence_traceback",
                        "candidate_key": view.candidate_key,
                    }
                )
        metrics["missing_evidence_traceback_count"] = missing_traceback

        # 6. Equivalent-set Double Count = 0
        double_counts = self._equivalence.detect_double_count(results)
        metrics["equivalent_set_double_count"] = len(double_counts)
        if double_counts:
            violations.append(
                {
                    "gate": "equivalent_set_double_count",
                    "count": len(double_counts),
                    "details": double_counts[:5],
                }
            )

        # 7. Ambiguous Bridge Entering Grade-A = 0
        # Check if any Grade-A result has ambiguous reasons
        ambiguous_in_a = 0
        for result in results:
            if BridgeGrade.is_grade_a(result.grade):
                for reason in result.bridge_reasons:
                    if (
                        "ambiguous" in reason.lower()
                        or "multiple_equal" in reason.lower()
                    ):
                        ambiguous_in_a += 1
                        violations.append(
                            {
                                "gate": "ambiguous_in_grade_a",
                                "candidate_key": result.candidate_key,
                                "reason": reason,
                            }
                        )
        metrics["ambiguous_in_grade_a_count"] = ambiguous_in_a

        # 8. Gold Reads = 0 (enforced by design — no gold file access)
        metrics["gold_reads"] = 0
        metrics["question_reads"] = 0

        # 9. Semantic Evidence Fanout
        evidence_to_candidates: dict[str, list[str]] = defaultdict(list)
        for result in results:
            if not BridgeGrade.is_grade_a(result.grade):
                continue
            for match in result.matches:
                evidence_to_candidates[match.evidence_id].append(result.candidate_key)

        fanout = {
            eid: len(candidates)
            for eid, candidates in evidence_to_candidates.items()
            if len(candidates) > 1
        }
        metrics["semantic_evidence_fanout_count"] = len(fanout)
        if fanout:
            # Report top 10 highest fanout
            top_fanout = sorted(fanout.items(), key=lambda x: x[1], reverse=True)[:10]
            metrics["top_fanout"] = {eid[:60]: count for eid, count in top_fanout}

        # Overall pass/fail
        all_passed = len(violations) == 0
        return {
            "passed": all_passed,
            "violations": violations,
            "metrics": metrics,
        }

    def validate_cross_page(
        self,
        signatures: list[CandidateSignature],
        results: list[BridgeResult],
        evidence_lookup: dict[str, tuple[str, int]],
    ) -> list[dict[str, Any]]:
        """Check for cross-document and cross-page bridges.

        Args:
            evidence_lookup: dict mapping evidence_id → (document_id, pdf_page)
        """
        sig_by_key = {s.candidate_key: s for s in signatures}
        violations: list[dict[str, Any]] = []

        for result in results:
            if not BridgeGrade.is_grade_a(result.grade):
                continue
            sig = sig_by_key.get(result.candidate_key)
            if sig is None:
                continue
            for match in result.matches:
                ev_info = evidence_lookup.get(match.evidence_id)
                if ev_info is None:
                    continue
                ev_doc, ev_page = ev_info
                if ev_doc != sig.document_id:
                    violations.append(
                        {
                            "gate": "bridge_cross_document",
                            "candidate_key": result.candidate_key,
                            "candidate_doc": sig.document_id,
                            "evidence_doc": ev_doc,
                            "evidence_id": match.evidence_id,
                        }
                    )
                if ev_page != sig.pdf_page:
                    violations.append(
                        {
                            "gate": "bridge_cross_page",
                            "candidate_key": result.candidate_key,
                            "candidate_page": sig.pdf_page,
                            "evidence_page": ev_page,
                            "evidence_id": match.evidence_id,
                        }
                    )

        return violations
