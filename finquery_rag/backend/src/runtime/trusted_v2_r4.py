from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_v2.adaptive import AdaptiveRAGStateV1, ReplanActionV1
from rag_v2.contracts.plan import SupervisorPlan
from rag_v2.supervisor import (
    EvidenceScope,
    EvidenceScopeClassification,
    canonical_entity_id,
    canonical_metric_id,
    classify_evidence_scope,
    extract_query_semantic_frame,
)
from src.pdf_retrieval_v4.candidate_direct_retriever import CandidateDirectRetriever
from src.pdf_retrieval_v4.candidate_query_builder import append_missing_query_terms
from src.pdf_retrieval_v4.planner import build_query_plan


class R4RetrievalCapabilityError(RuntimeError):
    """Base error for the V2-to-R4 capability boundary."""


class R4CandidateSchemaError(R4RetrievalCapabilityError):
    """Raised when a candidate cannot be materialized as structured evidence."""


@dataclass(frozen=True)
class R4RetrievalRequest:
    """The bounded runtime request understood by the R4 policy adapter."""

    request_id: str
    standalone_query: str
    plan: SupervisorPlan
    reason_code: str
    target_slots: tuple[str, ...] = ()
    retrieval_round: int = 0
    document_scope: tuple[str, ...] = ()
    constraints: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class R4RetrievalResult:
    """Candidate-only output from an R4 policy."""

    candidate_evidence: tuple[Mapping[str, Any], ...]
    candidate_ids: tuple[str, ...]
    retrieval_reason: str
    target_slots: tuple[str, ...]
    retrieval_round: int
    source_branch_metadata: Mapping[str, Any] = field(default_factory=dict)
    result_fingerprint: str | None = None


class R4Policy(Protocol):
    def retrieve(self, request: R4RetrievalRequest) -> R4RetrievalResult:
        ...


Materializer = Callable[[str], Mapping[str, Any]]


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return tuple(result)


def _slot_metric_matches(left: Any, right: Any) -> bool:
    left_id = canonical_metric_id(left)
    right_id = canonical_metric_id(right)
    if left_id is not None and right_id is not None:
        return left_id == right_id
    left_str = str(left or "").strip().casefold()
    right_str = str(right or "").strip().casefold()
    if not left_str or not right_str:
        return False
    if left_str == right_str:
        return True
    return left_str in right_str or right_str in left_str


def _slot_period_matches(left: Any, right: Any) -> bool:
    return str(left or "").strip().casefold() == str(right or "").strip().casefold()


_SOURCE_VIEW_LANE_ORDER = (
    "candidate_structured_bm25",
    "candidate_structured_dense",
    "candidate_raw_bm25",
    "candidate_raw_dense",
)


def _unique_texts(values: Iterable[Any]) -> tuple[str, ...]:
    return _stable_unique(str(value) for value in values if value is not None)


def _source_context_for_candidate(
    reader: Any,
    candidate_key: str,
    pool_item: Mapping[str, Any],
) -> tuple[dict[str, Any], EvidenceScopeClassification]:
    """Collect compact source-row context for one R4 candidate.

    R4 retrieval returns candidate keys and lane ranks.  The candidate-aligned
    metadata store is the authoritative bridge for row/header context; the
    adapter copies only structured fields needed by the Binder and semantic
    firewall, never raw answer text or model output.
    """

    lookup = getattr(reader, "view_for_candidate", None)
    if not callable(lookup):
        return {}, EvidenceScopeClassification()
    supporting = pool_item.get("supporting_view_ids")
    if not isinstance(supporting, Mapping):
        supporting = {}
    views: list[tuple[str, Mapping[str, Any]]] = []
    seen_view_ids: set[str] = set()
    for lane in _SOURCE_VIEW_LANE_ORDER:
        view_id = supporting.get(lane)
        if not view_id:
            continue
        try:
            view = lookup(lane, candidate_key)
        except Exception:
            view = None
        if not isinstance(view, Mapping):
            continue
        actual_view_id = str(view.get("view_id") or view_id)
        if actual_view_id in seen_view_ids:
            continue
        seen_view_ids.add(actual_view_id)
        views.append((lane, view))
    if not views:
        return {}, EvidenceScopeClassification()

    metric_paths: list[str] = []
    periods: list[str] = []
    row_ids: list[str] = []
    fact_ids: list[str] = []
    retrieval_texts: list[str] = []
    classifications: list[EvidenceScopeClassification] = []
    for _lane, view in views:
        metadata = view.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        metric_paths.extend(_unique_texts(metadata.get("metric_paths") or ()))
        periods.extend(_unique_texts(metadata.get("periods") or ()))
        row_ids.extend(_unique_texts(metadata.get("row_ids") or ()))
        fact_ids.extend(_unique_texts(metadata.get("fact_ids") or ()))
        text = view.get("retrieval_text")
        if isinstance(text, str) and text.strip():
            retrieval_texts.append(text)
        source = {
            **dict(metadata),
            "retrieval_text": text if isinstance(text, str) else "",
            "document_id": view.get("document_id") or metadata.get("document_id"),
        }
        classifications.append(classify_evidence_scope(source))

    known = [
        item for item in classifications if item.scope is not EvidenceScope.UNKNOWN
    ]
    known_scopes = {item.scope for item in known}
    if len(known_scopes) > 1:
        classification = EvidenceScopeClassification(
            EvidenceScope.UNKNOWN,
            source="conflicting_source_scope",
        )
    elif known:
        classification = known[0]
    else:
        classification = EvidenceScopeClassification()

    context: dict[str, Any] = {
        "source_view_ids": [item[1].get("view_id") for item in views],
        "source_lanes": [item[0] for item in views],
        "metric_paths": list(_unique_texts(metric_paths)),
        "row_path": list(_unique_texts(metric_paths)),
        "periods": list(_unique_texts(periods)),
        "row_ids": list(_unique_texts(row_ids)),
        "fact_ids": list(_unique_texts(fact_ids)),
        "retrieval_texts": list(dict.fromkeys(retrieval_texts)),
        "has_total_revenue_label": _is_total_revenue_context_raw(retrieval_texts, metric_paths),
        "scope": classification.scope.value,
        "scope_label": classification.scope_label,
        "scope_source": classification.source,
    }
    return context, classification


def _is_total_revenue_context(context: Mapping[str, Any]) -> bool:
    """Return true only for an explicitly labelled aggregate revenue row."""

    if context.get("has_total_revenue_label") is True:
        return True

    paths = {
        str(item).strip().casefold()
        for item in context.get("metric_paths", ())
        if str(item).strip()
    }
    if paths & {"total revenue", "total revenues", "total net sales"}:
        return True
    for text in context.get("retrieval_texts", ()):
        normalized = " ".join(str(text).casefold().split())
        if any(
            marker in normalized
            for marker in (
                "total revenue",
                "total revenues",
                "total net sales",
                "consolidated revenue",
            )
        ):
            return True
    return False


def _is_total_revenue_context_raw(
    retrieval_texts: Iterable[str],
    metric_paths: Iterable[str],
) -> bool:
    paths = {str(item).strip().casefold() for item in metric_paths if str(item).strip()}
    if paths & {"total revenue", "total revenues", "total net sales"}:
        return True
    for text in retrieval_texts:
        normalized = " ".join(str(text).casefold().split())
        if any(
            marker in normalized
            for marker in (
                "total revenue",
                "total revenues",
                "total net sales",
                "consolidated revenue",
            )
        ):
            return True
    return False


def _planner_slot_ids_for_targets(
    supervisor_plan: SupervisorPlan,
    query_plan: Any,
    target_slots: Sequence[str],
) -> tuple[str, ...]:
    """Map Supervisor slot IDs to deterministic QueryPlan slot IDs.

    The existing R4 planner uses stable role-oriented IDs (for example
    ``fact`` or ``current_period``), while Supervisor providers may emit
    request-specific IDs (for example ``revenue_fy2024``).  Selecting a slot
    pool by raw ID silently fell back to the unscoped raw pool whenever those
    IDs differed.  Match metric + period first, then use positional mapping
    only when the contracts expose the same slot count.
    """

    requested = [
        slot
        for slot in supervisor_plan.required_slots
        if not target_slots or slot.slot_id in set(target_slots)
    ]
    planner_slots = list(getattr(query_plan, "operand_slots", ()) or ())
    matched: list[str] = []
    used: set[str] = set()
    for expected in requested:
        for candidate in planner_slots:
            candidate_id = str(getattr(candidate, "slot_id", ""))
            if not candidate_id or candidate_id in used:
                continue
            if not _slot_metric_matches(
                expected.metric,
                getattr(candidate, "raw_metric_phrase", ""),
            ):
                continue
            if not _slot_period_matches(
                expected.period,
                getattr(candidate, "period", ""),
            ):
                continue
            matched.append(candidate_id)
            used.add(candidate_id)
            break

    if len(matched) == len(requested):
        return tuple(matched)
    if len(requested) == len(planner_slots):
        return tuple(
            str(getattr(slot, "slot_id", ""))
            for slot in planner_slots
            if str(getattr(slot, "slot_id", ""))
        )
    return tuple(matched)


class CandidateDirectR4Policy:
    """Adapt the existing CandidateDirectRetriever into the V2 R4 policy port.

    The retriever remains the canonical R4 policy.  This class only supplies
    the V2 request-to-QueryPlan mapping and rehydrates candidate keys into the
    structured candidate evidence required by the Binder.
    """

    def __init__(
        self,
        retriever: CandidateDirectRetriever,
        *,
        materializer: Materializer,
        document_scope: Sequence[str] = (),
    ) -> None:
        if not isinstance(retriever, CandidateDirectRetriever):
            raise TypeError("retriever must be CandidateDirectRetriever")
        self.retriever = retriever
        self.materializer = materializer
        self.document_scope = tuple(str(item) for item in document_scope if str(item))
        self.calls = 0

    @staticmethod
    def _rrf_pool_item(hit: Any, rank: int) -> dict[str, Any]:
        """Convert a slot RRF hit to the adapter's bounded pool shape."""
        if isinstance(hit, Mapping):
            candidate_key = str(hit.get("candidate_key") or "")
            rrf_score = hit.get("rrf_score")
            lane_ranks = dict(hit.get("lane_ranks") or {})
            supporting_view_ids = dict(hit.get("supporting_view_ids") or {})
        else:
            candidate_key = str(getattr(hit, "candidate_key", "") or "")
            rrf_score = getattr(hit, "rrf_score", None)
            lane_ranks = dict(getattr(hit, "lane_ranks", {}) or {})
            supporting_view_ids = dict(getattr(hit, "supporting_view_ids", {}) or {})

        return {
            "candidate_key": candidate_key,
            "rrf_score": rrf_score,
            "rank": rank,
            "lane_ranks": lane_ranks,
            "supporting_view_ids": supporting_view_ids,
        }

    def retrieve(self, request: R4RetrievalRequest) -> R4RetrievalResult:
        self.calls += 1
        targeted_slots = tuple(
            slot
            for slot in request.plan.required_slots
            if request.target_slots and slot.slot_id in request.target_slots
        )
        target_terms: list[str] = []
        for slot in targeted_slots:
            target_terms.extend(
                term
                for term in (slot.metric, slot.period)
                if term
            )
        if len(targeted_slots) == 1:
            # Recovery must search for the missing requirement itself, not
            # repeat the original multi-slot question. The latter can make the
            # legacy QueryPlan parser choose the first metric again, returning
            # the same crowded candidate pool indefinitely. This is a
            # deterministic slot query built from the Supervisor's canonical
            # RequiredSlot, never a second LLM plan.
            target = targeted_slots[0]
            targeted_question = " ".join(
                part for part in (target.metric, target.period) if part
            )
        elif request.target_slots:
            targeted_question = append_missing_query_terms(
                request.standalone_query,
                target_terms,
            )
        else:
            targeted_question = request.standalone_query
        query_plan = build_query_plan(
            targeted_question,
            tuple(request.document_scope or self.document_scope),
        )
        raw = self.retriever.retrieve(
            query_plan,
            document_scope=set(request.document_scope or self.document_scope),
        )
        slot_pools = raw.get("slot_pools", {})
        pool: list[Mapping[str, Any]] = []
        if request.target_slots and isinstance(slot_pools, Mapping):
            planner_slot_ids = _planner_slot_ids_for_targets(
                request.plan,
                query_plan,
                request.target_slots,
            )
            slot_values_list: list[list[Any]] = []
            for slot_id in planner_slot_ids:
                values = slot_pools.get(slot_id, ())
                if isinstance(values, Iterable):
                    slot_values_list.append(list(values))
            max_depth = max((len(v) for v in slot_values_list), default=0)
            existing_pool_keys: set[str] = set()
            for depth in range(max_depth):
                for slot_values in slot_values_list:
                    if depth < len(slot_values):
                        raw_item = slot_values[depth]
                        rank = depth + 1
                        item = (
                            raw_item
                            if isinstance(raw_item, Mapping)
                            else self._rrf_pool_item(raw_item, rank)
                        )
                        ckey = str(item.get("candidate_key") or "")
                        if ckey and ckey not in existing_pool_keys:
                            existing_pool_keys.add(ckey)
                            pool.append(item)
        if not pool:
            values = raw.get("candidate_direct_pool", ())
            if isinstance(values, Iterable):
                pool.extend(
                    item
                    if isinstance(item, Mapping)
                    else {"candidate_key": getattr(item, "candidate_key", "")}
                    for item in values
                )

        # Ensure slot-specific candidates from R4 slot pools are fairly
        # interleaved before candidate materialization and scope ordering,
        # preventing a single slot from starving operands in calculations.
        if not request.target_slots and request.plan.required_slots:
            slot_pool_values = raw.get("slot_pools", {})
            if isinstance(slot_pool_values, Mapping):
                existing = {
                    str(item.get("candidate_key"))
                    for item in pool
                    if isinstance(item, Mapping) and item.get("candidate_key")
                }
                slot_values_list = [
                    list(v) for v in slot_pool_values.values() if isinstance(v, Iterable)
                ]
                max_depth = max((len(v) for v in slot_values_list), default=0)
                limit = max(80, self.retriever.final_pool_k * 2)
                for depth in range(min(max_depth, limit)):
                    for slot_values in slot_values_list:
                        if depth < len(slot_values):
                            hit = slot_values[depth]
                            rank = depth + 1
                            item = self._rrf_pool_item(hit, rank)
                            candidate_key = item["candidate_key"]
                            if candidate_key and candidate_key not in existing:
                                pool.append(item)
                                existing.add(candidate_key)

        candidate_ids = _stable_unique(
            str(item.get("candidate_key", ""))
            for item in pool
            if isinstance(item, Mapping)
        )
        pool_by_candidate = {
            str(item.get("candidate_key")): item
            for item in pool
            if isinstance(item, Mapping) and item.get("candidate_key")
        }
        candidates: list[Mapping[str, Any]] = []
        for candidate_id in candidate_ids:
            try:
                materialized = self.materializer(candidate_id)
            except Exception as exc:
                raise R4CandidateSchemaError(
                    f"candidate_materialization_failed:{candidate_id}"
                ) from exc
            if not isinstance(materialized, Mapping):
                raise R4CandidateSchemaError(
                    f"candidate_materializer_returned_non_mapping:{candidate_id}"
                )
            candidate = dict(materialized)
            evidence_id = candidate.get("evidence_id") or candidate.get("fact_id")
            if not evidence_id:
                raise R4CandidateSchemaError(
                    f"candidate_missing_evidence_id:{candidate_id}"
                )
            candidate["evidence_id"] = str(evidence_id)
            candidate.setdefault("fact_id", str(evidence_id))
            candidate.setdefault("candidate_id", str(candidate_id))
            candidate["candidate_key"] = str(candidate_id)
            candidate["retrieval_reason"] = request.reason_code
            candidate["retrieval_round"] = request.retrieval_round
            source_context, source_scope = _source_context_for_candidate(
                self.retriever.reader,
                str(candidate_id),
                pool_by_candidate.get(str(candidate_id), {}),
            )
            if source_context:
                # Candidate metadata is source context for Binder selection;
                # it is never treated as admitted evidence by this adapter.
                existing_scope = classify_evidence_scope(candidate)
                if (
                    existing_scope.scope is not EvidenceScope.UNKNOWN
                    and source_scope.scope is not EvidenceScope.UNKNOWN
                    and existing_scope.scope is not source_scope.scope
                ):
                    candidate["scope_conflict"] = True
                    source_context["scope_conflict"] = True
                candidate["retrieval_context"] = source_context
                for key in (
                    "scope",
                    "scope_label",
                    "row_path",
                    "row_label",
                    "statement_type",
                    "table_title",
                    "document_id",
                    "pdf_page",
                ):
                    value = source_context.get(key)
                    if value is not None and not candidate.get(key):
                        candidate[key] = value
                if (
                    source_scope.scope is EvidenceScope.CONSOLIDATED
                    and _is_total_revenue_context(source_context)
                ):
                    metric = candidate.get("metric") or candidate.get("normalized_metric")
                    if (
                        canonical_metric_id(metric) is None
                        and str(metric or "").strip().casefold()
                        in {"total", "aggregate"}
                    ):
                        # The source row explicitly says Total Revenue. Keep
                        # the original metric for audit while exposing the
                        # canonical metric consumed by Binder/renderer.
                        candidate["source_metric"] = metric
                        candidate["metric"] = "Revenue"
                        candidate["normalized_metric"] = "Revenue"
                        candidate["metric_normalization"] = "structured_total_revenue"
            candidates.append(candidate)

        # For an unqualified financial request, a segment row must not win
        # simply because its metric/period text is more lexically similar.
        # Promote explicitly consolidated rows in the packet while retaining
        # every candidate for audit and allowing Binder to fail closed if no
        # safe aggregate exists. Explicit segment queries retain R4 order.
        try:
            frame = extract_query_semantic_frame(request.standalone_query)
        except (TypeError, ValueError):
            frame = None
        explicit_segment_label = False
        if frame is not None and not frame.scope_ids:
            query_text = " ".join(request.standalone_query.casefold().split())
            explicit_segment_label = any(
                classification.scope is EvidenceScope.SEGMENT
                and classification.scope_label
                and " ".join(str(classification.scope_label).casefold().split())
                in query_text
                for item in candidates
                for classification in (
                    classify_evidence_scope(item),
                )
            )
        def entity_match_priority(cand: Mapping[str, Any]) -> int:
            if not frame or not frame.entity_ids:
                return 0
            cand_entity = canonical_entity_id(cand.get("entity")) or str(cand.get("entity") or "").strip().casefold()
            if any(eid in cand_entity for eid in frame.entity_ids):
                return 0
            return 1

        if frame is not None and not frame.scope_ids and not explicit_segment_label:
            scope_priority = {
                EvidenceScope.CONSOLIDATED.value: 0,
                EvidenceScope.UNKNOWN.value: 1,
                EvidenceScope.SEGMENT.value: 2,
            }
            candidates = [
                item
                for _index, item in sorted(
                    enumerate(candidates),
                    key=lambda pair: (
                        entity_match_priority(pair[1]),
                        scope_priority.get(
                            classify_evidence_scope(pair[1]).scope.value,
                            1,
                        ),
                        pair[0],
                    ),
                )
            ]

        # Keep the Binder request bounded after the single-slot expansion.
        # Scope ordering is intentionally performed before this cap so a
        # source-verified aggregate is not discarded behind lexical
        # distractors.
        candidates = candidates[: self.retriever.final_pool_k]

        return R4RetrievalResult(
            candidate_evidence=tuple(candidates),
            candidate_ids=tuple(
                str(item.get("candidate_key"))
                for item in candidates
                if item.get("candidate_key")
            ),
            retrieval_reason=request.reason_code,
            target_slots=request.target_slots,
            retrieval_round=request.retrieval_round,
            source_branch_metadata={
                "policy": "candidate_direct_r4",
                "slot_pool_count": len(slot_pools) if isinstance(slot_pools, Mapping) else 0,
                "candidate_direct_pool_count": len(raw.get("candidate_direct_pool", ())),
                "query": targeted_question,
                "planner_slot_ids": list(
                    _planner_slot_ids_for_targets(
                        request.plan,
                        query_plan,
                        request.target_slots,
                    )
                ),
                "slot_query_variants": {
                    str(slot_id): [str(query) for query in queries]
                    for slot_id, queries in (raw.get("slot_query_variants", {}) or {}).items()
                    if isinstance(queries, Iterable)
                },
            },
        )


class R4RetrievalCapability:
    """TV2-02 RetrievalCapability backed by the existing R4 policy."""

    def __init__(
        self,
        policy: R4Policy,
        *,
        document_scope: Sequence[str] = (),
    ) -> None:
        if not hasattr(policy, "retrieve"):
            raise TypeError("policy must provide retrieve(request)")
        self.policy = policy
        self.document_scope = tuple(str(item) for item in document_scope if str(item))
        self.calls = 0
        self._trace: list[dict[str, Any]] = []
        self.last_result: R4RetrievalResult | None = None

    def retrieve(
        self,
        action: ReplanActionV1,
        state: AdaptiveRAGStateV1,
    ) -> tuple[Mapping[str, Any], ...]:
        raw_plan = state.plan.get("supervisor_plan")
        try:
            plan = SupervisorPlan.from_dict(raw_plan)
        except Exception as exc:
            raise R4RetrievalCapabilityError("invalid_supervisor_plan_for_r4") from exc
        self.calls += 1
        request = R4RetrievalRequest(
            request_id=state.request_id,
            standalone_query=state.normalized_query,
            plan=plan,
            reason_code=action.reason_code.value,
            target_slots=tuple(action.target_slots),
            retrieval_round=self.calls - 1,
            document_scope=self.document_scope,
            constraints=dict(action.constraints),
        )
        result = self.policy.retrieve(request)
        if not isinstance(result, R4RetrievalResult):
            raise R4RetrievalCapabilityError(
                "r4_policy_must_return_R4RetrievalResult"
            )
        candidates = tuple(result.candidate_evidence)
        normalized: list[Mapping[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise R4CandidateSchemaError("r4_candidate_must_be_mapping")
            evidence_id = candidate.get("evidence_id") or candidate.get("fact_id")
            if not evidence_id:
                raise R4CandidateSchemaError("r4_candidate_missing_evidence_id")
            item = dict(candidate)
            item["evidence_id"] = str(evidence_id)
            item.setdefault("fact_id", str(evidence_id))
            normalized.append(item)
        self.last_result = R4RetrievalResult(
            candidate_evidence=tuple(normalized),
            candidate_ids=_stable_unique(
                str(item.get("candidate_id") or item.get("evidence_id"))
                for item in normalized
            ),
            retrieval_reason=result.retrieval_reason,
            target_slots=result.target_slots,
            retrieval_round=result.retrieval_round,
            source_branch_metadata=dict(result.source_branch_metadata),
            result_fingerprint=result.result_fingerprint,
        )
        self._trace.append(
            {
                "round": result.retrieval_round,
                "reason_code": result.retrieval_reason,
                "target_slot_ids": list(result.target_slots),
                "candidate_count": len(normalized),
                "candidate_ids": list(self.last_result.candidate_ids),
                "source_branch_metadata": dict(result.source_branch_metadata),
            }
        )
        return tuple(normalized)

    def trace_snapshot(self) -> dict[str, Any]:
        records = [dict(item) for item in self._trace]
        return {
            "retrieval_rounds": records,
            "candidate_count_per_round": [
                int(item["candidate_count"]) for item in records
            ],
            "candidate_ids_per_round": [
                list(item["candidate_ids"]) for item in records
            ],
            "targeted_slot_ids": [
                slot_id
                for item in records
                for slot_id in item["target_slot_ids"]
            ],
        }


__all__ = [
    "CandidateDirectR4Policy",
    "R4CandidateSchemaError",
    "R4Policy",
    "R4RetrievalCapability",
    "R4RetrievalCapabilityError",
    "R4RetrievalRequest",
    "R4RetrievalResult",
]
