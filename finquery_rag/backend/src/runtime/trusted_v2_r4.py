from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_v2.adaptive import AdaptiveRAGStateV1, ReplanActionV1
from rag_v2.contracts.plan import SupervisorPlan
from src.pdf_retrieval_v4.candidate_direct_retriever import CandidateDirectRetriever
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

    def retrieve(self, request: R4RetrievalRequest) -> R4RetrievalResult:
        self.calls += 1
        target_terms = " ".join(
            f"{slot.metric} {slot.period}"
            for slot in request.plan.required_slots
            if not request.target_slots or slot.slot_id in request.target_slots
        )
        targeted_question = (
            f"{request.standalone_query} {target_terms}".strip()
            if request.target_slots
            else request.standalone_query
        )
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
            for slot_id in request.target_slots:
                values = slot_pools.get(slot_id, ())
                if isinstance(values, Iterable):
                    pool.extend(
                        item
                        if isinstance(item, Mapping)
                        else {"candidate_key": getattr(item, "candidate_key", "")}
                        for item in values
                    )
        if not pool:
            values = raw.get("candidate_direct_pool", ())
            if isinstance(values, Iterable):
                pool.extend(
                    item
                    if isinstance(item, Mapping)
                    else {"candidate_key": getattr(item, "candidate_key", "")}
                    for item in values
                )

        candidate_ids = _stable_unique(
            str(item.get("candidate_key", ""))
            for item in pool
            if isinstance(item, Mapping)
        )
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
            candidates.append(candidate)

        return R4RetrievalResult(
            candidate_evidence=tuple(candidates),
            candidate_ids=tuple(str(item) for item in candidate_ids),
            retrieval_reason=request.reason_code,
            target_slots=request.target_slots,
            retrieval_round=request.retrieval_round,
            source_branch_metadata={
                "policy": "candidate_direct_r4",
                "slot_pool_count": len(slot_pools) if isinstance(slot_pools, Mapping) else 0,
                "candidate_direct_pool_count": len(raw.get("candidate_direct_pool", ())),
                "query": targeted_question,
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
