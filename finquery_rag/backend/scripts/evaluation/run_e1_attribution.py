"""E1 final attribution: three arms, one pinned plan, one question at a time.

The published end-to-end chain releases 5 of 95 answerable questions, and R0
showed retrieval can put gold in the top 20 for 92% of them.  Those cannot both
describe the same pipeline, and the first in-process run found why: 92 of 120
questions are refused at `terminal_state='PLAN'` by the semantic alignment gate,
before any retrieval happens.  Only 28 ever search.

This measures the three stages that separate, with everything else held fixed.

  A  PINNED_PRODUCTION    gate ON, shipped retriever, replayed plan
  B  GATE_BYPASS_CONTROL  gate overridden, shipped retriever, same plan
  C  SLOT_RETRIEVAL       gate overridden, SlotBudgetRetriever, same plan

  A -> B  the semantic gate's capability cost
  B -> C  slot-aware retrieval's effect once the gate is out of the way

**B and C are diagnostics, not production results.**  They run with a gate that
has been overridden, and the coordinator records the *computed* verdict beside
the effective one (`computed_status`, `computed_allowed`, `computed_mismatches`,
`computed_unknown_query_fields`), so "the gate allowed this" stays separable from
"we said it did".  Every case row carries both.  Quoting a B or C number as what
the system does would be quoting a configuration the system does not run.

**A is the baseline, and it is reported against two different things.**  The
published 5 releases came from a live planner; this arm replays a frozen one, and
the planner is only 82.5% stable on slot identity (E1-0).  So A's release count
is the pinned-plan baseline, and its difference from 5 is the pinning cost --
not a regression and not noise to average away.

Everything else is identical across arms: the store, the pinned plan, the
Binder, admission, the validator, the finalizer, budgets, thresholds, and the
document scope (none -- `document_names=[]`, which is what the shipped HTTP path
resolves to and was the first confound found here).

Artifacts are case-level; every aggregate below is recomputed from them.

  TRUSTED_V2_SPECIALIST_DEVICE=cpu .venv/bin/python run_e1_attribution.py \\
      --out-dir /disk/qh/nano-finrag/artifacts/evaluation/e1-attribution
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _path in (str(_BACKEND_DIR), str(_BACKEND_DIR / "scripts" / "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

BENCH_DIR = _BACKEND_DIR / "benchmarks" / "tv2_canonical_v1"

#: Reason codes the semantic gate emits.  Kept apart from `PLAN_WRONG_METRIC`:
#: the planner asking for the wrong thing and the ontology not knowing the thing
#: are claims about different layers, and conflating them would hide which one
#: needs the work.
GATE_REASONS = {
    "QUERY_PLAN_SEMANTIC_MISMATCH": "SEMANTIC_MISMATCH",
    "QUERY_PLAN_SEMANTIC_AMBIGUOUS": "SEMANTIC_AMBIGUOUS",
    "QUERY_PLAN_SEMANTIC_UNKNOWN": "SEMANTIC_UNKNOWN",
    "SUPERVISOR_ABSTAIN": "SUPERVISOR_ABSTAIN",
    "INVALID_PLAN": "INVALID_PLAN",
}

ARMS: tuple[tuple[str, str], ...] = (
    ("A_pinned_production", "gate ON,   shipped retriever, pinned plan"),
    ("B_gate_bypass", "gate OVER, shipped retriever, pinned plan"),
    ("C_slot_retrieval", "gate OVER, SlotBudgetRetriever, pinned plan"),
)


class MissingPinnedPlan(RuntimeError):
    """Raised when a question reaches the supervisor with no pinned plan.

    Loud on purpose.  A missing plan must not become an empty one: an empty
    plan is a valid-looking plan with no slots, and every downstream stage
    would report a clean zero for a case that was never actually asked.
    """


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


# ---------------------------------------------------------------------------
# The pinned supervisor
# ---------------------------------------------------------------------------


def make_replay_supervisor(plans_by_question: Mapping[str, Mapping[str, Any]]):
    """A SupervisorService that serves the pinned plan instead of calling a model.

    A *subclass*, not a duck type: `build_trusted_v2_runtime` asserts
    `isinstance(supervisor, SupervisorService)`, and loosening that to admit a
    stand-in would change the production builder for the sake of an experiment.

    The pinned plan is the already-normalised, already-validated output of a real
    supervisor run, so it is served verbatim rather than re-normalised --
    re-normalising an already-normalised plan is a second transformation the
    production path does not apply.  `validate_plan_v2_01` is still run, because
    that one production does apply to every plan, and a pinned plan that no
    longer validates should stop the run rather than be served on trust.
    """
    from rag_v2.contracts.plan import SupervisorPlan
    from rag_v2.supervisor.service import SupervisorRun, SupervisorService
    from rag_v2.supervisor.plan_validator import validate_plan_v2_01

    frozen = {question: SupervisorPlan.from_dict(payload)
              for question, payload in plans_by_question.items()}

    class ReplaySupervisor(SupervisorService):
        def __init__(self) -> None:
            self.calls = 0
            self.served: list[str] = []

        def plan(self, question: str) -> SupervisorRun:
            self.calls += 1
            plan = frozen.get(question)
            if plan is None:
                raise MissingPinnedPlan(question)
            validate_plan_v2_01(plan)
            self.served.append(question)
            return SupervisorRun(question, plan, True, None, None)

    return ReplaySupervisor()


# ---------------------------------------------------------------------------
# The diagnostic gate override
# ---------------------------------------------------------------------------


def make_gate_override(question: str, plan: Any):
    """The production gate, with its verdict forced to ALIGNED.

    Built from the real `align_query_to_plan` rather than hand-assembled, so the
    object the coordinator receives is the production type carrying the
    production fields.  `allowed` is a property derived from `status`, so
    forcing the status is what makes the gate pass -- and the coordinator still
    records what the strict computation said, which is the whole reason this is
    reportable rather than a lie.
    """
    from rag_v2.supervisor.semantic_alignment import (
        SemanticAlignmentStatus,
        UnknownSemanticPolicy,
        align_query_to_plan,
    )

    computed = align_query_to_plan(
        question, plan, unknown_policy=UnknownSemanticPolicy.STRICT_DIRECT_FACT)
    return dataclasses.replace(computed, status=SemanticAlignmentStatus.ALIGNED)


# ---------------------------------------------------------------------------
# Arm C's retriever
# ---------------------------------------------------------------------------


def _slot_variants(request: Any) -> list[str]:
    """Every variant carries this slot's own entity, metric and period.

    The shipped query drops the entity on purpose -- its docstring records that
    an entity in an OR-tokenised query matches every row for the company and
    dilutes the specific fact out of the lane.  This arm puts it back, which is
    the variable under test, and adds the metric aliases.
    """
    from src.pdf_retrieval_v4.candidate_query_builder import _metric_aliases

    entity = getattr(request, "entity", None)
    period = getattr(request, "period", None)

    def render(term: str) -> str:
        parts = (entity, term, period)
        return " ".join(str(p).strip() for p in parts if p and str(p).strip())

    terms = [request.metric, *_metric_aliases(request.metric)]
    variants: list[str] = []
    for term in terms:
        text = render(term)
        if text and text not in variants:
            variants.append(text)
    return variants or [request.query]


def make_slot_budget_retriever(reader: Any):
    from src.pdf_retrieval_v4.candidate_direct_retriever import CandidateDirectRetriever
    from src.pdf_retrieval_v4.candidate_rrf import fuse_candidate_hits
    from src.pdf_retrieval_v4.candidate_slot_pool import build_slot_pool
    from src.pdf_retrieval_v4.candidate_view_index import (
        LANES,
        CandidateSearchHit,
    )

    class SlotBudgetRetriever(CandidateDirectRetriever):
        """Slot-local queries at an equal retrieved-candidate budget.

        One query per slot retrieves `lane_k` candidates per lane.  This arm
        searches N variants per slot at `lane_k // N`, so it looks at the same
        number of candidates and only differs in what it asked for.  Measured in
        E2: a gain that survives that budget is a gain from the queries and not
        from looking at more.
        """

        def retrieve_for_requests(
            self,
            requests: Sequence[Any],
            *,
            document_scope: set[str],
            total_k: int | None = None,
            slot_top_k: int | None = None,
            alias_expansion: bool = False,
        ) -> dict[str, Any]:
            allowed_keys = self._allowed_keys_for_scope(document_scope)
            slot_pools: dict[str, Any] = {}
            slot_queries: dict[str, list[str]] = {}

            for request in requests:
                variants = _slot_variants(request)
                slot_queries[request.slot_id] = list(variants)
                depth = max(1, self.lane_k // len(variants))
                per_lane: dict[str, list[CandidateSearchHit]] = {}
                for lane in LANES:
                    best: dict[str, int] = {}
                    order: list[str] = []
                    for query in variants:
                        for rank, hit in enumerate(
                            self.reader.search(
                                lane, query,
                                allowed_candidate_keys=allowed_keys[lane], k=depth),
                            1,
                        ):
                            key = hit.candidate_key
                            if not key:
                                continue
                            if key not in best:
                                best[key] = rank
                                order.append(key)
                            elif rank < best[key]:
                                best[key] = rank
                    per_lane[lane] = [
                        CandidateSearchHit(
                            candidate_key=key, view_id=key, lane=lane,
                            bm25_rank=best[key], dense_rank=None,
                            bm25_score=None, dense_score=None)
                        for key in sorted(order, key=lambda k: best[k])
                    ]
                slot_pools[request.slot_id] = fuse_candidate_hits(
                    per_lane, rrf_k=self.rrf_k)

            if not slot_pools:
                return {"candidate_direct_pool": [], "lane_hits": {}, "rrf_hits": [],
                        "slot_pools": {}, "slot_query_variants": {}}

            if len(slot_pools) == 1:
                pool = self._pool_from_rrf(next(iter(slot_pools.values())))
            else:
                merge_kwargs: dict[str, Any] = {
                    "total_k": total_k or max(80, self.final_pool_k * 2)}
                if slot_top_k is not None:
                    merge_kwargs["slot_top_k"] = slot_top_k
                pool = build_slot_pool(slot_pools, **merge_kwargs)

            return {"candidate_direct_pool": pool, "lane_hits": {}, "rrf_hits": [],
                    "slot_pools": slot_pools, "slot_query_variants": slot_queries}

    return SlotBudgetRetriever(reader)


# ---------------------------------------------------------------------------
# One case, one arm
# ---------------------------------------------------------------------------


def run_case(case_id, question, plan_payload, arm, resources, gold_ids, comparable):
    from src.runtime.query_lifecycle import QueryExecutionService
    from src.runtime.runtime_contract import FinancialQueryRequest
    from src.runtime.trusted_v2_production import build_trusted_v2_runtime_for_request

    from rag_v2.contracts.plan import SupervisorPlan

    request = FinancialQueryRequest(
        request_id=f"e1-{arm}-{case_id}",
        user_id="e1",
        session_id=f"__stateless__:e1-{arm}-{case_id}",
        original_query=question["question"],
        standalone_query=question["question"],
        query_as_resolved=True,
        conversation_metadata={},
        request_metadata={
            "document_names": [],   # the shipped path resolves to no scope
            "n_results": 5,
            "conversation_history": None,
            "memory_profile": None,
        },
    )

    gate_on = arm == "A_pinned_production"
    override = (None if gate_on
                else make_gate_override(question["question"],
                                        SupervisorPlan.from_dict(plan_payload)))
    factory = (make_slot_budget_retriever if arm == "C_slot_retrieval" else None)

    runtime = build_trusted_v2_runtime_for_request(
        None, request, resources=resources,
        alignment_override=override, retriever_factory=factory)
    result = asyncio.run(QueryExecutionService(runtime).execute(request))
    trace = (getattr(result, "debug_metadata", None) or {}).get("trace") or {}

    pool: list[str] = []
    for round_ids in (trace.get("candidate_ids_per_round") or []):
        pool.extend(str(item) for item in (round_ids or []))
    pool = _dedupe(pool)
    bound = _dedupe([str(x) for x in (trace.get("bound_evidence_ids") or [])])
    gold = set(gold_ids) if comparable else set()

    alignment = trace.get("semantic_alignment") or {}
    reasons = [str(r) for r in (trace.get("reason_codes") or result.reason_codes or [])]
    gate_code = next((GATE_REASONS[r] for r in reasons if r in GATE_REASONS), None)

    return {
        "case_id": case_id,
        "stratum": question.get("stratum"),
        "arm": arm,
        "question": question["question"],
        "comparable": comparable,
        "gold_ids": sorted(gold),
        "plan_slots": [slot.get("slot_id") for slot in
                       (plan_payload.get("required_slots") or [])],
        "plan_metrics": [slot.get("metric") for slot in
                         (plan_payload.get("required_slots") or [])],
        # -- gate -----------------------------------------------------------
        "gate_blocked": gate_code is not None,
        "gate_code": gate_code,
        "gate_reason_raw": next((r for r in reasons if r in GATE_REASONS), None),
        "effective_allowed": alignment.get("allowed"),
        "computed_status": alignment.get("computed_status") or alignment.get("status"),
        "computed_allowed": alignment.get("computed_allowed"),
        "computed_mismatches": list(alignment.get("computed_mismatches") or []),
        "computed_unknown_query_fields": list(
            alignment.get("computed_unknown_query_fields")
            or alignment.get("unknown_query_fields") or []),
        "overridden": not gate_on,
        # -- funnel ---------------------------------------------------------
        "terminal_state": trace.get("terminal_state"),
        "reason_codes": reasons,
        "retrieval_rounds": len(trace.get("retrieval_rounds") or []),
        "pool_size": len(pool),
        "gold_in_pool": len(gold & set(pool)),
        "binder_status_per_round": [str(x) for x in
                                    (trace.get("binder_status_per_round") or [])],
        "bound_evidence": len(bound),
        "gold_bound": len(gold & set(bound)),
        "missing_slot_ids": list(trace.get("missing_slot_ids") or []),
        "wrong_period_slots": list(trace.get("wrong_period_slots") or []),
        "missing_operand_slots": list(trace.get("missing_operand_slots") or []),
        "conflict_ids": list(trace.get("conflict_ids") or []),
        "validation_passed": bool(trace.get("validation_passed")),
        "validation_reason_codes": [str(x) for x in
                                    (trace.get("validation_reason_codes") or [])],
        "failed_checks": [str(x) for x in (trace.get("failed_checks") or [])],
        "release_status": trace.get("release_status") or result.release_status,
        "released": (trace.get("release_status") or result.release_status) == "RELEASED",
    }


def _funnel(rows: Sequence[dict]) -> dict[str, Any]:
    """Stage counts over the comparable answerable cases, recomputed from rows."""
    cases = [r for r in rows if r["comparable"] and not r.get("must_refuse")]
    total = len(cases)
    reached = [r for r in cases if r["retrieval_rounds"] or r["pool_size"]]
    in_pool = [r for r in reached if r["gold_in_pool"]]
    bound = [r for r in reached if r["gold_bound"]]
    complete = [r for r in reached if not r["missing_slot_ids"]]
    return {
        "cases": total,
        "gate_blocked": sum(1 for r in cases if r["gate_blocked"]),
        "reached_retrieval": len(reached),
        "gold_in_pool": len(in_pool),
        "gold_bound": len(bound),
        "slots_complete": len(complete),
        "validation_passed": sum(1 for r in reached if r["validation_passed"]),
        "released": sum(1 for r in cases if r["released"]),
        "p_released_given_in_pool": (
            round(sum(1 for r in in_pool if r["released"]) / len(in_pool), 4)
            if in_pool else None),
        "p_released_given_bound": (
            round(sum(1 for r in bound if r["released"]) / len(bound), 4)
            if bound else None),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=BENCH_DIR / "canonical-eval-v1.jsonl")
    parser.add_argument("--gold-evidence", type=Path,
                        default=BENCH_DIR / "gold-evidence-v1.jsonl")
    parser.add_argument("--pinned", type=Path,
                        default=Path("/disk/qh/nano-finrag/artifacts/evaluation/"
                                     "e1-pinned-plans/pinned-plans-v1.jsonl"))
    parser.add_argument("--v2-fact-store", type=Path,
                        default=Path("/disk/qh/nano-finrag/data/trusted-v2/fact-store/"
                                     "financial-facts.jsonl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arms", default="A,B,C")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    import run_p1_2_dual_track_benchmark as runner

    runner._load_deployment_env()

    from src.runtime.trusted_v2_production import _cached_resources

    questions = [json.loads(line)
                 for line in args.eval_set.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    gold_by_id = {row["id"]: row for row in
                  (json.loads(line) for line in
                   args.gold_evidence.read_text(encoding="utf-8").splitlines()
                   if line.strip())}
    pinned = [json.loads(line) for line in
              args.pinned.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[: args.limit]
        keep = {q["id"] for q in questions}
        pinned = [row for row in pinned if row["case_id"] in keep]

    plans_by_question: dict[str, Mapping[str, Any]] = {}
    for row in pinned:
        if row.get("plan"):
            plans_by_question[row["question"]] = row["plan"]

    wanted = [name for name, _ in ARMS if name.split("_")[0] in args.arms.split(",")]

    print("=" * 78)
    print("E1 FINAL ATTRIBUTION")
    print("=" * 78)
    print(f"  questions     {len(questions)}")
    print(f"  pinned plans  {len(plans_by_question)}")
    print(f"  arms          {wanted}")
    print(f"  device        {os.environ.get('TRUSTED_V2_SPECIALIST_DEVICE', '<unset>')}")
    print()
    if len(plans_by_question) < len({q["question"] for q in questions}):
        print("  !! some questions have no pinned plan; those cases will fail loudly")
        print()

    resources = _cached_resources(dict(os.environ))
    resources = dataclasses.replace(
        resources, supervisor=make_replay_supervisor(plans_by_question))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    all_rows: dict[str, list[dict]] = {}

    for arm in wanted:
        rows: list[dict] = []
        for position, question in enumerate(questions, 1):
            case_id = question["id"]
            entry = next((row for row in pinned if row["case_id"] == case_id), None)
            record = gold_by_id.get(case_id) or {}
            raw_ids = [str(f) for f in (record.get("fact_ids") or [])]
            comparable = bool(raw_ids) and all(f.startswith("v2fact:") for f in raw_ids)
            try:
                row = run_case(case_id, question, (entry or {}).get("plan") or {},
                               arm, resources, raw_ids, comparable)
            except MissingPinnedPlan as exc:
                row = {"case_id": case_id, "arm": arm, "stratum": question.get("stratum"),
                       "comparable": comparable, "gold_ids": raw_ids,
                       "error": f"MISSING_PINNED_PLAN: {exc}", "pool_size": 0,
                       "retrieval_rounds": 0, "gold_in_pool": 0, "gold_bound": 0,
                       "missing_slot_ids": [], "conflict_ids": [],
                       "validation_passed": False, "validation_reason_codes": [],
                       "failed_checks": [], "released": False,
                       "release_status": None, "terminal_state": "NO_PINNED_PLAN",
                       "reason_codes": [], "gate_blocked": False, "gate_code": None,
                       "bound_evidence": 0, "binder_status_per_round": [],
                       "wrong_period_slots": [], "missing_operand_slots": [],
                       "overridden": arm != "A_pinned_production",
                       "effective_allowed": None, "computed_status": None,
                       "computed_allowed": None, "computed_mismatches": [],
                       "computed_unknown_query_fields": [], "plan_slots": [],
                       "plan_metrics": [], "question": question["question"],
                       "gate_reason_raw": None}
            except Exception as exc:
                # The builder wraps its real failure in a generic
                # "could not build the request-scoped Trusted V2 runtime graph",
                # which is why the cause chain is walked here: the message that
                # says *what* was wrong survives only in `__cause__`.
                detail = f"{type(exc).__name__}: {exc}"
                cause = exc.__cause__
                while cause is not None:
                    detail += f"  <- {type(cause).__name__}: {cause}"
                    cause = cause.__cause__
                row = {"case_id": case_id, "arm": arm, "stratum": question.get("stratum"),
                       "comparable": comparable, "gold_ids": raw_ids,
                       "error": detail, "pool_size": 0,
                       "retrieval_rounds": 0, "gold_in_pool": 0, "gold_bound": 0,
                       "missing_slot_ids": [], "conflict_ids": [],
                       "validation_passed": False, "validation_reason_codes": [],
                       "failed_checks": [], "released": False,
                       "release_status": None, "terminal_state": "HARNESS_EXCEPTION",
                       "reason_codes": [], "gate_blocked": False, "gate_code": None,
                       "bound_evidence": 0, "binder_status_per_round": [],
                       "wrong_period_slots": [], "missing_operand_slots": [],
                       "overridden": arm != "A_pinned_production",
                       "effective_allowed": None, "computed_status": None,
                       "computed_allowed": None, "computed_mismatches": [],
                       "computed_unknown_query_fields": [], "plan_slots": [],
                       "plan_metrics": [], "question": question["question"],
                       "gate_reason_raw": None}
            row["must_refuse"] = bool(record.get("expected_failure_reason"))
            rows.append(row)
            if position % 20 == 0 or position == len(questions):
                print(f"    [{arm}] {position:>3}/{len(questions)} "
                      f"{time.perf_counter() - started:.0f}s", flush=True)
        all_rows[arm] = rows
        (args.out_dir / f"{arm}-cases.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                    for r in rows),
            encoding="utf-8", newline="\n")
        print(f"  {arm}: written {len(rows)} case rows", flush=True)
    print()

    # -- aggregates, recomputed from the case rows ---------------------------
    print("=" * 78)
    print("FUNNEL -- 75 comparable answerable cases")
    print("=" * 78)
    print(f"    {'stage':24}" + "".join(f"{arm.split('_')[0]:>12}" for arm in wanted))
    summary: dict[str, Any] = {}
    funnels = {arm: _funnel(all_rows[arm]) for arm in wanted}
    summary["funnels"] = funnels
    stages = ("cases", "gate_blocked", "reached_retrieval", "gold_in_pool",
              "gold_bound", "slots_complete", "validation_passed", "released")
    for stage in stages:
        print(f"    {stage:24}" + "".join(f"{funnels[a][stage]:>12}" for a in wanted))
    print()
    for arm in wanted:
        f = funnels[arm]
        print(f"    {arm:22} P(release|gold in pool) {f['p_released_given_in_pool']}"
              f"   P(release|gold bound) {f['p_released_given_bound']}")
    print()

    errors = collections.Counter(
        str(r.get("error", "")).split(":")[0] for a in wanted for r in all_rows[a]
        if r.get("error"))
    if errors:
        print(f"  harness errors: {dict(errors)}")
        print()

    print("  Arm A is the baseline.  Arms B and C run with the semantic gate")
    print("  overridden and are diagnostics: their numbers describe a configuration")
    print("  the system does not run.  `computed_status` on every case row is what")
    print("  the real gate said; `overridden` marks which rows were forced.")
    print()

    (args.out_dir / "e1-attribution-summary.json").write_text(
        json.dumps({"funnels": funnels,
                    "errors": dict(errors),
                    "arm_notes": {a: label for a, label in ARMS},
                    "pinned_plans": len(plans_by_question)},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    print(f"  written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
