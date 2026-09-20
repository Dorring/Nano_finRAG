"""NF-V3: retrieval Recall@K benchmark for the financial RAG system.

Measures, over the 120-case canonical evaluation set, how far the R4 four-lane
candidate index gets the evidence the gold says is required.  Six things are
reported, always separately and never merged into one number:

  Gold Evidence Recall@K          fraction of the case's gold ``fact_ids`` in top-K
  Gold Evidence Complete Recall@K the same, but a case counts only if ALL its
                                  gold ids are present
  Required-Slot Recall@K          fraction of the case's required slots whose own
                                  coordinate is retrieved
  Multi-Evidence Complete Recall@K cases needing N>=2 operands, hit only if all N
                                  slots are retrieved (a 2/3 is not a hit)
  Cross-Entity Slot Recall@K      Required-Slot Recall restricted to
                                  ``cross_entity_comparison``
  Retrieval-only MRR              rank of the first gold id in the fused list

Every one is computed per lane, for the 4-lane RRF hybrid, and (when a reranker
is reachable) for the reranked hybrid.  Every one is stratified by the four
strata plus an overall row.

**Gold never influences retrieval.**  The split is structural, not conventional:
``retrieve_case`` receives a question string and nothing else, returns a
``CaseRetrieval``, and is called for every case before ``score_case`` -- which
is the only function that has ever seen a gold record -- is called at all.  The
retrieval stage runs to completion first; scoring replays its output.

The one thing this script will not do is score against gold that does not exist
in the index.  ``--require-id-space`` (default) prints the resolution table
before any metric is computed: how many of the 120 cases have gold that resolves
to a candidate, and how many of those candidates the R4 index actually contains.
A stratum whose gold is in a different id space cannot be scored at all, and is
reported as STRUCTURALLY_UNMEASURABLE rather than as a retrieval failure -- a
zero there would be a fact about identifiers, not about retrieval.

  .venv/bin/python scripts/evaluation/run_nf_v3_retrieval_benchmark.py \\
      --out-dir /disk/qh/nano-finrag/artifacts/evaluation/nf-v3-retrieval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _path in (str(_BACKEND_DIR), str(_BACKEND_DIR / "scripts" / "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

BENCH_DIR = _BACKEND_DIR / "benchmarks" / "tv2_canonical_v1"

KS: tuple[int, ...] = (5, 10, 20)

#: Strata, in report order.  The overall row is appended after these.
STRATA: tuple[str, ...] = (
    "factual_lookup",
    "arithmetic_calculation",
    "cross_entity_comparison",
    "adversarial_abstention",
)

#: The id-space prefixes this benchmark knows how to resolve.
V2FACT_PREFIX = "v2fact:"
IXBRL_PREFIX = "ixbrl:"

#: Lane order, matching ``CandidateViewIndexReader.LANES``.
LANES: tuple[str, ...] = (
    "candidate_raw_bm25",
    "candidate_raw_dense",
    "candidate_structured_bm25",
    "candidate_structured_dense",
)

HYBRID = "rrf_hybrid"
HYBRID_RERANKED = "rrf_hybrid_reranked"

#: Retrieval configurations reported, in report order.
RETRIEVERS: tuple[str, ...] = LANES + (HYBRID, HYBRID_RERANKED)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Gold resolution -- the id-space question, answered before anything is scored
# ---------------------------------------------------------------------------


@dataclass
class Resolution:
    """Where one gold identifier ended up."""

    raw: str
    candidate_key: str | None
    id_space: str  # "v2fact" | "ixbrl" | "unresolved"
    resolved_by: str  # "candidate_key" | "alias" | "semantic" | "none"
    indexed: bool  # present in the R4 candidate index


class GoldResolver:
    """Resolve gold ``fact_ids`` to the candidate keys the R4 index is keyed on.

    Two stores, two id spaces, and they do not overlap.  The R4 index was built
    from ``financial-facts.jsonl`` (``v2fact:`` keys); the iXBRL store is a
    separate corpus keyed ``ixbrl:``.  This class resolves into either and says
    which, so a stratum that lands outside the index is visible as such instead
    of silently scoring zero.
    """

    def __init__(
        self,
        v2_index: Any,
        ixbrl_candidate_keys: set[str],
        index_candidate_keys: set[str],
    ) -> None:
        self.v2_index = v2_index
        self.ixbrl_candidate_keys = ixbrl_candidate_keys
        self.index_candidate_keys = index_candidate_keys
        # Keyed on the whole question, not just the id: the last-resort semantic
        # step reads (document_id, metric, period), so one raw id asked in two
        # different coordinates is two different questions.
        self._cache: dict[tuple[str, str, str, str], Resolution] = {}

    def resolve(
        self,
        raw: str,
        *,
        document_id: str | None = None,
        metric: str | None = None,
        period: str | None = None,
    ) -> Resolution:
        cache_key = (raw, str(document_id or ""), str(metric or ""), str(period or ""))
        if cache_key in self._cache:
            return self._cache[cache_key]
        result = self._resolve(raw, document_id, metric, period)
        self._cache[cache_key] = result
        return result

    def _resolve(
        self,
        raw: str,
        document_id: str | None,
        metric: str | None,
        period: str | None,
    ) -> Resolution:
        key: str | None = None
        via = "none"

        # 1. The identifier is already a candidate key of one of the two stores.
        if raw.startswith(IXBRL_PREFIX) and raw in self.ixbrl_candidate_keys:
            key, via = raw, "candidate_key"
        elif raw.startswith(V2FACT_PREFIX) and raw in self.v2_index.fact_alias_map:
            key, via = raw, "candidate_key"
        else:
            # 2. The runtime's own alias map: fact_id / evidence_id / citation_id
            #    and the bare hex suffix all collapse onto the candidate key.
            #    Reused verbatim from ``FactStoreGroundingIndex``.
            candidate = self.v2_index.resolve(raw)
            if candidate and candidate in self.v2_index.fact_alias_map:
                key, via = candidate, "alias"
            elif raw in self.ixbrl_candidate_keys:
                key, via = raw, "candidate_key"
            else:
                # 3. Last resort, and only the v2 store has the fields for it:
                #    the (document_id, metric, period) the gold names.
                if document_id and metric and period:
                    semantic = self.v2_index.semantic_fact_map.get(
                        (document_id, str(metric).lower(), str(period).upper())
                    )
                    if semantic:
                        key, via = semantic, "semantic"

        if key is None:
            return Resolution(raw, None, "unresolved", "none", False)
        id_space = "ixbrl" if key.startswith(IXBRL_PREFIX) else "v2fact"
        return Resolution(raw, key, id_space, via, key in self.index_candidate_keys)


def load_ixbrl_candidate_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for field_name in ("candidate_key", "fact_id"):
                value = record.get(field_name)
                if value:
                    keys.add(str(value))
    return keys


# ---------------------------------------------------------------------------
# Slot coordinates
# ---------------------------------------------------------------------------


def slot_coordinate_candidates(
    slot: Mapping[str, Any],
    coordinate_index: Mapping[tuple[tuple[str, ...], tuple[str, ...]], list[str]],
) -> list[str]:
    """The facts a slot's own coordinate describes, in the *fact_id* space.

    Shares ``build_p1_2_plan_fixtures``' coordinate folding so a slot selects
    exactly the facts the fixture builder counted.  The count is a fixture
    field; the *identifiers* are what a retrieval hit is tested against, and the
    fixtures do not carry them.

    Note the id space: the builder indexes on ``fact_id`` (``atomic:<hex>``),
    while the retrieval index is keyed on ``candidate_key`` (``v2fact:<hex>``).
    These are the same facts under two names, so every caller must pass the
    result through :func:`coordinate_candidate_keys` before comparing it with a
    retrieved candidate -- comparing the two directly is a comparison of
    identifiers, and it silently yields zero.
    """
    import build_p1_2_plan_fixtures as fixtures_builder

    return list(fixtures_builder._slot_candidates(slot, coordinate_index))


def coordinate_candidate_keys(raw_ids: Iterable[str], resolver: "GoldResolver") -> list[str]:
    """Translate coordinate ``fact_id``s into indexed ``candidate_key``s."""
    keys: list[str] = []
    for raw in raw_ids:
        resolution = resolver.resolve(str(raw))
        if resolution.candidate_key and resolution.indexed:
            keys.append(resolution.candidate_key)
    return list(dict.fromkeys(keys))


# ---------------------------------------------------------------------------
# Retrieval -- takes a question, returns candidates, cannot see gold
# ---------------------------------------------------------------------------


@dataclass
class CaseRetrieval:
    """Every ranked list produced for one question, and nothing else."""

    case_id: str
    query: str
    lane_keys: dict[str, list[str]] = field(default_factory=dict)
    fused_keys: list[str] = field(default_factory=list)
    reranked_keys: list[str] = field(default_factory=list)
    top5: list[str] = field(default_factory=list)

    def ranked(self, retriever: str) -> list[str]:
        if retriever in self.lane_keys:
            return self.lane_keys[retriever]
        if retriever == HYBRID:
            return self.fused_keys
        if retriever == HYBRID_RERANKED:
            return self.reranked_keys
        raise KeyError(retriever)


def _dedupe(keys: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _rerank_chunks(
    reader: Any,
    fused_keys: Sequence[str],
    supporting_view_ids: Mapping[str, Mapping[str, str]],
    fused_scores: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Materialise fused candidates into the dict shape the reranker expects.

    Text comes from a lane that already retrieved the candidate, so this adds no
    retrieval signal of its own -- it only hands the reranker the same evidence
    the fusion ranked.
    """

    def _lane_for(key: str) -> str | None:
        lanes = supporting_view_ids.get(key) or {}
        for lane in LANES:
            if lane in lanes:
                return lane
        return None

    chunks: list[dict[str, Any]] = []
    for key in fused_keys:
        lane = _lane_for(key)
        view = reader.view_for_candidate(lane, key) if lane else None
        metadata: dict[str, Any] = {}
        if view and view.get("metadata_json"):
            try:
                metadata = json.loads(view["metadata_json"]) or {}
            except (TypeError, json.JSONDecodeError):
                metadata = {}
        chunks.append(
            {
                "chunk_id": key,
                "doc_id": key,
                "content": (view or {}).get("retrieval_text", ""),
                "score": float(fused_scores.get(key, 0.0)),
                "metadata": {
                    "doc_name": (view or {}).get("document_id"),
                    "page": metadata.get("page") or metadata.get("pdf_page"),
                },
            }
        )
    return chunks


def retrieve_case(
    question: str,
    *,
    retriever: Any,
    reader: Any,
    allowed_keys: Mapping[str, set[str] | None],
    reranker: Any,
    rerank_depth: int,
) -> CaseRetrieval:
    """Retrieve for one question.  The signature is the guarantee: no gold.

    There is no gold parameter to pass and no store to consult, so no gold can
    reach the ranking even by accident.  Scoring happens in a second pass over
    the returned object.
    """

    from src.pdf_retrieval_v4.candidate_query_builder import build_all_queries
    from src.pdf_retrieval_v4.candidate_rrf import fuse_candidate_hits
    from src.pdf_retrieval_v4.planner import build_query_plan

    plan = build_query_plan(question, ())
    queries = build_all_queries(plan)
    raw_query = queries["raw_question"][0] if queries.get("raw_question") else question

    lane_hits = retriever._search_lanes(raw_query, dict(allowed_keys))
    lane_keys = {lane: _dedupe(hit.candidate_key for hit in lane_hits.get(lane, [])) for lane in LANES}

    fused = fuse_candidate_hits(lane_hits, rrf_k=retriever.rrf_k)
    fused_keys = _dedupe(hit.candidate_key for hit in fused)
    fused_scores = {hit.candidate_key: hit.rrf_score for hit in fused}
    supporting = {hit.candidate_key: hit.supporting_view_ids for hit in fused}

    reranked_keys: list[str] = []
    if reranker is not None:
        head = fused_keys[:rerank_depth]
        chunks = _rerank_chunks(reader, head, supporting, fused_scores)
        reranked_keys = _dedupe(
            chunk["chunk_id"] for chunk in reranker.rerank(raw_query, chunks)
        )

    return CaseRetrieval(
        case_id="",
        query=raw_query,
        lane_keys=lane_keys,
        fused_keys=fused_keys,
        reranked_keys=reranked_keys,
        top5=fused_keys[:5],
    )


# ---------------------------------------------------------------------------
# Scoring -- the only place gold is read
# ---------------------------------------------------------------------------


@dataclass
class CaseScore:
    case_id: str
    stratum: str
    ranked_lists: dict[str, list[str]]
    gold_ids: list[str]
    slot_targets: list[list[str]]
    slot_id_spaces: list[str]
    strata_searchable: bool


def _gold_hits(gold_ids: Sequence[str], ranked: Sequence[str], k: int) -> tuple[float, bool]:
    """(partial recall, complete recall) for one case at one K.

    Uses ``src.evaluation.metrics.recall_at_k`` by giving it ``ExpectedSource``
    objects keyed on the candidate ids and candidate dicts carrying the same
    field -- the module's own matcher, not a reimplementation of it.
    """
    from src.evaluation.metrics import recall_at_k
    from src.evaluation.schemas import ExpectedSource

    expected = [ExpectedSource(chunk_id=str(g)) for g in gold_ids]
    retrieved = [{"chunk_id": str(c)} for c in ranked[:k]]
    partial = recall_at_k(expected, retrieved, k)
    top = set(retrieved[i]["chunk_id"] for i in range(len(retrieved)))
    complete = all(str(g) in top for g in gold_ids)
    return partial, complete


def _slot_hits(slot_targets: Sequence[Sequence[str]], ranked: Sequence[str], k: int) -> float:
    """Fraction of slots whose own coordinate is in the top-K."""
    if not slot_targets:
        return 0.0
    top = set(str(c) for c in ranked[:k])
    hits = sum(
        1 for targets in slot_targets if targets and any(str(t) in top for t in targets)
    )
    return hits / len(slot_targets)


def _mrr(gold_ids: Sequence[str], ranked: Sequence[str]) -> float:
    from src.evaluation.metrics import mrr
    from src.evaluation.schemas import ExpectedSource

    return mrr(
        [ExpectedSource(chunk_id=str(g)) for g in gold_ids],
        [{"chunk_id": str(c)} for c in ranked],
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(
    scores: Sequence[CaseScore], retrievers: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Metrics per (stratum, retriever).  ``overall`` pools every stratum.

    Cases whose gold is unindexable are excluded from gold-evidence metrics and
    counted in ``unscorable_cases`` instead: averaging them in would report an
    identifier mismatch as a recall number.
    """

    out: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[CaseScore]] = {s: [c for c in scores if c.stratum == s] for s in STRATA}
    groups["overall"] = list(scores)

    for group_name, cases in groups.items():
        per_retriever: dict[str, Any] = {}
        for retriever in retrievers:
            gold_cases = [c for c in cases if c.gold_ids and c.strata_searchable]
            slot_cases = [c for c in cases if c.slot_targets and c.strata_searchable]
            multi_cases = [c for c in cases if len(c.slot_targets) >= 2 and c.strata_searchable]

            entry: dict[str, Any] = {
                "cases": len(cases),
                "gold_scored_cases": len(gold_cases),
                "slot_scored_cases": len(slot_cases),
                "multi_evidence_cases": len(multi_cases),
                "unscorable_cases": sum(1 for c in cases if not c.strata_searchable),
                "gold_recall": {},
                "gold_complete_recall": {},
                "slot_recall": {},
                "multi_evidence_complete_recall": {},
                "multi_evidence_partial_recall": {},
                "mrr": 0.0,
            }

            for k in KS:
                partials: list[float] = []
                completes: list[float] = []
                for case in gold_cases:
                    partial, complete = _gold_hits(
                        case.gold_ids, case.ranked_lists[retriever], k
                    )
                    partials.append(partial)
                    completes.append(1.0 if complete else 0.0)
                entry["gold_recall"][str(k)] = _mean(partials)
                entry["gold_complete_recall"][str(k)] = _mean(completes)

                entry["slot_recall"][str(k)] = _mean(
                    [_slot_hits(c.slot_targets, c.ranked_lists[retriever], k) for c in slot_cases]
                )

                multi_complete: list[float] = []
                multi_partial: list[float] = []
                for case in multi_cases:
                    hits = _slot_hits(case.slot_targets, case.ranked_lists[retriever], k)
                    multi_partial.append(hits)
                    # All N slots or nothing: a 2/3 is not a hit.  The partial
                    # number is kept beside it so the gap is visible rather than
                    # hidden behind one number.
                    multi_complete.append(1.0 if hits >= 1.0 else 0.0)
                entry["multi_evidence_complete_recall"][str(k)] = _mean(multi_complete)
                entry["multi_evidence_partial_recall"][str(k)] = _mean(multi_partial)

            entry["mrr"] = _mean(
                [_mrr(c.gold_ids, c.ranked_lists[retriever]) for c in gold_cases]
            )
            per_retriever[retriever] = entry
        out[group_name] = per_retriever
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_METRIC_FAMILIES: tuple[tuple[str, str], ...] = (
    ("gold_recall", "Gold Evidence Recall@K (partial: fraction of gold ids present)"),
    ("gold_complete_recall", "Gold Evidence Complete Recall@K (case counts only if ALL gold ids present)"),
    ("slot_recall", "Required-Slot Recall@K (slot hit when any fact at its own coordinate is retrieved)"),
    ("multi_evidence_complete_recall", "Multi-Evidence Complete Recall@K (N>=2 operands; all N or nothing)"),
    ("multi_evidence_partial_recall", "Multi-Evidence Partial Recall@K (same cases, mean slot fraction)"),
)


def _table(
    aggregate_out: Mapping[str, Mapping[str, Any]], metric: str, retrievers: Sequence[str]
) -> str:
    lines = [
        "| stratum | retriever | cases | scored | @5 | @10 | @20 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in list(STRATA) + ["overall"]:
        for retriever in retrievers:
            entry = aggregate_out[group][retriever]
            values = entry[metric]
            scored_key = (
                "gold_scored_cases"
                if metric.startswith("gold")
                else "slot_scored_cases"
            )
            lines.append(
                f"| {group} | {retriever} | {entry['cases']} | {entry[scored_key]} | "
                + " | ".join(f"{values[str(k)]:.3f}" for k in KS)
                + " |"
            )
    return "\n".join(lines)


def build_markdown(
    aggregate_out: Mapping[str, Mapping[str, Any]],
    resolution_summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    retrievers: Sequence[str],
) -> str:
    parts: list[str] = []
    parts.append("# NF-V3 retrieval Recall@K benchmark\n")
    parts.append(
        f"- run: `{manifest['timestamp']}`\n"
        f"- eval set: `{manifest['eval_set']}`\n"
        f"- fixtures: `{manifest['fixtures']}`\n"
        f"- index: `{manifest['index_dir']}`\n"
        f"- stores: `{manifest['v2_store']}` / `{manifest['ixbrl_store']}`\n"
        f"- lane depth: {manifest['lane_k']} per lane, RRF k={manifest['rrf_k']}\n"
        f"- reranker: {manifest['reranker']}\n"
    )

    parts.append("\n## 0. Id-space check (run before any metric)\n")
    parts.append(
        f"- cases: **{resolution_summary['cases']}**\n"
        f"- cases with at least one gold id resolved to a candidate key: "
        f"**{resolution_summary['cases_with_resolved_gold']} / {resolution_summary['cases']}**\n"
        f"- gold-id resolution rate: **{resolution_summary['gold_id_resolution_rate']:.4f}** "
        f"({resolution_summary['gold_ids_resolved']} / {resolution_summary['gold_ids_total']})\n"
        f"- of the resolved ids, present in the R4 index: "
        f"**{resolution_summary['gold_ids_indexed']}** "
        f"(rate {resolution_summary['gold_id_indexed_rate']:.4f})\n"
        f"- cases whose entire gold is indexable (scorable): "
        f"**{resolution_summary['scorable_cases']} / {resolution_summary['cases']}**\n"
    )
    parts.append("\n| stratum | cases | gold ids | resolved | indexed | id space | status |")
    parts.append("|---|---:|---:|---:|---:|---|---|")
    for row in resolution_summary["by_stratum"]:
        parts.append(
            f"| {row['stratum']} | {row['cases']} | {row['gold_ids']} | "
            f"{row['resolved']} | {row['indexed']} | {row['id_spaces']} | {row['status']} |"
        )

    parts.append("\n## 1. Metrics\n")
    parts.append(
        "\nThe `scored` column is how many cases the row's mean is taken over. "
        "A stratum whose gold is unindexable scores 0 cases, so its row is blank "
        "of meaning -- the zeros are the id-space finding restated, not a "
        "retrieval result.\n"
    )
    parts.append(
        f"\nSlot coordinates whose candidate count disagrees with the fixture's "
        f"recorded `coordinate_candidates`: "
        f"**{manifest['slot_coordinate_disagreements']}** of "
        f"{manifest['slot_coordinates_checked']}. "
        "These are the cross-entity slots, whose coordinates the fixture counts "
        "against the iXBRL store the R4 index does not contain.\n"
    )
    for metric, title in _METRIC_FAMILIES:
        parts.append(f"\n### {title}\n")
        parts.append(_table(aggregate_out, metric, retrievers))

    parts.append("\n### Retrieval-only MRR (first gold id, fused list)\n")
    parts.append("| stratum | retriever | scored | MRR |")
    parts.append("|---|---|---:|---:|")
    for group in list(STRATA) + ["overall"]:
        for retriever in retrievers:
            entry = aggregate_out[group][retriever]
            parts.append(
                f"| {group} | {retriever} | {entry['gold_scored_cases']} | {entry['mrr']:.4f} |"
            )

    parts.append("\n## 2. Sample triplets\n")
    parts.append(
        "Human eyeball check: if the retrieved ids and the gold ids are not the "
        "same kind of string, no number in this file means anything.\n"
    )
    for sample in manifest["samples"]:
        parts.append(f"\n**(a) `{sample['id']}` ({sample['stratum']})**\n")
        parts.append(f"- question: {sample['question']}")
        parts.append(
            f"- query: `{sample['query']}`\n"
            f"- gold ids: {sample['gold_ids']}\n"
            f"- gold -> candidate: {sample['gold_resolution']}\n"
            f"- top-5 retrieved (RRF): {sample['top5']}\n"
            f"- same id space: **{sample['same_id_space']}** | hit@5: **{sample['hit_at_5']}**"
        )
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolution_row(
    stratum: str,
    cases: Sequence[dict[str, Any]],
    gold_by_id: Mapping[str, Mapping[str, Any]],
    resolver: GoldResolver,
    eval_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gold_ids = 0
    resolved = 0
    indexed = 0
    spaces: set[str] = set()
    for case in cases:
        record = gold_by_id.get(case["id"], {})
        question_meta = eval_by_id.get(case["id"], {})
        for raw in record.get("fact_ids") or []:
            gold_ids += 1
            res = resolver.resolve(
                str(raw),
                document_id=question_meta.get("document_id"),
                metric=record.get("metric"),
                period=record.get("period"),
            )
            if res.candidate_key:
                resolved += 1
                spaces.add(res.id_space)
            if res.indexed:
                indexed += 1
    if not cases:
        status = "NOT_IN_SLICE"
    elif gold_ids == 0:
        status = "NO_GOLD (abstention)"
    elif indexed == 0:
        status = "STRUCTURALLY_UNMEASURABLE"
    elif indexed < gold_ids:
        status = "PARTIAL"
    else:
        status = "OK"
    return {
        "stratum": stratum,
        "cases": len(cases),
        "gold_ids": gold_ids,
        "resolved": resolved,
        "indexed": indexed,
        "id_spaces": ",".join(sorted(spaces)) or "none",
        "status": status,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=BENCH_DIR / "canonical-eval-v1.jsonl")
    parser.add_argument("--gold-evidence", type=Path, default=BENCH_DIR / "gold-evidence-v1.jsonl")
    parser.add_argument("--fixtures", type=Path, default=BENCH_DIR / "plan-fixtures-v8.jsonl")
    parser.add_argument(
        "--v2-fact-store",
        type=Path,
        default=Path("/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts.jsonl"),
    )
    parser.add_argument(
        "--ixbrl-fact-store",
        type=Path,
        default=Path(
            "/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts-ixbrl-v1.jsonl"
        ),
    )
    parser.add_argument("--index-dir", type=Path, default=None, help="Defaults to TRUSTED_V2_R4_INDEX_DIR")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lane-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank-depth", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the reranked hybrid even when a reranker is reachable.",
    )
    args = parser.parse_args(argv)

    import run_p1_2_dual_track_benchmark as runner

    runner._load_deployment_env()

    from run_tv2_canonical_benchmark import FactStoreGroundingIndex  # noqa: E402

    from src.pdf_retrieval_v4.candidate_direct_retriever import CandidateDirectRetriever
    from src.pdf_retrieval_v4.candidate_view_index import CandidateViewIndexReader
    from src.runtime.trusted_v2_production import _path_env

    environ = dict(os.environ)
    index_dir = (
        args.index_dir
        if args.index_dir is not None
        else _path_env(environ, "TRUSTED_V2_R4_INDEX_DIR", directory=True)
    )

    questions = _load_jsonl(args.eval_set)
    gold_by_id = {row["id"]: row for row in _load_jsonl(args.gold_evidence)}
    fixtures = {row["id"]: row for row in _load_jsonl(args.fixtures)}
    if args.limit:
        questions = questions[: args.limit]

    reader = CandidateViewIndexReader(index_dir)
    index_candidate_keys = {key for (_lane, key) in reader._candidate_to_view}
    v2_index = FactStoreGroundingIndex(args.v2_fact_store)
    ixbrl_keys = load_ixbrl_candidate_keys(args.ixbrl_fact_store)
    resolver = GoldResolver(v2_index, ixbrl_keys, index_candidate_keys)

    # -- the id-space report, printed before any retrieval happens -----------
    print()
    print("=" * 78)
    print("ID-SPACE CHECK -- run before any retrieval or scoring")
    print("=" * 78)
    print(f"  R4 index candidates: {len(index_candidate_keys)} (all keyed {V2FACT_PREFIX})")
    print(f"  v2 fact store rows:  {len(v2_index.fact_alias_map)} aliases")
    print(f"  ixbrl store keys:    {len(ixbrl_keys)}")
    overlap = len(ixbrl_keys & index_candidate_keys)
    print(f"  ixbrl keys present in the R4 index: {overlap}")

    eval_by_id = {q["id"]: q for q in questions}
    by_stratum = [
        _resolution_row(
            s, [q for q in questions if q.get("stratum") == s], gold_by_id, resolver, eval_by_id
        )
        for s in STRATA
    ]
    gold_ids_total = sum(row["gold_ids"] for row in by_stratum)
    gold_ids_resolved = sum(row["resolved"] for row in by_stratum)
    gold_ids_indexed = sum(row["indexed"] for row in by_stratum)
    cases = len(questions)
    cases_with_resolved = sum(
        1 for q in questions if any(
            resolver.resolve(str(raw), document_id=eval_by_id[q["id"]].get("document_id"),
                             metric=gold_by_id.get(q["id"], {}).get("metric"),
                             period=gold_by_id.get(q["id"], {}).get("period")).candidate_key
            for raw in (gold_by_id.get(q["id"], {}).get("fact_ids") or [])
        )
    )
    scorable_cases = 0
    for q in questions:
        golds = gold_by_id.get(q["id"], {}).get("fact_ids") or []
        if not golds:
            continue
        meta = eval_by_id[q["id"]]
        rec = gold_by_id.get(q["id"], {})
        if all(
            resolver.resolve(str(raw), document_id=meta.get("document_id"),
                             metric=rec.get("metric"), period=rec.get("period")).indexed
            for raw in golds
        ):
            scorable_cases += 1

    for row in by_stratum:
        flag = "" if row["status"] == "OK" else "   <<<"
        print(
            f"  {row['stratum']:26} cases={row['cases']:>3} gold_ids={row['gold_ids']:>3} "
            f"resolved={row['resolved']:>3} indexed={row['indexed']:>3} "
            f"space={row['id_spaces']:>8} {row['status']}{flag}"
        )
    print(
        f"  {'TOTAL':26} cases={cases:>3} gold_ids={gold_ids_total:>3} "
        f"resolved={gold_ids_resolved:>3} indexed={gold_ids_indexed:>3}"
    )
    resolution_summary = {
        "cases": cases,
        "cases_with_resolved_gold": cases_with_resolved,
        "gold_ids_total": gold_ids_total,
        "gold_ids_resolved": gold_ids_resolved,
        "gold_ids_indexed": gold_ids_indexed,
        "gold_id_resolution_rate": gold_ids_resolved / gold_ids_total if gold_ids_total else 0.0,
        "gold_id_indexed_rate": gold_ids_indexed / gold_ids_total if gold_ids_total else 0.0,
        "scorable_cases": scorable_cases,
        "ixbrl_keys_in_index": overlap,
        "by_stratum": by_stratum,
    }
    if gold_ids_indexed < gold_ids_total:
        print()
        print("  " + "!" * 74)
        print("  !! GOLD DOES NOT FULLY RESOLVE INTO THE RETRIEVAL ID SPACE.")
        print("  !! Strata marked STRUCTURALLY_UNMEASURABLE are not retrieval failures:")
        print("  !! their gold ids are absent from the index, so no rank exists to measure.")
        print("  " + "!" * 74)
    print()

    # -- reranker ------------------------------------------------------------
    from src.services.reranker import build_reranker  # noqa: E402

    reranker = None
    reranker_note = "disabled"
    reranker_name = os.environ.get("RAG_RERANKER") or "none"
    reranker_model = os.environ.get("RAG_RERANKER_MODEL") or ""
    if not args.no_rerank:
        try:
            reranker = build_reranker(reranker_name, reranker_model)
        except Exception as exc:  # pragma: no cover - configuration dependent
            reranker = None
            reranker_note = f"unavailable: {exc}"
        else:
            if reranker is None:
                reranker_note = "no reranker configured (build_reranker returned None)"
            else:
                reranker_note = f"{getattr(reranker, 'name', '?')}"

    print("=" * 78)
    print("RERANKER")
    print("=" * 78)
    print(f"  RAG_RERANKER={reranker_name!r} RAG_RERANKER_MODEL={reranker_model!r}")
    if reranker is None:
        print(f"  NO RERANKER REACHABLE ({reranker_note}) -- reranked Recall@K is SKIPPED, not faked.")
    else:
        print(f"  active reranker: {reranker_note}")
        if getattr(reranker, "name", "") == "heuristic":
            print("  NOTE: this is the dependency-free LEXICAL heuristic reranker, not a")
            print("        trained cross-encoder.  Its numbers are not cross-encoder numbers.")
    for module_name in ("sentence_transformers", "FlagEmbedding"):
        try:
            __import__(module_name)
        except ImportError:
            print(f"  cross-encoder backend {module_name!r}: NOT INSTALLED")
        else:
            print(f"  cross-encoder backend {module_name!r}: importable")
    print()

    # A reranked column is only reported when something actually reranked.
    retrievers: tuple[str, ...] = (
        RETRIEVERS if reranker is not None else RETRIEVERS[:-1]
    )

    # -- retrieval (gold is not in scope here) -------------------------------
    retriever = CandidateDirectRetriever(reader, rrf_k=args.rrf_k, lane_k=args.lane_k)
    retriever.final_pool_k = args.lane_k
    allowed = retriever._allowed_keys_for_scope(set())

    print("=" * 78)
    print(f"RETRIEVAL -- {len(questions)} questions, lane_k={args.lane_k}, rrf_k={args.rrf_k}")
    print("=" * 78)
    started = time.perf_counter()
    retrievals: dict[str, CaseRetrieval] = {}
    for position, question in enumerate(questions, 1):
        result = retrieve_case(
            question["question"],
            retriever=retriever,
            reader=reader,
            allowed_keys=allowed,
            reranker=reranker,
            rerank_depth=args.rerank_depth,
        )
        result.case_id = question["id"]
        retrievals[question["id"]] = result
        if position % 20 == 0 or position == len(questions):
            print(f"  [{position:>3}/{len(questions)}] retrieved", flush=True)
    elapsed = time.perf_counter() - started
    print(f"  retrieval complete in {elapsed:.1f}s\n")

    # -- scoring (the first and only place gold is read) ---------------------
    import build_p1_2_plan_fixtures as fixtures_builder  # noqa: E402

    coordinate_index = fixtures_builder.load_coordinate_index([args.v2_fact_store])
    scores: list[CaseScore] = []
    recorded_vs_computed: list[dict[str, Any]] = []
    for question in questions:
        case_id = question["id"]
        retrieval = retrievals[case_id]
        record = gold_by_id.get(case_id, {})
        fixture = fixtures.get(case_id, {})
        stratum = question.get("stratum") or record.get("stratum") or "unknown"

        gold_ids: list[str] = []
        gold_resolutions: list[Resolution] = []
        for raw in record.get("fact_ids") or []:
            res = resolver.resolve(
                str(raw),
                document_id=question.get("document_id"),
                metric=record.get("metric"),
                period=record.get("period"),
            )
            gold_ids.append(res.candidate_key or str(raw))
            gold_resolutions.append(res)

        slots = (fixture.get("plan") or {}).get("required_slots") or []
        gold_raw = [str(x) for x in (record.get("fact_ids") or [])]
        slot_targets: list[list[str]] = []
        slot_spaces: list[str] = []
        for index, slot in enumerate(slots):
            # A slot's own coordinate, not the gold row that happens to satisfy
            # it.  The fixture's own manifest records 68 slots whose coordinate
            # holds more than one fact; scoring those against the gold row alone
            # would call a retrieval that found the right quantity wrong.  The
            # gold-paired id is unioned in only as a guard against a gap between
            # the fixture's coordinate index and the gold's own id space.
            coordinate_raw = slot_coordinate_candidates(slot, coordinate_index)
            coordinate_targets = coordinate_candidate_keys(coordinate_raw, resolver)
            if len(gold_raw) == len(slots):
                res = gold_resolutions[index]
                paired = [res.candidate_key] if res.candidate_key else []
                targets = list(dict.fromkeys(coordinate_targets + paired))
                slot_spaces.append(res.id_space)
            else:
                targets = coordinate_targets
                slot_spaces.append("v2fact")
            slot_targets.append(targets)
            # Diagnostic only, and it compares like with like: the fixture's
            # recorded count describes the *coordinate*'s candidate set, so it is
            # checked against that set, never against the gold-paired target.
            recorded = (fixture.get("coordinate_candidates") or {}).get(slot["slot_id"])
            if recorded is not None and len(coordinate_raw) != recorded:
                recorded_vs_computed.append(
                    {
                        "case": case_id,
                        "slot_id": slot["slot_id"],
                        "recorded": recorded,
                        "computed": len(coordinate_raw),
                    }
                )

        strata_searchable = bool(gold_ids) and all(r.indexed for r in gold_resolutions)
        scores.append(
            CaseScore(
                case_id=case_id,
                stratum=stratum,
                ranked_lists={r: retrieval.ranked(r) for r in retrievers},
                gold_ids=gold_ids,
                slot_targets=slot_targets,
                slot_id_spaces=slot_spaces,
                strata_searchable=strata_searchable,
            )
        )

    aggregate_out = aggregate(scores, retrievers)

    # -- sample triplets -----------------------------------------------------
    samples: list[dict[str, Any]] = []
    for stratum in STRATA:
        picked = next(
            (q for q in questions if q.get("stratum") == stratum),
            None,
        )
        if picked is None:
            continue
        case_id = picked["id"]
        retrieval = retrievals[case_id]
        record = gold_by_id.get(case_id, {})
        raw_gold = [str(x) for x in (record.get("fact_ids") or [])]
        resolved = [
            resolver.resolve(
                str(raw),
                document_id=picked.get("document_id"),
                metric=record.get("metric"),
                period=record.get("period"),
            )
            for raw in raw_gold
        ]
        top5 = retrieval.top5
        hit = bool(raw_gold) and all(r.candidate_key in set(top5) for r in resolved)
        samples.append(
            {
                "id": case_id,
                "stratum": stratum,
                "question": picked["question"],
                "query": retrieval.query,
                "gold_ids": raw_gold,
                "gold_resolution": [
                    {
                        "gold": r.raw,
                        "candidate_key": r.candidate_key,
                        "id_space": r.id_space,
                        "resolved_by": r.resolved_by,
                        "indexed": r.indexed,
                    }
                    for r in resolved
                ],
                "top5": top5,
                # None, not False, when the case carries no gold at all: an
                # abstention has no id space to agree or disagree with.
                "same_id_space": (
                    None if not raw_gold else all(r.indexed for r in resolved)
                ),
                "hit_at_5": None if not raw_gold else hit,
            }
        )

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "eval_set": str(args.eval_set),
        "fixtures": str(args.fixtures),
        "index_dir": str(index_dir),
        "v2_store": str(args.v2_fact_store),
        "ixbrl_store": str(args.ixbrl_fact_store),
        "lane_k": args.lane_k,
        "rrf_k": args.rrf_k,
        "rerank_depth": args.rerank_depth,
        "reranker": reranker_note,
        "retrieval_seconds": round(elapsed, 2),
        "slot_coordinate_disagreements": len(recorded_vs_computed),
        "slot_coordinates_checked": sum(
            len((fixtures.get(q["id"], {}).get("plan") or {}).get("required_slots") or [])
            for q in questions
        ),
        "samples": samples,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": manifest,
        "resolution": resolution_summary,
        "metrics": aggregate_out,
        "recorded_vs_computed_slot_counts": recorded_vs_computed,
        "retrievers": list(retrievers),
        "ks": list(KS),
        "strata": list(STRATA),
    }
    (args.out_dir / "nf-v3-retrieval-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "nf-v3-retrieval.md").write_text(
        build_markdown(aggregate_out, resolution_summary, manifest, retrievers),
        encoding="utf-8",
    )

    # -- console summary -----------------------------------------------------
    print("=" * 78)
    print("RESULTS -- Gold Evidence Recall@K (partial) / Required-Slot Recall@K")
    print("=" * 78)
    header = f"{'stratum':26} {'retriever':28} " + " ".join(f"G@{k:<5}" for k in KS) + " " + " ".join(f"S@{k:<5}" for k in KS)
    print(header)
    for group in list(STRATA) + ["overall"]:
        for retriever in retrievers:
            entry = aggregate_out[group][retriever]
            g = " ".join(f"{entry['gold_recall'][str(k)]:.3f} " for k in KS)
            s = " ".join(f"{entry['slot_recall'][str(k)]:.3f} " for k in KS)
            print(f"{group:26} {retriever:28} {g} {s}")
    print()
    print("MULTI-EVIDENCE COMPLETE (all N slots) vs PARTIAL")
    print(f"{'stratum':26} {'retriever':28} " + " ".join(f"C@{k:<5}" for k in KS) + " " + " ".join(f"P@{k:<5}" for k in KS))
    for group in list(STRATA) + ["overall"]:
        for retriever in retrievers:
            entry = aggregate_out[group][retriever]
            c = " ".join(f"{entry['multi_evidence_complete_recall'][str(k)]:.3f} " for k in KS)
            p = " ".join(f"{entry['multi_evidence_partial_recall'][str(k)]:.3f} " for k in KS)
            print(f"{group:26} {retriever:28} {c} {p}")

    print()
    print("SAMPLE TRIPLETS")
    for sample in samples:
        print(f"  -- {sample['id']} ({sample['stratum']}) same_id_space={sample['same_id_space']} hit@5={sample['hit_at_5']}")
        print(f"     Q: {sample['question']}")
        print(f"     query: {sample['query']}")
        print(f"     gold:  {sample['gold_ids']}")
        print(f"     ->     {[r['candidate_key'] for r in sample['gold_resolution']]}")
        print(f"     top5:  {sample['top5']}")

    if recorded_vs_computed:
        print()
        print(f"NOTE: {len(recorded_vs_computed)} slot coordinates disagree with the fixture's "
              f"recorded coordinate_candidates count (see JSON).")

    print(f"\nwrote {args.out_dir / 'nf-v3-retrieval-metrics.json'}")
    print(f"wrote {args.out_dir / 'nf-v3-retrieval.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
