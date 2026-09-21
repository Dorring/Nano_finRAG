"""Whether a slot's own entity is reachable in the lanes at all.

`s3-compare-002` asks which of Apple and Visa had the higher FY2025 net income.
Both slots issue `Net income FY2025` -- the entity is deliberately not in the
query -- and the twenty-candidate pool that comes back holds seven Pfizer rows,
four Tesla rows, one Apple row and **no Visa row at all**.  A slot whose entity
never appears cannot be bound by anything downstream, so the question is whether
Visa is absent because the lanes do not rank it, or only because the pool is cut
before reaching it.

Those are different problems.  If the lanes hold Visa deeper down, isolating
each slot to its own entity recovers it and the query needs no change.  If they
do not, no eligibility rule helps and the demand has to be expressed differently.

  python probe_entity_reachability.py --index-dir <r4-index> --v2-fact-store <store>
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _path in (str(_BACKEND_DIR), str(_BACKEND_DIR / "scripts" / "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: The two slots of `tv2f01-s3-compare-002`, and the metric/period they share.
CASES = (
    ("tv2f01-s3-compare-002", "Net income", "FY2025", ("Apple", "Visa")),
    ("tv2f01-s3-compare-007", "Net income", "FY2025", ("JPMorganChase", "Pfizer")),
    ("tv2f01-s3-rank-001", "Net income", "FY2025",
     ("Tesla", "The Coca-Cola Company", "Visa")),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--v2-fact-store", type=Path, required=True)
    args = parser.parse_args(argv)

    import run_p1_2_dual_track_benchmark as runner

    # Without this the embedder resolves its model from the network, which is
    # unreachable here, and the run spends five retries per model file before
    # falling back to a mean-pooling stub -- a different retriever than the one
    # being measured.
    runner._load_deployment_env()

    from src.pdf_retrieval_v4.candidate_view_index import (
        LANES,
        CandidateViewIndexReader,
    )

    entities: dict[str, str] = {}
    for line in args.v2_fact_store.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = str(record.get("candidate_key") or "")
        if key:
            entities[key] = str(record.get("entity") or "")

    reader = CandidateViewIndexReader(args.index_dir)

    for case_id, metric, period, wanted in CASES:
        query = "%s %s" % (metric, period)
        print("=" * 78)
        print("%s  query=%r  want=%s" % (case_id, query, list(wanted)))
        print("=" * 78)
        hits: dict[str, int] = {}
        for lane in LANES:
            for rank, hit in enumerate(
                    reader.search(lane, query, allowed_candidate_keys=None, k=400), 1):
                key = hit.candidate_key
                if key and (key not in hits or rank < hits[key]):
                    hits[key] = rank
        tally: collections.Counter = collections.Counter()
        best: dict[str, int] = {}
        for key, rank in hits.items():
            entity = entities.get(key, "")
            tally[entity] += 1
            if entity not in best or rank < best[entity]:
                best[entity] = rank
        print("  candidates in lanes: %d" % len(hits))
        for entity in wanted:
            print("    %-24s rows=%-4d best_rank=%s"
                  % (entity, tally.get(entity, 0), best.get(entity, "-")))
        print("    top entities: %s"
              % [(e, n) for e, n in tally.most_common(6) if e])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
