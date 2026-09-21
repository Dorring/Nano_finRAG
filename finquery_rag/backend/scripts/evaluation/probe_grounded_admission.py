"""What the source-grounded vocabulary changes, case by case.

Runs the real gate twice on every blocked case -- once with the ontology alone,
once with the source's determinate row labels added -- and reports which cases
change verdict and which do not.  It builds the grounding exactly as production
does, from the authoritative store, so a label counts as grounded here only
because `SourceLabelGrounding` said so.

  python probe_grounded_admission.py --cases <A-cases.jsonl> --pinned <pinned.jsonl> \\
      --v2-fact-store <financial-facts.jsonl>
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


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--pinned", type=Path, required=True)
    parser.add_argument("--v2-fact-store", type=Path, required=True)
    args = parser.parse_args(argv)

    from rag_v2.contracts.plan import SupervisorPlan
    from rag_v2.supervisor.semantic_alignment import (
        UnknownSemanticPolicy,
        align_query_to_plan,
    )
    from src.finance.source_label_grounding import SourceLabelGrounding
    from src.runtime.trusted_v2_production import StructuredFactStore

    store = StructuredFactStore(args.v2_fact_store)
    grounding = SourceLabelGrounding(store)

    rows = [r for r in _load(args.cases)
            if r.get("comparable") and not r.get("must_refuse")]
    pinned = {r["case_id"]: r for r in _load(args.pinned)}

    policy = UnknownSemanticPolicy.STRICT_DIRECT_FACT
    changed: list[tuple[str, str, str, list[str]]] = []
    stayed: list[tuple[str, list[str]]] = []
    tally: collections.Counter = collections.Counter()

    for row in rows:
        if not row.get("gate_blocked"):
            continue
        entry = pinned.get(row["case_id"]) or {}
        if not entry.get("plan"):
            continue
        plan = SupervisorPlan.from_dict(entry["plan"])
        before = align_query_to_plan(row["question"], plan, unknown_policy=policy)
        labels = grounding(plan)
        after = align_query_to_plan(row["question"], plan, unknown_policy=policy,
                                    grounded_labels=labels)
        # Bucketed by the verdict now, not by whether it moved.  A case the
        # ontology alone now admits is not `blocked`, and reporting it under a
        # heading that says it is would understate the change by however many
        # the alias and span rules recovered on their own.
        if after.allowed:
            tally["allowed"] += 1
            changed.append((row["case_id"], before.status.value, after.status.value,
                            sorted(labels)))
        else:
            tally["still_blocked"] += 1
            stayed.append((row["case_id"], after))

    total = tally["allowed"] + tally["still_blocked"]
    print("=" * 78)
    print("SOURCE-GROUNDED ADMISSION -- applied to the %d blocked cases" % total)
    print("=" * 78)
    print("  %s" % dict(tally))
    print()
    for case_id, before, after, labels in sorted(changed):
        print("  %-26s %-9s -> %-9s  labels=%s" % (case_id, before, after, labels))
    print()
    print("  -- still blocked")
    for case_id, alignment in sorted(stayed):
        print("    %-26s %-10s mism=%-42s amb=%s"
              % (case_id, alignment.status.value,
                 str(list(alignment.mismatches)[:1])[:42],
                 list(alignment.ambiguous_query_fields)))
        print("        q_metrics=%s q_ops=%s plan_metrics=%s"
              % (list(alignment.query_metric_ids),
                 list(alignment.query_operation_ids),
                 list(alignment.plan_metric_ids)))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
