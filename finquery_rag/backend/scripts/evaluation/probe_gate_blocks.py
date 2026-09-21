"""Why each gate-blocked case is blocked, in the gate's own terms.

`gate_blocked = 48` is an aggregate over two different failures that need
different fixes:

  unknown query metric   the question's phrase is not in the ontology, so the
                         gate has no query-side id to compare with
  unrecognized plan slot the plan's slot metric is not in the ontology, so the
                         gate cannot check what the plan is aiming at

Both are "the vocabulary cannot name a source row label", but the first is a
query-side miss and the second a plan-side miss, and a fix that addresses one
will not move the other.  This prints the split with the raw phrases.

It also asks the question the ontology cannot: does the *store* know the label?
For each unresolved phrase it counts the facts whose own metric/row label
matches, and how many distinct metric identities those facts carry.  A label the
store resolves to exactly one identity is determinate for that entity; one that
resolves to several is genuinely ambiguous.  That distinction is not used to
change anything here -- it is measured, so the next step can be argued from
counts rather than from the labels that happen to be memorable.

  python probe_gate_blocks.py --cases <A-cases.jsonl> --pinned <pinned.jsonl> \\
      --v2-fact-store <financial-facts.jsonl>
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _path in (str(_BACKEND_DIR), str(_BACKEND_DIR / "scripts" / "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


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
        canonical_metric_id,
    )

    rows = [r for r in _load(args.cases)
            if r.get("comparable") and not r.get("must_refuse")]
    pinned = {r["case_id"]: r for r in _load(args.pinned)}

    # -- the store, indexed by its own row label ---------------------------
    label_index: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    sample: dict | None = None
    for record in _load(args.v2_fact_store):
        if sample is None:
            sample = record
        identity = str(record.get("metric") or "")
        for field in ("metric", "row_label", "metric_raw"):
            value = record.get(field)
            if value:
                label_index[_norm(value)][identity] += 1

    print("=" * 78)
    print("GATE BLOCK DIAGNOSIS")
    print("=" * 78)
    if sample:
        print("  store fields: %s" % sorted(sample.keys()))
    print()

    kinds: collections.Counter = collections.Counter()
    detail: list[tuple[str, str, str, dict]] = []

    for row in rows:
        if not row.get("gate_blocked"):
            continue
        entry = pinned.get(row["case_id"]) or {}
        plan_payload = entry.get("plan")
        if not plan_payload:
            kinds["NO_PINNED_PLAN"] += 1
            continue
        alignment = align_query_to_plan(
            row["question"], SupervisorPlan.from_dict(plan_payload),
            unknown_policy=UnknownSemanticPolicy.STRICT_DIRECT_FACT)
        if alignment.unknown_query_fields:
            kind = "QUERY_METRIC_UNKNOWN"
            phrases = [str(s.get("metric")) for s in
                       (plan_payload.get("required_slots") or [])]
        elif alignment.unknown_plan_metrics:
            kind = "PLAN_METRIC_UNKNOWN"
            phrases = list(alignment.unknown_plan_metrics)
        elif alignment.mismatches:
            kind = "ID_MISMATCH"
            phrases = [str(s.get("metric")) for s in
                       (plan_payload.get("required_slots") or [])]
        else:
            kind = "OTHER:%s" % alignment.status.value
            phrases = [str(s.get("metric")) for s in
                       (plan_payload.get("required_slots") or [])]
        kinds[kind] += 1
        detail.append((row["case_id"], kind, "; ".join(phrases),
                       {"mismatches": list(alignment.mismatches),
                        "unknown_plan": list(alignment.unknown_plan_metrics),
                        "query_metric_ids": list(alignment.query_metric_ids)}))

    print("  -- block kinds")
    for kind, count in kinds.most_common():
        print("    %-26s %4d" % (kind, count))
    print()

    print("  -- per case")
    for case_id, kind, phrases, extra in sorted(detail):
        print("    %-26s %-20s %s" % (case_id, kind, phrases[:70]))
        print("        q_ids=%s  mismatches=%s"
              % (extra["query_metric_ids"], extra["mismatches"][:3]))
    print()

    # -- what the store says about each unresolved phrase ------------------
    print("=" * 78)
    print("STORE GROUNDING for the unresolved phrases")
    print("=" * 78)
    print("  %-58s %6s %6s" % ("phrase", "facts", "ids"))
    seen: set[str] = set()
    for _, _, phrases, _ in sorted(detail):
        for phrase in phrases.split("; "):
            key = _norm(phrase)
            if not key or key in seen:
                continue
            seen.add(key)
            identities = label_index.get(key) or collections.Counter()
            print("  %-58s %6d %6d   %s"
                  % (phrase[:58], sum(identities.values()), len(identities),
                     list(identities)[:2] if len(identities) <= 2 else "AMBIGUOUS"))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
