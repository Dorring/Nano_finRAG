"""NF-V3: the end-to-end numbers, computed from the canonical 120-case run.

The headline this file exists to state honestly is the shape of the result, not its size:
the trusted chain releases very little and what it releases is correct.  Both halves have
to be reported together, because either alone is misleading -- a 5% release rate reads as a
broken system until you know 100% of it was right, and a 100% accuracy reads as a working
system until you know it answered 5 of 95 answerable questions.

Scoring is separated exactly as `src/evaluation/p1_2_dual_track.py` separates it:

  release     did the system answer, or refuse
  correctness of what it released -- never folded into the release rate
  refusal     correct refusals among the cases that should be refused

A release is scored against gold by the operation the case actually asks for: a value
comparison for factual, a tolerance comparison for arithmetic, a stated ordering for
comparison and ranking.  Scoring a percentage as a string would call `-28.13%` wrong
against a gold `-0.2813`, which is the same number.

  python score_nf_v3_final.py --predictions <jsonl> --gold <jsonl> --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path

#: A number in free text: optional sign, digits with optional thousands separators,
#: optional decimal part.  Percentages are handled by the operation, not here.
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

#: Capitalised words that begin a sentence rather than name a company, so the search for
#: "the entity that was higher" does not stop on them.
_SENTENCE_WORDS = frozenset({
    "the", "in", "for", "according", "what", "which", "how", "was", "were", "is", "are",
    "at", "of", "a", "an", "on", "by", "to", "and", "reported", "its", "their", "this",
    "that", "these", "both", "compared", "verified", "net", "income", "revenue", "total",
})


def numbers_in(text: str) -> list[float]:
    out = []
    for raw in _NUMBER.findall(str(text or "")):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def percentile(values: list[float], fraction: float) -> float | None:
    """Linear-interpolated percentile.  The runner's own report uses index lookup, which
    reports a real observation for p50 but not for p95 on small n."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 2)


def score_released(prediction: dict, gold: dict) -> str | None:
    """Correct / wrong / unscoreable, for one released answer.

    `unscoreable` is a real outcome and is not rounded to either side: a case whose gold
    carries nothing to compare against cannot be evidence for or against the system.
    """
    answer = str(prediction.get("answer") or "")
    operation = gold.get("operation")

    if operation == "comparison" or gold.get("expected_higher"):
        want = gold.get("expected_higher")
        if not want:
            return None
        # The claim is "X was higher".  X is the capitalised name nearest the word
        # `higher`, not the first capitalised word in the sentence -- an earlier version
        # of this captured `"The verified calculation indicates that Apple"` and scored a
        # correct answer wrong.
        index = answer.lower().find("higher")
        if index < 0:
            return None
        names = [n for n in re.findall(r"\b[A-Z][A-Za-z0-9&.\-]*\b", answer[:index])
                 if n.lower() not in _SENTENCE_WORDS]
        if not names:
            return None
        got = names[-1]
        return "correct" if got.lower() == want.lower() else "wrong"

    if gold.get("expected_ranking"):
        want = [str(e).lower() for e in gold["expected_ranking"]]
        positions = []
        for entity in want:
            index = answer.lower().find(entity)
            positions.append((index, entity))
        if any(index < 0 for index, _ in positions):
            return None
        return "correct" if [e for _, e in sorted(positions)] == want else "wrong"

    expected = gold.get("expected_value")
    if expected in (None, ""):
        return None
    tolerance = gold.get("tolerance")
    want = numbers_in(expected)
    got = numbers_in(answer)
    if not want or not got:
        return None
    if (
        operation in ("percentage_share", "growth_rate")
        or (tolerance and abs(want[0]) < 1.01)
    ):
        # A ratio, which the answer may state as a percentage of itself.  A
        # growth rate of `-2.0204` and `-202.04%` are one quantity in two
        # conventions, and the benchmark's gold fixes only one of them -- so
        # comparing them as written scores a correct answer wrong and reports a
        # rendering difference as a capability gap.
        #
        # This rescales a representation; it cannot rescue a wrong answer.  Two
        # operands from the wrong years still miss at every scale, and the
        # reciprocal a role inversion produces -- `-23.1429` where the gold is
        # `-0.0432` -- misses too, because `0.01 * -23.1429` is not `-0.0432`.
        for candidate in got:
            for scale in (1.0, 0.01):
                if abs(candidate * scale - want[0]) <= (tolerance or 0.0001):
                    return "correct"
        return "wrong"
    for candidate in got:
        if abs(candidate - want[0]) <= (tolerance or 0.005):
            return "correct"
    return "wrong"


def load_alias_map(fact_store: Path) -> dict[str, str]:
    """fact/evidence/citation id -> canonical candidate id, as the runtime resolves them.

    The two sides are in different namespaces -- gold says `v2fact:<hex>`, a citation says
    `citation:v2:<hex>` -- so comparing them as strings would report 0% precision on
    perfectly correct citations.  Resolution has to happen before scoring, and it happens
    here so the metric is about the citation and not about the naming.
    """
    aliases: dict[str, str] = {}
    for line in fact_store.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        candidate = record.get("candidate_id") or record.get("candidate_key")
        if not candidate:
            continue
        aliases[candidate] = candidate
        for key in ("fact_id", "evidence_id", "citation_id", "source_id",
                    "physical_source_id"):
            value = record.get(key)
            if not value:
                continue
            aliases[value] = candidate
            if ":" in value:
                aliases[value.split(":")[-1]] = candidate
    return aliases


def resolve(identifier: str, aliases: dict[str, str]) -> str:
    if identifier in aliases:
        return aliases[identifier]
    if ":" in identifier:
        return aliases.get(identifier.split(":")[-1], identifier)
    return identifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--fact-store", type=Path, default=None,
                        help="needed for citation precision/recall; without it the two "
                             "id namespaces cannot be compared")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    predictions = [json.loads(line)
                   for line in args.predictions.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    gold = {json.loads(line)["id"]: json.loads(line)
            for line in args.gold.read_text(encoding="utf-8").splitlines() if line.strip()}
    aliases = load_alias_map(args.fact_store) if args.fact_store else {}

    # `expected_failure_reason` is how the benchmark marks a case that must be refused.
    def must_refuse(case_id: str) -> bool:
        return bool((gold.get(case_id) or {}).get("expected_failure_reason"))

    strata = collections.defaultdict(lambda: collections.Counter())
    tally = collections.Counter()
    released_rows = []
    latency_by_stratum = collections.defaultdict(list)

    for prediction in predictions:
        case_id = prediction["id"]
        stratum = prediction["stratum"]
        released = prediction.get("release_status") == "RELEASED"
        refusal_expected = must_refuse(case_id)
        tally["total"] += 1
        strata[stratum]["total"] += 1

        if prediction.get("latency_ms") is not None:
            latency_by_stratum[stratum].append(prediction["latency_ms"])

        if refusal_expected:
            tally["should_refuse"] += 1
            strata[stratum]["should_refuse"] += 1
            if not released:
                tally["correct_refusal"] += 1
                strata[stratum]["correct_refusal"] += 1
            else:
                tally["false_release"] += 1
                strata[stratum]["false_release"] += 1
        else:
            tally["answerable"] += 1
            strata[stratum]["answerable"] += 1
            if not released:
                tally["answerable_refused"] += 1
                strata[stratum]["answerable_refused"] += 1

        if released:
            tally["released"] += 1
            strata[stratum]["released"] += 1
            verdict = score_released(prediction, gold.get(case_id) or {})
            if verdict:
                tally[f"released_{verdict}"] += 1
                strata[stratum][f"released_{verdict}"] += 1
            else:
                tally["released_unscoreable"] += 1
                strata[stratum]["released_unscoreable"] += 1
            citations = len(prediction.get("citation_ids") or [])
            tally["released_with_citation"] += 1 if citations else 0
            if not citations:
                tally["released_without_citation"] += 1
            # Citation precision and recall, never merged into one "citation accuracy":
            # precision asks whether a cited cell supports the claim, recall whether the
            # claim's own cells were cited.  A system can score 100% on either by citing
            # everything or nothing respectively.
            cited = [resolve(c, aliases) for c in (prediction.get("citation_ids") or [])]
            wanted = [resolve(f, aliases)
                      for f in (gold.get(case_id) or {}).get("fact_ids") or []]
            admitted = [resolve(e, aliases)
                        for e in (prediction.get("evidence_ids") or [])]
            if aliases and wanted:
                tally["cited_total"] += len(cited)
                tally["cited_in_gold"] += sum(1 for c in cited if c in wanted)
                tally["gold_total"] += len(wanted)
                tally["gold_cited"] += sum(1 for w in wanted if w in cited)
                # Two different questions, and conflating them misreports both.  `in gold`
                # asks whether the citation is the cell the benchmark names; `in evidence`
                # asks whether the citation is a cell the runtime actually admitted.  A
                # citation can be one without the other, and a system that cites a
                # different-but-real cell is a different thing from one that invents a
                # citation, which is what the second number is there to catch.
                tally["cited_total_ev"] += len(cited)
                tally["cited_in_evidence"] += sum(1 for c in cited if c in admitted)
            released_rows.append({
                "id": case_id, "stratum": stratum, "verdict": verdict,
                "citations": citations,
                "evidence": len(prediction.get("evidence_ids") or []),
                "answer": str(prediction.get("answer") or "")[:160],
            })

    latencies = [p["latency_ms"] for p in predictions
                 if p.get("latency_ms") is not None]
    report = {
        "phase": "NF-V3-final-e2e",
        "counts": dict(tally),
        "by_stratum": {k: dict(v) for k, v in strata.items()},
        "latency": {
            "n": len(latencies),
            "p50_ms": percentile(latencies, 0.50),
            "p95_ms": percentile(latencies, 0.95),
            "p99_ms": percentile(latencies, 0.99),
            "max_ms": round(max(latencies), 2) if latencies else None,
        },
        "latency_by_stratum": {k: {"n": len(v), "p50_ms": percentile(v, 0.50),
                                   "p95_ms": percentile(v, 0.95)}
                               for k, v in latency_by_stratum.items()},
        "released_cases": released_rows,
    }

    print("=== release, refusal, and what was released ===")
    print(f"    cases                       {tally['total']}")
    print(f"    answerable                  {tally['answerable']}")
    print(f"    should refuse               {tally['should_refuse']}")
    print(f"    released                    {tally['released']}")
    print()
    print(f"    answerable release rate     "
          f"{tally['released'] / max(1, tally['answerable']):.1%}")
    print(f"    answerable refusal rate     "
          f"{tally['answerable_refused'] / max(1, tally['answerable']):.1%}")
    print(f"    correct refusal (no-answer) "
          f"{tally['correct_refusal']}/{tally['should_refuse']}  "
          f"({tally['correct_refusal'] / max(1, tally['should_refuse']):.1%})")
    print(f"    FALSE RELEASE               {tally['false_release']}")
    print()
    scored = tally["released_correct"] + tally["released_wrong"]
    print(f"    released correct            {tally['released_correct']}/{scored}"
          f"   (unscoreable {tally['released_unscoreable']})")
    print(f"    released with a citation    "
          f"{tally['released_with_citation']}/{tally['released']}")
    if aliases:
        cited_total, gold_total = tally["cited_total"], tally["gold_total"]
        ev_total = tally["cited_total_ev"]
        print(f"    citation in its own evidence  {tally['cited_in_evidence']}/{ev_total}"
              f"   ({tally['cited_in_evidence'] / max(1, ev_total):.1%})"
              f"   <- the citation points at an admitted cell")
        print(f"    citation precision vs gold   {tally['cited_in_gold']}/{cited_total}"
              f"   ({tally['cited_in_gold'] / max(1, cited_total):.1%})"
              f"   <- the citation IS the benchmark's cell")
        print(f"    citation recall vs gold      {tally['gold_cited']}/{gold_total}"
              f"   ({tally['gold_cited'] / max(1, gold_total):.1%})")
    else:
        print("    citation precision/recall   not computed (no --fact-store: the gold "
              "and citation id namespaces cannot be compared without it)")
    print()
    print("=== per released case ===")
    for row in released_rows:
        print(f"    {row['id']:28} {row['stratum']:26} {str(row['verdict']):10} "
              f"cit={row['citations']} ev={row['evidence']}")
    print()
    print("=== latency (end to end, seconds per query) ===")
    print(f"    n {report['latency']['n']}   p50 {report['latency']['p50_ms']}ms   "
          f"p95 {report['latency']['p95_ms']}ms   p99 {report['latency']['p99_ms']}ms")
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "final-e2e-metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'final-e2e-metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
