"""E2-1: classify the 66 semantic-gate blocks.  Every one gets exactly one class.

E1 showed the end-to-end coverage bottleneck is the semantic alignment gate, not
retrieval: 66 of 75 comparable answerable cases are refused at
`terminal_state='PLAN'` before any search, and lifting the gate alone takes
releases from 3 to 43.  This asks whether that refusal is *right*.

The question is NOT "is the gold answer in the store".  It is the one the gate
is actually asking: **can the metric the plan names be resolved, deterministically
and from the source, to a concept the system has?**  A label that resolves is one
the gate should have accepted; a label that does not is one no amount of
ontology work can ground, and refusing it is the fail-closed behaviour working.

Four classes, and the rules that assign them are printed with the counts so each
one can be checked rather than believed:

  ALIGNMENT_NORMALIZATION_GAP  the ontology already names this metric -- the
                               canonical resolver returns an id -- and the case
                               was blocked regardless.  The failure is in the
                               alignment path, not in coverage.
  ONTOLOGY_VOCAB_GAP           the label is a well-formed quantity, the store
                               carries it, and the ontology simply has no entry.
                               Fixable by extending the ontology.
  CORRECT_FAIL_CLOSED          the label is not a concept: an extraction
                               artefact (footnote markers, a person's name, a
                               sentence fragment), or nothing in the store at
                               all.  The gate is right and nothing here is fixed.
  PLANNER_SEMANTIC_ERROR       the plan named something the question does not
                               contain, so the gate could not have matched it.

UNCLASSIFIED must be zero; the script fails loudly if it is not.

  python audit_semantic_alignment_blocks.py \\
      --cases <arm A cases>.jsonl --store financial-facts.jsonl --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _path in (str(_BACKEND_DIR), str(_BACKEND_DIR / "scripts" / "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: A footnote or table marker welded onto a row label by the extractor:
#: `Commercial(3)`, `Interest^{(f)}`, `Common stockholders' equity(f)`,
#: `Total nominal payments volume(4)`.  These are the P1.6-A defect -- the
#: extractor concatenates table furniture into `metric` -- and a label carrying
#: one is not the name of a quantity.
_FOOTNOTE = re.compile(r"\(\s*\d+\s*\)|\^\s*\{|\(\s*[a-z]\s*\)\s*$|\(\s*\d+\s*\)\s*$")

#: Labels that name a person or a role rather than a quantity.  The compensation
#: table is where the extractor files an executive's name as the metric; the
#: ambiguity survey recorded `NVIDIA / Colette M. Kress` as the example.
_PERSONLIKE = re.compile(
    r"^(mr|ms|mrs|dr)\.?\s|\b(ceo|cfo|coo|president|chief executive|"
    r"director|officer)\b",
    re.IGNORECASE,
)
_PERSON_NAME = re.compile(r"^[A-Z][a-z]+(\s+[A-Z]\.?)?\s+[A-Z][a-z]+$")

#: A label that opens with a filler word is a phrase lifted out of prose, not a
#: metric: `figure for Ajay K. Puri`, `State value`, `Direct Customer A value`.
_PHRASE_LEAD = re.compile(
    r"^(figure|value|amount|number|total of|balance of|the)\b", re.IGNORECASE)

#: Most actionable first.  A case takes the earliest class any of its slots
#: falls into, because the ontology is extended per concept and one
#: unresolvable slot still leaves the question unanswerable.
PRIORITY = ("ALIGNMENT_NORMALIZATION_GAP", "ONTOLOGY_VOCAB_GAP",
            "PLANNER_SEMANTIC_ERROR", "CORRECT_FAIL_CLOSED")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from rag_v2.supervisor.semantic_alignment import canonical_metric_id

    store_by_metric: collections.Counter = collections.Counter()
    store_by_entity_metric: collections.Counter = collections.Counter()
    for line in args.store.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        metric = str(record.get("metric") or "").strip()
        if metric:
            store_by_metric[metric] += 1
            store_by_entity_metric[
                (_norm(record.get("entity")), _norm(metric))] += 1

    rows = [json.loads(line) for line in
            args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    blocked = [r for r in rows
               if r.get("gate_blocked") and r.get("comparable") and not r.get("must_refuse")]

    print("=" * 78)
    print("E2-1 -- SEMANTIC ALIGNMENT GATE: IS THE REFUSAL RIGHT?")
    print("=" * 78)
    print(f"  blocked comparable answerable cases  {len(blocked)}")
    print()

    verdicts: list[dict] = []
    tally: collections.Counter = collections.Counter()
    rule_tally: collections.Counter = collections.Counter()

    for row in blocked:
        case_id = row["case_id"]
        question = str(row.get("question") or "")
        slots = list(zip(row.get("plan_slots") or [], row.get("plan_metrics") or []))
        per_slot: list[dict] = []
        case_class = None

        for slot_id, metric in slots:
            metric = str(metric or "")
            canonical = canonical_metric_id(metric)
            entity_metric_rows = sum(
                count for (entity, name), count in store_by_entity_metric.items()
                if name == _norm(metric))
            facts = {
                "canonical_id": canonical,
                "store_rows": store_by_metric.get(metric, 0),
                "store_rows_any_entity": entity_metric_rows,
                "in_question": _norm(metric) in _norm(question),
                "footnote_marker": bool(_FOOTNOTE.search(metric)),
                "personlike": bool(_PERSONLIKE.search(metric)) or bool(
                    _PERSON_NAME.match(metric)),
                "phrase_lead": bool(_PHRASE_LEAD.match(metric)),
            }
            # Rule order is the classification.  First match wins, and each rule
            # is a property of the label rather than a judgement about the case.
            if canonical:
                klass, rule = "ALIGNMENT_NORMALIZATION_GAP", "ontology_names_it"
            elif not facts["in_question"]:
                klass, rule = "PLANNER_SEMANTIC_ERROR", "metric_absent_from_question"
            elif facts["store_rows"] == 0:
                klass, rule = "CORRECT_FAIL_CLOSED", "nothing_in_store"
            elif facts["footnote_marker"]:
                klass, rule = "CORRECT_FAIL_CLOSED", "extraction_footnote_marker"
            elif facts["personlike"] or facts["phrase_lead"]:
                klass, rule = "CORRECT_FAIL_CLOSED", "label_is_not_a_quantity"
            else:
                klass, rule = "ONTOLOGY_VOCAB_GAP", "wellformed_and_in_store"
            facts["metric"] = metric
            facts["slot_id"] = slot_id
            facts["class"] = klass
            facts["rule"] = rule
            per_slot.append(facts)
            rule_tally[rule] += 1

        # A case is classified by the slots that actually BLOCKED it, not by its
        # most actionable slot.  A two-slot ratio question with one resolvable
        # leg and one raw label is blocked by the raw label; promoting it to
        # "the ontology names it, so the alignment path dropped it" would send
        # someone to fix a path that is working.  So the resolvable slots are
        # excluded first -- and a case survives to
        # ALIGNMENT_NORMALIZATION_GAP only if *every* slot resolved and it was
        # blocked anyway, which is the only shape that actually indicts the path.
        blocking = [slot for slot in per_slot if not slot["canonical_id"]]
        if not blocking:
            case_class = "ALIGNMENT_NORMALIZATION_GAP"
        else:
            case_class = None
            for slot in blocking:
                klass = slot["class"]
                if case_class is None or PRIORITY.index(klass) < PRIORITY.index(case_class):
                    case_class = klass

        tally[case_class] += 1
        verdicts.append({
            "case_id": case_id, "stratum": row.get("stratum"), "question": question,
            "class": case_class, "slots": per_slot,
            "computed_unknown_query_fields": row.get("computed_unknown_query_fields"),
            "computed_status": row.get("computed_status"),
            "gate_reason_raw": row.get("gate_reason_raw"),
        })

    print("  class                            cases   rule")
    by_class: dict[str, list[str]] = collections.defaultdict(list)
    for verdict in verdicts:
        for slot in verdict["slots"]:
            if slot["class"] == verdict["class"]:
                by_class[verdict["class"]].append(slot["rule"])
                break
    for klass in ("ALIGNMENT_NORMALIZATION_GAP", "ONTOLOGY_VOCAB_GAP",
                  "CORRECT_FAIL_CLOSED", "PLANNER_SEMANTIC_ERROR"):
        rules = collections.Counter(by_class.get(klass, []))
        print(f"    {klass:30} {tally.get(klass, 0):>4}   {dict(rules) if rules else ''}")
    print()

    print("  rule totals (per slot)")
    for rule, count in rule_tally.most_common():
        print(f"    {count:>4}  {rule}")
    print()

    for klass in ("ALIGNMENT_NORMALIZATION_GAP", "ONTOLOGY_VOCAB_GAP"):
        examples = [v for v in verdicts if v["class"] == klass][:12]
        if not examples:
            continue
        print(f"  {klass} -- sample")
        for verdict in examples:
            metrics = ", ".join(s["metric"][:52] for s in verdict["slots"])
            print(f"    {verdict['case_id']:26} {metrics}")
        print()

    unclassified = tally.get(None, 0)
    print(f"  UNCLASSIFIED: {unclassified}")
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "semantic-gate-blocks.json").write_text(
        json.dumps({"blocked": len(blocked), "by_class": dict(tally),
                    "rule_totals": dict(rule_tally), "cases": verdicts},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    print(f"  written to {args.out / 'semantic-gate-blocks.json'}")
    return 1 if unclassified else 0


if __name__ == "__main__":
    raise SystemExit(main())
