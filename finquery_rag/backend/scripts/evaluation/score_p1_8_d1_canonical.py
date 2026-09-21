"""D1-C1 canonical scorer.

One number token = optional accounting parens or sign, optional currency,
digits with separators, optional trailing percent.  Parenthesised negatives
keep their sign: `(1,694)` -> -1694, `1,694` -> 1694, `-1,694` -> -1694.
"""

import json
import os
import re
import sys

#: The benchmark's V2 artifact directory, overridable so a D1 arm that lives in
#: its own phase directory can be scored without being copied into the
#: benchmark's tree -- copying it there would make the scored bytes and the
#: sealed bytes the same files.
BASE = os.environ.get("P18_BASE", "/disk/qh/nano-finrag/artifacts/evaluation/p1-8-c-v2")
#: The gold.  A phase directory holds the run, not another benchmark, so the
#: gold defaults to V2's even when the arm is read from elsewhere.
GOLD = os.environ.get("P18_GOLD", BASE + "/gold-evidence-v1.jsonl")
ARM = sys.argv[1] if len(sys.argv) > 1 else "armB"

#: A number token.  The lookbehind must sit immediately before the digit, so
#: the currency and the space are matched *ahead* of it rather than inside the
#: thing the lookbehind inspects -- otherwise a match starting at the space in
#: `$ 2,039` is rejected and the number is silently skipped.
_NUMBER = r"(?:[$¥€]\s?)?\(?-?\d[\d,]*(?:\.\d+)?\)?%?"
_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9$])(" + _NUMBER + r")")
_CITATION_RE = re.compile(r"\[citation:[^\]]*\]")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


def parse(raw):
    text = str(raw if raw is not None else "").strip()
    percent = text.endswith("%")
    text = text.rstrip("%").replace("$", "").replace(",", "").replace(" ", "")
    negative = (text.startswith("(") and text.endswith(")")) or text.startswith("-")
    text = text.strip("()")
    if text.startswith("-"):
        negative = True
        text = text[1:]
    if not text:
        return None, percent
    try:
        value = float(text)
    except ValueError:
        return None, percent
    return (-value if negative else value), percent


def numbers(text):
    """Numeric tokens in `text`, with citation handles removed first.

    Year-shaped tokens are dropped only when they *are* contextual: a bare
    four-digit number with no currency and no percent, or a four-digit number
    sitting inside a `(…)` or `FY…` that is followed by a colon.  A bare
    ``2,039`` is a legitimate answer -- Pfizer's Xtandi net revenue is $2,039m
    and it is shaped exactly like a year -- so an unconditional `^(19|20)\\d{2}$`
    filter silently discards it.
    """

    stripped = _CITATION_RE.sub(" ", str(text or ""))
    out = []
    for match in _TOKEN_RE.finditer(stripped):
        raw = match.group(1)
        value, percent = parse(raw)
        if value is None:
            continue
        if _is_contextual_year(raw, stripped, match.end()):
            continue
        out.append((raw, value, percent))
    return out


def _is_contextual_year(raw, text, end):
    bare = raw.strip("()").lstrip("-").replace("$", "").replace(",", "").replace(" ", "").rstrip("%")
    if not _YEAR_RE.match(bare):
        return False
    if "$" in raw or raw.endswith("%"):
        return False
    tail = text[end:end + 2]
    return tail.lstrip().startswith(":") or not raw.strip().startswith("(")


def classify(answer, gold_value, operation):
    if operation in ("comparison", "ranking") or gold_value is None:
        return "UNSCORABLE", None, None
    target, _ = parse(gold_value)
    if target is None:
        return "UNSCORABLE", None, str(gold_value)
    normalized_gold = round(target, 6)
    tokens = numbers(answer)
    for _raw, value, percent in tokens:
        for candidate in ([value, value / 100.0] if percent else [value, value * 100.0]):
            if abs(candidate - target) <= max(0.005, abs(target) * 1e-3):
                return "STRICT_CORRECT", round(candidate, 6), normalized_gold
    for _raw, value, _percent in tokens:
        if abs(abs(value) - abs(target)) <= max(0.005, abs(target) * 1e-3):
            return "COMPARATOR_MISMATCH", round(value, 6), normalized_gold
    return "GENUINELY_INCORRECT", (round(tokens[0][1], 6) if tokens else None), normalized_gold


def main():
    gold = {}
    with open(GOLD, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                gold[row["id"]] = row
    rows = {}
    with open(BASE + "/" + ARM + "/replay-predictions.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row

    answerable = [c for c in gold if gold[c].get("expected_outcome") == "ANSWER"]
    released = [c for c in answerable if str(rows[c].get("release_status")) == "RELEASED"]

    tally = {}
    detail = []
    for case in released:
        verdict, normalized_answer, normalized_gold = classify(
            rows[case].get("answer"),
            gold[case].get("expected_value"),
            gold[case].get("operation"),
        )
        tally[verdict] = tally.get(verdict, 0) + 1
        detail.append((case, verdict, normalized_answer, normalized_gold,
                       gold[case].get("operation"), gold[case].get("expected_value"),
                       str(rows[case].get("answer"))[:56]))

    print("arm: %s   released: %d" % (ARM, len(released)))
    for name in ("STRICT_CORRECT", "COMPARATOR_MISMATCH", "GENUINELY_INCORRECT", "UNSCORABLE"):
        print("  %-22s %d" % (name, tally.get(name, 0)))
    print()
    print("non-STRICT_CORRECT:")
    for case, verdict, na, ng, op, gv, ans in detail:
        if verdict == "STRICT_CORRECT":
            continue
        print("  %-24s %-20s norm_ans=%-12s norm_gold=%-12s %s" % (case, verdict, na, ng, ans))
    correct = tally.get("STRICT_CORRECT", 0) + tally.get("UNSCORABLE", 0)
    print()
    print("canonical scorer: correct %d + relational %d = %d" % (
        tally.get("STRICT_CORRECT", 0), tally.get("UNSCORABLE", 0), correct))
    print("Released Accuracy = %d/%d = %.2f%%" % (correct, len(released), 100.0 * correct / len(released)))
    print("incorrect = %d" % (tally.get("COMPARATOR_MISMATCH", 0) + tally.get("GENUINELY_INCORRECT", 0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
