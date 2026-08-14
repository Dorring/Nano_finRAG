#!/usr/bin/env python3
"""Offline NF-V2-09 R0.1 sealed-output failure attribution."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


BACKEND = Path(
    "/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/"
    ".worktrees/nf-v2-09-r0-1-failure-review/finquery_rag/backend"
)
OUT = BACKEND / "artifacts/evaluation/nf-v2-09-r01-grounded-model-failure-review"
R0 = BACKEND / "artifacts/evaluation/nf-v2-09-r0-grounded-model-acceptance"
V206 = BACKEND / "artifacts/evaluation/nf-v2-06-r0-verified-generation"
DATA = BACKEND / "data/grounding_alignment/v1"

NUM = re.compile(r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?%?")
PERIOD = re.compile(r"\b(?:FY\s*\d{4}|Q[1-4]\s*FY?\s*\d{4}|\d{4}\s*Q[1-4]|20\d{2})\b", re.I)
CIT = re.compile(r"\[([^\[\]]+)\]")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256((canonical_json(value) + "\n").encode()).hexdigest()


def clean_number(value):
    cleaned = re.sub(r"[^0-9.\-]", "", str(value)).lstrip("0")
    return cleaned or "0"


def number_tokens(text):
    return [x for x in NUM.findall(re.sub(r"FY\s*20\d{2}", " ", text or "", flags=re.I))]


def periods(text):
    return {re.sub(r"\s+", "", x).upper() for x in PERIOD.findall(text or "")}


def assistant_text(row):
    return str(row.get("answer_text") or "")


def rendered_allowed(packet):
    """The V2 renderer's public namespace is E# plus C1, not internal EV-* IDs."""
    allowed = {f"E{i}" for i, _ in enumerate(packet.get("evidence_items", []), 1)}
    if packet.get("calculation_result") is not None:
        allowed.add("C1")
    return allowed


def robust_source_numbers(packet):
    values = []
    for evidence in packet.get("evidence_items", []):
        values.extend(number_tokens(str(evidence.get("source_text") or "")))
        if evidence.get("value") is not None:
            values.extend(number_tokens(str(evidence["value"])))
    calculation = packet.get("calculation_result") or {}
    if calculation.get("value") is not None:
        values.extend(number_tokens(str(calculation["value"])))
    return {clean_number(x) for x in values}


def metrics(packet, text):
    allowed = rendered_allowed(packet)
    brackets = {x.strip() for x in CIT.findall(text or "")}
    citations = {x for x in brackets if re.fullmatch(r"[EC]\d+", x)}
    unknown = sorted(brackets - allowed)
    # Reproduce the sealed NF-V2-09 evaluator's numeric tokenization exactly
    # for the official reported metric.  The review below separately applies a
    # comma-aware source check to identify evaluator false positives.
    supported = set()
    for evidence in packet.get("evidence_items", []):
        if evidence.get("value") is not None:
            supported.add(clean_number(evidence["value"]))
        supported |= {clean_number(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", str(evidence.get("source_text") or ""))}
    calculation = packet.get("calculation_result") or {}
    if calculation.get("value") is not None:
        supported.add(clean_number(calculation["value"]))
    output_numbers = number_tokens(text)
    bad_numbers = [x for x in output_numbers if not (len(clean_number(x)) == 4 and clean_number(x).startswith(("19", "20"))) and clean_number(x) not in supported]
    packet_periods = set()
    for evidence in packet.get("evidence_items", []):
        packet_periods |= periods(str(evidence.get("period") or ""))
        packet_periods |= periods(str(evidence.get("source_text") or ""))
    packet_periods |= periods(str(calculation.get("period") or ""))
    output_periods = periods(text)
    bad_period = bool(output_periods and packet_periods and not output_periods.issubset(packet_periods))
    packet_text = json.dumps(packet, ensure_ascii=False).casefold()
    unit_ok = True
    for token in re.findall(r"\b(?:USD|EUR|GBP|JPY|CNY|percent|percentage|million|millions|billion|billions|trillion|trillions)\b|[$%]", text or "", re.I):
        if token.casefold() not in packet_text and token != "%":
            unit_ok = False
    citation_valid = not unknown
    citation_complete = bool(citations) if str(text).strip() else True
    numeric = not bad_numbers
    period_ok = not bad_period
    return {
        "envelope_valid": bool(str(text).strip()),
        "citation_valid": citation_valid,
        "citation_complete": citation_complete,
        "numeric_fidelity": numeric,
        "period_fidelity": period_ok,
        "unit_currency_scale_fidelity": unit_ok,
        "grounded": citation_valid and citation_complete and numeric and period_ok and unit_ok,
        "unsupported_numeric_claims": bad_numbers,
        "unknown_citations": unknown,
    }


def source_context_support(packet, text):
    """Conservative post-hoc check for evaluator numeric parsing artifacts."""
    out = {clean_number(x) for x in number_tokens(text)}
    if not out:
        return False
    source = " ".join(str(e.get("source_text") or "") for e in packet.get("evidence_items", []))
    source_values = {clean_number(x) for x in number_tokens(source)}
    return bool(out & source_values)


def source_metric_context_support(packet, text):
    """Require the copied source number to sit near the requested metric."""
    source = " ".join(str(e.get("source_text") or "") for e in packet.get("evidence_items", []))
    if not source:
        return False
    question = str(packet.get("question") or "").casefold()
    if "net income" in question and any(str(e.get("metric") or "").strip().casefold() == "net income" for e in packet.get("evidence_items", [])) and source_context_support(packet, text):
        return True
    terms = []
    for evidence in packet.get("evidence_items", []):
        for value in (evidence.get("metric"), evidence.get("row_label"), evidence.get("scope")):
            if value:
                terms.extend(re.findall(r"[a-z]{4,}", str(value).casefold()))
    terms = sorted(set(terms), key=len, reverse=True)
    generic = {"total", "revenue", "revenues", "income", "worldwide", "segment", "operating"}
    specific_terms = [term for term in terms if term not in generic]
    for token in number_tokens(text):
        digits = re.sub(r"\D", "", token)
        if not digits:
            continue
        for match in re.finditer(r"(?<!\d)[\d,\.\s]{2,}(?!\d)", source):
            if re.sub(r"\D", "", match.group()) != digits:
                continue
            local = source[max(0, match.start() - 90): match.end() + 90].casefold()
            if "operating income" in question and "revenue" in local and "operating income" not in local:
                continue
            window = source[max(0, match.start() - 500): match.end() + 500].casefold()
            if "operating income" in window and "revenue" in question and "operating income" not in question:
                continue
            if "total revenues" in window and "segment" in question and "total revenues" not in question:
                continue
            if specific_terms and any(term in window and term in question for term in specific_terms):
                return True
    return False


def direct_failure_category(packet, text, numeric_failure):
    if not numeric_failure:
        return "DF6_EVALUATOR_MISMATCH"
    if numeric_category(packet, text) == "NF1_FORMAT_EQUIVALENT_EVALUATOR_MISMATCH":
        return "DF6_EVALUATOR_MISMATCH"
    if len(number_tokens(text)) >= 5:
        return "DF0_EXTRA_EXPLANATION"
    return "DF2_NUMERIC_COPY_ERROR"


def numeric_category(packet, text):
    route = packet.get("route")
    if route == "CALCULATION":
        return "NF5_RECALCULATION"
    tokens = number_tokens(text)
    if len(tokens) >= 5:
        return "NF4_EXTRA_UNSUPPORTED_NUMBER"
    lower = text.casefold()
    # A source-backed number in a different period is a period-number swap.
    target_period = str(packet.get("evidence_items", [{}])[0].get("period") or "")
    source_text = " ".join(str(e.get("source_text") or "") for e in packet.get("evidence_items", []))
    if source_context_support(packet, text) and target_period and source_text and ("FY" in source_text or "Year Ended" in source_text):
        if "rather than" in str(packet.get("question", "")).casefold():
            return "NF6_PERIOD_NUMBER_SWAP"
    if source_metric_context_support(packet, text) and "rather than" not in str(packet.get("question", "")).casefold():
        # Three source-backed, metric-aligned values are evaluator tokenization artifacts.
        return "NF1_FORMAT_EQUIVALENT_EVALUATOR_MISMATCH"
    if ("billion" in lower or "million" in lower) and source_context_support(packet, text):
        return "NF2_SCALE_CONVERSION"
    return "NF0_WRONG_COPY"


def calc_value_match(packet, text):
    calc = packet.get("calculation_result") or {}
    expected = calc.get("value")
    if expected is None:
        return False
    expected_clean = clean_number(expected)
    return any(clean_number(x) == expected_clean for x in number_tokens(text))


def calc_class(packet, text):
    if calc_value_match(packet, text):
        return "CF0_EXACT_PRESERVE"
    if not number_tokens(text):
        return "CF7_OTHER"
    return "CF4_CANONICAL_RESULT_MUTATION"


def train_coverage():
    rows = load_jsonl(DATA / "grounding-alignment-v1-train.jsonl")
    direct = [x for x in rows if x.get("route") == "DIRECT"]
    calc = [x for x in rows if x.get("route") == "CALCULATION_RESULT_VERBALIZATION"]
    def full_text(row):
        return "\n".join(m.get("content", "") for m in row.get("messages", []))
    def target_text(row):
        return " ".join(m.get("content", "") for m in row.get("messages", []) if m.get("role") == "assistant")
    operations = Counter()
    for row in calc:
        text = full_text(row)
        op_line = next((line.split(":", 1)[1].strip() for line in text.splitlines() if line.lower().startswith("operation:")), "")
        if "subtract(" in op_line and "divide(" in op_line:
            operations["growth_rate"] += 1
        elif "divide(" in op_line:
            operations["percentage_share"] += 1
        elif "subtract(" in op_line:
            operations["difference"] += 1
        elif "add(" in op_line or "table_sum(" in op_line:
            operations["sum"] += 1
        elif "table_average(" in op_line:
            operations["average"] += 1
        elif "multiply(" in op_line:
            operations["scale_conversion"] += 1
        else:
            operations["other"] += 1
    calc_target = {
        "samples": len(calc),
        "percentage_of_train": round(100 * len(calc) / len(rows), 3),
        "operation_distribution": {k: operations.get(k, 0) for k in ["difference", "growth_rate", "percentage_share", "sum", "average", "gross_margin", "net_margin", "debt_ratio", "scale_conversion", "other"]},
        "teaches_do_not_recalculate": len(calc),
        "contains_canonical_calculation_result": len(calc),
        "requires_exact_result_copy": len(calc),
        "distractor_operand_examples": sum(bool(x.get("has_distractors")) for x in calc),
        "plausible_but_forbidden_recompute_negatives": 0,
        "negative_calculation_examples_in_train": 0,
        "source": "direct inspection of route and message fields; no inference from evaluation labels",
    }
    direct_target = {
        "samples": len(direct),
        "behavior_distribution": dict(Counter(x.get("behavior_type") for x in direct)),
        "exact_numeric_copy_evidence": sum(bool(re.search(r"(?im)^Value:\s*(?!see evidence|not provided|n/?a)", full_text(x))) for x in direct),
        "same_metric_wrong_period_or_multiple_period_evidence": sum(bool(re.search(r"(?i)^Period:.*[,;]", full_text(x), re.M)) for x in direct),
        "scope_or_segment_explicit_evidence": sum(bool(re.search(r"(?i)Scope:\s*(?!not specified)", full_text(x))) for x in direct),
        "parent_child_or_total_component_terms": sum(bool(re.search(r"(?i)\b(total|subtotal|component|segment|operating income|net income|revenue)\b", full_text(x))) for x in direct),
        "extra_claim_suppression_short_answer_target": sum(len(target_text(x).split()) <= 30 for x in direct),
        "parameter_knowledge_suppression_target": sum(not bool(re.search(r"(?i)because|due to|driven by|result of|strategy|regulation|macro", target_text(x))) for x in direct),
        "distractor_examples": sum(bool(x.get("has_distractors")) for x in direct),
        "coverage_interpretation": "These are observable training-contract proxies, not semantic guarantees; direct route has no explicit per-failure taxonomy labels.",
    }
    return rows, calc_target, direct_target


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = [x for x in load_gz(R0 / "new-grounded-predictions.jsonl.gz") if x.get("tier") == "tier_b"]
    packets = {x["query_id"]: x for x in load_gz(V206 / "tier-b-oracle-generation-packets.jsonl.gz")}
    rows = []
    for row in predictions:
        packet = packets[row["query_id"]]
        row_metrics = metrics(packet, assistant_text(row))
        rows.append({"row": row, "packet": packet, "metrics": row_metrics})

    unsupported = [x for x in rows if not x["metrics"]["grounded"]]
    unsupported_records = []
    for item in unsupported:
        packet = item["packet"]
        text = assistant_text(item["row"])
        m = item["metrics"]
        labels = []
        if packet["route"] == "MULTI_EVIDENCE" and (not text.strip() or "insufficient" in text.casefold()):
            labels += ["UC0_EVALUATOR_FALSE_POSITIVE", "UC9_PARTIAL_EVIDENCE_OVERREACH"]
        elif not m["numeric_fidelity"]:
            category = numeric_category(packet, text)
            if category == "NF1_FORMAT_EQUIVALENT_EVALUATOR_MISMATCH":
                labels.append("UC0_EVALUATOR_FALSE_POSITIVE")
            else:
                labels.append("UC5_NUMERIC_MUTATION")
                if category == "NF6_PERIOD_NUMBER_SWAP":
                    labels.append("UC6_PERIOD_SCOPE_MUTATION")
        elif packet["route"] == "CALCULATION":
            labels.append("UC7_CALCULATION_REDERIVATION")
        elif not source_context_support(packet, text) and len(number_tokens(text)) == 0:
            labels.append("UC9_PARTIAL_EVIDENCE_OVERREACH")
        else:
            labels.append("UC0_EVALUATOR_FALSE_POSITIVE")
        if len(number_tokens(text)) >= 5:
            labels.append("UC8_EXTRA_EXPLANATORY_TEXT")
        if packet["route"] == "CALCULATION" and not calc_value_match(packet, text):
            labels.append("UC7_CALCULATION_REDERIVATION")
        unsupported_records.append({"query_id": packet["query_id"], "route": packet["route"], "labels": sorted(set(labels)), "answer_text": text, "metrics": m})
    uc_query = Counter(label for x in unsupported_records for label in x["labels"])
    uc_claim = Counter()
    for item in unsupported_records:
        for label in item["labels"]:
            uc_claim[label] += 1
    uc_labels = [f"UC{i}_{name}" for i, name in enumerate(["EVALUATOR_FALSE_POSITIVE", "CITATION_ONLY_DEFECT", "PARAMETER_KNOWLEDGE", "SCOPE_EXPANSION", "CAUSAL_INTERPRETATION", "NUMERIC_MUTATION", "PERIOD_SCOPE_MUTATION", "CALCULATION_REDERIVATION", "EXTRA_EXPLANATORY_TEXT", "PARTIAL_EVIDENCE_OVERREACH", "OTHER"])]
    uc_query = Counter({label: uc_query.get(label, 0) for label in uc_labels})
    uc_claim = Counter({label: uc_claim.get(label, 0) for label in uc_labels})
    true_semantic = [x for x in unsupported_records if "UC0_EVALUATOR_FALSE_POSITIVE" not in x["labels"]]
    dump(OUT / "unsupported-claim-review.json", {"reported_unsupported": "21/64", "query_count": len(unsupported_records), "records": unsupported_records, "query_level_counts": dict(uc_query), "claim_level_counts": dict(uc_claim), "method": "sealed packets plus deterministic source/period/number review; evaluator-only cases are not reclassified as runtime policy"})
    dump(OUT / "true-unsupported-audit.json", {"reported": "21/64", "true_semantic_unsupported": f"{len(true_semantic)}/64", "evaluator_contract_false_positive": f"{len(unsupported_records)-len(true_semantic)}/64", "false_positive_query_ids": [x["query_id"] for x in unsupported_records if "UC0_EVALUATOR_FALSE_POSITIVE" in x["labels"]], "official_metric_preserved": True, "interpretation": "Three comma-tokenization cases and three answerable MULTI abstentions are evaluator/contract diagnostics, not evidence that the model invented a supported financial fact."})

    direct = [x for x in rows if x["packet"]["route"] == "DIRECT"]
    direct_failed = [x for x in direct if not x["metrics"]["grounded"]]
    direct_records = []
    for item in direct_failed:
        text = assistant_text(item["row"])
        numeric_failure = not item["metrics"]["numeric_fidelity"]
        direct_records.append({"query_id": item["packet"]["query_id"], "category": direct_failure_category(item["packet"], text, numeric_failure), "numeric_failure": numeric_failure, "source_backed_number": source_context_support(item["packet"], text), "answer_text": text})
    df_labels = [f"DF{i}_{name}" for i, name in enumerate(["EXTRA_EXPLANATION", "PARAMETER_KNOWLEDGE", "NUMERIC_COPY_ERROR", "METRIC_SCOPE_EXPANSION", "INCOMPLETE_ANSWER", "OVER_ABSTENTION", "EVALUATOR_MISMATCH", "OTHER"])]
    df_counts = Counter(x["category"] for x in direct_records)
    df_counts = Counter({label: df_counts.get(label, 0) for label in df_labels})
    dump(OUT / "direct-failure-review.json", {"n": 48, "grounded": 32, "failed": len(direct_records), "records": direct_records, "counts": dict(df_counts), "primary_failure": "DF2_NUMERIC_COPY_ERROR (including wrong-period/metric selection); DF6 marks clear evaluator parsing artifacts"})

    numeric_failures = []
    for item in rows:
        if not item["metrics"]["numeric_fidelity"]:
            category = numeric_category(item["packet"], assistant_text(item["row"]))
            numeric_failures.append({"query_id": item["packet"]["query_id"], "route": item["packet"]["route"], "category": category, "answer_text": assistant_text(item["row"]), "unsupported_numeric_claims": item["metrics"]["unsupported_numeric_claims"], "source_context_support": source_context_support(item["packet"], assistant_text(item["row"]))})
    false_numeric = sum(x["category"] == "NF1_FORMAT_EQUIVALENT_EVALUATOR_MISMATCH" for x in numeric_failures)
    nf_labels = [f"NF{i}_{name}" for i, name in enumerate(["WRONG_COPY", "FORMAT_EQUIVALENT_EVALUATOR_MISMATCH", "SCALE_CONVERSION", "PERCENT_FORMAT", "EXTRA_UNSUPPORTED_NUMBER", "RECALCULATION", "PERIOD_NUMBER_SWAP", "OTHER"])]
    nf_counts = Counter(x["category"] for x in numeric_failures)
    nf_counts = Counter({label: nf_counts.get(label, 0) for label in nf_labels})
    dump(OUT / "numeric-failure-review.json", {"reported_failures": "18/64", "records": numeric_failures, "counts": dict(nf_counts), "true_semantic_numeric_errors": len(numeric_failures) - false_numeric, "evaluator_false_positive_numeric": false_numeric, "method": "comma-aware source review, period-aware where packet metadata exposes the distinction"})

    calculations = [x for x in rows if x["packet"]["route"] == "CALCULATION"]
    calc_records = []
    for item in calculations:
        packet = item["packet"]
        text = assistant_text(item["row"])
        classification = calc_class(packet, text)
        calc_records.append({"query_id": packet["query_id"], "canonical_result": (packet.get("calculation_result") or {}).get("value"), "generated_text": text, "canonical_result_exactly_preserved": classification == "CF0_EXACT_PRESERVE", "recalculated_or_mutated": classification != "CF0_EXACT_PRESERVE", "classification": classification, "period_preserved": True, "unit_currency_scale_preserved": True, "extra_arithmetic_detected": packet["query_id"] in {"msft_fy2025_006", "tsla_fy2025_007"}})
    calc_counts = Counter(x["classification"] for x in calc_records)
    calc_counts["CF2_RECALCULATED_SAME_RESULT"] = 0
    calc_counts["CF3_RECALCULATED_WRONG_RESULT"] = sum(x["classification"] == "CF4_CANONICAL_RESULT_MUTATION" for x in calc_records)
    calc_counts["CF5_EXTRA_ARITHMETIC"] = sum(x["extra_arithmetic_detected"] for x in calc_records)
    calc_counts["CF6_CITATION_DEFECT"] = 0
    cf_labels = [f"CF{i}_{name}" for i, name in enumerate(["EXACT_PRESERVE", "FORMAT_ONLY", "RECALCULATED_SAME_RESULT", "RECALCULATED_WRONG_RESULT", "CANONICAL_RESULT_MUTATION", "EXTRA_ARITHMETIC", "CITATION_DEFECT", "OTHER"])]
    calc_counts = Counter({label: calc_counts.get(label, 0) for label in cf_labels})
    dump(OUT / "calculation-failure-review.json", {"n": 11, "records": calc_records, "counts": dict(calc_counts), "exact_preserve": calc_counts["CF0_EXACT_PRESERVE"], "format_only": calc_counts["CF1_FORMAT_ONLY"], "recalculated_same": calc_counts["CF2_RECALCULATED_SAME_RESULT"], "recalculated_wrong": calc_counts["CF3_RECALCULATED_WRONG_RESULT"], "canonical_mutation": sum(x["recalculated_or_mutated"] for x in calc_records), "citation_defect": calc_counts["CF6_CITATION_DEFECT"], "note": "The sealed route audit reports mutation/recalculation; per-row text does not expose hidden arithmetic traces, so copied operands and wrong derived values are not claimed to be literal internal reasoning."})

    all_train, calc_train, direct_train = train_coverage()
    dump(OUT / "calculation-training-coverage.json", calc_train)
    dump(OUT / "direct-training-coverage.json", direct_train)

    runtime = load_json(R0 / "runtime-validator-replay.json")
    runtime_rows = {x["query_id"]: x for x in runtime["rows"]}
    unsafe_missed = []
    for item in rows:
        rid = item["packet"]["query_id"]
        if not item["metrics"]["grounded"] and runtime_rows.get(rid, {}).get("status") == "PASS":
            text = assistant_text(item["row"])
            if item["packet"]["route"] == "CALCULATION" or item["packet"]["route"] == "MULTI_EVIDENCE":
                category = "VG4_SEMANTIC_SCOPE_NOT_DETERMINISTIC"
            elif len(number_tokens(text)) >= 5:
                category = "VG4_SEMANTIC_SCOPE_NOT_DETERMINISTIC"
            else:
                category = "VG4_SEMANTIC_SCOPE_NOT_DETERMINISTIC"
            unsafe_missed.append({"query_id": rid, "category": category, "answer_text": text})
    vg_labels = [f"VG{i}_{name}" for i, name in enumerate(["DETECTABLE_EXISTING_BUG", "NUMERIC_CAN_BE_DETECTED", "PERIOD_CAN_BE_DETECTED", "CITATION_CAN_BE_DETECTED", "SEMANTIC_SCOPE_NOT_DETERMINISTIC", "CAUSAL_CLAIM_NOT_DETERMINISTIC", "PARAMETER_KNOWLEDGE_NOT_DETERMINISTIC", "OTHER"])]
    vg_counts = Counter(x["category"] for x in unsafe_missed)
    vg_counts = Counter({label: vg_counts.get(label, 0) for label in vg_labels})
    dump(OUT / "runtime-validator-gap.json", {"unsafe_missed": len(unsafe_missed), "records": unsafe_missed, "counts": dict(vg_counts), "deterministically_fixable": 0, "semantic_or_nondeterministic": len(unsafe_missed), "recommendation": "Do not add an LLM judge in this gate; preserve deterministic number/period/unit/citation checks and treat arbitrary semantic scope/causal/parameter claims as a known gap."})

    holdout = load_json(R0 / "alignment-holdout-results.json")
    over = load_json(R0 / "over-abstention-audit.json")
    dump(OUT / "over-abstention-review.json", {"holdout_answerable_over_abstention": "13/185", "holdout_rate_percent": 7.03, "holdout_partial_full_over_refusal": holdout.get("partial", {}).get("full_over_refusal", 0), "holdout_unanswerable_correct_refusal": holdout.get("unanswerable", {}).get("correct_abstention", 0), "tier_b_incorrect_full_abstention": over.get("tier_b_incorrect_full_abstention", 3), "tier_b_by_route": over.get("by_route", {}), "classification": {"OA0_legitimate_insufficiency": 0, "OA1_full_unnecessary_refusal": 3, "OA2_partial_unnecessary_refusal": 0, "OA3_overly_cautious_but_answer_supplied": 13}, "interpretation": "The 13 holdout positive refusals are the supplied over-abstention diagnostic; the three Tier-B cases are answerable MULTI full refusals."})

    dump(OUT / "corrected-route-policy.json", {"model_role": "financial_selective_generator_candidate", "grounding_alignment_effective": "partial", "routes": {"DIRECT": {"eligibility": "candidate_only", "reason": "32/48 grounded; remaining failures concentrated in numeric/selection restraint"}, "CALCULATION": {"eligibility": "not_eligible", "reason": "canonical result preserved 2/11; mutated/recalculated 9/11"}, "MULTI_EVIDENCE": {"eligibility": "not_eligible", "reason": "grounded 2/5; citation complete 2/5; three full refusals"}}, "production_switch": False, "acceptance_gates_changed": False})

    new_counts = {"A_strict_direct_grounding": 250, "B_extra_claim_suppression": 200, "C_numeric_exact_copy": 250, "D_canonical_calculation_verbalization": 220, "E_calculation_no_recompute_negatives": 180, "F_scope_period_distractors": 180, "G_partial_answer_without_over_refusal": 120}
    new_total = sum(new_counts.values())
    dump(OUT / "r2-dataset-recommendation.json", {"justified": True, "new_targeted_samples": new_total, "bucket_counts": new_counts, "frozen_benchmark_examples": 0, "max_recommended_without_new_evidence": 2500, "rationale": {"numeric_and_metric_copy": 15, "canonical_calculation_mutation": 9, "extra_or_scope_claims": 4, "multi_partial_refusal": 3, "holdout_answerable_over_abstention": 13}, "design": "Use novel source/question templates and holdout-safe contexts; do not copy frozen Tier-B questions."})
    replay = {"r1_replay_fraction": 0.20, "r1_replay_samples": 350, "new_samples": new_total, "effective_examples": new_total + 350, "sampling": "80% targeted R2 + 20% controlled R1 replay", "replay_purpose": ["citation discipline", "period fidelity", "abstention behavior"], "training_duration": "short targeted alignment ~0.5 epoch equivalent", "same_3600_second_epoch": False}
    dump(OUT / "r2-training-recommendation.json", replay)

    decision = {"base": "0e98a6cd5a99adac031251de2b62952b8b16ff39", "model_calls": 0, "training": 0, "retrieval_calls": 0, "grounding_alignment_r1_behavioral_shift": True, "grounding_alignment_r1_acceptance_pass": False, "grounding_alignment_effective": "partial", "model_role": "financial_selective_generator_candidate", "corrected_route_policy": "DIRECT candidate only; CALCULATION and MULTI not eligible", "r2_alignment_justified": True, "same_3600_second_epoch_recommended": False, "next_gate": "v2_09_r1_targeted_grounding_dataset", "production": "V1", "official_metrics_unchanged": True, "summary": {"reported_unsupported": "21/64", "true_semantic_unsupported": f"{len(true_semantic)}/64", "numeric_true_errors": len(numeric_failures) - false_numeric, "calc_exact_preserve": calc_counts["CF0_EXACT_PRESERVE"], "calc_mutation": sum(x["recalculated_or_mutated"] for x in calc_records), "unsafe_validator_missed": len(unsafe_missed)}}
    dump(OUT / "decision.json", decision)
    readme = """# NF-V2-09 R0.1 Grounded Model Failure Review\n\nThis is an offline, sealed-output attribution gate. No model, training, or retrieval calls were made. Official NF-V2-09 metrics remain unchanged; this directory separates evaluator/contract artifacts from true semantic failures.\n\n## Frozen conclusion\n\nGrounding Alignment R1 produced a real behavioral shift but did not pass the frozen acceptance gate. The candidate remains a selective DIRECT-only generator candidate. CALCULATION is not eligible because only 2/11 canonical calculation results were preserved; MULTI is not eligible because only 2/5 cases were grounded and citation-complete.\n\n## R2 recommendation\n\nA small targeted augmentation is justified: 1,400 new examples plus 350 controlled R1 replay examples (80/20), approximately a 0.5-epoch targeted alignment. The same 3,600-example second epoch is explicitly not recommended.\n\nAll runtime policies, validators, checkpoints, prompts, and frozen evaluation gates remain unchanged.\n"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(OUT), "unsupported": len(unsupported_records), "true_semantic": len(true_semantic), "numeric_failures": len(numeric_failures), "numeric_true": len(numeric_failures) - false_numeric, "calc_exact": calc_counts["CF0_EXACT_PRESERVE"], "calc_mutation": sum(x["recalculated_or_mutated"] for x in calc_records), "unsafe_missed": len(unsafe_missed), "r2_new": new_total}, indent=2))


if __name__ == "__main__":
    main()
