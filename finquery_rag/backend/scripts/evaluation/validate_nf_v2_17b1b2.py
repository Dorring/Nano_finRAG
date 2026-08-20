from __future__ import annotations
import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[4]
OUT = BASE / "finquery_rag/backend/artifacts/evaluation/nf-v2-17-fresh-blind-eval"
A5 = BASE / "finquery_rag/backend/artifacts/evaluation/nf-v2-17-financial-corpus-v2"
QUOTAS = {
    "SINGLE_EVIDENCE_FACT": 30,
    "MULTI_EVIDENCE": 20,
    "DETERMINISTIC_CALCULATION": 15,
    "TEMPORAL_PERIOD": 15,
    "AGENTIC_REPLAN": 15,
    "VERSION_TEMPORAL": 10,
    "CONFLICT_AMBIGUITY": 5,
    "NO_ANSWER_FAIL_CLOSED": 10,
}
EXPECTED = {
    "corpus": "3ef3d8e772dfb2d4e2594d18efe3c101c4a4a3bb108e0faa0d75d11c667421a3",
    "freeze": "63620b2183c4635f1ecff974935bc81a4d8ce678c72e72e94155d8f0a96e6929",
    "reservation": "8708ecf5b0f5ee056cf003238a510345c96cce720a41709d5eeb0c5d47e1dc23",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name):
    return [
        json.loads(x)
        for x in (OUT / name).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def sidecar(name):
    return (OUT / (name + ".sha256")).read_text(encoding="utf-8").split()[0]


def fail(msg):
    raise AssertionError(msg)


items = rows("fresh-blind-eval-v1.jsonl")
runtime = rows("fresh-blind-questions-v1.jsonl")
gold = rows("fresh-blind-gold-evidence-v1.jsonl")
refs = rows("fresh-blind-reference-answers-v1.jsonl")
anns = rows("fresh-blind-annotations-v1.jsonl")
if (
    len(items) != 120
    or len(runtime) != 120
    or len(gold) != 120
    or len(refs) != 120
    or len(anns) != 120
):
    fail("row count mismatch")
ids = [x["question_id"] for x in items]
if ids != [f"FBV1-{i:03d}" for i in range(1, 121)] or len(set(ids)) != 120:
    fail("question ids/order mismatch")
questions = [x["question"] for x in items]
norm = [
    " ".join("".join(c.lower() if c.isalnum() else " " for c in q).split())
    for q in questions
]
if len(set(questions)) != 120 or len(set(norm)) != 120:
    fail("duplicate questions in final pack")
task_counts = Counter(x["primary_task_type"] for x in items)
if dict(task_counts) != QUOTAS:
    fail(f"task quota mismatch: {dict(task_counts)}")
answer_counts = Counter(x["answerability"] for x in items)
if answer_counts != Counter({"ANSWERABLE": 105, "UNANSWERABLE": 15}):
    fail(f"answerability mismatch: {dict(answer_counts)}")
required = {
    "question_id",
    "question",
    "primary_task_type",
    "secondary_task_tags",
    "answerability",
    "company",
    "ticker",
    "entity_scope",
    "document_scope",
    "temporal_scope",
    "required_slots",
    "gold_evidence_ids",
    "gold_document_ids",
    "reference_answer",
    "expected_replan",
    "expected_replan_reason",
    "expected_tool_capabilities",
    "expected_conflict_state",
    "expected_terminal_state",
    "calculation_contract",
    "annotation_status",
    "review_status",
    "difficulty_notes",
}
for item in items:
    if not required.issubset(item):
        fail(f"missing item fields {item['question_id']}")
    if (
        item["review_status"] != "ACCEPTED"
        or item["annotation_status"] != "PASS1_CONSTRUCTED_PASS2_VERIFIED"
    ):
        fail(f"review status {item['question_id']}")
    if item["answerability"] == "ANSWERABLE" and not item["gold_evidence_ids"]:
        fail(f"answerable without Gold {item['question_id']}")
    if item["answerability"] == "UNANSWERABLE" and item["gold_evidence_ids"]:
        fail(f"unanswerable with Gold {item['question_id']}")
gold_by = {x["question_id"]: x for x in gold}
ref_by = {x["question_id"]: x for x in refs}
ann_by = {x["question_id"]: x for x in anns}
if set(gold_by) != set(ids) or set(ref_by) != set(ids) or set(ann_by) != set(ids):
    fail("sidecar ids mismatch")
provenance = 0
for item in items:
    g = gold_by[item["question_id"]]
    if g["answerability"] != item["answerability"]:
        fail(f"Gold answerability mismatch {item['question_id']}")
    if [e["evidence_id"] for e in g["gold_evidence"]] != item["gold_evidence_ids"]:
        fail(f"Gold evidence id mismatch {item['question_id']}")
    for e in g["gold_evidence"]:
        for field in [
            "evidence_id",
            "document_id",
            "chunk_id",
            "accession_number",
            "raw_sha256",
            "content",
            "company",
            "ticker",
            "period_semantics",
        ]:
            if field not in e or e[field] in (None, ""):
                fail(f"missing provenance {item['question_id']}:{field}")
        if not (e.get("period_end") or e.get("report_period_end")):
            fail(f"missing period provenance {item['question_id']}")
        provenance += 1
    if item["primary_task_type"] == "DETERMINISTIC_CALCULATION":
        c = item["calculation_contract"]
        if (
            not c
            or not c.get("operation")
            or not c.get("canonical_result")
            or len(c.get("operands", [])) < 2
        ):
            fail(f"calculation contract incomplete {item['question_id']}")
        for operand in c["operands"]:
            if operand.get("evidence_id") not in item["gold_evidence_ids"]:
                fail(f"calculation operand provenance {item['question_id']}")
    if item["primary_task_type"] == "TEMPORAL_PERIOD" and not item[
        "temporal_scope"
    ].get("period_semantics"):
        fail(f"temporal annotation incomplete {item['question_id']}")
replans = [x for x in items if x["expected_replan"]]
if len(replans) != 15:
    fail("replan count mismatch")
config_sha = sidecar("evaluation-config-v1.json")
for x in replans:
    if x[
        "retrieval_config_sha_for_replan"
    ] != config_sha or "top5_chunk_ids" not in x.get("replan_observation", {}):
        fail(f"replan not tied to frozen config {x['question_id']}")
if sum(x["primary_task_type"] == "CONFLICT_AMBIGUITY" for x in items) != 5:
    fail("conflict task count mismatch")
runtime_forbidden = {
    "gold_evidence_ids",
    "gold_document_ids",
    "required_slots",
    "reference_answer",
    "expected_replan",
    "expected_conflict_state",
    "expected_terminal_state",
    "calculation_contract",
    "distractor_evidence_ids",
}
for x in runtime:
    if set(x) != {"question_id", "question", "authorized_corpus"}:
        fail(f"runtime projection keys {x['question_id']}")
    if any(k in json.dumps(x, ensure_ascii=False) for k in runtime_forbidden):
        fail(f"Gold leakage in runtime projection {x['question_id']}")
leak = json.loads(
    (OUT / "fresh-blind-leakage-audit-v1.json").read_text(encoding="utf-8")
)
for k in [
    "exact_duplicate_questions",
    "normalized_duplicate_questions",
    "high_lexical_similarity",
    "same_question_same_answer_fact",
    "blocking_leakage",
    "benchmark_answer_leakage",
]:
    if leak.get(k) != 0:
        fail(f"leakage {k}={leak.get(k)}")
review = json.loads((OUT / "fresh-blind-review-v1.json").read_text(encoding="utf-8"))
if (
    review.get("two_pass_qc") is not True
    or review.get("manual_packet_items", 0) < 20
    or review.get("manual_packet_items", 0) > 30
):
    fail("review gate mismatch")
freeze = json.loads(
    (OUT / "fresh-blind-evaluation-freeze.json").read_text(encoding="utf-8")
)
for key, name in [
    ("question_sha", "fresh-blind-questions-v1.jsonl"),
    ("gold_sha", "fresh-blind-gold-evidence-v1.jsonl"),
    ("reference_sha", "fresh-blind-reference-answers-v1.jsonl"),
    ("annotation_sha", "fresh-blind-annotations-v1.jsonl"),
    ("eval_sha", "fresh-blind-eval-v1.jsonl"),
]:
    if freeze.get(key) != sha(OUT / name) or freeze.get(key) != sidecar(name):
        fail(f"freeze hash mismatch {key}")
if (
    freeze.get("corpus_freeze_sha") != EXPECTED["freeze"]
    or freeze.get("searchable_corpus_sha") != EXPECTED["corpus"]
    or freeze.get("fresh_blind_reservation_sha") != EXPECTED["reservation"]
):
    fail("upstream freeze hash mismatch")
for name in [
    "trace-schema-v1.json",
    "metric-registry-v1.json",
    "evaluation-config-v1.json",
]:
    if sha(OUT / name) != sidecar(name):
        fail(f"schema/config hash mismatch {name}")
decision = json.loads((OUT / "b1-b2-decision.json").read_text(encoding="utf-8"))
if (
    decision.get("decision") != "FRESH_BLIND_PACK_ACCEPTED"
    or decision.get("accepted_questions") != 120
    or decision.get("model_calls") != 0
    or decision.get("training") != 0
    or decision.get("final_system_execution_performed") is not False
    or decision.get("gold_runtime_isolation") != "PASS"
):
    fail("decision gate mismatch")
report = {
    "status": "VALIDATION_PASS",
    "accepted": len(items),
    "task_counts": dict(task_counts),
    "answerability": dict(answer_counts),
    "gold_evidence_objects": provenance,
    "gold_provenance_complete": True,
    "questions_unique_exact": True,
    "questions_unique_normalized": True,
    "runtime_projection_isolated": True,
    "replans_tied_to_config": len(replans),
    "leakage_all_blocking_fields_zero": True,
    "two_pass_qc": True,
    "manual_review_packet_items": review["manual_packet_items"],
    "freeze_hashes_match": True,
    "upstream_freezes_match": True,
    "model_calls": 0,
    "training": 0,
    "final_system_execution_performed": False,
}
(OUT / "validation-report.json").write_text(
    json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, ensure_ascii=False, sort_keys=True))
