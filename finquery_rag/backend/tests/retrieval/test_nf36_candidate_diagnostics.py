from pathlib import Path

from src.evaluation.evaluation import EvaluationCase, Prediction, diagnose_candidate_stages
from src.retrieval.candidate_fusion import weighted_rrf
from src.retrieval.query_profile import QueryVariant, build_query_variants, profile_query


def test_original_query_is_preserved_and_compact_weight_is_lower():
    profile = profile_query(
        "What percentage of total revenue came from Alpha and Beta in 2023?",
        is_numeric=True,
    )
    variants = build_query_variants(profile)
    assert variants[0].text == profile.original_query
    assert variants[0].weight > variants[1].weight
    assert "2023" in variants[1].text


def test_fact_and_no_answer_queries_do_not_expand():
    fact = profile_query("What is the title of this document?", is_numeric=False)
    no_answer = profile_query("Does this document mention an unrelated person?", is_numeric=False)
    assert len(build_query_variants(fact)) == 1
    assert len(build_query_variants(no_answer)) == 1


def test_multi_document_query_enables_compact_variant():
    profile = profile_query(
        "Compare cash and revenue across documents for 2023.",
        is_numeric=True,
    )
    assert profile.is_multi_document
    assert len(build_query_variants(profile)) == 2


def test_weighted_rrf_dedupes_rows_skips_cells_and_is_deterministic():
    original = QueryVariant("original", "cash 2023", 1.0)
    compact = QueryVariant("compact", "cash 2023", 0.85)
    row = {"doc_id": "row:report:7", "metadata": {"type": "table_row"}}
    cell = {"doc_id": "cell:report:7:1", "metadata": {"type": "table_cell"}}
    first = weighted_rrf([(original, "dense", [row, cell]), (compact, "bm25", [row])])
    second = weighted_rrf([(original, "dense", [row, cell]), (compact, "bm25", [row])])
    assert first == second
    assert [item["doc_id"] for item in first] == ["row:report:7"]
    assert len(first[0]["retrieval_provenance"]) == 2


def test_stage_classification_distinguishes_rerank_loss():
    case = EvaluationCase.from_dict({
        "id": "generic-case",
        "question": "What was revenue?",
        "expected_sources": [{"filename": "report.pdf", "page": 3}],
    })
    prediction = Prediction.from_dict({
        "id": "generic-case",
        "answer": "",
        "retrieval_debug": {
            "candidate_stages": {
                "bm25": [{"document_id": "report.pdf", "page": 3, "evidence_id": "x"}],
                "dense": [],
                "rrf": [{"document_id": "report.pdf", "page": 3, "evidence_id": "x"}],
                "reranker": [],
                "final": [],
            }
        },
    })
    report = diagnose_candidate_stages([case], {"generic-case": prediction})
    assert report["summary"]["lost_during_rerank"] == 1


def test_nf36_production_modules_do_not_contain_case_specific_markers():
    root = Path(__file__).parents[2] / "src" / "retrieval"
    text = chr(10).join((root / name).read_text() for name in (
        "query_profile.py", "candidate_fusion.py", "candidate_trace.py",
    ))
    forbidden = ("WIPO", "PDF Solutions", "page 10", "p10", "case_id ==")
    assert all(marker not in text for marker in forbidden)
