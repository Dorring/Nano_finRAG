"""Apply the PDF-audited Draft corrections without promoting Golden data.

The corrections are based on the eight exact PDFs recorded in the benchmark
corpus.  They update physical 1-based PDF pages and displayed table metadata,
but deliberately leave source identity binding and negative-evidence review
pending.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.evaluation.annotation_contract import annotation_contract_report, build_annotation_worklist
from scripts.evaluation.benchmark_foundation import load_jsonl, write_jsonl


VERIFIED_AT = "2026-08-02"


def _binding(
    page: int,
    printed_page: int,
    section: str,
    table_title: str,
    row_label: str,
    column_header: str,
) -> dict[str, Any]:
    return {
        "page": page,
        "printed_page": printed_page,
        "section": section,
        "table_title": table_title,
        "row_label": row_label,
        "column_header": column_header,
    }


SOURCE_BINDINGS: dict[str, list[dict[str, Any]]] = {
    "aapl_fy2025_001": [_binding(32, 29, "Consolidated Statements of Operations", "Consolidated Statements of Operations", "Total net sales", "September 27, 2025")],
    "aapl_fy2025_002": [_binding(26, 23, "Products and Services Performance", "Net sales by category", "Services", "2025")],
    "aapl_fy2025_003": [_binding(27, 24, "Gross Margin", "Products and Services gross margin and gross margin percentage", "Total gross margin percentage", "2025")],
    "aapl_fy2025_004": [_binding(26, 23, "Products and Services Performance", "Net sales by category", "iPhone", "2025")],
    "aapl_fy2025_005": [_binding(26, 23, "Products and Services Performance", "Net sales by category", "Mac", "2025")],
    "aapl_fy2025_006": [
        _binding(26, 23, "Products and Services Performance", "Net sales by category", "Total net sales", "2024"),
        _binding(26, 23, "Products and Services Performance", "Net sales by category", "Total net sales", "2025"),
    ],
    "aapl_fy2025_007": [
        _binding(26, 23, "Products and Services Performance", "Net sales by category", "iPhone", "2025"),
        _binding(26, 23, "Products and Services Performance", "Net sales by category", "Services", "2025"),
    ],
    "aapl_fy2025_009": [_binding(26, 23, "Products and Services Performance", "Net sales by category", "iPhone", "2024")],
    "jpm_fy2025_001": [_binding(199, 167, "Consolidated balance sheets", "Consolidated balance sheets", "Total assets", "2025")],
    "jpm_fy2025_002": [_binding(199, 167, "Consolidated balance sheets", "Consolidated balance sheets", "Total liabilities", "2025")],
    "jpm_fy2025_003": [_binding(197, 165, "Consolidated statements of income", "Consolidated statements of income", "Net income", "2025")],
    "jpm_fy2025_004": [_binding(97, 65, "Consumer & Community Banking", "Selected income statement data", "Net income", "2025")],
    "jpm_fy2025_005": [_binding(101, 69, "Commercial & Investment Bank", "Selected income statement data", "Net income", "2025")],
    "jpm_fy2025_006": [
        _binding(199, 167, "Consolidated balance sheets", "Consolidated balance sheets", "Total assets", "2024"),
        _binding(199, 167, "Consolidated balance sheets", "Consolidated balance sheets", "Total assets", "2025"),
    ],
    "jpm_fy2025_007": [
        _binding(97, 65, "Consumer & Community Banking", "Selected income statement data", "Net income", "2025"),
        _binding(199, 167, "Consolidated balance sheets", "Consolidated balance sheets", "Total liabilities", "2025"),
    ],
    "jpm_fy2025_009": [_binding(97, 65, "Consumer & Community Banking", "Selected income statement data", "Net income", "2024")],
    "ko_fy2025_001": [_binding(63, 61, "Consolidated Statements of Income", "Consolidated Statements of Income", "Net Operating Revenues", "2025")],
    "ko_fy2025_002": [_binding(63, 61, "Consolidated Statements of Income", "Consolidated Statements of Income", "Operating Income", "2025")],
    "ko_fy2025_003": [_binding(63, 61, "Consolidated Statements of Income", "Consolidated Statements of Income", "Net Income Attributable to Shareowners of The Coca-Cola Company", "2025")],
    "ko_fy2025_004": [_binding(119, 117, "Operating segments", "Operations by operating segment and Corporate", "Total net operating revenues", "EMEA / 2025")],
    "ko_fy2025_005": [_binding(119, 117, "Operating segments", "Operations by operating segment and Corporate", "Total net operating revenues", "North America / 2025")],
    "ko_fy2025_006": [
        _binding(63, 61, "Consolidated Statements of Income", "Consolidated Statements of Income", "Net Operating Revenues", "2024"),
        _binding(63, 61, "Consolidated Statements of Income", "Consolidated Statements of Income", "Net Operating Revenues", "2025"),
    ],
    "ko_fy2025_007": [
        _binding(119, 117, "Operating segments", "Operations by operating segment and Corporate", "Third party", "EMEA / 2025"),
        _binding(119, 117, "Operating segments", "Operations by operating segment and Corporate", "Total net operating revenues", "Consolidated / 2025"),
    ],
    "ko_fy2025_009": [_binding(119, 117, "Operating segments", "Operations by operating segment and Corporate", "Total net operating revenues", "EMEA / 2024")],
    "msft_fy2025_001": [_binding(80, 79, "Segment Information and Geographic Data", "Segment revenue, cost of revenue, operating expenses, and operating income", "Total revenue", "2025")],
    "msft_fy2025_002": [_binding(80, 79, "Segment Information and Geographic Data", "Segment revenue, cost of revenue, operating expenses, and operating income", "Productivity and Business Processes revenue", "2025")],
    "msft_fy2025_003": [_binding(80, 79, "Segment Information and Geographic Data", "Segment revenue, cost of revenue, operating expenses, and operating income", "More Personal Computing revenue", "2025")],
    "msft_fy2025_004": [_binding(80, 79, "Segment Information and Geographic Data", "Segment revenue, cost of revenue, operating expenses, and operating income", "Intelligent Cloud revenue", "2025")],
    "msft_fy2025_005": [_binding(80, 79, "Segment Information and Geographic Data", "Segment revenue, cost of revenue, operating expenses, and operating income", "Productivity and Business Processes operating income", "2025")],
    "msft_fy2025_006": [
        _binding(80, 79, "Segment Information and Geographic Data", "Segment revenue, cost of revenue, operating expenses, and operating income", "Total revenue", "2024"),
        _binding(80, 79, "Segment Information and Geographic Data", "Segment revenue, cost of revenue, operating expenses, and operating income", "Total revenue", "2025"),
    ],
    "msft_fy2025_007": [
        _binding(80, 79, "Segment Information and Geographic Data", "Segment revenue, cost of revenue, operating expenses, and operating income", "Intelligent Cloud revenue", "2025"),
        _binding(80, 79, "Segment Information and Geographic Data", "Segment revenue, cost of revenue, operating expenses, and operating income", "Total revenue", "2025"),
    ],
    "msft_fy2025_009": [_binding(80, 79, "Segment Information and Geographic Data", "Segment revenue, cost of revenue, operating expenses, and operating income", "Intelligent Cloud revenue", "2024")],
    "nvda_fy2025_001": [_binding(144, 52, "Consolidated Statements of Income", "Consolidated Statements of Income", "Revenue", "Jan 26, 2025")],
    "nvda_fy2025_002": [_binding(172, 80, "Revenue by End Market", "Revenue by End Market", "Gaming", "Jan 26, 2025")],
    "nvda_fy2025_003": [_binding(130, 38, "Fiscal Year 2025 Summary", "Fiscal Year 2025 Summary", "Gross margin", "Jan 26, 2025")],
    "nvda_fy2025_004": [_binding(172, 80, "Revenue by End Market", "Revenue by End Market", "Data Center", "Jan 26, 2025")],
    "nvda_fy2025_005": [_binding(172, 80, "Revenue by End Market", "Revenue by End Market", "Automotive", "Jan 26, 2025")],
    "nvda_fy2025_006": [
        _binding(144, 52, "Consolidated Statements of Income", "Consolidated Statements of Income", "Revenue", "Jan 28, 2024"),
        _binding(144, 52, "Consolidated Statements of Income", "Consolidated Statements of Income", "Revenue", "Jan 26, 2025"),
    ],
    "nvda_fy2025_007": [
        _binding(172, 80, "Revenue by End Market", "Revenue by End Market", "Data Center", "Jan 26, 2025"),
        _binding(130, 38, "Fiscal Year 2025 Summary", "Fiscal Year 2025 Summary", "Gross margin", "Jan 26, 2025"),
    ],
    "nvda_fy2025_009": [_binding(172, 80, "Revenue by End Market", "Revenue by End Market", "Data Center", "Jan 28, 2024")],
    "pfe_fy2024_001": [_binding(60, 51, "Consolidated Statements of Operations", "Consolidated Statements of Operations", "Total revenues", "2024")],
    "pfe_fy2024_002": [_binding(45, 36, "Total Revenues - Selected Product Discussion", "Selected product discussion", "Comirnaty - Worldwide", "2024")],
    "pfe_fy2024_003": [_binding(44, 35, "Total Revenues - Selected Product Discussion", "Selected product discussion", "Eliquis - Worldwide", "2024")],
    "pfe_fy2024_004": [_binding(45, 36, "Total Revenues - Selected Product Discussion", "Selected product discussion", "Paxlovid - Worldwide", "2024")],
    "pfe_fy2024_005": [_binding(44, 35, "Total Revenues - Selected Product Discussion", "Selected product discussion", "Prevnar family - Worldwide", "2024")],
    "pfe_fy2024_006": [
        _binding(60, 51, "Consolidated Statements of Operations", "Consolidated Statements of Operations", "Total revenues", "2023"),
        _binding(60, 51, "Consolidated Statements of Operations", "Consolidated Statements of Operations", "Total revenues", "2024"),
    ],
    "pfe_fy2024_007": [
        _binding(45, 36, "Total Revenues - Selected Product Discussion", "Selected product discussion", "Comirnaty - Worldwide", "2024"),
        _binding(60, 51, "Consolidated Statements of Operations", "Consolidated Statements of Operations", "Research and development expenses", "2024"),
    ],
    "pfe_fy2024_009": [_binding(45, 36, "Total Revenues - Selected Product Discussion", "Selected product discussion", "Paxlovid - Worldwide", "2023")],
    "tsla_fy2025_001": [_binding(69, 49, "Consolidated Statements of Operations", "Consolidated Statements of Operations", "Total revenues", "2025")],
    "tsla_fy2025_002": [_binding(69, 49, "Consolidated Statements of Operations", "Consolidated Statements of Operations", "Services and other", "2025")],
    "tsla_fy2025_003": [_binding(69, 49, "Consolidated Statements of Operations", "Consolidated Statements of Operations", "Automotive regulatory credits", "2025")],
    "tsla_fy2025_004": [_binding(55, 35, "Results of Operations - Revenues", "Revenues", "Total automotive revenues", "2025")],
    "tsla_fy2025_005": [_binding(55, 35, "Results of Operations - Revenues", "Revenues", "Energy generation and storage segment revenue", "2025")],
    "tsla_fy2025_006": [
        _binding(69, 49, "Consolidated Statements of Operations", "Consolidated Statements of Operations", "Total revenues", "2024"),
        _binding(69, 49, "Consolidated Statements of Operations", "Consolidated Statements of Operations", "Total revenues", "2025"),
    ],
    "tsla_fy2025_007": [
        _binding(55, 35, "Results of Operations - Revenues", "Revenues", "Total automotive revenues", "2025"),
        _binding(55, 35, "Results of Operations - Revenues", "Revenues", "Energy generation and storage segment revenue", "2025"),
    ],
    "tsla_fy2025_009": [_binding(55, 35, "Results of Operations - Revenues", "Revenues", "Total automotive revenues", "2024")],
    "v_fy2025_001": [_binding(75, 60, "Consolidated Statements of Operations", "Consolidated Statements of Operations", "Net revenue", "2025")],
    "v_fy2025_002": [_binding(4, 2, "Operational highlights", "Operational highlights", "Total volume", "FY 2025")],
    "v_fy2025_003": [_binding(4, 2, "Operational highlights", "Operational highlights", "Transactions processed on Visa's networks", "FY 2025")],
    "v_fy2025_004": [_binding(4, 2, "Operational highlights", "Operational highlights", "Payments volume", "FY 2025")],
    "v_fy2025_005": [_binding(75, 60, "Consolidated Statements of Operations", "Consolidated Statements of Operations", "Net income", "2025")],
    "v_fy2025_006": [
        _binding(75, 60, "Consolidated Statements of Operations", "Consolidated Statements of Operations", "Net revenue", "2024"),
        _binding(75, 60, "Consolidated Statements of Operations", "Consolidated Statements of Operations", "Net revenue", "2025"),
    ],
    "v_fy2025_007": [
        _binding(4, 2, "Operational highlights", "Operational highlights", "Payments volume", "FY 2025"),
        _binding(4, 2, "Operational highlights", "Operational highlights", "Transactions processed on Visa's networks", "FY 2025"),
    ],
    "v_fy2025_009": [_binding(4, 2, "Operational highlights", "Operational highlights", "Payments volume", "FY 2024")],
}


QUESTION_CORRECTIONS = {
    "nvda_fy2025_004": "In the Revenue by End Market table, what was Data Center revenue in FY2025?",
    "nvda_fy2025_005": "In the Revenue by End Market table, what was Automotive revenue in FY2025?",
    "tsla_fy2025_005": "In the Results of Operations revenue table, what was Energy generation and storage segment revenue in FY2025?",
    "tsla_fy2025_007": "Which was higher in FY2025: Tesla's Total automotive revenues or Energy generation and storage segment revenue, and by how much?",
    "v_fy2025_003": "How many transactions were processed on Visa's networks in FY2025?",
    "v_fy2025_004": "In the Operational highlights table, what was payments volume in FY2025?",
}


NO_ANSWER_FINDINGS = {
    "aapl_fy2025_008": "No model-accuracy disclosure found; general AI references do not contain an accuracy metric.",
    "jpm_fy2025_008": "No individual-client contract amount found; client-specific hits concern credit exposure.",
    "ko_fy2025_008": "No named-plant output or production-volume disclosure found.",
    "msft_fy2025_008": "No customer-level contract renewal rate found.",
    "nvda_fy2025_008": "Customer concentration percentages exist, but no customer-level Data Center revenue amount.",
    "pfe_fy2024_008": "No guaranteed realized FY2026 financial result found.",
    "tsla_fy2025_008": "Resale-value guarantees exist, but no FY2026 customer-specific purchase price.",
    "v_fy2025_008": "No internal employee-performance accuracy metric found.",
}


def apply_pdf_verification(
    questions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    question_by_id = {str(item["case_id"]): item for item in questions}
    label_by_id = {str(item["case_id"]): item for item in labels}
    review_by_id = {str(item["case_id"]): item for item in reviews}
    if len(question_by_id) != 72 or len(label_by_id) != 72 or len(review_by_id) != 72:
        raise ValueError("PDF verification requires the complete 72-case Draft set")

    for case_id, question_text in QUESTION_CORRECTIONS.items():
        question = question_by_id[case_id]
        question["question"] = question_text
        question["semantic_group"] = f"{question['document_scope'][0]}:{case_id.rsplit('_', 1)[-1]}:{question_text.casefold()}"
        question["question_revision"] = 3
        label_by_id[case_id]["question_revision"] = 3
        label_by_id[case_id].setdefault("review_plan", {})["question_revision"] = 3
        review_by_id[case_id]["question_revision"] = 3

    question_by_id["tsla_fy2025_007"]["comparison_operands"] = [
        "Total automotive revenues",
        "Energy generation and storage segment revenue",
    ]
    tsla_answer = label_by_id["tsla_fy2025_007"]["expected_answer"]
    tsla_answer["display_value"] = "Total automotive revenues were higher by $56.755 billion"
    tsla_answer["text"] = "Total automotive revenues were higher by $56.755 billion"
    tsla_answer["components"] = ["Total automotive revenues", "Energy generation and storage segment revenue"]
    tsla_calculation = label_by_id["tsla_fy2025_007"]["calculation"]
    tsla_calculation["comparison"] = "Total automotive revenues higher"
    tsla_calculation["operands"][0]["metric"] = "Total automotive revenues"
    tsla_calculation["operands"][1]["metric"] = "Energy generation and storage segment revenue"

    source_count = 0
    for case_id, bindings in SOURCE_BINDINGS.items():
        sources = label_by_id[case_id].get("expected_sources", [])
        if len(sources) != len(bindings):
            raise ValueError(f"{case_id}: expected {len(bindings)} sources, got {len(sources)}")
        for source, binding in zip(sources, bindings, strict=True):
            source.update(binding)
            source.update(
                {
                    "pdf_page_verified": True,
                    "pdf_content_verified": True,
                    "pdf_verification_method": "pdf_text_and_visual",
                    "pdf_verified_at": VERIFIED_AT,
                    "candidate_identity_status": "pending",
                    "source_verified": False,
                    "candidate_key": None,
                    "evidence_id": None,
                    "row_id": None,
                }
            )
            source_count += 1
        label_by_id[case_id]["review_notes"] = (
            "Answer value and physical PDF evidence were checked against the exact ingested PDF; "
            "candidate identity binding remains pending, so source_verified and Golden promotion remain false."
        )

    for case_id, finding in NO_ANSWER_FINDINGS.items():
        review = label_by_id[case_id].setdefault("no_answer_review", {})
        review["automated_pdf_full_text_scan_completed"] = True
        review["automated_pdf_scan_status"] = "provisionally_supported"
        review["automated_pdf_scan_finding"] = finding
        review["human_negative_evidence_reviewed"] = False

    updated_questions = [question_by_id[str(item["case_id"])] for item in questions]
    updated_labels = [label_by_id[str(item["case_id"])] for item in labels]
    updated_reviews = [review_by_id[str(item["case_id"])] for item in reviews]
    worklist = build_annotation_worklist(updated_questions, updated_labels, updated_reviews)
    report = annotation_contract_report(updated_questions, updated_labels, updated_reviews)
    report.update(
        {
            "artifact_schema": "financial-rag-v1/pdf-verification-corrections/v1",
            "question_correction_count": len(QUESTION_CORRECTIONS),
            "pdf_verified_source_record_count": source_count,
            "candidate_identity_pending_count": source_count,
            "no_answer_provisional_scan_count": len(NO_ANSWER_FINDINGS),
            "golden_promotion_allowed": False,
        }
    )
    if source_count != 80 or report["pdf_verified_source_record_count"] != 80:
        raise ValueError("Expected all 80 answerable source records to have PDF verification")
    if report["verified_source_record_count"] != 0 or report["golden_case_count"] != 0:
        raise ValueError("PDF verification must not promote source or Golden status")
    return updated_questions, updated_labels, updated_reviews, {"worklist": worklist, "report": report}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("benchmarks/financial_rag_v1/data/questions.draft.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("benchmarks/financial_rag_v1/data/labels.draft.jsonl"))
    parser.add_argument("--review", type=Path, default=Path("benchmarks/financial_rag_v1/data/review-status.jsonl"))
    parser.add_argument("--worklist", type=Path, default=Path("benchmarks/financial_rag_v1/data/annotation-worklist.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/evaluation/nf-eval-02/pdf-verification-update-report.json"))
    args = parser.parse_args()
    questions, labels, reviews, result = apply_pdf_verification(
        load_jsonl(args.questions),
        load_jsonl(args.labels),
        load_jsonl(args.review),
    )
    write_jsonl(args.questions, questions)
    write_jsonl(args.labels, labels)
    write_jsonl(args.review, reviews)
    write_jsonl(args.worklist, result["worklist"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
