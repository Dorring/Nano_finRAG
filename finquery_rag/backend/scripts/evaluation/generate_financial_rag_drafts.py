"""Generate clearly marked, source-located Draft QA records for human review.

The generated answers are intentionally placeholders.  This script never
creates Golden or Sealed labels and never calls the RAG system or a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.evaluation.benchmark_foundation import load_json, validate_dataset, write_jsonl
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.evaluation.benchmark_foundation import load_json, validate_dataset, write_jsonl


FOCUS: dict[str, dict[str, str]] = {
    "Microsoft": {"metric": "segment revenue", "table_metric": "Intelligent Cloud revenue", "second": "operating income"},
    "Apple": {"metric": "total net sales", "table_metric": "iPhone net sales", "second": "gross margin"},
    "NVIDIA": {"metric": "revenue", "table_metric": "Data Center revenue", "second": "gross margin"},
    "JPMorganChase": {"metric": "total assets", "table_metric": "Consumer & Community Banking net income", "second": "total liabilities"},
    "Tesla": {"metric": "total revenues", "table_metric": "Automotive revenues", "second": "Energy generation and storage revenues"},
    "The Coca-Cola Company": {"metric": "net operating revenues", "table_metric": "Europe segment revenue", "second": "operating income"},
    "Visa": {"metric": "net revenues", "table_metric": "payment volume", "second": "processed transactions"},
    "Pfizer": {"metric": "total revenues", "table_metric": "product revenue", "second": "research and development expense"},
}


def locate_pages(pdf_path: Path, terms: list[str]) -> list[int]:
    try:
        import fitz
    except ImportError:
        return [1]
    pages: list[int] = []
    with fitz.open(pdf_path) as pdf:
        normalized_terms = [term.casefold() for term in terms]
        for index, page in enumerate(pdf, start=1):
            text = page.get_text("text").casefold()
            if any(term in text for term in normalized_terms):
                pages.append(index)
    return pages or [1]


def source(document: dict[str, Any], page: int, *, evidence_type: str, row_label: str, column_header: str, table_title: str) -> dict[str, Any]:
    return {
        "document_id": document["document_id"],
        "filename": document["filename"],
        "page": page,
        "section": table_title,
        "table_title": table_title,
        "row_label": row_label,
        "column_header": column_header,
        "period": column_header,
        "unit": "USD",
        "scale": "draft_pending_review",
        "candidate_key": None,
        "evidence_id": None,
        "row_id": None,
        "evidence_type": evidence_type,
        "source_verified": False,
    }


def expected_answer(*, answer_type: str, period: str) -> dict[str, Any]:
    percentage = answer_type == "percentage"
    return {
        "text": None,
        "canonical_value": None,
        "currency": None if percentage else "USD",
        "unit": "percentage" if percentage else "currency",
        "scale": "1",
        "period": period,
        "tolerance": None,
        "draft_placeholder": True,
    }


def build_records(corpus: dict[str, Any], pdf_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    generated_at = datetime.now(timezone.utc).isoformat()
    questions_per_document: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    for document in sorted(corpus["documents"], key=lambda item: item["document_id"]):
        company = document["company"]
        year = int(document["fiscal_year"])
        focus = FOCUS[company]
        pdf_path = pdf_dir / document["filename"]
        second_pages = locate_pages(pdf_path, [focus["second"]])
        specs = [
            ("fact", "currency", f"What was {focus['metric']} reported by {company} for FY{year}?", [focus["metric"]], False, False, "single_document"),
            ("fact", "currency", f"How much {focus['metric']} did {company} report in FY{year}?", [focus["metric"]], False, False, "single_document"),
            ("fact", "currency", f"What amount of {focus['metric']} is disclosed in {company}'s FY{year} report?", [focus["metric"]], False, False, "single_document"),
            ("table_fact", "currency", f"In the {focus['table_metric']} table, what value is shown for FY{year}?", [focus["table_metric"]], False, False, "table_fact"),
            ("table_fact", "currency", f"Which FY{year} column value is reported for {focus['table_metric']}?", [focus["table_metric"]], False, False, "table_fact"),
            ("calculation", "percentage", f"What was the year-over-year growth rate of {focus['metric']} from FY{year - 1} to FY{year}?", [focus["metric"]], True, False, "calculation"),
            ("multi_source", "comparison", f"Compare {focus['table_metric']} with {focus['second']} reported by {company} for FY{year}.", [focus["table_metric"], focus["second"]], False, True, "multi_source"),
            ("no_answer", "no_answer", f"Does {company}'s FY{year} report disclose the accuracy of a proprietary internal AI model?", [], False, False, "no_answer"),
            ("unit_period_trap", "currency", f"Using the table's reported scale, what was {focus['table_metric']} for FY{year - 1} rather than FY{year}?", [focus["table_metric"]], False, False, "unit_scale_period_trap"),
        ]
        for index, (kind, answer_type, question_text, terms, requires_calculation, multiple_sources, category) in enumerate(specs, start=1):
            case_id = f"{document['document_id']}_{index:03d}"
            answerable = kind != "no_answer"
            scope = [document["document_id"]]
            question = {
                "case_id": case_id,
                "benchmark_id": "financial-rag-v1",
                "company": company,
                "document_scope": scope,
                "question": question_text,
                "answerable": answerable,
                "answer_type": answer_type,
                "category": ["draft", category],
                "difficulty": "medium" if kind not in {"fact", "no_answer"} else "easy",
                "requires_calculation": requires_calculation,
                "requires_multiple_sources": multiple_sources,
                "draft_status": "generated",
                "authoring_method": "human_or_assisted",
                "generation_timestamp": generated_at,
            }
            questions.append(question)
            questions_per_document[document["document_id"]] += 1
            category_counts[category] += 1
            answer_type_counts[answer_type] += 1
            if not answerable:
                label = {
                    "case_id": case_id,
                    "expected_answer": {"text": "Information not available (draft; verify manually).", "canonical_value": None, "currency": None, "unit": None, "scale": "1", "period": f"FY{year}", "tolerance": None, "draft_placeholder": True},
                    "expected_sources": [],
                    "calculation": None,
                    "expected_no_answer": True,
                    "label_status": "draft",
                    "review_status": "unreviewed",
                    "review_notes": "Must manually search the relevant report sections before promotion.",
                }
            else:
                pages = locate_pages(pdf_path, terms)
                sources = [source(document, pages[0], evidence_type="table_row" if "table" in category or category == "calculation" else "text", row_label=terms[0], column_header=f"FY{year}", table_title=category)]
                if multiple_sources:
                    second = second_pages[0] if second_pages else pages[0]
                    sources.append(source(document, second, evidence_type="table_row", row_label=terms[1], column_header=f"FY{year}", table_title=category))
                calculation = None
                if requires_calculation:
                    calculation = {
                        "operation": "growth_rate",
                        "operands": [{"value": None, "period": str(year - 1), "source_index": 0}, {"value": None, "period": str(year), "source_index": 0}],
                        "formula": "(new-old)/old",
                        "draft_placeholder": True,
                    }
                label = {
                    "case_id": case_id,
                    "expected_answer": expected_answer(answer_type=answer_type, period=f"FY{year}"),
                    "expected_sources": sources,
                    "calculation": calculation,
                    "expected_no_answer": False,
                    "label_status": "draft",
                    "review_status": "unreviewed",
                    "review_notes": "Answer, unit, period, table row and source must be human verified.",
                }
            labels.append(label)
            reviews.append({
                "case_id": case_id,
                "question_reviewed": False,
                "answer_reviewed": False,
                "source_reviewed": False,
                "calculation_reviewed": not requires_calculation,
                "reviewer": None,
                "review_notes": "Draft generated; not eligible for Baseline or Golden evaluation.",
                "ready_for_golden": False,
            })
    report = {
        "benchmark_id": "financial-rag-v1",
        "document_count": len(corpus["documents"]),
        "draft_question_count": len(questions),
        "questions_per_document": dict(sorted(questions_per_document.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "answer_type_counts": dict(sorted(answer_type_counts.items())),
        "answerable_count": sum(int(item["answerable"]) for item in questions),
        "no_answer_count": sum(int(not item["answerable"]) for item in questions),
        "calculation_count": sum(int(item["requires_calculation"]) for item in questions),
        "multi_source_count": sum(int(item["requires_multiple_sources"]) for item in questions),
        "duplicate_case_ids": 0,
        "schema_errors": 0,
        "golden_case_count": 0,
        "draft_only": True,
        "generated_at": generated_at,
    }
    return questions, labels, reviews, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("benchmarks/financial_rag_v1/corpus.json"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("runtime/benchmark/financial_rag_v1/pdfs"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/financial_rag_v1/data"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/evaluation/nf-eval-01/draft-dataset-report.json"))
    args = parser.parse_args()
    corpus = load_json(args.corpus)
    questions, labels, reviews, report = build_records(corpus, args.pdf_dir)
    write_jsonl(args.out_dir / "questions.draft.jsonl", questions)
    write_jsonl(args.out_dir / "labels.draft.jsonl", labels)
    write_jsonl(args.out_dir / "review-status.jsonl", reviews)
    validation = validate_dataset(corpus=corpus, questions=questions, labels=labels, review_records=reviews, draft=True)
    report["schema_errors"] = len(validation["errors"])
    report["duplicate_case_ids"] = len(validation["duplicate_question_ids"]) + len(validation["duplicate_label_ids"])
    report["validation"] = validation
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if validation["schema_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
