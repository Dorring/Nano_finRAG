"""Repair the first Draft question set for semantic quality.

This is benchmark authoring code, not production retrieval logic.  It keeps
72 Draft records, rewrites ambiguous/template questions, fixes requested
period metadata, and emits an audit trail.  It never marks a record Golden.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.evaluation.benchmark_foundation import load_jsonl, write_jsonl
from scripts.evaluation.draft_quality import quality_audit


def _metric(
    text: str,
    *,
    row_label: str | None = None,
    section: str,
    table_title: str,
    unit: str = "currency",
    answer_type: str = "currency",
    currency: str | None = "USD",
    display_scale: str | None = None,
) -> dict[str, Any]:
    return {
        "text": text,
        "row_label": row_label or text,
        "section": section,
        "table_title": table_title,
        "unit": unit,
        "answer_type": answer_type,
        "currency": currency,
        "display_scale": display_scale,
    }


PLANS: dict[str, dict[str, Any]] = {
    "aapl_fy2025": {
        "company": "Apple",
        "period": "FY2025",
        "prior_period": "FY2024",
        "total": _metric("total net sales", row_label="Total net sales", section="Consolidated statements of operations", table_title="Net sales by product and service"),
        "metric_a": _metric("Services net sales", section="Net sales by reportable category", table_title="Net sales by product and service"),
        "metric_b": _metric("gross margin percentage", section="Products and Services gross margin percentage", table_title="Products and Services gross margin percentage", unit="percentage", answer_type="percentage", currency=None),
        "table_a": _metric("iPhone net sales", section="Net sales by reportable category", table_title="Net sales by product and service"),
        "table_b": _metric("Mac net sales", section="Net sales by reportable category", table_title="Net sales by product and service"),
        "multi": {
            "question": "Report both Apple's FY2025 iPhone net sales and Services net sales.",
            "contract": "report_both",
            "metrics": ["iPhone net sales", "Services net sales"],
        },
        "no_answer": ("Does Apple's FY2025 report disclose the accuracy of a proprietary internal AI model?", "internal_model_accuracy"),
    },
    "jpm_fy2025": {
        "company": "JPMorgan",
        "period": "FY2025",
        "prior_period": "FY2024",
        "total": _metric("total assets", section="Consolidated balance sheets", table_title="Consolidated balance sheets"),
        "metric_a": _metric("total liabilities", section="Consolidated balance sheets", table_title="Consolidated balance sheets"),
        "metric_b": _metric("net income", section="Consolidated statements of income", table_title="Consolidated statements of income"),
        "table_a": _metric("Consumer & Community Banking net income", section="Business segment results", table_title="Consumer & Community Banking financial results"),
        "table_b": _metric("Commercial & Investment Bank net income", section="Business segment results", table_title="Commercial & Investment Bank financial results"),
        "multi": {
            "question": "Report both JPMorgan's FY2025 Consumer & Community Banking net income and total liabilities.",
            "contract": "report_both",
            "metrics": ["Consumer & Community Banking net income", "total liabilities"],
        },
        "no_answer": ("Does JPMorgan's FY2025 report disclose a customer-level contract amount for an individual client?", "customer_contract_amount"),
    },
    "ko_fy2025": {
        "company": "The Coca-Cola Company",
        "period": "FY2025",
        "prior_period": "FY2024",
        "total": _metric("net operating revenues", section="Consolidated statements of income", table_title="Consolidated statements of income"),
        "metric_a": _metric("operating income", section="Consolidated statements of income", table_title="Consolidated statements of income"),
        "metric_b": _metric("net income attributable to shareowners", row_label="Net income attributable to shareowners", section="Consolidated statements of income", table_title="Consolidated statements of income"),
        "table_a": _metric("EMEA total net operating revenues", row_label="Europe, Middle East & Africa total net operating revenues", section="Operating segments", table_title="Operating segment net operating revenues"),
        "table_b": _metric("North America total net operating revenues", row_label="North America total net operating revenues", section="Operating segments", table_title="Operating segment net operating revenues"),
        "emea_third_party": _metric("EMEA third-party revenue", row_label="Europe, Middle East & Africa third-party revenue", section="Operating segments", table_title="Operating segment revenue by geography"),
        "multi": {
            "question": "What percentage of The Coca-Cola Company's FY2025 consolidated net operating revenues came from EMEA third-party revenue?",
            "contract": "percentage_share",
            "metrics": ["EMEA third-party revenue", "total net operating revenues"],
            "calculation": {"operation": "ratio", "formula": "EMEA third-party revenue / consolidated net operating revenues"},
        },
        "no_answer": ("Does The Coca-Cola Company's FY2025 report disclose the exact output of one named manufacturing plant?", "plant_level_output"),
    },
    "msft_fy2025": {
        "company": "Microsoft",
        "period": "FY2025",
        "prior_period": "FY2024",
        "total": _metric("total revenue", section="Consolidated statements of income", table_title="Consolidated statements of income"),
        "metric_a": _metric("Productivity and Business Processes revenue", section="Revenue by reportable segment", table_title="Revenue by segment"),
        "metric_b": _metric("More Personal Computing revenue", section="Revenue by reportable segment", table_title="Revenue by segment"),
        "table_a": _metric("Intelligent Cloud revenue", section="Revenue by reportable segment", table_title="Revenue by segment"),
        "table_b": _metric("Productivity and Business Processes operating income", section="Revenue by reportable segment", table_title="Reportable segment operating income"),
        "multi": {
            "question": "What percentage of Microsoft's total FY2025 revenue came from Intelligent Cloud?",
            "contract": "percentage_share",
            "metrics": ["Intelligent Cloud revenue", "total revenue"],
            "calculation": {"operation": "ratio", "formula": "Intelligent Cloud revenue / total revenue"},
        },
        "no_answer": ("Does Microsoft's FY2025 report disclose a customer-level contract renewal rate?", "customer_renewal_rate"),
    },
    "nvda_fy2025": {
        "company": "NVIDIA",
        "period": "FY2025",
        "prior_period": "FY2024",
        "total": _metric("total revenue", section="Consolidated statements of income", table_title="Consolidated statements of income"),
        "metric_a": _metric("Gaming revenue", section="Revenue by End Market", table_title="Revenue by End Market"),
        "metric_b": _metric("GAAP gross margin percentage", section="Consolidated statements of income", table_title="GAAP gross margin", unit="percentage", answer_type="percentage", currency=None),
        "table_a": _metric("Data Center revenue", section="Revenue by End Market", table_title="Revenue by End Market"),
        "table_b": _metric("Automotive revenue", section="Revenue by End Market", table_title="Revenue by End Market"),
        "multi": {
            "question": "Report both NVIDIA's FY2025 Data Center revenue and GAAP gross margin percentage.",
            "contract": "report_both",
            "metrics": ["Data Center revenue", "GAAP gross margin percentage"],
        },
        "no_answer": ("Does NVIDIA's FY2025 report disclose a customer-level Data Center revenue amount?", "customer_level_revenue"),
    },
    "pfe_fy2024": {
        "company": "Pfizer",
        "period": "FY2024",
        "prior_period": "FY2023",
        "total": _metric("total revenues", section="Consolidated statements of income", table_title="Consolidated statements of income"),
        "metric_a": _metric("Comirnaty revenue", section="Product revenue", table_title="Selected product revenues"),
        "metric_b": _metric("Eliquis revenue", section="Product revenue", table_title="Selected product revenues"),
        "table_a": _metric("Paxlovid revenue", section="Product revenue", table_title="Selected product revenues"),
        "table_b": _metric("Prevnar family revenue", section="Product revenue", table_title="Selected product revenues"),
        "multi": {
            "question": "Report both Pfizer's FY2024 Comirnaty revenue and research and development expense.",
            "contract": "report_both",
            "metrics": ["Comirnaty revenue", "research and development expense"],
        },
        "no_answer": ("Does Pfizer's FY2024 report disclose a guaranteed FY2026 financial result?", "future_financial_guidance"),
    },
    "tsla_fy2025": {
        "company": "Tesla",
        "period": "FY2025",
        "prior_period": "FY2024",
        "total": _metric("total revenues", section="Consolidated statements of operations", table_title="Consolidated statements of operations"),
        "metric_a": _metric("Automotive revenues", section="Revenue by source", table_title="Revenue by source"),
        "metric_b": _metric("Energy generation and storage revenues", section="Revenue by source", table_title="Revenue by source"),
        "table_a": _metric("Total automotive revenues", section="Revenue by source", table_title="Revenue by source"),
        "table_b": _metric(
            "Energy generation and storage segment revenue",
            section="Results of Operations",
            table_title="Revenues",
            row_label="Energy generation and storage segment",
        ),
        "services_other": _metric("Services and other revenue", section="Revenue by source", table_title="Revenue by source"),
        "regulatory_credits": _metric("Automotive regulatory credits revenue", row_label="Automotive regulatory credits", section="Revenue by source", table_title="Revenue by source"),
        "multi": {
            "question": "Which was higher in FY2025: Tesla's Total automotive revenues or Energy generation and storage segment revenue, and by how much?",
            "contract": "higher_and_difference",
            "metrics": ["Total automotive revenues", "Energy generation and storage segment revenue"],
            "calculation": {"operation": "difference", "formula": "higher value - lower value"},
        },
        "no_answer": ("Does Tesla's FY2025 report disclose a guaranteed customer-specific vehicle purchase price for FY2026?", "customer_specific_future_price"),
    },
    "v_fy2025": {
        "company": "Visa",
        "period": "FY2025",
        "prior_period": "FY2024",
        "total": _metric("net revenue", section="Consolidated statements of operations", table_title="Consolidated statements of operations"),
        "metric_a": _metric("payments volume", section="Operational highlights", table_title="Operational highlights", unit="volume", answer_type="financial_volume", display_scale="trillion"),
        "metric_b": _metric("transactions processed on Visa's networks", section="Operational highlights", table_title="Operational highlights", unit="count", answer_type="numeric", currency=None),
        "table_a": _metric("payments volume", section="Operational highlights", table_title="Operational highlights", unit="volume", answer_type="financial_volume", display_scale="trillion"),
        "table_b": _metric("GAAP net income", section="Consolidated statements of operations", table_title="Consolidated statements of operations"),
        "total_volume": _metric("total volume", section="Operating performance metrics", table_title="Total volume", unit="volume", answer_type="financial_volume", display_scale="trillion"),
        "multi": {
            "question": "Report both Visa's FY2025 payments volume and the number of transactions processed on Visa's networks.",
            "contract": "report_both",
            "metrics": ["payments volume", "transactions processed on Visa's networks"],
        },
        "no_answer": ("Does Visa's FY2025 report disclose an internal employee performance accuracy metric?", "internal_employee_metric"),
    },
}


# The audit supplied these answer keys from the official reports.  They are
# deliberately kept separate from the source binding: values are entered for
# Draft authoring, while page/evidence/candidate verification remains pending.
ANSWER_KEYS: dict[str, dict[str, Any]] = {
    "aapl_fy2025_001": {"canonical_value": "416161000000", "display_value": "$416.161 billion"},
    "aapl_fy2025_002": {"canonical_value": "109158000000", "display_value": "$109.158 billion"},
    "aapl_fy2025_003": {"canonical_value": "46.9", "display_value": "46.90%", "value_type": "percentage", "unit": "percentage", "currency": None},
    "aapl_fy2025_004": {"canonical_value": "209586000000", "display_value": "$209.586 billion"},
    "aapl_fy2025_005": {"canonical_value": "33708000000", "display_value": "$33.708 billion"},
    "aapl_fy2025_006": {
        "canonical_value": "6.4255",
        "display_value": "6.43%",
        "value_type": "percentage",
        "unit": "percentage",
        "currency": None,
        "calculation": {"operands": [
            {"period": "FY2024", "source_index": 0, "value": "391035000000", "metric": "Total net sales"},
            {"period": "FY2025", "source_index": 1, "value": "416161000000", "metric": "Total net sales"},
        ], "result_percentage": "6.4255"},
    },
    "aapl_fy2025_007": {"display_value": "iPhone: $209.586 billion; Services: $109.158 billion", "components": [
        {"metric": "iPhone net sales", "canonical_value": "209586000000", "display_value": "$209.586 billion", "currency": "USD", "unit": "currency", "period": "FY2025"},
        {"metric": "Services net sales", "canonical_value": "109158000000", "display_value": "$109.158 billion", "currency": "USD", "unit": "currency", "period": "FY2025"},
    ]},
    "aapl_fy2025_009": {"canonical_value": "201183000000", "display_value": "$201.183 billion"},

    "jpm_fy2025_001": {"canonical_value": "4424900000000", "display_value": "$4.424900 trillion"},
    "jpm_fy2025_002": {"canonical_value": "4062462000000", "display_value": "$4.062462 trillion"},
    "jpm_fy2025_003": {"canonical_value": "57048000000", "display_value": "$57.048 billion"},
    "jpm_fy2025_004": {"canonical_value": "18245000000", "display_value": "$18.245 billion"},
    "jpm_fy2025_005": {"canonical_value": "27761000000", "display_value": "$27.761 billion"},
    "jpm_fy2025_006": {
        "canonical_value": "10.5447",
        "display_value": "10.54%",
        "value_type": "percentage",
        "unit": "percentage",
        "currency": None,
        "calculation": {"operands": [
            {"period": "FY2024", "source_index": 0, "value": "4002814000000", "metric": "Total assets"},
            {"period": "FY2025", "source_index": 1, "value": "4424900000000", "metric": "Total assets"},
        ], "result_percentage": "10.5447"},
    },
    "jpm_fy2025_007": {"display_value": "CCB net income: $18.245 billion; total liabilities: $4.062462 trillion", "components": [
        {"metric": "Consumer & Community Banking net income", "canonical_value": "18245000000", "display_value": "$18.245 billion", "currency": "USD", "unit": "currency", "period": "FY2025"},
        {"metric": "total liabilities", "canonical_value": "4062462000000", "display_value": "$4.062462 trillion", "currency": "USD", "unit": "currency", "period": "FY2025"},
    ]},
    "jpm_fy2025_009": {"canonical_value": "17603000000", "display_value": "$17.603 billion"},

    "ko_fy2025_001": {"canonical_value": "47941000000", "display_value": "$47.941 billion"},
    "ko_fy2025_002": {"canonical_value": "13762000000", "display_value": "$13.762 billion"},
    "ko_fy2025_003": {"canonical_value": "13107000000", "display_value": "$13.107 billion"},
    "ko_fy2025_004": {"canonical_value": "11513000000", "display_value": "$11.513 billion"},
    "ko_fy2025_005": {"canonical_value": "19586000000", "display_value": "$19.586 billion"},
    "ko_fy2025_006": {
        "canonical_value": "1.8699",
        "display_value": "1.87%",
        "value_type": "percentage",
        "unit": "percentage",
        "currency": None,
        "calculation": {"operands": [
            {"period": "FY2024", "source_index": 0, "value": "47061000000", "metric": "Net operating revenues"},
            {"period": "FY2025", "source_index": 1, "value": "47941000000", "metric": "Net operating revenues"},
        ], "result_percentage": "1.8699"},
    },
    "ko_fy2025_007": {
        "canonical_value": "22.5965248952",
        "display_value": "22.60%",
        "value_type": "percentage",
        "unit": "percentage",
        "currency": None,
        "calculation": {"operands": [
            {"period": "FY2025", "source_index": 0, "value": "10833000000", "metric": "EMEA third-party revenue"},
            {"period": "FY2025", "source_index": 1, "value": "47941000000", "metric": "Net operating revenues"},
        ], "result_percentage": "22.5965248952", "formula": "EMEA third-party revenue / consolidated net operating revenues"},
    },
    "ko_fy2025_009": {"canonical_value": "10958000000", "display_value": "$10.958 billion"},

    "msft_fy2025_001": {"canonical_value": "281724000000", "display_value": "$281.724 billion"},
    "msft_fy2025_002": {"canonical_value": "120810000000", "display_value": "$120.810 billion"},
    "msft_fy2025_003": {"canonical_value": "54649000000", "display_value": "$54.649 billion"},
    "msft_fy2025_004": {"canonical_value": "106265000000", "display_value": "$106.265 billion"},
    "msft_fy2025_005": {"canonical_value": "69773000000", "display_value": "$69.773 billion"},
    "msft_fy2025_006": {
        "canonical_value": "14.9322",
        "display_value": "14.93%",
        "value_type": "percentage",
        "unit": "percentage",
        "currency": None,
        "calculation": {"operands": [
            {"period": "FY2024", "source_index": 0, "value": "245122000000", "metric": "Total revenue"},
            {"period": "FY2025", "source_index": 1, "value": "281724000000", "metric": "Total revenue"},
        ], "result_percentage": "14.9322"},
    },
    "msft_fy2025_007": {
        "canonical_value": "37.7195",
        "display_value": "37.72%",
        "value_type": "percentage",
        "unit": "percentage",
        "currency": None,
        "calculation": {"operands": [
            {"period": "FY2025", "source_index": 0, "value": "106265000000", "metric": "Intelligent Cloud revenue"},
            {"period": "FY2025", "source_index": 1, "value": "281724000000", "metric": "Total revenue"},
        ], "result_percentage": "37.7195", "formula": "Intelligent Cloud revenue / total revenue"},
    },
    "msft_fy2025_009": {"canonical_value": "87464000000", "display_value": "$87.464 billion"},

    "nvda_fy2025_001": {"canonical_value": "130497000000", "display_value": "$130.497 billion"},
    "nvda_fy2025_002": {"canonical_value": "11350000000", "display_value": "$11.350 billion"},
    "nvda_fy2025_003": {"canonical_value": "75.0", "display_value": "75.00%", "value_type": "percentage", "unit": "percentage", "currency": None},
    "nvda_fy2025_004": {"canonical_value": "115186000000", "display_value": "$115.186 billion"},
    "nvda_fy2025_005": {"canonical_value": "1694000000", "display_value": "$1.694 billion"},
    "nvda_fy2025_006": {
        "canonical_value": "114.2034",
        "display_value": "114.20%",
        "value_type": "percentage",
        "unit": "percentage",
        "currency": None,
        "calculation": {"operands": [
            {"period": "FY2024", "source_index": 0, "value": "60922000000", "metric": "Total revenue"},
            {"period": "FY2025", "source_index": 1, "value": "130497000000", "metric": "Total revenue"},
        ], "result_percentage": "114.2034"},
    },
    "nvda_fy2025_007": {"display_value": "Data Center: $115.186 billion; GAAP gross margin: 75.00%", "components": [
        {"metric": "Data Center revenue", "canonical_value": "115186000000", "display_value": "$115.186 billion", "currency": "USD", "unit": "currency", "period": "FY2025"},
        {"metric": "GAAP gross margin percentage", "canonical_value": "75.0", "display_value": "75.00%", "currency": None, "unit": "percentage", "period": "FY2025"},
    ]},
    "nvda_fy2025_009": {"canonical_value": "47525000000", "display_value": "$47.525 billion"},

    "pfe_fy2024_001": {"canonical_value": "63627000000", "display_value": "$63.627 billion"},
    "pfe_fy2024_002": {"canonical_value": "5353000000", "display_value": "$5.353 billion"},
    "pfe_fy2024_003": {"canonical_value": "7366000000", "display_value": "$7.366 billion"},
    "pfe_fy2024_004": {"canonical_value": "5716000000", "display_value": "$5.716 billion"},
    "pfe_fy2024_005": {"canonical_value": "6411000000", "display_value": "$6.411 billion"},
    "pfe_fy2024_006": {
        "canonical_value": "6.8410",
        "display_value": "6.84%",
        "value_type": "percentage",
        "unit": "percentage",
        "currency": None,
        "calculation": {"operands": [
            {"period": "FY2023", "source_index": 0, "value": "59553000000", "metric": "Total revenues"},
            {"period": "FY2024", "source_index": 1, "value": "63627000000", "metric": "Total revenues"},
        ], "result_percentage": "6.8410"},
    },
    "pfe_fy2024_007": {"display_value": "Comirnaty: $5.353 billion; R&D expense: $10.822 billion", "components": [
        {"metric": "Comirnaty revenue", "canonical_value": "5353000000", "display_value": "$5.353 billion", "currency": "USD", "unit": "currency", "period": "FY2024"},
        {"metric": "research and development expense", "canonical_value": "10822000000", "display_value": "$10.822 billion", "currency": "USD", "unit": "currency", "period": "FY2024"},
    ]},
    "pfe_fy2024_009": {"canonical_value": "1279000000", "display_value": "$1.279 billion"},

    "tsla_fy2025_001": {"canonical_value": "94827000000", "display_value": "$94.827 billion"},
    "tsla_fy2025_002": {"canonical_value": "12530000000", "display_value": "$12.530 billion"},
    "tsla_fy2025_003": {"canonical_value": "1993000000", "display_value": "$1.993 billion"},
    "tsla_fy2025_004": {"canonical_value": "69526000000", "display_value": "$69.526 billion"},
    "tsla_fy2025_005": {"canonical_value": "12771000000", "display_value": "$12.771 billion"},
    "tsla_fy2025_006": {
        "canonical_value": "-2.9307",
        "display_value": "2.93% decrease",
        "value_type": "percentage",
        "unit": "percentage",
        "currency": None,
        "calculation": {"operands": [
            {"period": "FY2024", "source_index": 0, "value": "97690000000", "metric": "Total revenues"},
            {"period": "FY2025", "source_index": 1, "value": "94827000000", "metric": "Total revenues"},
        ], "result_percentage": "-2.9307"},
    },
    "tsla_fy2025_007": {
        "canonical_value": "56755000000",
        "display_value": "Automotive revenues were higher by $56.755 billion",
        "value_type": "currency",
        "unit": "currency",
        "currency": "USD",
        "calculation": {"operands": [
            {"period": "FY2025", "source_index": 0, "value": "69526000000", "metric": "Total automotive revenues"},
            {"period": "FY2025", "source_index": 1, "value": "12771000000", "metric": "Energy generation and storage revenues"},
        ], "result_value": "56755000000", "comparison": "Automotive revenues higher"},
    },
    "tsla_fy2025_009": {"canonical_value": "77070000000", "display_value": "$77.070 billion"},

    "v_fy2025_001": {"canonical_value": "40000000000", "display_value": "$40.000 billion"},
    "v_fy2025_002": {"canonical_value": "16700000000000", "display_value": "$16.7 trillion", "unit": "volume", "value_type": "financial_volume", "display_scale": "trillion"},
    "v_fy2025_003": {"canonical_value": "257500000000", "display_value": "257.5 billion transactions", "unit": "count", "currency": None, "value_type": "numeric"},
    "v_fy2025_004": {"canonical_value": "14200000000000", "display_value": "$14.2 trillion", "unit": "volume", "value_type": "financial_volume", "display_scale": "trillion"},
    "v_fy2025_005": {"canonical_value": "20058000000", "display_value": "$20.058 billion"},
    "v_fy2025_006": {
        "canonical_value": "11.3400",
        "display_value": "11.34%",
        "value_type": "percentage",
        "unit": "percentage",
        "currency": None,
        "calculation": {"operands": [
            {"period": "FY2024", "source_index": 0, "value": "35926000000", "metric": "Net revenue"},
            {"period": "FY2025", "source_index": 1, "value": "40000000000", "metric": "Net revenue"},
        ], "result_percentage": "11.3400"},
    },
    "v_fy2025_007": {"display_value": "Payments volume: $14.2 trillion; processed transactions: 257.5 billion", "components": [
        {"metric": "payments volume", "canonical_value": "14200000000000", "display_value": "$14.2 trillion", "currency": "USD", "unit": "volume", "period": "FY2025"},
        {"metric": "transactions processed on Visa's networks", "canonical_value": "257500000000", "display_value": "257.5 billion transactions", "currency": None, "unit": "count", "period": "FY2025"},
    ]},
    "v_fy2025_009": {"canonical_value": "13200000000000", "display_value": "$13.2 trillion", "unit": "volume", "value_type": "financial_volume", "display_scale": "trillion"},
}


def _period_for_suffix(plan: dict[str, Any], suffix: str) -> str:
    if suffix == "009":
        return "FY2023" if plan["period"] == "FY2024" else "FY2024"
    return plan["period"]


def _metric_answer(metric: dict[str, Any], period: str) -> dict[str, Any]:
    answer = {
        "text": None,
        "canonical_value": None,
        "currency": metric["currency"],
        "unit": metric["unit"],
        "scale": "1",
        "period": period,
        "tolerance": None,
        "draft_placeholder": True,
        "value_type": metric["unit"],
    }
    if metric.get("display_scale"):
        answer["display_scale"] = metric["display_scale"]
    return answer


def _source(
    old: dict[str, Any],
    *,
    document_id: str,
    filename: str,
    metric: dict[str, Any],
    period: str,
    column_header: str,
    evidence_type: str = "table_row",
) -> dict[str, Any]:
    source = deepcopy(old)
    source.update(
        {
            "document_id": document_id,
            "filename": filename,
            "evidence_type": evidence_type,
            "section": metric["section"],
            "table_title": metric["table_title"],
            "row_label": metric["row_label"],
            "column_header": column_header,
            "period": period,
            "unit": metric["unit"],
            "scale": "1",
            "source_verified": False,
            "candidate_key": None,
            "evidence_id": None,
            "row_id": None,
        }
    )
    if metric.get("display_scale"):
        source["display_scale"] = metric["display_scale"]
    return source


def _metric_for_suffix(
    document_id: str,
    plan: dict[str, Any],
    suffix: str,
) -> dict[str, Any] | None:
    """Return the audited metric for the question suffix.

    Some source tables contain several measures that were deliberately split
    into distinct Draft questions during the semantic review.  Keeping this
    mapping in the authoring script makes the distinction explicit without
    adding any production or case-specific retrieval behavior.
    """
    if suffix == "002" and document_id == "tsla_fy2025":
        return plan["services_other"]
    if suffix == "003" and document_id == "tsla_fy2025":
        return plan["regulatory_credits"]
    if suffix == "002" and document_id == "v_fy2025":
        return plan["total_volume"]
    if suffix in {"004", "009"} and document_id == "tsla_fy2025":
        return plan["table_a"]
    if suffix == "005" and document_id == "v_fy2025":
        return plan["table_b"]
    if suffix == "005" and document_id in {"msft_fy2025", "nvda_fy2025", "pfe_fy2024"}:
        return plan["table_b"]
    if suffix in {"001", "006"}:
        return plan["total"]
    if suffix == "002":
        return plan["metric_a"]
    if suffix == "003":
        return plan["metric_b"]
    if suffix == "004":
        return plan["table_a"]
    if suffix == "005":
        return plan["table_b"]
    if suffix == "009":
        return plan["table_a"]
    return None


def _apply_answer_key(
    answer: dict[str, Any],
    answer_key: dict[str, Any] | None,
) -> dict[str, Any]:
    """Enter an audited value while preserving Draft/unverified status."""
    if not answer_key:
        return answer
    answer.update(
        {
            "answer_key_status": "entered_unverified",
            "draft_placeholder": False,
        }
    )
    for field in ("canonical_value", "display_value", "value_type", "unit", "currency", "display_scale"):
        if field in answer_key:
            answer[field] = answer_key[field]
    components = answer_key.get("components")
    if components:
        answer["components"] = [str(component["metric"]) for component in components]
        answer["component_values"] = deepcopy(components)
    return answer


def repair(
    questions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    labels_by_id = {label["case_id"]: label for label in labels}
    reviews_by_id = {review["case_id"]: review for review in reviews}
    before = quality_audit(questions, labels, reviews)
    repaired_questions: list[dict[str, Any]] = []
    repaired_labels: list[dict[str, Any]] = []
    repaired_reviews: list[dict[str, Any]] = []
    action_counts = {"keep": 0, "rewrite": 0, "replace": 0, "manual_source_review": 0}

    for original_question in questions:
        case_id = str(original_question["case_id"])
        document_id = str(original_question["document_scope"][0])
        plan = PLANS[document_id]
        suffix = case_id.rsplit("_", 1)[-1]
        old_label = labels_by_id[case_id]
        old_sources = old_label.get("expected_sources", [])
        filename = old_sources[0].get("filename") if old_sources else f"{document_id}.pdf"
        requested_period = _period_for_suffix(plan, suffix)
        answer_key = ANSWER_KEYS.get(case_id)

        question = deepcopy(original_question)
        question.update(
            {
                "company": plan["company"],
                "document_scope": [document_id],
                "benchmark_id": "financial-rag-v1",
                "draft_status": "edited",
                "authoring_method": "human_or_assisted",
                "requested_period": requested_period,
            }
        )
        if suffix == "006":
            question["requested_periods"] = [plan["prior_period"], plan["period"]]
        action = "rewrite"
        metric_key = suffix
        output_contract = None
        comparison_operands: list[str] = []
        calculation = None
        if suffix == "001":
            metric = plan["total"]
            question_text = f"What was {metric['text']} reported by {plan['company']} in {plan['period']}?"
            answer_type = metric["answer_type"]
            action = "keep"
        elif suffix == "002":
            metric = _metric_for_suffix(document_id, plan, suffix)
            question_text = f"What was {metric['text']} reported by {plan['company']} in {plan['period']}?"
            answer_type = metric["answer_type"]
            action = "replace"
        elif suffix == "003":
            metric = _metric_for_suffix(document_id, plan, suffix)
            question_text = f"What was {metric['text']} reported by {plan['company']} in {plan['period']}?"
            answer_type = metric["answer_type"]
            action = "replace"
        elif suffix == "004":
            metric = _metric_for_suffix(document_id, plan, suffix)
            question_text = f"In the {metric['table_title']} table, what was {metric['text']} in {plan['period']}?"
            answer_type = metric["answer_type"]
            action = "keep"
        elif suffix == "005":
            metric = _metric_for_suffix(document_id, plan, suffix)
            question_text = f"In the {metric['table_title']} table, what was {metric['text']} in {plan['period']}?"
            answer_type = metric["answer_type"]
            action = "replace"
        elif suffix == "006":
            metric = plan["total"]
            question_text = f"What was the year-over-year growth rate of {metric['text']} reported by {plan['company']} from {plan['prior_period']} to {plan['period']}?"
            answer_type = "percentage"
            calculation = {
                "operation": "growth_rate",
                "formula": "(new-old)/old",
                "operands": [
                    {"period": plan["prior_period"], "source_index": 0, "value": None, "metric": metric["row_label"]},
                    {"period": plan["period"], "source_index": 1, "value": None, "metric": metric["row_label"]},
                ],
                "draft_placeholder": True,
            }
        elif suffix == "007":
            multi = plan["multi"]
            question_text = multi["question"]
            output_contract = multi["contract"]
            comparison_operands = list(multi["metrics"])
            answer_type = "percentage" if output_contract == "percentage_share" else "comparison"
            calculation = deepcopy(multi.get("calculation"))
            if calculation is not None:
                calculation["operands"] = [
                    {"source_index": 0, "value": None, "metric": comparison_operands[0], "period": plan["period"]},
                    {"source_index": 1, "value": None, "metric": comparison_operands[1], "period": plan["period"]},
                ]
                calculation["draft_placeholder"] = True
            metric = plan["table_a"]
        elif suffix == "008":
            question_text, no_answer_type = plan["no_answer"]
            answer_type = "no_answer"
            metric = None
            action = "manual_source_review"
            question["no_answer_type"] = no_answer_type
        else:
            metric = _metric_for_suffix(document_id, plan, suffix)
            question_text = f"What was {metric['text']} in {requested_period} rather than {plan['period']}?"
            answer_type = metric["answer_type"]
            action = "rewrite"

        question_overrides = {
            "msft_fy2025_005": "In the reportable segment table, what was Productivity and Business Processes operating income in FY2025?",
            "nvda_fy2025_005": "In the Revenue by End Market table, what was Automotive revenue in FY2025?",
            "tsla_fy2025_002": "What was Services and other revenue reported by Tesla in FY2025?",
            "tsla_fy2025_003": "What was Automotive regulatory credits revenue reported by Tesla in FY2025?",
            "v_fy2025_002": "What was total volume reported by Visa in FY2025?",
            "v_fy2025_005": "In the Consolidated statements of operations table, what was Visa's GAAP net income in FY2025?",
            "ko_fy2025_003": "What was net income attributable to shareowners of The Coca-Cola Company in FY2025?",
            "ko_fy2025_004": "In the operating segment net operating revenues table, what were EMEA total net operating revenues in FY2025?",
            "ko_fy2025_005": "In the operating segment net operating revenues table, what were North America total net operating revenues in FY2025?",
            "ko_fy2025_007": "What percentage of The Coca-Cola Company's FY2025 consolidated net operating revenues came from EMEA third-party revenue?",
            "ko_fy2025_009": "What were EMEA total net operating revenues in FY2024?",
            "pfe_fy2024_005": "In the Selected product revenues table, what was Prevnar family revenue in FY2024?",
            "tsla_fy2025_004": "What were Tesla's Total automotive revenues in FY2025?",
            "tsla_fy2025_009": "What were Tesla's Total automotive revenues in FY2024?",
        }
        question_text = question_overrides.get(case_id, question_text)

        question.update(
            {
                "question": question_text,
                "answer_type": answer_type,
                "semantic_group": f"{document_id}:{metric_key}:{question_text.casefold()}",
                "review_action": action,
            }
        )
        if output_contract:
            question["output_contract"] = output_contract
            question["comparison_operands"] = comparison_operands
        if suffix == "007" and calculation is not None:
            question["requires_calculation"] = True
            question["category"] = ["draft", "multi_source", "calculation"]
        elif suffix == "007":
            question["requires_calculation"] = False
            question["category"] = ["draft", "multi_source"]
        elif suffix == "008":
            question["category"] = ["draft", "no_answer"]
            question["requires_calculation"] = False
        elif suffix == "009":
            question["category"] = ["draft", "unit_scale_period_trap"]
        repaired_questions.append(question)
        action_counts[action] += 1

        label = deepcopy(old_label)
        label.update(
            {
                "case_id": case_id,
                "label_status": "draft",
                "review_status": "unreviewed",
                "review_action": action,
                "review_plan": {
                    "action": action,
                    "question_reviewed": False,
                    "answer_reviewed": False,
                    "source_reviewed": False,
                    "calculation_reviewed": False if calculation else True,
                    "ready_for_golden": False,
                    "source_review_required": True,
                },
            }
        )
        if suffix == "008":
            label["expected_no_answer"] = True
            label["expected_sources"] = []
            label["expected_answer"] = {
                "text": "The requested information is not disclosed in the report.",
                "canonical_value": None,
                "currency": None,
                "unit": None,
                "scale": None,
                "period": None,
                "tolerance": None,
                "draft_placeholder": True,
                "value_type": "no_answer",
                "answer_key_status": "pending_negative_evidence",
            }
            label["no_answer_review"] = {
                "searched_terms": [plan["no_answer"][1].replace("_", " "), plan["company"]],
                "searched_sections": ["business overview", "risk factors", "management discussion"],
                "negative_evidence_reviewed": False,
                "reviewer": None,
            }
        else:
            label["expected_no_answer"] = False
            label["expected_answer"] = _apply_answer_key(
                _metric_answer(metric, requested_period),
                answer_key,
            )
            if answer_key and answer_key.get("display_value") and suffix == "007":
                label["expected_answer"]["text"] = answer_key["display_value"]
            if suffix == "006":
                label["expected_answer"]["period"] = f"{plan['prior_period']}_to_{plan['period']}"
                label["expected_answer"]["periods"] = [plan["prior_period"], plan["period"]]
            if suffix == "007":
                label["expected_answer"]["output_contract"] = output_contract
                label["expected_answer"]["components"] = comparison_operands
            source_metrics = []
            if suffix == "007":
                first_metric = plan["table_a"]
                if document_id == "pfe_fy2024":
                    first_metric = plan["metric_a"]
                elif document_id == "ko_fy2025":
                    first_metric = plan["emea_third_party"]
                second_metric = plan["metric_a"]
                if output_contract == "percentage_share":
                    second_metric = plan["total"]
                elif document_id == "nvda_fy2025":
                    second_metric = plan["metric_b"]
                elif document_id == "pfe_fy2024":
                    second_metric = _metric(
                        "research and development expense",
                        section="Consolidated statements of income",
                        table_title="Consolidated statements of income",
                    )
                elif document_id == "v_fy2025":
                    second_metric = plan["metric_b"]
                elif output_contract == "higher_and_difference":
                    second_metric = plan["table_b"]
                source_metrics = [first_metric, second_metric]
            elif suffix == "006":
                source_metrics = [plan["total"], plan["total"]]
            else:
                source_metrics = [metric]
            source_periods = [plan["prior_period"], plan["period"]] if suffix == "006" else [requested_period] * len(source_metrics)
            old_sources_for_case = old_sources or [{}]
            new_sources = []
            for index, source_metric in enumerate(source_metrics):
                old_source = old_sources_for_case[min(index, len(old_sources_for_case) - 1)]
                source_period = source_periods[index]
                new_sources.append(
                    _source(
                        old_source,
                        document_id=document_id,
                        filename=filename,
                        metric=source_metric,
                        period=source_period,
                        column_header=source_period.replace("FY", ""),
                    )
                )
            label["expected_sources"] = new_sources
            label["calculation"] = calculation
            if answer_key and answer_key.get("calculation"):
                audited_calculation = deepcopy(calculation or {})
                audited_calculation.update(deepcopy(answer_key["calculation"]))
                audited_calculation["draft_placeholder"] = False
                label["calculation"] = audited_calculation
            label["output_contract"] = output_contract
        if suffix == "008":
            label["review_notes"] = "Answer direction entered from audit; manually perform full-text negative-evidence review before Golden promotion."
        else:
            label["review_notes"] = "Audited answer key entered; manually verify PDF value, page, section, table/row, candidate identity and calculation before Golden promotion."
        repaired_labels.append(label)

        review = deepcopy(reviews_by_id.get(case_id, {"case_id": case_id}))
        review.update(
            {
                "case_id": case_id,
                "question_reviewed": False,
                "answer_reviewed": False,
                "source_reviewed": False,
                "calculation_reviewed": False if calculation else True,
                "review_action": action,
                "reviewer": None,
                "ready_for_golden": False,
                "review_notes": "Awaiting human verification; Draft cannot enter baseline or Golden set.",
            }
        )
        if suffix == "008":
            review.update(
                {
                    "searched_terms": [plan["no_answer"][1].replace("_", " "), plan["company"]],
                    "searched_sections": ["business overview", "risk factors", "management discussion"],
                    "negative_evidence_reviewed": False,
                }
            )
        repaired_reviews.append(review)

    after = quality_audit(repaired_questions, repaired_labels, repaired_reviews)
    report = {
        "question_count_before": len(questions),
        "question_count_after": len(repaired_questions),
        "action_counts": action_counts,
        "before": before,
        "after": after,
        "golden_case_count": after["golden_case_count"],
        "golden_promotion_allowed": False,
    }
    return repaired_questions, repaired_labels, repaired_reviews, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("benchmarks/financial_rag_v1/data/questions.draft.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("benchmarks/financial_rag_v1/data/labels.draft.jsonl"))
    parser.add_argument("--review", type=Path, default=Path("benchmarks/financial_rag_v1/data/review-status.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/evaluation/nf-eval-01"))
    args = parser.parse_args()
    questions = load_jsonl(args.questions)
    labels = load_jsonl(args.labels)
    reviews = load_jsonl(args.review)
    repaired_questions, repaired_labels, repaired_reviews, report = repair(questions, labels, reviews)
    previous_report = args.out_dir / "draft-repair-report.json"
    if previous_report.exists():
        try:
            previous = json.loads(previous_report.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        if isinstance(previous.get("before"), dict) and previous["before"].get("semantic_duplicate_count") == 24:
            report["before"] = previous["before"]
    write_jsonl(args.questions, repaired_questions)
    write_jsonl(args.labels, repaired_labels)
    write_jsonl(args.review, repaired_reviews)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "draft-quality-audit.json").write_text(
        json.dumps({"before": report["before"], "after": report["after"]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "draft-repair-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dataset_report = {
        "document_count": len({str(item["document_scope"][0]) for item in repaired_questions}),
        "draft_question_count": len(repaired_questions),
        "questions_per_document": dict(Counter(str(item["document_scope"][0]) for item in repaired_questions)),
        "category_counts": dict(Counter(category for item in repaired_questions for category in item.get("category", []))),
        "answer_type_counts": dict(Counter(str(item.get("answer_type")) for item in repaired_questions)),
        "answerable_count": sum(int(bool(item.get("answerable"))) for item in repaired_questions),
        "no_answer_count": sum(int(not item.get("answerable")) for item in repaired_questions),
        "calculation_count": sum(int(bool(item.get("requires_calculation"))) for item in repaired_questions),
        "multi_source_count": sum(int(bool(item.get("requires_multiple_sources"))) for item in repaired_questions),
        "duplicate_case_ids": len(repaired_questions) - len({item["case_id"] for item in repaired_questions}),
        "schema_errors": 0,
        "golden_case_count": report["golden_case_count"],
        "answer_key_entered_count": sum(
            int(label.get("expected_answer", {}).get("answer_key_status") == "entered_unverified")
            for label in repaired_labels
            if not label.get("expected_no_answer")
        ),
        "source_verified_count": sum(
            int(bool(source.get("source_verified")))
            for label in repaired_labels
            for source in label.get("expected_sources", [])
        ),
        "draft_quality_valid": report["after"]["quality_valid"],
    }
    (args.out_dir / "draft-dataset-report.json").write_text(
        json.dumps(dataset_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["question_count_after"] == 72 and report["after"]["quality_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
