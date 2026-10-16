"""Ingest audited 10-K financial facts directly from SEC EDGAR iXBRL API into persistent SQLite,

and perform accounting integrity verification and cross-company comparative analysis.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from decimal import Decimal
from typing import Any


# Ensure backend root is on sys.path
backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from rag_v2.sources.sql_fact_store import SqlFactStore

COMPANIES = [
    {"symbol": "AAPL", "name": "Apple", "cik": "0000320193"},
    {"symbol": "MSFT", "name": "Microsoft", "cik": "0000789019"},
    {"symbol": "TSLA", "name": "Tesla", "cik": "0001318605"},
    {"symbol": "NVDA", "name": "Nvidia", "cik": "0001045810"},
    {"symbol": "AMZN", "name": "Amazon", "cik": "0001018724"},
    {"symbol": "GOOGL", "name": "Google", "cik": "0001652044"},
    {"symbol": "META", "name": "Meta", "cik": "0001326801"},
    {"symbol": "JPM", "name": "JPMorgan", "cik": "0000019617"},
    {"symbol": "KO", "name": "Coca-Cola", "cik": "0000021344"},
    {"symbol": "WMT", "name": "Walmart", "cik": "0000104169"},
    {"symbol": "XOM", "name": "ExxonMobil", "cik": "0000034088"},
    {"symbol": "NFLX", "name": "Netflix", "cik": "0001065280"},
]

GAAP_METRIC_MAP = {
    "total revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenuesNetOfInterestExpense",
    ],
    "net income": [
        "NetIncomeLoss",
    ],
    "operating income": [
        "OperatingIncomeLoss",
    ],
    "gross profit": [
        "GrossProfit",
    ],
    "total assets": [
        "Assets",
    ],
    "total liabilities": [
        "Liabilities",
    ],
    "stockholders' equity": [
        "StockholdersEquity",
    ],
    "diluted eps": [
        "EarningsPerShareDiluted",
    ],
    "cash and cash equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "operating cash flows": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "research and development": [
        "ResearchAndDevelopmentExpense",
    ],
}


def fetch_sec_company_facts(cik: str) -> dict[str, Any] | None:
    """Fetch all XBRL facts for a given CIK from SEC EDGAR API."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
    headers = {
        "User-Agent": "NanoFinRAG research@nanofin.com",
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read()
            if content[:2] == b"\x1f\x8b":
                content = gzip.decompress(content)
            return json.loads(content.decode("utf-8"))
    except Exception as e:
        print(f"Error fetching CIK {cik}: {e}")
        return None


def extract_annual_facts(company: dict[str, str], edgar_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract 10-K audited annual metrics for the last 5 fiscal years."""
    facts = []
    us_gaap = edgar_data.get("facts", {}).get("us-gaap", {})
    entity_name = company["name"]
    entity_id = company["symbol"].lower()

    for canonical_metric, gaap_tags in GAAP_METRIC_MAP.items():
        found_records: list[dict[str, Any]] = []
        unit_type = "USD"

        for tag in gaap_tags:
            if tag in us_gaap:
                tag_units = us_gaap[tag].get("units", {})
                if canonical_metric == "diluted eps":
                    for u in tag_units:
                        if "USD" in u or "shares" in u:
                            found_records.extend(tag_units[u])
                            unit_type = "USD/share"
                elif "USD" in tag_units:
                    found_records.extend(tag_units["USD"])
                    unit_type = "USD"

        if not found_records:
            continue

        fy_map: dict[str, dict[str, Any]] = {}
        for rec in found_records:
            form = rec.get("form", "")
            if form != "10-K":
                continue
            end = str(rec.get("end", ""))
            start = rec.get("start")
            val = rec.get("val")
            filed = str(rec.get("filed", ""))
            if not end or val is None:
                continue

            # If start is present (duration metric), ensure it represents a full year (>= 300 days)
            if start:
                try:
                    s_dt = datetime.fromisoformat(str(start))
                    e_dt = datetime.fromisoformat(end)
                    if (e_dt - s_dt).days < 300:
                        continue
                except Exception:
                    pass

            # Determine fiscal year from frame or period end
            frame = str(rec.get("frame", ""))
            if len(frame) >= 6 and (frame.startswith("CY") or frame.startswith("FY")):
                fy_str = f"FY{frame[2:6]}"
            else:
                fy_str = f"FY{end[:4]}"

            # Keep latest filed value for each fiscal year
            if fy_str not in fy_map or filed > str(fy_map[fy_str].get("filed", "")):
                fy_map[fy_str] = rec



        for fy_str, rec in sorted(fy_map.items()):
            val = rec["val"]
            if canonical_metric == "diluted eps":
                scale = "units"
                val_str = str(val)
            else:
                scale = "million"
                val_dec = (Decimal(str(val)) / Decimal("1000000")).quantize(Decimal("1"))
                val_str = str(val_dec)

            facts.append({
                "entity": entity_name,
                "entity_id": entity_id,
                "metric": canonical_metric,
                "period": fy_str,
                "value": val_str,
                "unit": unit_type,
                "scale": scale,
                "scope": "consolidated",
                "source_doc": f"SEC 10-K (filed {rec.get('filed', 'N/A')})",
            })

    return facts


def run_checks_and_comparisons(store: SqlFactStore) -> None:
    """Run accounting integrity checks and comparative financial benchmarking."""
    print("\n" + "=" * 80)
    print("1. 会计勾稽关系平衡校验 (Accounting Identity & Integrity Verification)")
    print("=" * 80)

    balance_checks_passed = 0
    balance_checks_total = 0

    for comp in COMPANIES:
        ent = comp["name"]
        for fy in ["FY2024", "FY2023", "FY2022"]:
            assets_fact = store.query_fact(ent, "total assets", fy)
            liab_fact = store.query_fact(ent, "total liabilities", fy)
            equity_fact = store.query_fact(ent, "stockholders' equity", fy)

            if assets_fact and liab_fact and equity_fact:
                balance_checks_total += 1
                assets = Decimal(assets_fact["value"])
                liab = Decimal(liab_fact["value"])
                equity = Decimal(equity_fact["value"])
                diff = assets - (liab + equity)
                diff_pct = (abs(diff) / assets) * Decimal("100") if assets > 0 else Decimal("0")

                # Discrepancy <= 2% (due to minor non-controlling interests / redeemable equity)
                status = "PASS" if diff_pct <= Decimal("2.0") else "WARN"
                if status == "PASS":
                    balance_checks_passed += 1

                print(
                    f"[{status}] {ent:12} {fy} | Assets: ${assets:>9}M | Liab: ${liab:>9}M | Equity: ${equity:>9}M | Diff: ${diff:>6}M ({diff_pct:.2f}%)"
                )

    print(f"\n-> 资产负债表平衡检验通过率: {balance_checks_passed}/{balance_checks_total} ({balance_checks_passed/balance_checks_total*100:.1f}%)")

    print("\n" + "=" * 80)
    print("2. 2024财年各巨头核心业绩横向横比 (Cross-Company Performance Benchmarking FY2024)")
    print("=" * 80)
    print(f"{'Company':12} | {'Revenue ($M)':>13} | {'Net Income ($M)':>15} | {'Net Margin':>11} | {'Operating Cash ($M)':>20}")
    print("-" * 80)

    ranked = []
    for comp in COMPANIES:
        ent = comp["name"]
        rev_f = store.query_fact(ent, "total revenue", "FY2024") or store.query_fact(ent, "total revenue", "FY2023")
        ni_f = store.query_fact(ent, "net income", "FY2024") or store.query_fact(ent, "net income", "FY2023")
        cf_f = store.query_fact(ent, "operating cash flows", "FY2024") or store.query_fact(ent, "operating cash flows", "FY2023")

        rev = Decimal(rev_f["value"]) if rev_f else Decimal("0")
        ni = Decimal(ni_f["value"]) if ni_f else Decimal("0")
        cf = Decimal(cf_f["value"]) if cf_f else Decimal("0")
        margin = (ni / rev * Decimal("100")) if rev > Decimal("0") else Decimal("0")
        period_label = rev_f["period"] if rev_f else "N/A"

        ranked.append({
            "name": f"{ent} ({period_label})",
            "rev": rev,
            "ni": ni,
            "margin": margin,
            "cf": cf,
        })

    # Sort by Revenue descending
    ranked.sort(key=lambda x: x["rev"], reverse=True)
    for r in ranked:
        print(f"{r['name']:12} | ${r['rev']:>12,f}M | ${r['ni']:>14,f}M | {r['margin']:>10.2f}% | ${r['cf']:>19,f}M")

    print("\n" + "=" * 80)
    print("3. 用户关注场景验证：可口可乐 (Coca-Cola) 历年净利润与营收趋势分析")
    print("=" * 80)
    ko_trend = []
    for fy in ["FY2020", "FY2021", "FY2022", "FY2023", "FY2024"]:
        rev_f = store.query_fact("Coca-Cola", "total revenue", fy)
        ni_f = store.query_fact("Coca-Cola", "net income", fy)
        op_f = store.query_fact("Coca-Cola", "operating income", fy)
        if rev_f and ni_f:
            ko_trend.append({
                "period": fy,
                "revenue": Decimal(rev_f["value"]),
                "net_income": Decimal(ni_f["value"]),
                "operating_income": Decimal(op_f["value"]) if op_f else Decimal("0"),
            })

    print(f"{'Fiscal Period':14} | {'Revenue ($M)':>14} | {'Operating Inc ($M)':>18} | {'Net Income ($M)':>16} | {'Net Margin':>11}")
    print("-" * 80)
    for row in ko_trend:
        margin = (row["net_income"] / row["revenue"] * Decimal("100")) if row["revenue"] > 0 else Decimal("0")
        print(f"{row['period']:14} | ${row['revenue']:>13,f}M | ${row['operating_income']:>17,f}M | ${row['net_income']:>15,f}M | {margin:>10.2f}%")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Ingest SEC EDGAR iXBRL facts into SQLite fact store.")
    parser.add_argument("--symbol", type=str, default=None, help="Stock ticker symbol (e.g. AAPL, INTC, MSFT)")
    parser.add_argument("--name", type=str, default=None, help="Company display name (e.g. Intel)")
    parser.add_argument("--cik", type=str, default=None, help="SEC EDGAR 10-digit CIK (e.g. 0000050863)")
    parser.add_argument("--db", type=str, default=None, help="Path to SQLite database file")
    args = parser.parse_args()

    db_path = args.db or os.path.join(backend_root, "data", "financial_facts.sqlite")
    print(f"Opening SQLite database at: {db_path}")
    store = SqlFactStore(db_path)

    target_companies = COMPANIES
    if args.symbol and args.cik:
        name = args.name or args.symbol.upper()
        target_companies = [{"symbol": args.symbol.upper(), "name": name, "cik": args.cik}]

    total_ingested = 0
    for comp in target_companies:
        print(f"Fetching SEC EDGAR iXBRL facts for {comp['name']} ({comp['symbol']}, CIK: {comp['cik']})...")
        edgar = fetch_sec_company_facts(comp["cik"])
        if edgar:
            company_facts = extract_annual_facts(comp, edgar)
            store.upsert_facts(company_facts)
            print(f"  -> Ingested {len(company_facts)} facts for {comp['name']}.")
            total_ingested += len(company_facts)
        # SEC rate limit: <= 10 requests per second
        time.sleep(0.3)

    print(f"\nSuccessfully populated/updated {total_ingested} audited 10-K financial facts into {db_path}!")

    if len(target_companies) > 1:
        # Run verification and comparison across standard portfolio
        run_checks_and_comparisons(store)
    store.close()


if __name__ == "__main__":
    main()

