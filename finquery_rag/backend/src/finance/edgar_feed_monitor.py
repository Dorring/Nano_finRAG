"""Automated SEC EDGAR Feed Monitor and Incremental Ingestion Pipeline.

Periodically or on-demand polls SEC EDGAR official submission APIs / RSS feeds,
detects new 10-K / 10-Q filings for tracked blue-chip companies, and
automatically extracts and ingests audited financial facts into the SQLite
fact store (financial_facts.sqlite) with zero manual intervention.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping
import urllib.request

logger = logging.getLogger("nano_finrag.edgar_feed")

DEFAULT_CIK_MAP: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "META": "0001326801",
    "TSLA": "0001318605",
    "JPM": "0000019617",
    "WMT": "0000104169",
    "KO": "0000021344",
    "XOM": "0000034088",
    "NFLX": "0001065280",
}

DEFAULT_USER_AGENT = "NanoFinRAG/3.0 (research@nano-finrag.internal; SEC EDGAR Monitor)"

SEC_8K_ITEM_DESCRIPTIONS: dict[str, str] = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition (Earnings / Operations)",
    "2.03": "Creation of a Direct Financial Obligation or Off-Balance Sheet Arrangement",
    "3.02": "Unregistered Sales of Equity Securities",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure of Directors or Certain Officers; Election of Directors",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}


class EdgarFeedMonitor:
    """Monitor SEC EDGAR company submissions and orchestrate incremental ingestion."""

    def __init__(
        self,
        db_path: Path | str,
        cik_map: dict[str, str] | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.db_path = Path(db_path)
        self.cik_map = dict(cik_map or DEFAULT_CIK_MAP)
        self.user_agent = user_agent

    def fetch_company_submissions(self, cik: str) -> dict[str, Any] | None:
        """Fetch SEC EDGAR submissions JSON for a CIK."""
        padded_cik = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("Failed to fetch EDGAR submissions for CIK %s: %s", cik, exc)
            return None
        return None

    def get_ingested_accession_numbers(self) -> set[str]:
        """Query fact store for already ingested filing accession numbers / sources."""
        if not self.db_path.exists():
            return set()
        seen = set()
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ingested_filings'"
            )
            if cursor.fetchone():
                cursor.execute("SELECT accession_number FROM ingested_filings")
                seen = {row[0] for row in cursor.fetchall()}
            cursor.close()
        except Exception:
            pass
        finally:
            if conn:
                conn.close()
        return seen

    def mark_filing_ingested(
        self,
        ticker: str,
        accession_number: str,
        form: str,
        filing_date: str,
        facts_count: int,
        event_summary: str = "",
    ) -> None:
        """Record an accession number as ingested to ensure idempotency."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingested_filings (
                    accession_number TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    form TEXT NOT NULL,
                    filing_date TEXT,
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    facts_count INTEGER DEFAULT 0,
                    event_summary TEXT DEFAULT ''
                )
                """
            )
            try:
                conn.execute("ALTER TABLE ingested_filings ADD COLUMN event_summary TEXT DEFAULT ''")
            except Exception:
                pass
            conn.execute(
                """
                INSERT OR REPLACE INTO ingested_filings (accession_number, ticker, form, filing_date, facts_count, event_summary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (accession_number, ticker, form, filing_date, facts_count, event_summary),
            )
            conn.commit()
        finally:
            conn.close()

    def discover_new_filings(
        self,
        target_tickers: list[str] | None = None,
        target_forms: tuple[str, ...] = ("10-K", "10-Q", "8-K"),
    ) -> list[dict[str, Any]]:
        """Identify new filings on EDGAR not yet recorded in SQLite."""
        tickers = target_tickers or list(self.cik_map.keys())
        ingested = self.get_ingested_accession_numbers()
        new_filings: list[dict[str, Any]] = []

        for ticker in tickers:
            cik = self.cik_map.get(ticker.upper())
            if not cik:
                continue
            data = self.fetch_company_submissions(cik)
            if not data or "filings" not in data or "recent" not in data["filings"]:
                continue
            recent = data["filings"]["recent"]
            forms = recent.get("form", [])
            accessions = recent.get("accessionNumber", [])
            dates = recent.get("filingDate", [])
            report_dates = recent.get("reportDate", [])
            items_list = recent.get("items", [])

            for idx, form in enumerate(forms):
                if form not in target_forms:
                    continue
                acc_num = accessions[idx] if idx < len(accessions) else ""
                if not acc_num or acc_num in ingested:
                    continue
                filing_date = dates[idx] if idx < len(dates) else ""
                report_date = report_dates[idx] if idx < len(report_dates) else ""

                items_val = ""
                if idx < len(items_list):
                    items_val = str(items_list[idx] or "").strip()

                event_summary = ""
                if form == "8-K" and items_val:
                    item_keys = [it.strip() for it in items_val.replace(";", ",").split(",") if it.strip()]
                    descs = [f"Item {k}: {SEC_8K_ITEM_DESCRIPTIONS.get(k, 'Material Corporate Event')}" for k in item_keys]
                    event_summary = "; ".join(descs)

                filing_info = {
                    "ticker": ticker.upper(),
                    "cik": cik,
                    "form": form,
                    "accession_number": acc_num,
                    "filing_date": filing_date,
                    "report_date": report_date,
                }
                if event_summary:
                    filing_info["event_summary"] = event_summary
                    filing_info["items"] = items_val

                new_filings.append(filing_info)

            time.sleep(0.12)  # Respect SEC EDGAR 10 requests/sec rate limit

        return new_filings

    def sync(
        self, target_tickers: list[str] | None = None
    ) -> dict[str, Any]:
        """Poll and ingest newly discovered filings."""
        from scripts.ingest_sec_facts import ingest_company_facts

        new_filings = self.discover_new_filings(target_tickers=target_tickers)
        synced_count = 0
        new_facts_count = 0

        for filing in new_filings:
            ticker = filing["ticker"]
            cik = filing["cik"]
            form = filing["form"]
            event_summary = filing.get("event_summary", "")
            logger.info("Ingesting new filing %s for %s (%s)", form, ticker, filing["accession_number"])
            if form == "8-K":
                self.mark_filing_ingested(
                    ticker=ticker,
                    accession_number=filing["accession_number"],
                    form=form,
                    filing_date=filing["filing_date"],
                    facts_count=0,
                    event_summary=event_summary,
                )
                synced_count += 1
                continue

            try:
                count = ingest_company_facts(
                    ticker=ticker,
                    cik=cik,
                    db_path=self.db_path,
                )
                self.mark_filing_ingested(
                    ticker=ticker,
                    accession_number=filing["accession_number"],
                    form=form,
                    filing_date=filing["filing_date"],
                    facts_count=count,
                    event_summary=event_summary,
                )
                synced_count += 1
                new_facts_count += count
            except Exception as exc:
                logger.error("Failed to ingest facts for %s: %s", ticker, exc)

        return {
            "status": "COMPLETED",
            "new_filings_found": len(new_filings),
            "filings_synced": synced_count,
            "facts_ingested": new_facts_count,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="SEC EDGAR Feed Monitor & Auto Ingestion")
    parser.add_argument("--db", type=str, default="data/financial_facts.sqlite", help="SQLite fact store path")
    parser.add_argument("--tickers", nargs="+", help="Tickers to monitor")
    parser.add_argument("--poll", action="store_true", help="Execute sync now")
    args = parser.parse_args()

    monitor = EdgarFeedMonitor(db_path=args.db)
    if args.poll:
        result = monitor.sync(target_tickers=args.tickers)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
