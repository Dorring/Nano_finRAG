"""Lightweight local SQL wide table fact store for structured financial data."""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any



# Standard canonical metric mapping to handle minor reporting variations
_METRIC_CANONICAL_MAP: dict[str, str] = {
    "total net sales": "total revenue",
    "net sales": "total revenue",
    "revenue": "total revenue",
    "revenues": "total revenue",
    "total revenues": "total revenue",
    "operating income": "operating income",
    "operating profit": "operating income",
    "net income": "net income",
    "net earnings": "net income",
    "gross profit": "gross profit",
    "total assets": "total assets",
    "total liabilities": "total liabilities",
    "stockholders' equity": "stockholders' equity",
    "shareholders' equity": "stockholders' equity",
    "total equity": "stockholders' equity",
    "diluted eps": "diluted eps",
    "diluted earnings per share": "diluted eps",
    "cost of sales": "cost of sales",
    "cost of revenue": "cost of sales",
    "research and development": "research and development",
    "r&d": "research and development",
    "r and d": "research and development",
    "rd": "research and development",
    "selling, general and administrative": "selling, general and administrative",
    "sg&a": "selling, general and administrative",
    "sg and a": "selling, general and administrative",
    "sga": "selling, general and administrative",
    "cash and cash equivalents": "cash and cash equivalents",
    "cash": "cash and cash equivalents",
    "operating cash flow": "operating cash flows",
    "operating cash flows": "operating cash flows",
    "cash flows from operating activities": "operating cash flows",
}

# Standard ticker normalization
_ENTITY_TICKER_MAP: dict[str, str] = {
    "apple": "aapl",
    "apple inc": "aapl",
    "apple inc.": "aapl",
    "aapl": "aapl",
    "microsoft": "msft",
    "microsoft corp": "msft",
    "microsoft corporation": "msft",
    "msft": "msft",
    "tesla": "tsla",
    "tesla inc": "tsla",
    "tesla inc.": "tsla",
    "tsla": "tsla",
    "nvidia": "nvda",
    "nvidia corp": "nvda",
    "nvidia corporation": "nvda",
    "nvda": "nvda",
    "amazon": "amzn",
    "amazon com": "amzn",
    "amazon.com inc": "amzn",
    "amzn": "amzn",
    "google": "googl",
    "alphabet": "googl",
    "alphabet inc": "googl",
    "googl": "googl",
    "goog": "googl",
    "meta": "meta",
    "meta platforms": "meta",
    "facebook": "meta",
    "jpmorgan": "jpm",
    "jpmorgan chase": "jpm",
    "jpmorgan chase & co": "jpm",
    "jpm": "jpm",
    "coca-cola": "ko",
    "coca cola": "ko",
    "coke": "ko",
    "ko": "ko",
    "walmart": "wmt",
    "walmart inc": "wmt",
    "wmt": "wmt",
    "exxon": "xom",
    "exxonmobil": "xom",
    "exxon mobil": "xom",
    "xom": "xom",
    "netflix": "nflx",
    "nflx": "nflx",
}




def normalize_metric(metric: str) -> str:
    """Normalize metric string into canonical accounting representation."""
    norm = metric.strip().lower()
    norm = norm.replace("&", " and ")
    cleaned = re.sub(r"[^\w\s']", "", norm)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return _METRIC_CANONICAL_MAP.get(cleaned, cleaned)



def normalize_entity(entity: str) -> tuple[str, str]:
    """Normalize entity into (display_name, entity_id/ticker)."""
    clean_ent = entity.strip()
    norm = clean_ent.lower()
    ticker = _ENTITY_TICKER_MAP.get(norm, norm.replace(" ", "_"))
    return clean_ent, ticker


def normalize_period(period: str) -> str:
    """Normalize period representation to standard FYYYYY or YYYY."""
    p = period.strip().upper()
    m = re.search(r"\b(20\d{2})\b", p)
    if m:
        year = m.group(1)
        if "FY" in p or "FISCAL" in p:
            return f"FY{year}"
        return f"FY{year}"
    return p


DEFAULT_FACT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "financial_facts.sqlite",
)


class SqlFactStore:
    """Lightweight SQLite-backed structured financial fact wide table."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is not None:
            self.db_path = db_path
        elif os.path.exists(DEFAULT_FACT_DB_PATH):
            self.db_path = DEFAULT_FACT_DB_PATH
        else:
            self.db_path = ":memory:"

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()


    def _init_db(self) -> None:
        """Create schema and lookup indices."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS financial_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                metric TEXT NOT NULL,
                period TEXT NOT NULL,
                value TEXT NOT NULL,
                unit TEXT NOT NULL DEFAULT 'USD',
                scale TEXT NOT NULL DEFAULT 'million',
                scope TEXT NOT NULL DEFAULT 'consolidated',
                source_doc TEXT NOT NULL DEFAULT '10-K iXBRL',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_facts_lookup
            ON financial_facts(entity_id, metric, period, scope);
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_facts_entity_name
            ON financial_facts(entity, metric, period, scope);
            """
        )
        self._conn.commit()

    def upsert_fact(
        self,
        *,
        entity: str,
        metric: str,
        period: str,
        value: str | int | float,
        unit: str = "USD",
        scale: str = "million",
        scope: str = "consolidated",
        source_doc: str = "10-K iXBRL",
    ) -> None:
        """Insert or update a single financial fact."""
        clean_ent, entity_id = normalize_entity(entity)
        norm_m = normalize_metric(metric)
        norm_p = normalize_period(period)
        val_str = str(value).strip()

        cursor = self._conn.cursor()
        # Delete existing duplicate fact if any to guarantee uniqueness
        cursor.execute(
            """
            DELETE FROM financial_facts
            WHERE entity_id = ? AND metric = ? AND period = ? AND scope = ?
            """,
            (entity_id, norm_m, norm_p, scope),
        )
        cursor.execute(
            """
            INSERT INTO financial_facts
            (entity, entity_id, metric, period, value, unit, scale, scope, source_doc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (clean_ent, entity_id, norm_m, norm_p, val_str, unit, scale, scope, source_doc),
        )
        self._conn.commit()

    def upsert_facts(self, facts: Iterable[Mapping[str, Any]]) -> None:
        """Batch upsert multiple facts."""
        for f in facts:
            self.upsert_fact(
                entity=str(f["entity"]),
                metric=str(f["metric"]),
                period=str(f["period"]),
                value=str(f["value"]),
                unit=str(f.get("unit", "USD")),
                scale=str(f.get("scale", "million")),
                scope=str(f.get("scope", "consolidated")),
                source_doc=str(f.get("source_doc", "10-K iXBRL")),
            )

    def query_fact(
        self,
        entity: str,
        metric: str,
        period: str,
        scope: str = "consolidated",
    ) -> dict[str, Any] | None:
        """Query a fact using normalized entity, metric, and period."""
        clean_ent, entity_id = normalize_entity(entity)
        norm_m = normalize_metric(metric)
        norm_p = normalize_period(period)
        norm_scope = scope.strip().lower() or "consolidated"

        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT entity, entity_id, metric, period, value, unit, scale, scope, source_doc
            FROM financial_facts
            WHERE (entity_id = ? OR LOWER(entity) = LOWER(?))
              AND metric = ?
              AND period = ?
              AND scope = ?
            LIMIT 1
            """,
            (entity_id, clean_ent, norm_m, norm_p, norm_scope),
        )
        row = cursor.fetchone()
        if row is not None:
            return dict(row)
        return None

    def as_evidence_packet(
        self,
        fact: Mapping[str, Any],
        slot_id: str,
        candidate_id: str | None = None,
    ) -> dict[str, Any]:
        """Convert a fact row into an EvidencePacket-compatible dictionary."""
        cid = candidate_id or f"sql_fact_{fact['entity_id']}_{fact['metric']}_{fact['period']}"
        source_doc = fact.get("source_doc", "10-K iXBRL")
        return {
            "evidence_id": cid,
            "candidate_id": cid,
            "metric": fact["metric"],
            "value": str(fact["value"]),
            "period": fact["period"],
            "entity": fact["entity"],
            "entity_id": fact.get("entity_id"),
            "scope": fact.get("scope", "consolidated"),
            "unit": fact.get("unit", "USD"),
            "currency": fact.get("unit", "USD") if fact.get("unit") in ("USD", "EUR", "CNY", "GBP") else "USD",
            "scale": fact.get("scale", "million"),
            "source": "SQL_FACT_STORE",
            "document_id": source_doc,
            "citation_id": f"[{source_doc}] {fact['entity']} {fact['period']} {fact['metric']}",
            "page": 0,
            "slots": (str(slot_id),),
            "slot_ids": (str(slot_id),),
            "source_text": (
                f"{fact['entity']} reported {fact['metric']} of {fact['value']} "
                f"{fact.get('scale', 'million')} {fact.get('unit', 'USD')} for {fact['period']}."
            ),
            "temporal": {
                "entity": fact["entity"],
                "metric": fact["metric"],
                "value": str(fact["value"]),
                "period_semantics": "ANNUAL",
                "fiscal_year": fact["period"].replace("FY", ""),
                "source": "SQL_FACT_STORE",
            },
            "metadata": {
                "store": "sqlite_ixbrl_wide_table",
                "source_doc": source_doc,
            },
        }

    def seed_benchmark_data(self) -> None:
        """Seed standard SEC 10-K audited figures for testing and production benchmarks."""
        benchmark_facts = [
            # Apple (AAPL) Audited Figures
            {"entity": "Apple", "metric": "total revenue", "period": "FY2024", "value": "391035"},
            {"entity": "Apple", "metric": "net income", "period": "FY2024", "value": "93736"},
            {"entity": "Apple", "metric": "operating income", "period": "FY2024", "value": "123216"},
            {"entity": "Apple", "metric": "gross profit", "period": "FY2024", "value": "180683"},
            {"entity": "Apple", "metric": "total assets", "period": "FY2024", "value": "364980"},
            {"entity": "Apple", "metric": "total liabilities", "period": "FY2024", "value": "308030"},
            {"entity": "Apple", "metric": "total revenue", "period": "FY2023", "value": "383285"},
            {"entity": "Apple", "metric": "net income", "period": "FY2023", "value": "96995"},
            {"entity": "Apple", "metric": "total revenue", "period": "FY2022", "value": "394328"},
            # Microsoft (MSFT) Audited Figures
            {"entity": "Microsoft", "metric": "total revenue", "period": "FY2024", "value": "245122"},
            {"entity": "Microsoft", "metric": "net income", "period": "FY2024", "value": "88136"},
            {"entity": "Microsoft", "metric": "operating income", "period": "FY2024", "value": "109433"},
            {"entity": "Microsoft", "metric": "total assets", "period": "FY2024", "value": "512163"},
            # Tesla (TSLA) Audited Figures
            {"entity": "Tesla", "metric": "total revenue", "period": "FY2023", "value": "96773"},
            {"entity": "Tesla", "metric": "net income", "period": "FY2023", "value": "14997"},
            {"entity": "Tesla", "metric": "total assets", "period": "FY2023", "value": "106618"},
        ]
        self.upsert_facts(benchmark_facts)

    def close(self) -> None:
        """Close SQLite connection."""
        self._conn.close()
