"""
SEC EDGAR data fetcher.

Fetches 10-K / 10-Q XBRL data from the SEC EDGAR API for US-listed companies.
Used as a verification layer to cross-check yfinance financial statement data.
"""

from __future__ import annotations

import json
from typing import Any

import requests

from src.data.data_cache import DataCache, TTL_FINANCIALS
from src.data.http_cache import get_http_session
from src.utils.logger import get_logger

logger = get_logger(__name__)

# SEC requires a User-Agent identifying the requestor
_SEC_HEADERS = {
    "User-Agent": "IntrensicValuator/0.1 (contact@example.com)",
    "Accept": "application/json",
}

# Mapping from common SEC XBRL concepts to our internal field names
_XBRL_CONCEPT_MAP: dict[str, str] = {
    # Income Statement
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "Revenues": "revenue",
    "SalesRevenueNet": "revenue",
    "NetIncomeLoss": "net_income",
    "OperatingIncomeLoss": "ebit",
    "InterestExpense": "interest_expense",
    "IncomeTaxExpenseBenefit": "tax_expense",
    # Balance Sheet
    "Assets": "total_assets",
    "Liabilities": "total_liabilities",
    "StockholdersEquity": "total_equity",
    "LongTermDebt": "long_term_debt",
    "CashAndCashEquivalentsAtCarryingValue": "cash_and_equivalents",
    "CommonStockSharesOutstanding": "shares_outstanding",
    # Cash Flow
    "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capital_expenditure",
}


class SECFetcher:
    """Fetch XBRL-tagged financial data from SEC EDGAR.

    Parameters
    ----------
    cache : DataCache | None
        Application-layer cache.
    """

    BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts"

    def __init__(self, cache: DataCache | None = None) -> None:
        self.cache = cache or DataCache()
        self.session = get_http_session()

    # ------------------------------------------------------------------
    # CIK lookup
    # ------------------------------------------------------------------

    def _get_cik(self, ticker: str) -> str | None:
        """Look up the CIK (Central Index Key) for a ticker symbol."""
        cache_key = f"sec:cik:{ticker}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached if cached != "__none__" else None

        try:
            # SEC company tickers JSON
            url = "https://www.sec.gov/files/company_tickers.json"
            resp = self.session.get(url, headers=_SEC_HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            ticker_upper = ticker.upper()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker_upper:
                    cik_str = str(entry["cik_str"]).zfill(10)
                    self.cache.set(cache_key, cik_str, ttl_seconds=TTL_FINANCIALS)
                    return cik_str

            self.cache.set(cache_key, "__none__", ttl_seconds=TTL_FINANCIALS)
            return None
        except Exception as exc:
            logger.error(f"CIK lookup failed for {ticker}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Company facts (XBRL)
    # ------------------------------------------------------------------

    def fetch_company_facts(self, ticker: str, use_cache: bool = True) -> dict:
        """Fetch the full company facts XBRL-JSON from SEC EDGAR.

        Returns the raw JSON as a dict.
        """
        cache_key = f"sec:facts:{ticker}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        cik = self._get_cik(ticker)
        if cik is None:
            logger.warning(f"Could not find CIK for {ticker}")
            return {}

        url = f"{self.BASE_URL}/CIK{cik}.json"
        try:
            resp = self.session.get(url, headers=_SEC_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            self.cache.set(cache_key, data, ttl_seconds=TTL_FINANCIALS)
            logger.info(f"Fetched SEC facts for {ticker} (CIK {cik})")
            return data
        except Exception as exc:
            logger.error(f"SEC facts fetch failed for {ticker}: {exc}")
            return {}

    # ------------------------------------------------------------------
    # Extract key metrics from XBRL
    # ------------------------------------------------------------------

    def extract_key_metrics(self, ticker: str, use_cache: bool = True) -> dict[str, float]:
        """Extract key financial metrics from SEC XBRL data.

        Pulls the most recent annual (10-K) values for each mapped concept.
        """
        cache_key = f"sec:metrics:{ticker}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        facts = self.fetch_company_facts(ticker, use_cache=use_cache)
        if not facts:
            return {}

        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        if not us_gaap:
            logger.warning(f"No US-GAAP facts found for {ticker}")
            return {}

        metrics: dict[str, float] = {}

        for xbrl_concept, our_field in _XBRL_CONCEPT_MAP.items():
            if xbrl_concept in us_gaap:
                concept_data = us_gaap[xbrl_concept]
                units = concept_data.get("units", {})

                # Prefer USD, then shares
                for unit_key in ("USD", "shares", "USD/shares"):
                    if unit_key in units:
                        filings = units[unit_key]
                        # Get the most recent 10-K (annual) filing
                        annuals = [
                            f for f in filings
                            if f.get("form") in ("10-K", "10-K/A") and f.get("fy")
                        ]
                        if annuals:
                            # Sort by fiscal year descending
                            annuals.sort(key=lambda f: f.get("fy", 0), reverse=True)
                            metrics[our_field] = float(annuals[0]["val"])
                        break

        self.cache.set(cache_key, metrics, ttl_seconds=TTL_FINANCIALS)
        return metrics

    # ------------------------------------------------------------------
    # Cross-check against yfinance
    # ------------------------------------------------------------------

    def cross_check(
        self, ticker: str, yahoo_metrics: dict, use_cache: bool = True
    ) -> dict:
        """Compare yfinance metrics against SEC EDGAR data.

        Returns a dict with discrepancies flagged.
        """
        sec_metrics = self.extract_key_metrics(ticker, use_cache=use_cache)
        if not sec_metrics:
            return {"status": "no_sec_data", "discrepancies": []}

        discrepancies = []
        # Check key fields
        checks = [
            ("revenue", "Total Revenue"),
            ("net_income", "Net Income"),
            ("operating_cash_flow", "Operating Cash Flow"),
        ]

        for sec_field, label in checks:
            sec_val = sec_metrics.get(sec_field)
            yahoo_val = yahoo_metrics.get(sec_field.lower())

            if sec_val and yahoo_val and sec_val != 0:
                pct_diff = abs(sec_val - yahoo_val) / abs(sec_val)
                if pct_diff > 0.05:  # >5% discrepancy
                    discrepancies.append({
                        "field": label,
                        "sec_value": round(sec_val, 2),
                        "yahoo_value": round(yahoo_val, 2),
                        "pct_difference": round(pct_diff * 100, 2),
                    })

        return {
            "status": "ok" if not discrepancies else "discrepancies_found",
            "discrepancies": discrepancies,
            "sec_metrics_available": len(sec_metrics),
        }
