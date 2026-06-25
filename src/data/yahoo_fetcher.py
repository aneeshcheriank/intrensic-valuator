"""
Yahoo Finance data fetcher.

Wraps the yfinance library to pull stock price data, financial statements,
company info, and peer companies. Results are cached via DataCache to avoid
redundant API calls.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from src.data.data_cache import DataCache, TTL_FINANCIALS, TTL_PRICE
from src.utils.logger import get_logger

logger = get_logger(__name__)


class YahooFinanceFetcher:
    """Fetch financial data from Yahoo Finance via yfinance.

    All methods accept an optional *use_cache* parameter (default True).
    When True, the result is cached in the application-layer DataCache.
    """

    def __init__(self, cache: DataCache | None = None) -> None:
        self.cache = cache or DataCache()

    # ------------------------------------------------------------------
    # Company info
    # ------------------------------------------------------------------

    def fetch_company_info(self, ticker: str, use_cache: bool = True) -> dict:
        """Return company metadata: name, sector, industry, market cap, shares, etc."""
        cache_key = f"yfinance:{ticker}:company_info"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached

        logger.info(f"Fetching company info for {ticker}...")
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        result = {
            "ticker": ticker.upper(),
            "company_name": info.get("longName") or info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "country": info.get("country", "United States"),
            "market_cap": _safe_float(info.get("marketCap")),
            "enterprise_value": _safe_float(info.get("enterpriseValue")),
            "shares_outstanding": _safe_float(info.get("sharesOutstanding")),
            "beta": _safe_float(info.get("beta"), default=1.0),
            "current_price": _safe_float(info.get("currentPrice")),
            "previous_close": _safe_float(info.get("previousClose")),
            "fifty_two_week_high": _safe_float(info.get("fiftyTwoWeekHigh")),
            "fifty_two_week_low": _safe_float(info.get("fiftyTwoWeekLow")),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", ""),
            "website": info.get("website", ""),
            "description": info.get("longBusinessSummary", ""),
        }
        self.cache.set(cache_key, result, ttl_seconds=TTL_PRICE)
        return result

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------

    def fetch_price_data(self, ticker: str, use_cache: bool = True) -> dict:
        """Return current price, beta, and recent price history."""
        cache_key = f"yfinance:{ticker}:price_data"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        logger.info(f"Fetching price data for {ticker}...")
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        hist = stock.history(period="1y")

        result = {
            "current_price": _safe_float(info.get("currentPrice")),
            "previous_close": _safe_float(info.get("previousClose")),
            "beta": _safe_float(info.get("beta"), default=1.0),
            "fifty_day_avg": _safe_float(info.get("fiftyDayAverage")),
            "two_hundred_day_avg": _safe_float(info.get("twoHundredDayAverage")),
            "year_high": _safe_float(info.get("fiftyTwoWeekHigh")),
            "year_low": _safe_float(info.get("fiftyTwoWeekLow")),
            "price_history": hist["Close"].tail(252).tolist() if not hist.empty else [],
        }
        self.cache.set(cache_key, result, ttl_seconds=TTL_PRICE)
        return result

    # ------------------------------------------------------------------
    # Financial statements
    # ------------------------------------------------------------------

    def fetch_financials(self, ticker: str, use_cache: bool = True) -> dict[str, pd.DataFrame]:
        """Return IS, BS, CF as DataFrames (annual + quarterly)."""
        cache_key = f"yfinance:{ticker}:financials"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        logger.info(f"Fetching financial statements for {ticker}...")
        stock = yf.Ticker(ticker)

        result = {
            "income_statement_annual": _df_or_empty(stock.financials),
            "income_statement_quarterly": _df_or_empty(stock.quarterly_financials),
            "balance_sheet_annual": _df_or_empty(stock.balance_sheet),
            "balance_sheet_quarterly": _df_or_empty(stock.quarterly_balance_sheet),
            "cash_flow_annual": _df_or_empty(stock.cashflow),
            "cash_flow_quarterly": _df_or_empty(stock.quarterly_cashflow),
        }
        self.cache.set(cache_key, result, ttl_seconds=TTL_FINANCIALS)
        return result

    # ------------------------------------------------------------------
    # Key financial metrics (extracted from statements)
    # ------------------------------------------------------------------

    def fetch_key_metrics(self, ticker: str, use_cache: bool = True) -> dict:
        """Return a flat dict of key valuation inputs extracted from financials."""
        cache_key = f"yfinance:{ticker}:key_metrics"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        financials = self.fetch_financials(ticker, use_cache=use_cache)
        info = self.fetch_company_info(ticker, use_cache=use_cache)

        cf = financials["cash_flow_annual"]
        bs = financials["balance_sheet_annual"]
        inc = financials["income_statement_annual"]

        # Extract latest annual values
        ocf = _get_latest(cf, "Operating Cash Flow")
        capex = _get_latest(cf, "Capital Expenditure")
        if capex:
            capex = abs(capex)

        revenue = _get_latest(inc, "Total Revenue")
        net_income = _get_latest(inc, "Net Income")
        ebit = _get_latest(inc, "EBIT")
        ebitda_val = _get_latest(inc, "EBITDA")
        interest_expense = _get_latest(inc, "Interest Expense")
        tax_expense = _get_latest(inc, "Tax Provision")
        pretax_income = _get_latest(inc, "Pretax Income")

        total_debt = _get_latest(bs, "Total Debt")
        if total_debt == 0.0:
            total_debt = _get_latest(bs, "Long Term Debt") + _get_latest(bs, "Short Long Term Debt")
        cash_equiv = _get_latest(bs, "Cash And Cash Equivalents") or _get_latest(bs, "Cash")
        total_assets = _get_latest(bs, "Total Assets")
        total_equity = _get_latest(bs, "Stockholders Equity") or _get_latest(bs, "Total Equity")

        shares = info.get("shares_outstanding", 0.0)
        market_cap = info.get("market_cap", 0.0)
        current_price = info.get("current_price", 0.0)

        # Derived metrics
        fcf = ocf - capex if ocf else 0.0
        fcf_margin = fcf / revenue if revenue else 0.0
        tax_rate = tax_expense / pretax_income if pretax_income else 0.21
        eps = net_income / shares if shares else 0.0
        bvps = total_equity / shares if shares else 0.0
        icr = ebit / abs(interest_expense) if interest_expense else 99.0

        result = {
            "ticker": ticker.upper(),
            "revenue": revenue or 0.0,
            "net_income": net_income or 0.0,
            "ebit": ebit or 0.0,
            "ebitda": ebitda_val or (ebit or 0.0),
            "operating_cash_flow": ocf or 0.0,
            "capital_expenditure": capex or 0.0,
            "free_cash_flow": fcf,
            "fcf_margin": fcf_margin,
            "total_debt": total_debt or 0.0,
            "cash_and_equivalents": cash_equiv or 0.0,
            "total_assets": total_assets or 0.0,
            "total_equity": total_equity or 0.0,
            "interest_expense": abs(interest_expense or 0.0),
            "tax_rate": tax_rate,
            "eps": eps,
            "book_value_per_share": bvps,
            "interest_coverage_ratio": icr,
            "shares_outstanding": shares,
            "market_cap": market_cap,
            "current_price": current_price,
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "beta": info.get("beta", 1.0),
        }
        self.cache.set(cache_key, result, ttl_seconds=TTL_FINANCIALS)
        return result

    # ------------------------------------------------------------------
    # Peer companies
    # ------------------------------------------------------------------

    def fetch_peer_companies(self, ticker: str, use_cache: bool = True) -> list[str]:
        """Return a list of peer ticker symbols."""
        cache_key = f"yfinance:{ticker}:peers"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        logger.info(f"Finding peers for {ticker}...")
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # Try direct peers from yfinance
        peers: list[str] = info.get("sectorPeers", []) or info.get("peerList", [])

        # If none, try the ticker's recommendations or similar
        if not peers:
            try:
                recs = stock.recommendations
                if recs is not None and not recs.empty:
                    # Get peer tickers from analyst coverage
                    pass
            except Exception:
                pass

        # Fallback: use the industry key to find similar companies
        if not peers:
            industry = info.get("industry", "")
            sector = info.get("sector", "")
            logger.warning(
                f"No direct peers for {ticker} (industry={industry}, sector={sector}). "
                f"Will rely on manual peer specification."
            )
            peers = []

        result = [p.upper() for p in peers[:12]]  # cap at 12
        self.cache.set(cache_key, result, ttl_seconds=TTL_FINANCIALS)
        return result

    # ------------------------------------------------------------------
    # Peer financials (for relative valuation)
    # ------------------------------------------------------------------

    def fetch_peer_metrics(
        self, peer_tickers: list[str], use_cache: bool = True
    ) -> list[dict]:
        """Fetch key valuation multiples for a list of peer tickers."""
        results = []
        for pt in peer_tickers:
            try:
                info = self.fetch_company_info(pt, use_cache=use_cache)
                metrics = self.fetch_key_metrics(pt, use_cache=use_cache)
                price = info.get("current_price", 0.0) or metrics.get("current_price", 0.0)
                eps = metrics.get("eps", 0.0)
                ebitda_val = metrics.get("ebitda", 0.0)
                ev = info.get("enterprise_value", 0.0)
                bvps = metrics.get("book_value_per_share", 0.0)

                results.append({
                    "ticker": pt,
                    "pe_ratio": price / eps if eps > 0 else None,
                    "ev_ebitda": ev / ebitda_val if ebitda_val > 0 else None,
                    "pb_ratio": price / bvps if bvps > 0 else None,
                    "market_cap": metrics.get("market_cap", 0.0),
                    "revenue": metrics.get("revenue", 0.0),
                })
            except Exception as exc:
                logger.warning(f"Failed to fetch peer metrics for {pt}: {exc}")

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely coerce a value to float, returning *default* on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _df_or_empty(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return *df* if it's a valid DataFrame, otherwise an empty one."""
    if df is not None and not df.empty:
        return df
    return pd.DataFrame()


def _get_latest(df: pd.DataFrame, field: str) -> float:
    """Extract the most recent non-null value for *field* from a financials DataFrame.

    yfinance format: index=field names, columns=dates (most recent first).
    Tries multiple possible field name matches since yfinance field names
    vary across stocks and statement types.
    """
    if df is None or df.empty:
        return 0.0

    # Direct match
    if field in df.index:
        row = df.loc[field]
        if hasattr(row, "dropna"):
            valid = row.dropna()
            if not valid.empty:
                return float(valid.iloc[0])

    # Fuzzy match: try common variations
    aliases: dict[str, list[str]] = {
        "Total Revenue": ["Total Revenue", "Revenue", "Total Revenues", "Sales Revenue Net", "Revenues"],
        "Operating Cash Flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Net Cash Provided By Operating Activities", "Operating Cash Flow"],
        "Capital Expenditure": ["Capital Expenditure", "Capital Expenditures", "Purchase Of Property Plant And Equipment", "Payments To Acquire Property Plant And Equipment", "Capital Expenditure Reported"],
        "Net Income": ["Net Income", "Net Income Loss", "Net Income From Continuing And Discontinued Operation", "Net Income Common Stockholders"],
        "EBIT": ["EBIT", "Operating Income", "Operating Income Loss"],
        "EBITDA": ["EBITDA", "Normalized EBITDA", "Reconciled EBITDA"],
        "Interest Expense": ["Interest Expense", "Interest Expense Non Operating", "Interest Expense Net"],
        "Tax Provision": ["Tax Provision", "Income Tax Expense", "Tax Effect Of Unusual Items"],
        "Pretax Income": ["Pretax Income", "Income Before Tax", "Income From Continuing Operations Before Income Taxes"],
        "Total Debt": ["Total Debt", "Total Long Term Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligations"],
        "Cash And Cash Equivalents": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"],
        "Total Assets": ["Total Assets", "Assets", "Total Assets Non Current"],
        "Stockholders Equity": ["Stockholders Equity", "Total Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"],
        "Free Cash Flow": ["Free Cash Flow"],
    }

    for alias in aliases.get(field, [field]):
        if alias in df.index:
            row = df.loc[alias]
            if hasattr(row, "dropna"):
                valid = row.dropna()
                if not valid.empty:
                    return float(valid.iloc[0])

    return 0.0
