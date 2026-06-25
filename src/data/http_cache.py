"""
HTTP-level caching via ``requests-cache``.

Provides a singleton ``CachedSession`` that transparently caches HTTP
responses from data-source REST APIs.  TTLs are set per URL pattern to
match the freshness requirements in the architecture.

Used by data fetcher modules instead of raw ``requests.get()`` — all
responses are cached automatically without the fetcher needing to care.
"""

from __future__ import annotations

import requests_cache

from src.utils.config import get_config

# ---------------------------------------------------------------------------
# URL-pattern → TTL mapping (seconds)
# ---------------------------------------------------------------------------

_URL_TTL_MAP: dict[str, int | str] = {
    # Yahoo Finance (yfinance uses its own internal caching, but we can
    # still cache any direct HTTP calls).
    "*.yahoo.com/*": 86_400,  # 1 day

    # SEC EDGAR — filings change rarely, but we use a moderate TTL so
    # we don't miss a Friday filing over the weekend.
    "data.sec.gov/*": 86_400,  # 1 day
    "www.sec.gov/*": 86_400,

    # FRED (St. Louis Fed) — rates update daily, macro series vary.
    "api.stlouisfed.org/*": 604_800,  # 7 days

    # World Bank — annual / quarterly indicators.
    "api.worldbank.org/*": 2_592_000,  # 30 days

    # FMP — analyst estimates, DCFs.
    "financialmodelingprep.com/*": 604_800,  # 7 days

    # Alpha Vantage.
    "www.alphavantage.co/*": 604_800,  # 7 days
}

# ---------------------------------------------------------------------------
# Singleton session
# ---------------------------------------------------------------------------

_session: requests_cache.CachedSession | None = None


def get_http_session() -> requests_cache.CachedSession:
    """Return (or create) the process-wide cached HTTP session.

    The session caches responses in a SQLite database inside the project's
    cache directory.  TTLs are applied per URL pattern (see ``_URL_TTL_MAP``).

    Usage in fetcher modules::

        from src.data.http_cache import get_http_session

        session = get_http_session()
        resp = session.get("https://api.stlouisfed.org/fred/series/observations", params=...)
    """
    global _session
    if _session is not None:
        return _session

    cfg = get_config()
    cfg.ensure_cache_dir()

    _session = requests_cache.CachedSession(
        cache_name=str(cfg.http_cache_path),
        backend="sqlite",
        urls_expire_after=_URL_TTL_MAP,
        allowable_methods=("GET", "HEAD"),
        allowable_codes=(200, 301, 302, 404),  # cache 404s briefly to avoid re-hitting bad URLs
        match_headers=False,  # only URL + params matter
        stale_if_error=True,  # serve stale cache if the remote is unreachable
    )

    # Set a sensible user-agent so APIs don't block us.
    _session.headers.update(
        {
            "User-Agent": "IntrensicValuator/0.1 (contact@example.com)",
            "Accept": "application/json",
        }
    )

    return _session


def clear_http_cache() -> None:
    """Remove all cached HTTP responses (forces fresh calls next time)."""
    session = get_http_session()
    session.cache.clear()
