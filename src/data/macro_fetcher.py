"""
Macroeconomic data fetchers.

Pulls macro indicators from:
  - FRED (Federal Reserve Economic Data) via fredapi
  - World Bank API via wbgapi

Data is cached aggressively — macro data changes monthly or quarterly.
"""

from __future__ import annotations

from src.data.data_cache import DataCache, TTL_MACRO_GDP, TTL_MACRO_RATES
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MacroFetcher:
    """Fetch macroeconomic data from FRED and World Bank.

    Parameters
    ----------
    cache : DataCache | None
        Application-layer cache. Created automatically if omitted.
    """

    def __init__(self, cache: DataCache | None = None) -> None:
        self.cache = cache or DataCache()
        self.config = get_config()

    # ------------------------------------------------------------------
    # FRED helpers
    # ------------------------------------------------------------------

    def _get_fred(self) -> object | None:
        """Return a fredapi Fred instance, or None if the key is missing."""
        if not self.config.has_fred:
            logger.warning("FRED API key not configured — macro data will be limited")
            return None
        try:
            from fredapi import Fred
        except ImportError:
            logger.error("fredapi not installed")
            return None
        return Fred(api_key=self.config.fred_api_key)

    def _fred_series(self, series_id: str, cache_key: str, ttl: int) -> float:
        """Fetch the most recent value of a FRED series, with caching."""
        cached = self.cache.get(cache_key)
        if cached is not None:
            return float(cached)

        fred = self._get_fred()
        if fred is None:
            return 0.0

        try:
            series = fred.get_series(series_id)
            if series.empty:
                logger.warning(f"FRED series {series_id} returned empty")
                return 0.0
            # Most recent non-null value
            latest: float = float(series.dropna().iloc[-1])
            self.cache.set(cache_key, latest, ttl_seconds=ttl)
            logger.info(f"FRED {series_id} = {latest:.4f}")
            return latest
        except Exception as exc:
            logger.error(f"FRED series {series_id} failed: {exc}")
            return 0.0

    # ------------------------------------------------------------------
    # Risk-Free Rate
    # ------------------------------------------------------------------

    def fetch_risk_free_rate(self, use_cache: bool = True) -> float:
        """Return the 10-Year US Treasury yield (DGS10)."""
        if not use_cache:
            self.cache.delete("fred:DGS10:daily")
        return self._fred_series(
            "DGS10",
            "fred:DGS10:daily",
            TTL_MACRO_RATES,
        )

    # ------------------------------------------------------------------
    # US GDP Growth
    # ------------------------------------------------------------------

    def fetch_gdp_growth(self, country: str = "US", use_cache: bool = True) -> float:
        """Return annual GDP growth rate as a decimal (e.g., 0.025 = 2.5%).

        For the US, pulls from FRED (GDPC1 = real GDP, compute YoY % change).
        For other countries, pulls from World Bank API.
        """
        cache_key = f"macro:gdp_growth:{country}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return float(cached)

        if country.upper() == "US":
            value = self._fred_gdp_growth(cache_key)
        else:
            value = self._worldbank_gdp_growth(country, cache_key)

        return value

    def _fred_gdp_growth(self, cache_key: str) -> float:
        """Compute US GDP growth from FRED GDPC1 series."""
        fred = self._get_fred()
        if fred is None:
            return 0.025  # sensible default

        try:
            gdp = fred.get_series("GDPC1")  # Real GDP, billions of chained 2017 $
            if len(gdp) < 2:
                return 0.025
            # YoY growth from the two most recent quarterly values
            recent = gdp.dropna().iloc[-1]
            year_ago = gdp.dropna().iloc[-5] if len(gdp.dropna()) >= 5 else gdp.dropna().iloc[0]
            growth = (recent - year_ago) / abs(year_ago) if year_ago != 0 else 0.025
            growth = max(-0.10, min(0.15, growth))
            self.cache.set(cache_key, float(growth), ttl_seconds=TTL_MACRO_GDP)
            logger.info(f"US GDP growth = {growth:.4f}")
            return float(growth)
        except Exception as exc:
            logger.error(f"FRED GDP growth failed: {exc}")
            return 0.025

    # ------------------------------------------------------------------
    # World Bank
    # ------------------------------------------------------------------

    def _worldbank_gdp_growth(self, country: str, cache_key: str) -> float:
        """Fetch GDP growth from World Bank API."""
        try:
            import wbgapi as wb
        except ImportError:
            logger.error("wbgapi not installed")
            return 0.03

        try:
            # Map country name to ISO code (simple heuristic)
            country_map = {
                "US": "USA", "UNITED STATES": "USA",
                "CN": "CHN", "CHINA": "CHN",
                "IN": "IND", "INDIA": "IND",
                "GB": "GBR", "UK": "GBR", "UNITED KINGDOM": "GBR",
                "JP": "JPN", "JAPAN": "JPN",
                "DE": "DEU", "GERMANY": "DEU",
                "BR": "BRA", "BRAZIL": "BRA",
                "FR": "FRA", "FRANCE": "FRA",
                "CA": "CAN", "CANADA": "CAN",
            }
            iso = country_map.get(country.upper(), country.upper()[:3])

            # NY.GDP.MKTP.KD.ZG = GDP growth (annual %)
            df = wb.data.DataFrame("NY.GDP.MKTP.KD.ZG", iso, time=range(2019, 2025))
            if df.empty:
                return 0.03

            recent = df.iloc[-1].dropna()
            if recent.empty:
                return 0.03

            growth = float(recent.iloc[-1]) / 100.0  # World Bank returns percentages
            growth = max(-0.10, min(0.15, growth))
            self.cache.set(cache_key, growth, ttl_seconds=TTL_MACRO_GDP)
            logger.info(f"{country} GDP growth (WB) = {growth:.4f}")
            return growth
        except Exception as exc:
            logger.error(f"World Bank GDP growth for {country} failed: {exc}")
            return 0.03

    # ------------------------------------------------------------------
    # Inflation
    # ------------------------------------------------------------------

    def fetch_inflation(self, country: str = "US", use_cache: bool = True) -> float:
        """Return the most recent annual inflation rate as a decimal."""
        cache_key = f"macro:inflation:{country}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return float(cached)

        if country.upper() == "US":
            # FRED: CPIAUCSL (CPI All Urban Consumers) YoY
            fred = self._get_fred()
            if fred is not None:
                try:
                    cpi = fred.get_series("CPIAUCSL")
                    if len(cpi) >= 13:
                        recent = cpi.dropna().iloc[-1]
                        year_ago = cpi.dropna().iloc[-13]
                        inflation = (recent - year_ago) / abs(year_ago)
                        inflation = max(-0.05, min(0.20, inflation))
                        self.cache.set(cache_key, float(inflation), ttl_seconds=TTL_MACRO_GDP)
                        return float(inflation)
                except Exception:
                    pass

        # World Bank fallback
        try:
            import wbgapi as wb
            country_map = {"US": "USA", "IN": "IND", "CN": "CHN", "GB": "GBR", "JP": "JPN"}
            iso = country_map.get(country.upper(), country.upper()[:3])
            df = wb.data.DataFrame("FP.CPI.TOTL.ZG", iso, time=range(2019, 2025))
            if not df.empty:
                recent = df.iloc[-1].dropna()
                if not recent.empty:
                    inflation = float(recent.iloc[-1]) / 100.0
                    inflation = max(-0.05, min(0.20, inflation))
                    self.cache.set(cache_key, inflation, ttl_seconds=TTL_MACRO_GDP)
                    return inflation
        except Exception:
            pass

        logger.warning(f"Could not fetch inflation for {country}, using 3% default")
        return 0.03

    # ------------------------------------------------------------------
    # Comprehensive macro snapshot
    # ------------------------------------------------------------------

    def fetch_macro_snapshot(self, country: str = "US", use_cache: bool = True) -> dict:
        """Return a complete macro snapshot for the given country."""
        cache_key = f"macro:snapshot:{country}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        result = {
            "country": country,
            "risk_free_rate": self.fetch_risk_free_rate(use_cache),
            "gdp_growth": self.fetch_gdp_growth(country, use_cache),
            "inflation": self.fetch_inflation(country, use_cache),
        }
        # Default equity risk premium based on country development
        if country.upper() == "US":
            result["equity_risk_premium"] = 0.05
        elif country.upper() in ("IN", "BR", "CN", "ZA", "TR", "RU"):
            result["equity_risk_premium"] = 0.075
        else:
            result["equity_risk_premium"] = 0.06

        self.cache.set(cache_key, result, ttl_seconds=TTL_MACRO_RATES)
        return result
