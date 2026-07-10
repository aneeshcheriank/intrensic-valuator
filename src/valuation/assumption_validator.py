"""
Assumption Validation Layer — Pre-DCF Guardrail.

Before agent-generated assumptions enter the DCF engine, they pass through
a three-tier validation:

  1. Historical Range Check — Is the assumption within the company's 5-year range?
  2. Industry Benchmark Check — Is the assumption within 2× the industry median?
  3. Statistical Confidence Bands — GREEN (<1σ), AMBER (1-2σ), RED (>2σ → capped)

RED-flagged assumptions are capped at 2σ from the historical mean and
reported transparently. Confidence scoring is penalized for each flag.

This prevents hallucinated or unreasonably optimistic/pessimistic assumptions
from silently corrupting the valuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationBand(str, Enum):
    """Traffic-light band for assumption validation."""
    GREEN = "GREEN"    # Within 1σ — passes cleanly
    AMBER = "AMBER"    # 1-2σ — flagged, needs justification
    RED = "RED"        # >2σ — capped at 2σ bound


@dataclass
class ValidationResult:
    """Result of validating a single assumption."""
    parameter: str
    agent_value: float
    capped_value: float          # value after capping (same as agent_value if GREEN)
    band: ValidationBand
    historical_mean: float | None = None
    historical_std: float | None = None
    industry_median: float | None = None
    z_score: float | None = None  # (agent_value - mean) / std
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "parameter": self.parameter,
            "agent_value": self.agent_value,
            "capped_value": self.capped_value,
            "band": self.band.value,
            "historical_mean": self.historical_mean,
            "historical_std": self.historical_std,
            "industry_median": self.industry_median,
            "z_score": self.z_score,
            "message": self.message,
        }


@dataclass
class ValidationReport:
    """Aggregate validation report for all assumptions entering the DCF."""
    results: list[ValidationResult] = field(default_factory=list)
    green_count: int = 0
    amber_count: int = 0
    red_count: int = 0
    overall_band: ValidationBand = ValidationBand.GREEN

    def to_dict(self) -> dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "green_count": self.green_count,
            "amber_count": self.amber_count,
            "red_count": self.red_count,
            "overall_band": self.overall_band.value,
        }


# ---------------------------------------------------------------------------
# Default historical ranges by parameter
#
# These are fallback reference ranges derived from Damodaran's cross-sectional
# data. When company-specific historical data is available (from the 5-year
# financials), those are used preferentially.
# ---------------------------------------------------------------------------

_DEFAULT_REFERENCE_RANGES: dict[str, dict[str, float]] = {
    "revenue_growth": {
        "mean": 0.05,    # 5% — typical US large-cap revenue growth
        "std": 0.06,     # 6% — wide dispersion across sectors
        "min": -0.10,    # -10% — declining businesses
        "max": 0.25,     # 25% — high-growth outliers
    },
    "fcf_margin": {
        "mean": 0.15,    # 15% — median FCF margin across all US stocks
        "std": 0.10,     # 10% — wide dispersion (software vs retail)
        "min": 0.0,      # 0% — breakeven
        "max": 0.45,     # 45% — exceptional software/royalty businesses
    },
    "country_risk_premium": {
        "mean": 0.005,   # 50 bps — median CRP (most developed markets)
        "std": 0.015,    # 150 bps
        "min": 0.0,      # US = 0
        "max": 0.10,     # 1000 bps — distressed/frontier markets
    },
    "industry_growth_rate": {
        "mean": 0.04,    # 4% — median industry growth
        "std": 0.04,     # 4%
        "min": -0.02,    # -2% — declining industries
        "max": 0.20,     # 20% — high-growth emerging industries
    },
    "industry_beta": {
        "mean": 1.0,     # Market beta
        "std": 0.35,     # 0.35 dispersion
        "min": 0.3,      # Defensive sectors
        "max": 2.5,      # Cyclical/volatile sectors
    },
    "company_specific_risk_premium": {
        "mean": 0.015,   # 150 bps — median company-specific risk
        "std": 0.015,    # 150 bps
        "min": 0.0,      # Exceptional quality
        "max": 0.05,     # 500 bps — distressed
    },
}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class AssumptionValidator:
    """Validate agent-generated assumptions before they enter the DCF engine.

    Parameters
    ----------
    historical_financials : dict | None
        Company-specific historical data. Keys should match parameter names
        with '_historical' suffix and contain a list of annual values.
        Example: ``{"revenue_growth_historical": [0.05, 0.07, 0.06, 0.08, 0.09]}``
    industry_benchmarks : dict | None
        Industry median values for each parameter.
        Example: ``{"revenue_growth": 0.06, "fcf_margin": 0.18}``
    """

    def __init__(
        self,
        historical_financials: dict | None = None,
        industry_benchmarks: dict | None = None,
    ) -> None:
        self.historical = historical_financials or {}
        self.industry = industry_benchmarks or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        parameter: str,
        agent_value: float,
        historical_key: str | None = None,
        industry_key: str | None = None,
    ) -> ValidationResult:
        """Validate a single assumption.

        Parameters
        ----------
        parameter : str
            Name of the parameter (e.g. "revenue_growth", "fcf_margin").
        agent_value : float
            The value generated by the LLM agent.
        historical_key : str | None
            Key in ``historical_financials`` for company-specific data.
            Defaults to ``"{parameter}_historical"``.
        industry_key : str | None
            Key in ``industry_benchmarks``. Defaults to ``parameter``.

        Returns
        -------
        ValidationResult
        """
        hk = historical_key or f"{parameter}_historical"
        ik = industry_key or parameter

        hist_data = self.historical.get(hk, [])
        ind_median = self.industry.get(ik, None)

        ref = _DEFAULT_REFERENCE_RANGES.get(parameter, {})

        # Compute historical stats
        if hist_data and len(hist_data) >= 3:
            import numpy as np
            hist_mean = float(np.mean(hist_data))
            hist_std = float(np.std(hist_data, ddof=1))
            if hist_std < 1e-9:
                hist_std = abs(hist_mean) * 0.1 or ref.get("std", 0.01)
        else:
            hist_mean = ref.get("mean", 0.0)
            hist_std = ref.get("std", 0.01)

        if ind_median is None:
            ind_median = ref.get("mean", hist_mean)

        # Compute z-score relative to historical distribution
        if hist_std > 0:
            z_score = (agent_value - hist_mean) / hist_std
        else:
            z_score = 0.0

        # Determine band
        abs_z = abs(z_score)
        if abs_z <= 1.0:
            band = ValidationBand.GREEN
        elif abs_z <= 2.0:
            band = ValidationBand.AMBER
        else:
            band = ValidationBand.RED

        # Industry check — if >2× industry median, escalate band
        if ind_median and ind_median > 0:
            ratio = agent_value / ind_median
            if ratio > 2.0 and band == ValidationBand.GREEN:
                band = ValidationBand.AMBER
            elif ratio > 3.0 and band == ValidationBand.AMBER:
                band = ValidationBand.RED

        # Cap RED values at 2σ from historical mean
        if band == ValidationBand.RED:
            capped_value = hist_mean + (2.0 * hist_std * (1 if z_score > 0 else -1))
            # Ensure caps respect absolute sanity bounds
            capped_value = max(ref.get("min", capped_value), min(ref.get("max", capped_value), capped_value))
        else:
            capped_value = agent_value

        # Build message
        if band == ValidationBand.GREEN:
            message = (
                f"{parameter}={agent_value:.4f} is within 1σ of historical mean "
                f"({hist_mean:.4f} ± {hist_std:.4f}). Passes cleanly."
            )
        elif band == ValidationBand.AMBER:
            message = (
                f"{parameter}={agent_value:.4f} is {abs_z:.1f}σ from historical mean "
                f"({hist_mean:.4f}). Industry median: {ind_median:.4f}. "
                f"Requires justification in evidence chain."
            )
        else:
            message = (
                f"{parameter}={agent_value:.4f} is {abs_z:.1f}σ from historical mean — "
                f"CAPPED at {capped_value:.4f} (2σ bound). "
                f"Original value exceeds statistical plausibility range. "
                f"Industry median: {ind_median:.4f}."
            )

        return ValidationResult(
            parameter=parameter,
            agent_value=agent_value,
            capped_value=capped_value,
            band=band,
            historical_mean=hist_mean,
            historical_std=hist_std,
            industry_median=ind_median,
            z_score=z_score,
            message=message,
        )

    def validate_all(self, assumptions: dict[str, float]) -> ValidationReport:
        """Validate a batch of assumptions and return a report.

        Parameters
        ----------
        assumptions : dict[str, float]
            Mapping of parameter_name → agent_value.
            Example:
            ``{"revenue_growth": 0.12, "fcf_margin": 0.25,
            "country_risk_premium": 0.03}``

        Returns
        -------
        ValidationReport
        """
        report = ValidationReport()

        for param, value in assumptions.items():
            result = self.validate(param, value)
            report.results.append(result)

            if result.band == ValidationBand.GREEN:
                report.green_count += 1
            elif result.band == ValidationBand.AMBER:
                report.amber_count += 1
            else:
                report.red_count += 1

        # Overall band: worst case wins
        if report.red_count > 0:
            report.overall_band = ValidationBand.RED
        elif report.amber_count > 0:
            report.overall_band = ValidationBand.AMBER

        return report

    def get_capped_assumptions(self, assumptions: dict[str, float]) -> dict[str, float]:
        """Return assumptions with RED values capped at 2σ.

        This is the primary integration point — call this before passing
        assumptions to the DCF engine.

        Parameters
        ----------
        assumptions : dict[str, float]
            Raw agent-generated assumptions.

        Returns
        -------
        dict[str, float]
            Assumptions with RED values capped.
        """
        report = self.validate_all(assumptions)
        capped = {}
        for result in report.results:
            capped[result.parameter] = result.capped_value
        return capped

    def confidence_penalty(self, report: ValidationReport) -> float:
        """Compute confidence penalty (0-1) from validation flags.

        Each RED flag costs 0.03-0.04 (3-4 pts on 0-100 scale).
        Each AMBER flag costs 0.01-0.02 (1-2 pts).
        Maximum penalty is capped at 0.10 (10 pts).

        Returns a value between 0.0 (no penalty) and 0.10 (max penalty).
        """
        penalty = report.red_count * 0.035 + report.amber_count * 0.015
        return min(penalty, 0.10)
