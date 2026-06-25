"""
Monte Carlo simulation for DCF valuation uncertainty quantification.

Runs N iterations of the DCF model with randomly sampled inputs to produce
a distribution of intrinsic values.  The distribution's standard deviation
feeds into the confidence score calculation.
"""

from __future__ import annotations

import numpy as np

from src.valuation.dcf_model import DCFModel, decay_growth_rates


class MonteCarloSimulation:
    """Monte Carlo DCF simulator.

    Parameters
    ----------
    dcf_model : DCFModel
        A pre-configured DCF model instance with base financials.
    base_inputs : dict
        Central-tendency inputs:
        - ``wacc`` (float)
        - ``company_growth`` (float)
        - ``industry_growth`` (float)
        - ``terminal_growth`` (float)
        - ``fcf_margin`` (float)
    iterations : int
        Number of simulation runs (default 10,000).
    seed : int | None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        dcf_model: DCFModel,
        base_inputs: dict,
        iterations: int = 10_000,
        seed: int | None = None,
    ) -> None:
        self.dcf = dcf_model
        self.base = base_inputs
        self.iterations = iterations
        self.rng = np.random.default_rng(seed)

        # Results storage (populated by run())
        self.intrinsic_values: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Input sampling
    # ------------------------------------------------------------------

    def _sample_growth(self, base: float, std: float = 0.015) -> float:
        """Sample a growth rate from a normal distribution, clipped to sane bounds."""
        val = self.rng.normal(base, std)
        return float(np.clip(val, 0.0, 0.40))

    def _sample_margin(self, base: float, std: float = 0.02) -> float:
        """Sample an FCF margin, clipped."""
        val = self.rng.normal(base, std)
        return float(np.clip(val, 0.0, 0.60))

    def _sample_wacc(self, base: float, std: float = 0.01) -> float:
        """Sample a WACC, clipped to positive values."""
        val = self.rng.normal(base, std)
        return float(np.clip(val, 0.01, 0.30))

    def _sample_terminal_growth(self, base: float, std: float = 0.005) -> float:
        """Sample a terminal growth rate."""
        val = self.rng.normal(base, std)
        return float(np.clip(val, 0.005, 0.06))

    # ------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the Monte Carlo simulation.

        Returns a dict with distribution statistics.
        """
        results = np.zeros(self.iterations)
        projection_years = self.dcf.projection_years

        for i in range(self.iterations):
            # Sample inputs
            company_g = self._sample_growth(self.base.get("company_growth", 0.05))
            industry_g = self._sample_growth(self.base.get("industry_growth", 0.04))
            terminal_g = self._sample_terminal_growth(
                self.base.get("terminal_growth", 0.025)
            )
            wacc = self._sample_wacc(self.base.get("wacc", 0.10))
            margin = self._sample_margin(self.base.get("fcf_margin", 0.15))

            # Generate decayed growth rates
            growth_rates = decay_growth_rates(
                company_g, industry_g, terminal_g, projection_years
            )
            margins = [margin] * projection_years

            # Run DCF
            dcf_result = self.dcf.run(
                revenue_growth_rates=growth_rates,
                fcf_margins=margins,
                wacc=wacc,
                terminal_growth=terminal_g,
            )

            results[i] = dcf_result["intrinsic_value_per_share"]

        self.intrinsic_values = results

        return self.statistics()

    # ------------------------------------------------------------------
    # Distribution statistics
    # ------------------------------------------------------------------

    def statistics(self) -> dict:
        """Compute distribution statistics from simulation results."""
        if self.intrinsic_values is None:
            raise RuntimeError("Must call run() before statistics()")

        values = self.intrinsic_values
        finite = values[np.isfinite(values)]

        if len(finite) == 0:
            return {"error": "No finite intrinsic values produced"}

        return {
            "iterations": self.iterations,
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "std_dev": float(np.std(finite)),
            "percentile_5": float(np.percentile(finite, 5)),
            "percentile_10": float(np.percentile(finite, 10)),
            "percentile_25": float(np.percentile(finite, 25)),
            "percentile_75": float(np.percentile(finite, 75)),
            "percentile_90": float(np.percentile(finite, 90)),
            "percentile_95": float(np.percentile(finite, 95)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "skewness": float(_skewness(finite)),
            "fair_value_low": float(np.percentile(finite, 10)),
            "fair_value_high": float(np.percentile(finite, 90)),
        }


class ScenarioAnalysis:
    """Bull / Base / Bear scenario analysis for a DCF model.

    Each scenario adjusts key inputs by predefined offsets.
    """

    SCENARIO_DEFS = {
        "Bull": {
            "growth_mult": 1.20,
            "margin_add": 0.02,  # +200 bps
            "wacc_add": -0.01,  # −100 bps
            "terminal_add": 0.005,  # +50 bps
        },
        "Base": {
            "growth_mult": 1.0,
            "margin_add": 0.0,
            "wacc_add": 0.0,
            "terminal_add": 0.0,
        },
        "Bear": {
            "growth_mult": 0.80,
            "margin_add": -0.02,  # −200 bps
            "wacc_add": 0.015,  # +150 bps
            "terminal_add": -0.005,  # −50 bps
        },
    }

    def __init__(self, dcf_model: DCFModel, base_inputs: dict) -> None:
        self.dcf = dcf_model
        self.base = base_inputs

    def run(self) -> dict:
        """Run all three scenarios and return a comparison."""
        results = {}
        projection_years = self.dcf.projection_years

        for name, adj in self.SCENARIO_DEFS.items():
            company_g = self.base.get("company_growth", 0.05) * adj["growth_mult"]
            industry_g = self.base.get("industry_growth", 0.04)
            terminal_g = self.base.get("terminal_growth", 0.025) + adj["terminal_add"]
            wacc = self.base.get("wacc", 0.10) + adj["wacc_add"]
            margin = self.base.get("fcf_margin", 0.15) + adj["margin_add"]

            margin = max(0.01, min(0.60, margin))
            wacc = max(0.01, min(0.30, wacc))

            growth_rates = decay_growth_rates(
                company_g, industry_g, terminal_g, projection_years
            )
            margins = [margin] * projection_years

            dcf_result = self.dcf.run(
                revenue_growth_rates=growth_rates,
                fcf_margins=margins,
                wacc=wacc,
                terminal_growth=terminal_g,
            )

            results[name] = {
                "intrinsic_value_per_share": dcf_result["intrinsic_value_per_share"],
                "revenue_growth": company_g,
                "fcf_margin": margin,
                "wacc": wacc,
                "terminal_growth": terminal_g,
            }

        return results


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _skewness(x: np.ndarray) -> float:
    """Compute sample skewness."""
    n = len(x)
    if n < 3:
        return 0.0
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * np.sum(((x - mean) / std) ** 3))
