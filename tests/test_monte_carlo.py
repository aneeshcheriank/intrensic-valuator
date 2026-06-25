"""Unit tests for Monte Carlo simulation and scenario analysis."""

import numpy as np
import pytest

from src.valuation.dcf_model import DCFModel
from src.valuation.monte_carlo import MonteCarloSimulation, ScenarioAnalysis


@pytest.fixture
def dcf() -> DCFModel:
    return DCFModel(
        {
            "revenue": 100_000,
            "operating_cash_flow": 25_000,
            "capital_expenditure": -5_000,
            "total_debt": 20_000,
            "cash_and_equivalents": 10_000,
            "shares_outstanding": 1_000,
        },
        projection_years=5,
    )


@pytest.fixture
def base_inputs() -> dict:
    return {
        "company_growth": 0.08,
        "industry_growth": 0.05,
        "terminal_growth": 0.025,
        "wacc": 0.10,
        "fcf_margin": 0.20,
    }


class TestMonteCarlo:
    def test_run_produces_distribution(self, dcf, base_inputs):
        mc = MonteCarloSimulation(dcf, base_inputs, iterations=500, seed=42)
        stats = mc.run()
        assert stats["iterations"] == 500
        assert stats["mean"] > 0
        assert stats["std_dev"] >= 0
        # Percentiles should be ordered
        assert stats["percentile_10"] <= stats["median"] <= stats["percentile_90"]

    def test_reproducibility(self, dcf, base_inputs):
        mc1 = MonteCarloSimulation(dcf, base_inputs, iterations=200, seed=42)
        mc2 = MonteCarloSimulation(dcf, base_inputs, iterations=200, seed=42)
        assert abs(mc1.run()["mean"] - mc2.run()["mean"]) < 0.01

    def test_statistics_keys(self, dcf, base_inputs):
        mc = MonteCarloSimulation(dcf, base_inputs, iterations=100, seed=1)
        stats = mc.run()
        expected = [
            "iterations", "mean", "median", "std_dev",
            "percentile_5", "percentile_10", "percentile_25",
            "percentile_75", "percentile_90", "percentile_95",
            "min", "max", "skewness", "fair_value_low", "fair_value_high",
        ]
        for key in expected:
            assert key in stats, f"Missing: {key}"

    def test_statistics_raises_before_run(self, dcf, base_inputs):
        mc = MonteCarloSimulation(dcf, base_inputs, iterations=100)
        with pytest.raises(RuntimeError, match="Must call run"):
            mc.statistics()

    def test_higher_growth_higher_mean(self, dcf, base_inputs):
        low = MonteCarloSimulation(
            dcf, {**base_inputs, "company_growth": 0.03}, iterations=200, seed=42
        )
        high = MonteCarloSimulation(
            dcf, {**base_inputs, "company_growth": 0.15}, iterations=200, seed=42
        )
        assert high.run()["mean"] > low.run()["mean"]

    def test_higher_wacc_lower_mean(self, dcf, base_inputs):
        low_wacc = MonteCarloSimulation(
            dcf, {**base_inputs, "wacc": 0.06}, iterations=200, seed=42
        )
        high_wacc = MonteCarloSimulation(
            dcf, {**base_inputs, "wacc": 0.18}, iterations=200, seed=42
        )
        assert high_wacc.run()["mean"] < low_wacc.run()["mean"]

    def test_returns_finite_values(self, dcf, base_inputs):
        mc = MonteCarloSimulation(dcf, base_inputs, iterations=200, seed=42)
        stats = mc.run()
        assert np.isfinite(stats["mean"])
        assert np.isfinite(stats["std_dev"])


class TestScenarioAnalysis:
    def test_three_scenarios(self, dcf, base_inputs):
        sa = ScenarioAnalysis(dcf, base_inputs)
        results = sa.run()
        assert "Bull" in results
        assert "Base" in results
        assert "Bear" in results
        for name in ["Bull", "Base", "Bear"]:
            assert "intrinsic_value_per_share" in results[name]

    def test_bull_highest_bear_lowest(self, dcf, base_inputs):
        sa = ScenarioAnalysis(dcf, base_inputs)
        results = sa.run()
        assert (
            results["Bull"]["intrinsic_value_per_share"]
            > results["Base"]["intrinsic_value_per_share"]
            > results["Bear"]["intrinsic_value_per_share"]
        )

    def test_each_scenario_has_required_keys(self, dcf, base_inputs):
        sa = ScenarioAnalysis(dcf, base_inputs)
        results = sa.run()
        for name in ["Bull", "Base", "Bear"]:
            r = results[name]
            for key in ["intrinsic_value_per_share", "revenue_growth", "fcf_margin", "wacc", "terminal_growth"]:
                assert key in r, f"Missing {key} in {name}"
