"""Unit tests for the DCF valuation model."""

import math

import pytest

from src.valuation.dcf_model import DCFModel, decay_growth_rates


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_financials() -> dict:
    return {
        "revenue": 100_000,  # $M
        "operating_cash_flow": 30_000,
        "capital_expenditure": -8_000,
        "total_debt": 25_000,
        "cash_and_equivalents": 15_000,
        "shares_outstanding": 1_000,  # millions
    }


@pytest.fixture
def dcf(sample_financials) -> DCFModel:
    return DCFModel(sample_financials, projection_years=5)


# ---------------------------------------------------------------------------
# FCF calculation
# ---------------------------------------------------------------------------


class TestFCFCalculation:
    def test_basic_fcf(self, dcf):
        assert dcf.calculate_fcf() == 22_000

    def test_negative_fcf_when_capex_exceeds_ocf(self):
        fin = {
            "revenue": 50_000,
            "operating_cash_flow": 5_000,
            "capital_expenditure": -12_000,
            "total_debt": 10_000,
            "cash_and_equivalents": 5_000,
            "shares_outstanding": 500,
        }
        d = DCFModel(fin)
        assert d.calculate_fcf() == -7_000

    def test_zero_fcf_when_ocf_equals_capex(self):
        fin = {
            "revenue": 50_000,
            "operating_cash_flow": 10_000,
            "capital_expenditure": -10_000,
            "total_debt": 0,
            "cash_and_equivalents": 0,
            "shares_outstanding": 100,
        }
        d = DCFModel(fin)
        assert d.calculate_fcf() == 0


# ---------------------------------------------------------------------------
# FCF projection
# ---------------------------------------------------------------------------


class TestFCFProjection:
    def test_constant_growth(self, dcf):
        rates = [0.05] * 5
        margins = [0.22] * 5
        fcfs = dcf.project_fcf(rates, margins)
        assert len(fcfs) == 5
        # Year 1: 100000 * 1.05 * 0.22 = 23,100
        assert abs(fcfs[0] - 23_100) < 1

    def test_growth_mismatch_length_raises(self, dcf):
        with pytest.raises(ValueError, match="revenue_growth_rates must have length"):
            dcf.project_fcf([0.05, 0.06], [0.22] * 5)

    def test_margin_mismatch_length_raises(self, dcf):
        with pytest.raises(ValueError, match="fcf_margins must have length"):
            dcf.project_fcf([0.05] * 5, [0.22])

    def test_declining_growth_produces_decaying_fcf(self, dcf):
        rates = [0.10, 0.08, 0.06, 0.04, 0.03]
        margins = [0.22] * 5
        fcfs = dcf.project_fcf(rates, margins)
        # FCF should still grow in absolute terms but rate should slow
        assert fcfs[-1] > fcfs[0]


# ---------------------------------------------------------------------------
# Terminal value
# ---------------------------------------------------------------------------


class TestTerminalValue:
    def test_perpetuity_method(self, dcf):
        tv = dcf.calculate_terminal_value(final_fcf=30_000, wacc=0.10, terminal_growth=0.025)
        # TV = 30000 * 1.025 / (0.10 - 0.025) = 30750 / 0.075 = 410,000
        assert abs(tv - 410_000) < 1

    def test_perpetuity_returns_nan_when_wacc_le_growth(self, dcf):
        tv = dcf.calculate_terminal_value(final_fcf=30_000, wacc=0.02, terminal_growth=0.025)
        assert math.isnan(tv)

    def test_exit_multiple_method(self, dcf):
        tv = dcf.calculate_terminal_value_exit_multiple(
            final_year_revenue=120_000, final_year_fcf_margin=0.22, ev_revenue_multiple=3.0
        )
        assert tv == 360_000


# ---------------------------------------------------------------------------
# Discounting
# ---------------------------------------------------------------------------


class TestDiscount:
    def test_present_value(self, dcf):
        fcf_array = [23_100, 24_255, 25_467, 26_741, 28_078]
        tv = 410_000
        wacc = 0.10
        ev = dcf.discount_cashflows(fcf_array, tv, wacc)
        # Should be positive and reasonable
        assert ev > 0
        assert ev > sum(fcf_array)  # TV dominates

    def test_higher_wacc_lower_ev(self, dcf):
        fcf_array = [23_100] * 5
        tv = 300_000
        ev_low = dcf.discount_cashflows(fcf_array, tv, 0.08)
        ev_high = dcf.discount_cashflows(fcf_array, tv, 0.15)
        assert ev_low > ev_high


# ---------------------------------------------------------------------------
# Equity value & intrinsic value
# ---------------------------------------------------------------------------


class TestEquityValue:
    def test_equity_value_formula(self, dcf):
        ev = 350_000
        equity = dcf.calculate_equity_value(ev)
        expected = 350_000 - 25_000 + 15_000  # EV - Debt + Cash
        assert equity == expected

    def test_intrinsic_value_per_share(self, dcf):
        equity = 340_000
        iv = dcf.calculate_intrinsic_value_per_share(equity)
        assert iv == 340.0  # 340,000 / 1,000 shares

    def test_zero_shares_returns_nan(self, dcf):
        dcf.financials["shares_outstanding"] = 0
        assert math.isnan(dcf.calculate_intrinsic_value_per_share(340_000))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_run_returns_all_keys(self, dcf):
        result = dcf.run(wacc=0.10, terminal_growth=0.025)
        expected_keys = [
            "base_fcf", "base_revenue", "base_fcf_margin",
            "projected_fcfs", "terminal_value_perpetuity", "terminal_value_exit",
            "terminal_value", "terminal_value_method", "terminal_value_pct_of_ev",
            "enterprise_value", "equity_value", "intrinsic_value_per_share",
            "wacc", "terminal_growth",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_run_with_custom_growth(self, dcf):
        result = dcf.run(
            revenue_growth_rates=[0.08, 0.07, 0.06, 0.05, 0.04],
            fcf_margins=[0.25] * 5,
            wacc=0.09,
            terminal_growth=0.03,
        )
        assert result["intrinsic_value_per_share"] > 0
        assert result["terminal_value_pct_of_ev"] > 0

    def test_bull_bear_asymmetry(self, dcf):
        bull = dcf.run(
            revenue_growth_rates=[0.10] * 5,
            fcf_margins=[0.30] * 5,
            wacc=0.08,
            terminal_growth=0.03,
        )
        bear = dcf.run(
            revenue_growth_rates=[0.02] * 5,
            fcf_margins=[0.15] * 5,
            wacc=0.14,
            terminal_growth=0.02,
        )
        assert bull["intrinsic_value_per_share"] > bear["intrinsic_value_per_share"]


# ---------------------------------------------------------------------------
# Growth decay helper
# ---------------------------------------------------------------------------


class TestGrowthDecay:
    def test_decays_toward_terminal(self):
        rates = decay_growth_rates(0.12, 0.06, 0.025, 5)
        assert len(rates) == 5
        # Should start near company growth and end near terminal
        assert rates[0] > rates[-1]
        assert rates[-1] >= 0.025

    def test_all_positive(self):
        rates = decay_growth_rates(0.05, 0.04, 0.025, 5)
        assert all(r > 0 for r in rates)

    def test_single_year(self):
        rates = decay_growth_rates(0.10, 0.06, 0.03, 1)
        assert len(rates) == 1
        assert 0.06 <= rates[0] <= 0.10  # in the range
