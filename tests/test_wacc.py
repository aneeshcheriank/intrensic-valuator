"""Unit tests for the WACC calculator."""

import pytest

from src.valuation.wacc_calculator import (
    WACCCalculator,
    estimate_credit_spread,
    estimate_size_premium,
    unlever_beta,
    relever_beta,
)


class TestCreditSpread:
    def test_aaa_rating(self):
        spread, rating = estimate_credit_spread(10.0)
        assert rating == "AAA"
        assert spread == pytest.approx(0.007, abs=1e-6)  # 70 bps

    def test_aa_rating(self):
        spread, rating = estimate_credit_spread(8.0)
        assert rating == "AA"
        assert spread == 0.01

    def test_bb_rating(self):
        spread, rating = estimate_credit_spread(3.5)
        assert rating == "BB"
        assert spread == 0.035

    def test_distressed(self):
        spread, rating = estimate_credit_spread(-1.0)
        assert rating == "D"
        assert spread == 0.12

    def test_zero_icr(self):
        spread, rating = estimate_credit_spread(0.0)
        assert rating == "D"


class TestSizePremium:
    def test_mega_cap(self):
        assert estimate_size_premium(500_000) == 0.0

    def test_large_cap(self):
        assert estimate_size_premium(50_000) == 0.01

    def test_mid_cap(self):
        assert estimate_size_premium(5_000) == 0.02

    def test_small_cap(self):
        assert estimate_size_premium(500) == 0.03

    def test_micro_cap(self):
        assert estimate_size_premium(100) == 0.05


class TestBeta:
    def test_unlever(self):
        ul = unlever_beta(levered_beta=1.5, debt=50, equity=100, tax_rate=0.25)
        # ul = 1.5 / (1 + 0.75 * 0.5) = 1.5 / 1.375 = 1.091
        assert abs(ul - 1.091) < 0.01

    def test_relever(self):
        lev = relever_beta(unlevered_beta=1.091, debt=50, equity=100, tax_rate=0.25)
        assert abs(lev - 1.5) < 0.01

    def test_unlever_zero_debt(self):
        ul = unlever_beta(1.2, 0, 100, 0.21)
        assert ul == 1.2

    def test_relever_zero_equity(self):
        lev = relever_beta(1.0, 50, 0, 0.21)
        assert lev == 1.0  # fallback


class TestWACCCalculator:
    @pytest.fixture
    def calc(self):
        return WACCCalculator()

    def test_cost_of_equity_us_large_cap(self, calc):
        re_ = calc.calculate_cost_of_equity(
            risk_free_rate=0.042,
            equity_risk_premium=0.05,
            beta=1.0,
            country_risk_premium=0.0,
            size_premium=0.0,
        )
        assert abs(re_ - 0.092) < 0.001  # 4.2% + 5% = 9.2%

    def test_cost_of_equity_with_crp(self, calc):
        re_ = calc.calculate_cost_of_equity(
            risk_free_rate=0.042,
            equity_risk_premium=0.05,
            beta=1.15,
            country_risk_premium=0.02,  # 200 bps India
            size_premium=0.02,  # 200 bps small cap
            company_specific_premium=0.005,  # 50 bps
        )
        # Re = 4.2 + 1.15*5.0 + 2.0 + 2.0 + 0.5 = 4.2 + 5.75 + 4.5 = 14.45%
        assert abs(re_ - 0.1445) < 0.001

    def test_cost_of_debt(self, calc):
        rd = calc.calculate_cost_of_debt(
            risk_free_rate=0.042, credit_spread=0.013, tax_rate=0.21
        )
        # (4.2% + 1.3%) * (1 - 0.21) = 5.5% * 0.79 = 4.345%
        assert abs(rd - 0.04345) < 0.001

    def test_wacc_basic(self, calc):
        wacc = calc.calculate_wacc(
            cost_of_equity=0.10, cost_of_debt=0.04, equity_value=500, debt_value=100
        )
        # (500/600)*10% + (100/600)*4% = 8.33% + 0.67% = 9.0%
        assert abs(wacc - 0.09) < 0.001

    def test_wacc_zero_total(self, calc):
        wacc = calc.calculate_wacc(0.10, 0.04, 0, 0)
        assert wacc == 0.10  # fallback

    def test_full_calculate_returns_all_keys(self, calc):
        result = calc.calculate(
            risk_free_rate=0.042,
            equity_risk_premium=0.05,
            beta=1.1,
            market_cap=500_000,
            total_debt=50_000,
            interest_expense=2_000,
            ebit=40_000,
            tax_rate=0.21,
        )
        expected = [
            "risk_free_rate", "equity_risk_premium", "beta",
            "country_risk_premium", "size_premium", "company_specific_premium",
            "cost_of_equity", "interest_coverage_ratio", "synthetic_rating",
            "credit_spread", "cost_of_debt", "equity_weight", "debt_weight",
            "wacc", "market_cap", "total_debt", "tax_rate",
        ]
        for key in expected:
            assert key in result, f"Missing: {key}"

    def test_aaa_rating_for_high_icr(self, calc):
        """Apple-like profile: massive EBIT, low interest."""
        result = calc.calculate(
            risk_free_rate=0.042, beta=1.1,
            market_cap=3_000_000, total_debt=100_000,
            interest_expense=4_000, ebit=130_000,  # ICR = 32.5
            tax_rate=0.16,
        )
        assert result["synthetic_rating"] == "AAA"

    def test_increasing_crp_increases_wacc(self, calc):
        base = calc.calculate(
            risk_free_rate=0.05, beta=1.0, market_cap=100_000, total_debt=20_000,
            interest_expense=1_000, ebit=10_000,
            country_risk_premium=0.0,
        )
        with_crp = calc.calculate(
            risk_free_rate=0.05, beta=1.0, market_cap=100_000, total_debt=20_000,
            interest_expense=1_000, ebit=10_000,
            country_risk_premium=0.03,  # 300 bps
        )
        assert with_crp["wacc"] > base["wacc"]
        assert with_crp["cost_of_equity"] > base["cost_of_equity"]
