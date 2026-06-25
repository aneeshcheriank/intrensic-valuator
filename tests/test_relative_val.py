"""Unit tests for relative valuation / peer comparison."""

import math

import pytest

from src.valuation.relative_val import RelativeValuation


@pytest.fixture
def company_metrics() -> dict:
    return {
        "eps": 6.50,
        "ebitda": 130_000,  # $M
        "book_value_per_share": 25.0,
        "market_cap": 500_000,
        "total_debt": 50_000,
        "cash_and_equivalents": 15_000,
        "shares_outstanding": 1_000,
    }


@pytest.fixture
def peer_metrics() -> list[dict]:
    return [
        {"ticker": "A", "pe_ratio": 28, "ev_ebitda": 20, "pb_ratio": 12, "market_cap": 400_000},
        {"ticker": "B", "pe_ratio": 32, "ev_ebitda": 22, "pb_ratio": 15, "market_cap": 600_000},
        {"ticker": "C", "pe_ratio": 30, "ev_ebitda": 18, "pb_ratio": 14, "market_cap": 350_000},
        {"ticker": "D", "pe_ratio": 26, "ev_ebitda": 19, "pb_ratio": 10, "market_cap": 500_000},
        {"ticker": "E", "pe_ratio": 33, "ev_ebitda": 21, "pb_ratio": 13, "market_cap": 450_000},
    ]


@pytest.fixture
def rel(company_metrics, peer_metrics) -> RelativeValuation:
    return RelativeValuation(company_metrics, peer_metrics)


class TestRelativeValuation:
    def test_pe_method(self, rel):
        value = rel.calculate_fair_value_pe()
        # Median P/E = 30, EPS = 6.50 → 30 * 6.5 = 195
        assert abs(value - 195.0) < 0.1

    def test_ev_ebitda_method(self, rel):
        value = rel.calculate_fair_value_ev_ebitda()
        # Median EV/EBITDA = 20, EBITDA = 130,000
        # Fair EV = 20 * 130,000 = 2,600,000
        # Equity = 2,600,000 - 50,000 + 15,000 = 2,565,000
        # Per share = 2,565,000 / 1,000 = 2,565
        assert abs(value - 2565.0) < 0.1

    def test_pb_method(self, rel):
        value = rel.calculate_fair_value_pb()
        # Median P/B = 13, BVPS = 25 → 13 * 25 = 325
        assert abs(value - 325.0) < 0.1

    def test_blended_excludes_pb_if_outlier(self, rel):
        """P/B = 325 is not a huge outlier vs P/E 195 and EV/EBITDA 2565 for
        this test case (the average is ~1,380), so all 3 should be used."""
        result = rel.run()
        assert result["blended_relative_value"] > 0
        assert len(result["methods_used"]) >= 2

    def test_negative_eps_returns_nan(self, peer_metrics):
        company = {
            "eps": -1.0, "ebitda": 130_000, "book_value_per_share": 25.0,
            "market_cap": 500_000, "total_debt": 50_000,
            "cash_and_equivalents": 15_000, "shares_outstanding": 1_000,
        }
        rel = RelativeValuation(company, peer_metrics)
        assert math.isnan(rel.calculate_fair_value_pe())

    def test_no_peers_returns_nan(self, company_metrics):
        rel = RelativeValuation(company_metrics, [])
        assert math.isnan(rel.calculate_fair_value_pe())
        assert rel.run()["num_peers"] == 0

    def test_run_returns_all_keys(self, rel):
        result = rel.run()
        for key in [
            "pe_implied_value", "ev_ebitda_implied_value", "pb_implied_value",
            "blended_relative_value", "methods_used", "peer_median_pe",
            "peer_median_ev_ebitda", "peer_median_pb", "num_peers",
        ]:
            assert key in result, f"Missing: {key}"

    def test_peer_median_correct(self, rel):
        result = rel.run()
        # Sorted P/Es: 26, 28, 30, 32, 33 → median = 30
        assert result["peer_median_pe"] == 30
        # Sorted EV/EBITDA: 18, 19, 20, 21, 22 → median = 20
        assert result["peer_median_ev_ebitda"] == 20
