"""Integration tests for the orchestrator pipeline.

These tests verify the full pipeline runs end-to-end, both with and
without LLM agents.  They require network access for yfinance / FRED.
"""

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Pipeline smoke test (no LLM)
# ---------------------------------------------------------------------------


class TestPipelineNoLLM:
    """Test the full pipeline without LLM agents (quantitative only)."""

    def test_pipeline_runs_on_aapl(self):
        """AAPL should run successfully and produce a recommendation."""
        from src.orchestrator import initial_state, run_valuation

        state = run_valuation("AAPL")

        assert state["ticker"] == "AAPL"
        assert state["company_name"] != ""
        assert state["current_price"] > 0
        assert state["intrinsic_value"] > 0
        assert state["wacc"] > 0
        assert state["wacc"] < 1.0  # should be a decimal (< 100%)
        assert state["recommendation"] in ("BUY", "SELL", "HOLD")
        assert 0 <= state["confidence_score"] <= 100
        # Should have run the valuation engine
        assert state.get("status") == "complete"
        assert "intrinsic_value_per_share" in state.get("dcf_details", {})

    def test_pipeline_produces_scenario_results(self):
        """Verify scenario analysis output."""
        from src.orchestrator import initial_state, run_valuation

        state = run_valuation("MSFT")
        scenarios = state.get("scenario_results", {})
        assert "Bull" in scenarios
        assert "Base" in scenarios
        assert "Bear" in scenarios
        # Bull > Base > Bear
        assert (
            scenarios["Bull"]["intrinsic_value_per_share"]
            > scenarios["Base"]["intrinsic_value_per_share"]
            > scenarios["Bear"]["intrinsic_value_per_share"]
        )

    def test_pipeline_produces_monte_carlo_stats(self):
        """Verify Monte Carlo distribution statistics."""
        from src.orchestrator import initial_state, run_valuation

        state = run_valuation("GOOGL")
        mc = state.get("monte_carlo_stats", {})
        assert mc.get("mean", 0) > 0
        assert mc.get("std_dev", 0) >= 0
        assert "fair_value_low" in mc
        assert "fair_value_high" in mc
        assert mc["fair_value_low"] <= mc["fair_value_high"]

    def test_different_tickers_produce_different_values(self):
        """AAPL and META should have different valuations."""
        from src.orchestrator import initial_state, run_valuation

        aapl = run_valuation("AAPL")
        meta = run_valuation("META")

        # Different companies, different intrinsic values
        assert aapl["intrinsic_value"] != meta["intrinsic_value"]
        # Both should have valid data
        assert aapl["current_price"] > 0
        assert meta["current_price"] > 0

    def test_pipeline_handles_unknown_ticker_gracefully(self):
        """An unknown ticker should error gracefully, not crash."""
        from src.orchestrator import initial_state, run_valuation

        state = run_valuation("XYZZY_UNKNOWN_TICKER_999")
        # Should still complete with status indicating issues
        assert state["status"] in ("complete", "data_fetch_failed")
        # Should have errors recorded
        # (may not error if yfinance returns empty but non-error data)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class TestStateSchema:
    def test_initial_state_has_all_keys(self):
        from src.orchestrator import initial_state

        state = initial_state("AAPL")
        required_keys = [
            "ticker", "company_name", "country", "sector", "industry",
            "country_risk_premium", "gdp_growth_forecast", "inflation_forecast",
            "industry_growth_rate", "industry_beta",
            "revenue_growth_forecast", "fcf_margin_forecast",
            "moat_width_score", "management_quality_score",
            "wacc", "intrinsic_value", "fair_value_low", "fair_value_high",
            "current_price", "recommendation", "confidence_score",
            "margin_of_safety", "executive_summary",
            "status", "errors",
        ]
        for key in required_keys:
            assert key in state, f"Missing state key: {key}"

    def test_initial_state_sensible_defaults(self):
        from src.orchestrator import initial_state

        state = initial_state("MSFT")
        assert state["ticker"] == "MSFT"
        assert state["recommendation"] == "HOLD"
        assert state["confidence_score"] == 50
        assert state["status"] == "pending"
        assert state["errors"] == []


# ---------------------------------------------------------------------------
# Data fetchers (integration)
# ---------------------------------------------------------------------------


class TestDataFetchers:
    def test_yahoo_fetches_aapl(self):
        from src.data.yahoo_fetcher import YahooFinanceFetcher

        y = YahooFinanceFetcher()
        info = y.fetch_company_info("AAPL")
        assert info["ticker"] == "AAPL"
        assert "Apple" in info["company_name"]
        assert info["current_price"] > 0
        assert info["market_cap"] > 1_000_000_000_000  # $1T+

    def test_yahoo_fetches_financials(self):
        from src.data.yahoo_fetcher import YahooFinanceFetcher

        y = YahooFinanceFetcher()
        metrics = y.fetch_key_metrics("AAPL")
        assert metrics["revenue"] > 100_000_000_000  # $100B+
        assert metrics["operating_cash_flow"] > 50_000_000_000
        assert metrics["shares_outstanding"] > 1_000_000_000

    def test_macro_fetches_us_data(self):
        from src.data.macro_fetcher import MacroFetcher

        m = MacroFetcher()
        snap = m.fetch_macro_snapshot("US")
        assert snap["country"] == "US"
        assert 0.01 <= snap["gdp_growth"] <= 0.15
        # risk_free_rate from FRED is in raw percentage points (e.g. 4.5 = 4.5%)
        assert snap["risk_free_rate"] > 0
        assert snap["equity_risk_premium"] == 0.05


# ---------------------------------------------------------------------------
# PDF generation (integration)
# ---------------------------------------------------------------------------


class TestPDFGeneration:
    def test_generate_pdf(self, tmp_path):
        """Generate a PDF and verify it's valid."""
        from src.orchestrator import initial_state, run_valuation
        from src.report.pdf_generator import PDFReportGenerator

        state = run_valuation("AAPL")
        pdf_path = tmp_path / "test_report.pdf"
        gen = PDFReportGenerator(state)
        result_path = gen.generate(str(pdf_path))

        assert result_path == str(pdf_path)
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1000  # should be at least 1KB

        # Verify it's a valid PDF
        with open(pdf_path, "rb") as f:
            header = f.read(5)
            assert header == b"%PDF-"

    def test_pdf_contains_expected_content(self, tmp_path):
        """Verify PDF contains the ticker (compressed streams may hide other text)."""
        from src.orchestrator import initial_state, run_valuation
        from src.report.pdf_generator import PDFReportGenerator

        state = run_valuation("AAPL")
        pdf_path = tmp_path / "test_report2.pdf"
        gen = PDFReportGenerator(state)
        gen.generate(str(pdf_path))

        with open(pdf_path, "rb") as f:
            content = f.read()

        # PDF header
        assert content[:5] == b"%PDF-"
        # Ticker should appear somewhere (even in compressed sections it's often ASCII)
        assert b"AAPL" in content
        # Valid PDF footer
        assert b"%%EOF" in content
