"""
Discounted Cash Flow valuation engine.

Calculates intrinsic value per share from projected free cash flows
using the standard DCF methodology.

All formulas follow the CFA Institute / Damodaran approach:
  FCF = Operating Cash Flow − Capital Expenditure
  EV  = Σ FCF_t / (1+WACC)^t + TV / (1+WACC)^n
  IV  = (EV − Debt + Cash) / Shares Outstanding
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DCFModel:
    """Core DCF valuation engine.

    Parameters
    ----------
    financials : dict
        Must contain keys:
        - ``operating_cash_flow`` (float, $M)
        - ``capital_expenditure`` (float, $M)
        - ``total_debt`` (float, $M)
        - ``cash_and_equivalents`` (float, $M)
        - ``shares_outstanding`` (float, millions)
        - ``revenue`` (float, $M) — trailing twelve months
    projection_years : int
        Number of explicit projection years (default 5).
    """

    def __init__(
        self,
        financials: dict,
        projection_years: int = 5,
    ) -> None:
        self.financials = financials
        self.projection_years = projection_years

        # Derived base values
        self.base_fcf: float = self.calculate_fcf()
        self.base_revenue: float = financials.get("revenue", 0.0)
        self.base_fcf_margin: float = (
            self.base_fcf / self.base_revenue if self.base_revenue > 0 else 0.0
        )

    # ------------------------------------------------------------------
    # Step 1: Base FCF
    # ------------------------------------------------------------------

    def calculate_fcf(self) -> float:
        """Return base Free Cash Flow = OCF − CapEx."""
        ocf = self.financials.get("operating_cash_flow", 0.0)
        capex = abs(self.financials.get("capital_expenditure", 0.0))
        return ocf - capex

    # ------------------------------------------------------------------
    # Step 2: Project future FCFs
    # ------------------------------------------------------------------

    def project_fcf(
        self,
        revenue_growth_rates: list[float] | None = None,
        fcf_margins: list[float] | None = None,
    ) -> np.ndarray:
        """Project FCF for each explicit year.

        Parameters
        ----------
        revenue_growth_rates : list[float]
            Growth rate per year (as decimals, e.g. 0.08 = 8%).
            Length must equal ``projection_years``.
        fcf_margins : list[float], optional
            FCF margin per year. If None, the base margin is held constant.

        Returns
        -------
        np.ndarray
            Projected FCF per year, length ``projection_years``.
        """
        if revenue_growth_rates is None:
            revenue_growth_rates = [0.03] * self.projection_years
        if fcf_margins is None:
            fcf_margins = [self.base_fcf_margin] * self.projection_years

        if len(revenue_growth_rates) != self.projection_years:
            raise ValueError(
                f"revenue_growth_rates must have length {self.projection_years}"
            )
        if len(fcf_margins) != self.projection_years:
            raise ValueError(
                f"fcf_margins must have length {self.projection_years}"
            )

        revenues = np.zeros(self.projection_years)
        fcfs = np.zeros(self.projection_years)

        revenues[0] = self.base_revenue * (1.0 + revenue_growth_rates[0])
        fcfs[0] = revenues[0] * fcf_margins[0]

        for t in range(1, self.projection_years):
            revenues[t] = revenues[t - 1] * (1.0 + revenue_growth_rates[t])
            fcfs[t] = revenues[t] * fcf_margins[t]

        return fcfs

    # ------------------------------------------------------------------
    # Step 3: Terminal Value
    # ------------------------------------------------------------------

    def calculate_terminal_value(
        self,
        final_fcf: float,
        wacc: float,
        terminal_growth: float = 0.025,
    ) -> float:
        """Calculate terminal value using the perpetuity growth method.

        TV = FCF_final × (1 + g) / (WACC − g)

        If WACC <= terminal_growth, the formula is undefined; we return NaN
        and the caller should use an alternative method.
        """
        if wacc <= terminal_growth:
            return float("nan")
        return final_fcf * (1.0 + terminal_growth) / (wacc - terminal_growth)

    def calculate_terminal_value_exit_multiple(
        self,
        final_year_revenue: float,
        final_year_fcf_margin: float,
        ev_revenue_multiple: float = 3.0,
    ) -> float:
        """Alternative terminal value using exit multiple.

        TV = Final_Year_Revenue × EV/Revenue_Multiple
        """
        return final_year_revenue * ev_revenue_multiple

    # ------------------------------------------------------------------
    # Step 4: Present Value
    # ------------------------------------------------------------------

    def discount_cashflows(
        self,
        fcf_array: np.ndarray,
        terminal_value: float,
        wacc: float,
    ) -> float:
        """Discount projected FCFs and terminal value to present.

        Returns enterprise value.
        """
        pv_fcfs = 0.0
        for t in range(len(fcf_array)):
            pv_fcfs += fcf_array[t] / ((1.0 + wacc) ** (t + 1))

        pv_tv = terminal_value / ((1.0 + wacc) ** len(fcf_array))

        return pv_fcfs + pv_tv

    # ------------------------------------------------------------------
    # Step 5: Equity Value → Intrinsic Value Per Share
    # ------------------------------------------------------------------

    def calculate_equity_value(self, enterprise_value: float) -> float:
        """Equity Value = EV − Debt + Cash."""
        debt = self.financials.get("total_debt", 0.0)
        cash = self.financials.get("cash_and_equivalents", 0.0)
        minority_interest = self.financials.get("minority_interest", 0.0)
        return enterprise_value - debt + cash - minority_interest

    def calculate_intrinsic_value_per_share(self, equity_value: float) -> float:
        """Intrinsic value per share = Equity Value / Diluted Shares."""
        shares = self.financials.get("shares_outstanding", 1.0)
        if shares <= 0:
            return float("nan")
        return equity_value / shares

    # ------------------------------------------------------------------
    # Full DCF pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        revenue_growth_rates: list[float] | None = None,
        fcf_margins: list[float] | None = None,
        wacc: float = 0.10,
        terminal_growth: float = 0.025,
    ) -> dict:
        """Execute the full DCF valuation pipeline.

        Returns a dict with all intermediate results for transparency.
        """
        if revenue_growth_rates is None:
            revenue_growth_rates = [0.05] * self.projection_years
        if fcf_margins is None:
            fcf_margins = [self.base_fcf_margin] * self.projection_years

        # Project
        fcf_array = self.project_fcf(revenue_growth_rates, fcf_margins)

        # Terminal value
        final_fcf = fcf_array[-1]
        tv_perpetuity = self.calculate_terminal_value(final_fcf, wacc, terminal_growth)

        # Alternative: exit multiple
        final_year_revenue = self.base_revenue
        for i in range(self.projection_years):
            final_year_revenue *= (1.0 + revenue_growth_rates[i])
        tv_exit = self.calculate_terminal_value_exit_multiple(
            final_year_revenue, fcf_margins[-1]
        )

        # Blend terminal values (70% perpetuity, 30% exit)
        if np.isnan(tv_perpetuity):
            terminal_value = tv_exit
            tv_method = "exit_multiple"
        else:
            terminal_value = 0.7 * tv_perpetuity + 0.3 * tv_exit
            tv_method = "blended"

        # Discount
        enterprise_value = self.discount_cashflows(fcf_array, terminal_value, wacc)

        # Equity
        equity_value = self.calculate_equity_value(enterprise_value)
        intrinsic_value = self.calculate_intrinsic_value_per_share(equity_value)

        # Terminal value % of EV
        pv_tv = terminal_value / ((1.0 + wacc) ** self.projection_years)
        tv_pct = (pv_tv / enterprise_value * 100) if enterprise_value > 0 else 0.0

        return {
            "base_fcf": self.base_fcf,
            "base_revenue": self.base_revenue,
            "base_fcf_margin": self.base_fcf_margin,
            "projected_fcfs": fcf_array.tolist(),
            "terminal_value_perpetuity": tv_perpetuity,
            "terminal_value_exit": tv_exit,
            "terminal_value": terminal_value,
            "terminal_value_method": tv_method,
            "terminal_value_pct_of_ev": tv_pct,
            "enterprise_value": enterprise_value,
            "equity_value": equity_value,
            "intrinsic_value_per_share": intrinsic_value,
            "wacc": wacc,
            "terminal_growth": terminal_growth,
        }


# ------------------------------------------------------------------
# Growth rate decay helper
# ------------------------------------------------------------------


def decay_growth_rates(
    company_growth: float,
    industry_growth: float,
    terminal_growth: float,
    projection_years: int = 5,
) -> list[float]:
    """Generate growth rates using a continuous 3-knot linear spline.

    Knots:
      Year 0  → Company_Growth   (firm-specific starting point)
      Year 3  → Industry_Growth  (convergence to sector norm)
      Year 5+ → Terminal_Growth  (long-run GDP growth)

    Years 1-3: Company_Growth → Industry_Growth (linear interpolation)
    Years 4-5: Industry_Growth → Terminal_Growth (linear decay toward perpetuity)

    At year 3, the rate EQUALS industry_growth — a mathematically seamless
    transition that preserves the top-down 3-layer anchoring.

    This ensures projections are grounded: no company can grow above its
    industry forever, and every industry converges to macro fundamentals.
    """
    rates = []
    # First segment: Company → Industry (years 1-3)
    segment1_end = min(3, projection_years)
    for year in range(1, segment1_end + 1):
        frac = year / segment1_end  # 1/3, 2/3, 3/3=1.0
        rate = company_growth + frac * (industry_growth - company_growth)
        rates.append(rate)

    # Second segment: Industry → Terminal (years 4+)
    remaining = projection_years - segment1_end
    if remaining > 0:
        for step in range(1, remaining + 1):
            frac = step / (remaining + 1)  # gradual decay, doesn't fully reach terminal
            rate = industry_growth + frac * (terminal_growth - industry_growth)
            rates.append(rate)

    return rates
