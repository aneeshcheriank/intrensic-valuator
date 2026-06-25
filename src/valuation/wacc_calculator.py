"""
Weighted Average Cost of Capital (WACC) calculator.

This is THE integration point where all three layers of the top-down
analysis converge into a single discount rate.

Country  → Risk-Free Rate, Country Risk Premium
Industry → Beta (unlevered, relevered to company D/E)
Company  → Size Premium, Company-Specific Risk Premium

All formulas follow the standard CAPM / Hamada / synthetic-rating approach.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Synthetic credit rating from interest coverage ratio
# ---------------------------------------------------------------------------

# Mapping: (min_icr, spread_bps, rating_label)
_CREDIT_SPREAD_TABLE: list[tuple[float, float, str]] = [
    (8.5, 0.70, "AAA"),
    (6.5, 1.00, "AA"),
    (5.5, 1.30, "A"),
    (4.25, 2.00, "BBB"),
    (3.0, 3.50, "BB"),
    (2.5, 5.00, "B"),
    (2.0, 7.00, "CCC"),
    (1.5, 9.00, "CC"),
    (0.0, 12.00, "D"),
]


def estimate_credit_spread(
    interest_coverage_ratio: float,
    operating_margin: float | None = None,
) -> tuple[float, str]:
    """Estimate a synthetic credit spread from the interest coverage ratio.

    Interest Coverage Ratio = EBIT / Interest Expense

    Returns (spread_percent, rating_label).
      Example: (0.013, "A") means a 1.3% spread over the risk-free rate.
    """
    if interest_coverage_ratio <= 0:
        return (0.12, "D")  # 1200 bps — distressed

    for min_icr, spread, rating in _CREDIT_SPREAD_TABLE:
        if interest_coverage_ratio >= min_icr:
            return (spread / 100.0, rating)

    return (0.12, "D")


# ---------------------------------------------------------------------------
# Size premium lookup
# ---------------------------------------------------------------------------


def estimate_size_premium(market_cap_millions: float) -> float:
    """Return a size premium as a decimal based on market cap.

    Mid/Large cap ($2B+)  → 0%
    Small cap ($250M-$2B)  → 1-3%
    Micro cap (<$250M)     → 3-6%
    """
    if market_cap_millions >= 200_000:
        return 0.0
    elif market_cap_millions >= 10_000:
        return 0.01
    elif market_cap_millions >= 2_000:
        return 0.02
    elif market_cap_millions >= 250:
        return 0.03
    else:
        return 0.05


# ---------------------------------------------------------------------------
# Hamada formula for levering/unlevering beta
# ---------------------------------------------------------------------------


def unlever_beta(levered_beta: float, debt: float, equity: float, tax_rate: float) -> float:
    """Unlever a beta using Hamada's formula.

    β_unlevered = β_levered / (1 + (1 - tax) × D/E)
    """
    if equity <= 0:
        return levered_beta
    de_ratio = debt / equity
    return levered_beta / (1.0 + (1.0 - tax_rate) * de_ratio)


def relever_beta(unlevered_beta: float, debt: float, equity: float, tax_rate: float) -> float:
    """Re-lever a beta using Hamada's formula.

    β_levered = β_unlevered × (1 + (1 - tax) × D/E)
    """
    if equity <= 0:
        return unlevered_beta
    de_ratio = debt / equity
    return unlevered_beta * (1.0 + (1.0 - tax_rate) * de_ratio)


# ---------------------------------------------------------------------------
# WACC Calculator
# ---------------------------------------------------------------------------


class WACCCalculator:
    """Compute WACC from component inputs.

    Typical usage::

        wacc_calc = WACCCalculator()
        result = wacc_calc.calculate(
            risk_free_rate=0.042,
            equity_risk_premium=0.05,
            beta=1.15,
            market_cap=3_100_000,    # $M
            total_debt=110_000,       # $M
            interest_expense=4_000,   # $M
            ebit=125_000,             # $M
            tax_rate=0.21,
            country_risk_premium=0.0,
            company_specific_premium=0.0,
        )
    """

    def calculate_cost_of_equity(
        self,
        risk_free_rate: float,
        equity_risk_premium: float,
        beta: float,
        country_risk_premium: float = 0.0,
        size_premium: float = 0.0,
        company_specific_premium: float = 0.0,
    ) -> float:
        """CAPM with country risk and size premiums.

        Re = Rf + β × ERP + CRP + Size_Premium + Company_Specific_Premium
        """
        return (
            risk_free_rate
            + beta * equity_risk_premium
            + country_risk_premium
            + size_premium
            + company_specific_premium
        )

    def calculate_cost_of_debt(
        self,
        risk_free_rate: float,
        credit_spread: float,
        tax_rate: float,
    ) -> float:
        """After-tax cost of debt.

        Rd = (Rf + Credit_Spread) × (1 − Tax_Rate)
        """
        pre_tax = risk_free_rate + credit_spread
        return pre_tax * (1.0 - tax_rate)

    def calculate_wacc(
        self,
        cost_of_equity: float,
        cost_of_debt: float,
        equity_value: float,
        debt_value: float,
    ) -> float:
        """Weighted average cost of capital.

        WACC = (E/V) × Re + (D/V) × Rd

        If both E and D are zero, returns cost_of_equity as a fallback.
        """
        total_value = equity_value + debt_value
        if total_value <= 0:
            return cost_of_equity
        return (equity_value / total_value) * cost_of_equity + (
            debt_value / total_value
        ) * cost_of_debt

    def calculate(
        self,
        risk_free_rate: float,
        equity_risk_premium: float = 0.05,
        beta: float = 1.0,
        market_cap: float = 0.0,
        total_debt: float = 0.0,
        interest_expense: float = 0.0,
        ebit: float = 0.0,
        tax_rate: float = 0.21,
        country_risk_premium: float = 0.0,
        company_specific_premium: float = 0.0,
    ) -> dict:
        """Run the full WACC calculation pipeline.

        Returns a dict with all intermediate values for transparency.
        """
        # Size premium from market cap
        size_premium = estimate_size_premium(market_cap)

        # Cost of equity
        cost_of_equity = self.calculate_cost_of_equity(
            risk_free_rate=risk_free_rate,
            equity_risk_premium=equity_risk_premium,
            beta=beta,
            country_risk_premium=country_risk_premium,
            size_premium=size_premium,
            company_specific_premium=company_specific_premium,
        )

        # Credit spread from synthetic rating
        icr = ebit / interest_expense if interest_expense > 0 else 99.0
        credit_spread, rating = estimate_credit_spread(icr)

        # Cost of debt
        cost_of_debt = self.calculate_cost_of_debt(
            risk_free_rate=risk_free_rate,
            credit_spread=credit_spread,
            tax_rate=tax_rate,
        )

        # Capital structure weights
        equity_value = market_cap
        debt_value = total_debt

        # WACC
        wacc = self.calculate_wacc(
            cost_of_equity=cost_of_equity,
            cost_of_debt=cost_of_debt,
            equity_value=equity_value,
            debt_value=debt_value,
        )

        return {
            "risk_free_rate": risk_free_rate,
            "equity_risk_premium": equity_risk_premium,
            "beta": beta,
            "country_risk_premium": country_risk_premium,
            "size_premium": size_premium,
            "company_specific_premium": company_specific_premium,
            "cost_of_equity": cost_of_equity,
            "interest_coverage_ratio": icr,
            "synthetic_rating": rating,
            "credit_spread": credit_spread,
            "cost_of_debt": cost_of_debt,
            "equity_weight": equity_value / (equity_value + debt_value) if (equity_value + debt_value) > 0 else 1.0,
            "debt_weight": debt_value / (equity_value + debt_value) if (equity_value + debt_value) > 0 else 0.0,
            "wacc": wacc,
            "market_cap": market_cap,
            "total_debt": total_debt,
            "tax_rate": tax_rate,
        }
