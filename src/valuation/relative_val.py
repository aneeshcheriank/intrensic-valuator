"""
Relative valuation via comparable company analysis.

Computes fair value using peer group multiples:
  - P/E (Price-to-Earnings)
  - EV/EBITDA
  - P/B (Price-to-Book)

The median of peer multiples is applied to the target company's metrics
to derive an implied fair value.  This serves as a sanity check on the
DCF valuation (which is sensitive to long-term assumptions).
"""

from __future__ import annotations

import statistics

import numpy as np
import pandas as pd


class RelativeValuation:
    """Comparable company analysis.

    Parameters
    ----------
    company_metrics : dict
        Metrics for the target company. Must contain:
        - ``eps`` (float) — trailing twelve months EPS
        - ``ebitda`` (float, $M)
        - ``book_value_per_share`` (float)
        - ``market_cap`` (float, $M)
        - ``total_debt`` (float, $M)
        - ``cash_and_equivalents`` (float, $M)
    peer_metrics : list[dict]
        List of peer metric dicts. Each must contain:
        - ``ticker`` (str)
        - ``pe_ratio`` (float)
        - ``ev_ebitda`` (float)
        - ``pb_ratio`` (float)
        - ``market_cap`` (float, $M)
    """

    def __init__(
        self,
        company_metrics: dict,
        peer_metrics: list[dict],
    ) -> None:
        self.company = company_metrics
        self.peers = peer_metrics

    # ------------------------------------------------------------------
    # Peer median multiples
    # ------------------------------------------------------------------

    def _peer_median(self, key: str) -> float:
        """Return the median of *key* across all peers."""
        values = [p[key] for p in self.peers if p.get(key) is not None and not np.isnan(p.get(key, float("nan")))]
        if not values:
            return float("nan")
        return statistics.median(values)

    # ------------------------------------------------------------------
    # Implied fair values
    # ------------------------------------------------------------------

    def calculate_fair_value_pe(self) -> float:
        """Fair Price = Median(Peer P/E) × Company TTM EPS."""
        peer_pe = self._peer_median("pe_ratio")
        eps = self.company.get("eps", 0.0)
        if eps <= 0:
            return float("nan")
        return peer_pe * eps

    def calculate_fair_value_ev_ebitda(self) -> float:
        """Fair EV = Median(Peer EV/EBITDA) × Company EBITDA.

        Then convert to per-share: (Fair EV − Debt + Cash) / Shares.
        """
        peer_multiple = self._peer_median("ev_ebitda")
        ebitda = self.company.get("ebitda", 0.0)
        if ebitda <= 0:
            return float("nan")

        fair_ev = peer_multiple * ebitda
        equity_value = (
            fair_ev
            - self.company.get("total_debt", 0.0)
            + self.company.get("cash_and_equivalents", 0.0)
        )
        shares = self.company.get("shares_outstanding", 1.0)
        if shares <= 0:
            return float("nan")
        return equity_value / shares

    def calculate_fair_value_pb(self) -> float:
        """Fair Price = Median(Peer P/B) × Company Book Value Per Share."""
        peer_pb = self._peer_median("pb_ratio")
        bvps = self.company.get("book_value_per_share", 0.0)
        if bvps <= 0:
            return float("nan")
        return peer_pb * bvps

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute all three relative valuation methods.

        Returns a dict with each method's result and a blended average.
        P/B is excluded if it's an outlier (>2x the average of the other
        two methods, common for asset-light companies).
        """
        pe_value = self.calculate_fair_value_pe()
        ev_ebitda_value = self.calculate_fair_value_ev_ebitda()
        pb_value = self.calculate_fair_value_pb()

        values = []
        labels = []
        if not np.isnan(pe_value):
            values.append(pe_value)
            labels.append("P/E")
        if not np.isnan(ev_ebitda_value):
            values.append(ev_ebitda_value)
            labels.append("EV/EBITDA")

        # Include P/B only if it's not an extreme outlier
        if not np.isnan(pb_value) and values:
            avg_other = statistics.mean(values)
            if avg_other > 0 and abs(pb_value - avg_other) / avg_other < 1.0:
                values.append(pb_value)
                labels.append("P/B")
        elif not np.isnan(pb_value) and not values:
            values.append(pb_value)
            labels.append("P/B")

        blended = statistics.mean(values) if values else float("nan")

        return {
            "pe_implied_value": pe_value,
            "ev_ebitda_implied_value": ev_ebitda_value,
            "pb_implied_value": pb_value,
            "blended_relative_value": blended,
            "methods_used": labels,
            "peer_median_pe": self._peer_median("pe_ratio"),
            "peer_median_ev_ebitda": self._peer_median("ev_ebitda"),
            "peer_median_pb": self._peer_median("pb_ratio"),
            "num_peers": len(self.peers),
        }
