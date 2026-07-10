"""
Backtesting Framework — Core Infrastructure.

Tracks historical recommendations against actual returns and performs
component-level validation to answer the key questions:

  - Does Country Analysis improve valuation accuracy?
  - Does Industry Analysis improve recommendations?
  - Does Management scoring improve forecasting?
  - Does Monte Carlo improve confidence calibration?
  - Does Relative Valuation reduce DCF error?
  - Does blending DCF + Relative beat pure DCF?

Without these measurements it is impossible to determine whether
additional model complexity creates additional value.

Architecture:
  BacktestStore  — SQLite-backed record of every recommendation
  BacktestRunner — Replay historical valuations and measure accuracy
  ComponentTest — A/B test individual components
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Backtest Record Store
# ---------------------------------------------------------------------------


class BacktestStore:
    """Persistent store for tracking every recommendation against actual returns.

    Uses SQLite (same pattern as DataCache) for durability. Each record
    captures the full valuation state at the time of recommendation plus
    the realized return at each checkpoint (1m, 3m, 6m, 1y).
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "cache" / "backtest.db")
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                company_name TEXT,
                recommendation_date TEXT NOT NULL,
                recommendation TEXT NOT NULL CHECK(recommendation IN ('BUY','SELL','HOLD')),
                confidence_score INTEGER,
                intrinsic_value REAL,
                current_price REAL,
                margin_of_safety REAL,
                wacc REAL,
                recommendation_price REAL,
                validation_flags TEXT,          -- JSON array
                confidence_breakdown TEXT,       -- JSON object
                valuation_state_json TEXT,       -- Full state snapshot
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS return_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recommendation_id INTEGER NOT NULL,
                checkpoint_days INTEGER NOT NULL,  -- 30, 90, 180, 365
                actual_price REAL,
                actual_return REAL,
                excess_return REAL,                -- vs S&P 500
                checked_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (recommendation_id) REFERENCES recommendations(id)
            );

            CREATE TABLE IF NOT EXISTS component_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                component_enabled TEXT NOT NULL,   -- e.g., 'country_agent', 'monte_carlo'
                component_disabled TEXT,
                ticker TEXT NOT NULL,
                intrinsic_value_enabled REAL,
                intrinsic_value_disabled REAL,
                error_reduction REAL,              -- positive = component helps
                test_date TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_rec_ticker ON recommendations(ticker);
            CREATE INDEX IF NOT EXISTS idx_rec_date ON recommendations(recommendation_date);
            CREATE INDEX IF NOT EXISTS idx_cp_rec_id ON return_checkpoints(recommendation_id);
            CREATE INDEX IF NOT EXISTS idx_ct_test ON component_tests(test_name);
        """)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_recommendation(
        self,
        ticker: str,
        state: dict,
    ) -> int:
        """Store a recommendation for future backtesting.

        Returns the record ID for checkpoint updates.
        """
        rec_price = state.get("current_price", 0.0)
        val_flags = json.dumps(state.get("validation_flags", []))
        conf_breakdown = json.dumps(state.get("confidence_breakdown", {}))
        state_json = json.dumps(_serializable_state(state))

        cursor = self.conn.execute(
            """INSERT INTO recommendations
               (ticker, company_name, recommendation_date, recommendation,
                confidence_score, intrinsic_value, current_price,
                margin_of_safety, wacc, recommendation_price,
                validation_flags, confidence_breakdown, valuation_state_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ticker.upper(),
                state.get("company_name", ""),
                datetime.now().strftime("%Y-%m-%d"),
                state.get("recommendation", "HOLD"),
                state.get("confidence_score", 50),
                state.get("intrinsic_value", 0.0),
                state.get("current_price", 0.0),
                state.get("margin_of_safety", 0.0),
                state.get("wacc", 0.10),
                rec_price,
                val_flags,
                conf_breakdown,
                state_json,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def record_checkpoint(
        self,
        recommendation_id: int,
        days: int,
        actual_price: float,
        benchmark_price: float | None = None,
    ) -> None:
        """Record a return checkpoint for a previous recommendation."""
        # Get the original recommendation price
        row = self.conn.execute(
            "SELECT recommendation_price, recommendation FROM recommendations WHERE id = ?",
            (recommendation_id,),
        ).fetchone()

        if not row or row[0] <= 0:
            return

        rec_price = row[0]
        rec = row[1]

        # Calculate actual return (signed by recommendation direction)
        # BUY: +return means correct, SELL: -return means correct
        raw_return = (actual_price - rec_price) / rec_price
        if rec == "SELL":
            actual_return = -raw_return  # Invert for SELL recommendations
        elif rec == "HOLD":
            actual_return = -abs(raw_return)  # HOLD wants minimal movement
        else:
            actual_return = raw_return

        excess_return = actual_return
        if benchmark_price and benchmark_price > 0:
            bench_return = (benchmark_price - rec_price) / rec_price
            excess_return = actual_return - bench_return

        self.conn.execute(
            """INSERT INTO return_checkpoints
               (recommendation_id, checkpoint_days, actual_price, actual_return, excess_return)
               VALUES (?, ?, ?, ?, ?)""",
            (recommendation_id, days, actual_price, actual_return, excess_return),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Read — Aggregate Statistics
    # ------------------------------------------------------------------

    def hit_rate(self, min_confidence: int = 0, min_days: int = 90) -> dict:
        """Calculate recommendation accuracy metrics.

        A 'hit' for BUY = price went up, SELL = price went down,
        HOLD = price stayed within ±5%.
        """
        rows = self.conn.execute(
            """SELECT r.recommendation, r.confidence_score, r.recommendation_price,
                      c.actual_price, c.checkpoint_days
               FROM recommendations r
               JOIN return_checkpoints c ON r.id = c.recommendation_id
               WHERE r.confidence_score >= ? AND c.checkpoint_days >= ?
               AND r.recommendation_price > 0""",
            (min_confidence, min_days),
        ).fetchall()

        if not rows:
            return {"total": 0, "hit_rate": 0.0, "avg_return": 0.0}

        hits = 0
        total_return = 0.0
        for rec, conf, rec_price, actual_price, days in rows:
            pct_change = (actual_price - rec_price) / rec_price
            total_return += pct_change
            if rec == "BUY" and pct_change > 0:
                hits += 1
            elif rec == "SELL" and pct_change < 0:
                hits += 1
            elif rec == "HOLD" and abs(pct_change) < 0.05:
                hits += 1

        return {
            "total": len(rows),
            "hits": hits,
            "hit_rate": hits / len(rows) if rows else 0.0,
            "avg_return": total_return / len(rows) if rows else 0.0,
        }

    def confidence_calibration(self) -> dict:
        """Check if high-confidence recommendations actually perform better."""
        rows = self.conn.execute(
            """SELECT r.confidence_score, c.actual_return
               FROM recommendations r
               JOIN return_checkpoints c ON r.id = c.recommendation_id
               WHERE c.checkpoint_days >= 90 AND c.actual_return IS NOT NULL"""
        ).fetchall()

        if len(rows) < 10:
            return {"status": "insufficient_data", "count": len(rows)}

        # Split into high (>70) and low (<=70) confidence buckets
        high_conf = [r[1] for r in rows if r[0] >= 70]
        low_conf = [r[1] for r in rows if r[0] < 70]

        result = {"count": len(rows)}
        if high_conf:
            result["high_confidence_avg_return"] = float(np.mean(high_conf))
            result["high_confidence_hit_rate"] = float(np.mean([1 if r > 0 else 0 for r in high_conf]))
        if low_conf:
            result["low_confidence_avg_return"] = float(np.mean(low_conf))
            result["low_confidence_hit_rate"] = float(np.mean([1 if r > 0 else 0 for r in low_conf]))
        if high_conf and low_conf:
            result["calibration_delta"] = result.get("high_confidence_avg_return", 0) - result.get("low_confidence_avg_return", 0)

        return result

    def component_attribution(self) -> dict:
        """Measure which analysis layers add the most value."""
        tests = self.conn.execute(
            "SELECT component_enabled, AVG(error_reduction) FROM component_tests GROUP BY component_enabled"
        ).fetchall()

        if not tests:
            return {"status": "no_component_tests_run"}

        return {
            test[0]: round(test[1], 4)
            for test in tests
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Overall backtest store statistics."""
        rec_count = self.conn.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0]
        cp_count = self.conn.execute("SELECT COUNT(*) FROM return_checkpoints").fetchone()[0]
        ct_count = self.conn.execute("SELECT COUNT(*) FROM component_tests").fetchone()[0]

        # Get date range
        first_date = self.conn.execute(
            "SELECT MIN(recommendation_date) FROM recommendations"
        ).fetchone()[0]

        return {
            "total_recommendations": rec_count,
            "total_checkpoints": cp_count,
            "total_component_tests": ct_count,
            "first_recommendation": first_date or "N/A",
            "db_path": self.db_path,
        }

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Backtest Runner
# ---------------------------------------------------------------------------


class BacktestRunner:
    """Replay historical valuations and measure accuracy.

    Usage::

        runner = BacktestRunner(store)
        runner.record_current_recommendation(state)

        # ... time passes ...

        runner.check_price(ticker, current_price, benchmark_price)
        print(runner.store.hit_rate())
    """

    def __init__(self, store: BacktestStore | None = None) -> None:
        self.store = store or BacktestStore()
        self._pending: dict[str, int] = {}  # ticker → recommendation_id

    def record_recommendation(self, ticker: str, state: dict) -> int:
        """Record a recommendation and track it for future checkpoints."""
        rec_id = self.store.record_recommendation(ticker, state)
        self._pending[ticker.upper()] = rec_id
        return rec_id

    def check_price(
        self,
        ticker: str,
        current_price: float,
        benchmark_price: float | None = None,
        days_since: int | None = None,
    ) -> None:
        """Update all pending recommendations for a ticker with current price."""
        ticker = ticker.upper()
        rec_id = self._pending.get(ticker)
        if rec_id is None:
            return

        # Default checkpoints: 30, 90, 180, 365 days
        for d in [30, 90, 180, 365]:
            if days_since is None or d <= days_since:
                self.store.record_checkpoint(rec_id, d, current_price, benchmark_price)


# ---------------------------------------------------------------------------
# Component-Level A/B Testing
# ---------------------------------------------------------------------------


class ComponentTester:
    """Test whether individual components improve valuation accuracy.

    Runs paired valuations (component enabled vs disabled) and records
    the error reduction for each component.
    """

    def __init__(self, store: BacktestStore | None = None) -> None:
        self.store = store or BacktestStore()

    def test_component(
        self,
        test_name: str,
        ticker: str,
        state_enabled: dict,
        state_disabled: dict,
        known_fair_value: float | None = None,
    ) -> dict:
        """Compare valuation with and without a component.

        Parameters
        ----------
        test_name : str
            e.g. "country_analysis", "monte_carlo", "blended_dcf_relative"
        ticker : str
        state_enabled : dict
            Full valuation state with the component enabled.
        state_disabled : dict
            Full valuation state with the component disabled.
        known_fair_value : float | None
            If available, the known eventual price (for historical backtesting).

        Returns
        -------
        dict with error_reduction metric
        """
        iv_enabled = state_enabled.get("intrinsic_value", 0.0)
        iv_disabled = state_disabled.get("intrinsic_value", 0.0)

        if known_fair_value and known_fair_value > 0:
            error_enabled = abs(iv_enabled - known_fair_value) / known_fair_value
            error_disabled = abs(iv_disabled - known_fair_value) / known_fair_value
            error_reduction = error_disabled - error_enabled  # positive = component helps
        else:
            # Without a known fair value, measure stability: which is closer to
            # the blended value? (less extreme = more reasonable)
            avg = (iv_enabled + iv_disabled) / 2
            if avg > 0:
                error_reduction = (
                    abs(iv_disabled - avg) - abs(iv_enabled - avg)
                ) / avg
            else:
                error_reduction = 0.0

        # Store result
        self.store.conn.execute(
            """INSERT INTO component_tests
               (test_name, component_enabled, component_disabled, ticker,
                intrinsic_value_enabled, intrinsic_value_disabled, error_reduction)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                test_name,
                test_name,
                f"no_{test_name}",
                ticker.upper(),
                iv_enabled,
                iv_disabled,
                error_reduction,
            ),
        )
        self.store.conn.commit()

        return {
            "test_name": test_name,
            "ticker": ticker,
            "iv_enabled": iv_enabled,
            "iv_disabled": iv_disabled,
            "error_reduction": error_reduction,
            "verdict": "component_adds_value" if error_reduction > 0 else "component_may_not_help",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serializable_state(state: dict) -> dict:
    """Strip non-serializable values from state for storage."""
    safe = {}
    for key, value in state.items():
        try:
            json.dumps(value)
            safe[key] = value
        except (TypeError, OverflowError):
            safe[key] = str(value)[:500]
    return safe
