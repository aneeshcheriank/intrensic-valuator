"""
LangGraph orchestrator for the top-down valuation agent pipeline.

The graph executes:
    DataFetch → Country → Industry → Company
              → AssumptionValidation → Valuation → Recommendation.

Each agent node is an LLM call with structured output.  The Valuation
node is pure Python computation (no LLM).  State flows between nodes
automatically via LangGraph's state management.

Key architectural guardrails (v1):
  - Assumption Validation Layer: GREEN/AMBER/RED bands before DCF
  - No-override principle: qualitative factors affect confidence, not fair value
  - Evidence-chain requirement: every adjustment must cite verifiable evidence
  - 7-factor confidence scoring: forecast precision, model agreement, data quality,
    historical stability, analyst consensus, macro uncertainty, assumption validation
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from src.data.data_cache import DataCache
from src.data.macro_fetcher import MacroFetcher
from src.data.yahoo_fetcher import YahooFinanceFetcher
from src.valuation.assumption_validator import AssumptionValidator, ValidationReport
from src.valuation.backtester import BacktestStore
from src.valuation.dcf_model import DCFModel, decay_growth_rates
from src.valuation.monte_carlo import MonteCarloSimulation, ScenarioAnalysis
from src.valuation.relative_val import RelativeValuation
from src.valuation.wacc_calculator import WACCCalculator
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# State schema (plain dict with known keys — compatible with LangGraph)
# ---------------------------------------------------------------------------


class ValuationState(dict):
    """Typed state that flows through the agent graph.

    We extend dict for LangGraph compatibility while providing typed access.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def initial_state(ticker: str) -> ValuationState:
    """Create the starting state for a valuation run."""
    return ValuationState(
        ticker=ticker.upper(),
        company_name="",
        country="",
        sector="",
        industry="",
        # --- Country layer ---
        country_risk_premium=0.0,
        gdp_growth_forecast=0.025,
        inflation_forecast=0.03,
        political_stability_score=7.0,
        currency_risk_adj=0.0,
        risk_free_rate=0.04,
        macro_narrative="",
        country_evidence_chain={},
        # --- Industry layer ---
        industry_growth_rate=0.05,
        industry_beta=1.0,
        competitive_intensity_score=5.0,
        regulatory_risk_score=5.0,
        disruption_risk_score=5.0,
        peer_tickers=[],
        industry_fcf_margin_avg=0.15,
        industry_narrative="",
        industry_evidence_chain={},
        # --- Company layer ---
        revenue_growth_forecast=0.05,
        fcf_margin_forecast=0.15,
        moat_width_score=5.0,
        management_quality_score=5.0,
        financial_health_score=5.0,
        company_specific_risk_premium=0.0,
        roic=0.10,
        debt_to_equity=0.5,
        company_narrative="",
        key_company_drivers=[],
        key_company_risks=[],
        company_evidence_chain={},
        growth_attribution={},
        # --- Financial data ---
        financials={},
        statutory_tax_rate=0.21,  # US statutory federal rate (default)
        # --- Assumption Validation ---
        validation_report={},
        validation_flags=[],
        # --- Valuation ---
        wacc=0.10,
        intrinsic_value=0.0,
        fair_value_low=0.0,
        fair_value_high=0.0,
        current_price=0.0,
        dcf_details={},
        scenario_results={},
        monte_carlo_stats={},
        relative_val_details={},
        # --- Output ---
        recommendation="HOLD",
        confidence_score=50,
        confidence_breakdown={},
        margin_of_safety=0.0,
        executive_summary="",
        key_drivers=[],
        key_risks=[],
        binary_risk_flags=[],
        # --- Metadata ---
        status="pending",
        errors=[],
    )


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------


def _get_llm(temperature: float = 0.1):
    """Return a LangChain ChatDeepSeek instance."""
    config = get_config()
    try:
        from langchain_deepseek import ChatDeepSeek
    except ImportError:
        raise ImportError(
            "langchain-deepseek is required. Install with: pip install langchain-deepseek"
        )

    if not config.has_deepseek:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not configured. Set it in .env to use agentic features."
        )

    return ChatDeepSeek(
        model="deepseek-chat",
        temperature=temperature,
        api_key=config.deepseek_api_key,
    )


def _parse_json_output(text: str) -> dict:
    """Extract a JSON object from an LLM response.

    Handles cases where the model wraps JSON in markdown fences or
    includes explanatory text.
    """
    # Try to find JSON in code fences first
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # Try to find a raw JSON object
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort: try the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON from LLM output: {text[:500]}...")
        raise ValueError("LLM did not produce valid JSON")


def _web_search(query: str, num_results: int = 5) -> str:
    """Perform a web search using DuckDuckGo (free, no API key).

    Falls back gracefully if the library is unavailable.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo-search not installed — web search disabled")
        return "Web search unavailable (duckduckgo-search not installed)."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
            if not results:
                return "No web search results found."

            lines = []
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                lines.append(f"- {title}\n  {body}\n  URL: {href}")
            return "\n\n".join(lines)
    except Exception as exc:
        logger.warning(f"Web search failed: {exc}")
        return f"Web search error: {exc}"


# ---------------------------------------------------------------------------
# Agent node functions
# ---------------------------------------------------------------------------


def _fetch_data_node(state: ValuationState) -> ValuationState:
    """Pre-processing node: pull all financial data from yfinance."""
    ticker = state["ticker"]
    logger.info(f"[Data Fetch] Pulling data for {ticker}...")

    yahoo = YahooFinanceFetcher()
    macro = MacroFetcher()

    try:
        # Company info and financials
        info = yahoo.fetch_company_info(ticker)
        metrics = yahoo.fetch_key_metrics(ticker)
        peers = yahoo.fetch_peer_companies(ticker)

        state["company_name"] = info.get("company_name", ticker)
        state["sector"] = info.get("sector", "Unknown")
        state["industry"] = info.get("industry", "Unknown")
        state["country"] = info.get("country", "United States")
        state["current_price"] = info.get("current_price", 0.0)
        state["financials"] = metrics
        state["peer_tickers"] = peers

        # Populate defaults from actual data (overridden by LLM agents if available)
        state["fcf_margin_forecast"] = metrics.get("fcf_margin", 0.15)
        state["revenue_growth_forecast"] = metrics.get("revenue_growth", 0.05)
        state["roic"] = metrics.get("roic", 0.10)
        state["debt_to_equity"] = metrics.get("debt_to_equity", 0.5)
        state["industry_beta"] = metrics.get("beta", 1.0)
        state["industry_growth_rate"] = metrics.get("revenue_growth", 0.04)

        # Statutory tax rate — use the jurisdiction's statutory corporate rate
        # (Damodaran convention: statutory rate in WACC, not effective rate)
        state["statutory_tax_rate"] = _get_statutory_tax_rate(state["country"])

        # Macro snapshot for the identified country
        macro_snap = macro.fetch_macro_snapshot(state["country"])
        state["risk_free_rate"] = macro_snap.get("risk_free_rate", 0.04) / 100.0 if macro_snap.get("risk_free_rate", 0) > 1 else macro_snap.get("risk_free_rate", 0.04)
        if state["risk_free_rate"] < 0.001:
            state["risk_free_rate"] = 0.04
        state["gdp_growth_forecast"] = macro_snap.get("gdp_growth", 0.025)
        state["inflation_forecast"] = macro_snap.get("inflation", 0.03)
        state["equity_risk_premium"] = macro_snap.get("equity_risk_premium", 0.05)

        logger.info(f"[Data Fetch] Complete. {state['company_name']} | {state['industry']} | {state['country']} | ${state['current_price']:.2f}")
        state["status"] = "data_fetched"

    except Exception as exc:
        logger.error(f"[Data Fetch] Failed: {exc}")
        state["errors"].append(f"Data fetch error: {exc}")
        state["status"] = "data_fetch_failed"

    return state


def _country_agent_node(state: ValuationState) -> ValuationState:
    """Country/Macro analysis via LLM."""
    if not get_config().has_deepseek:
        logger.warning("[Country Agent] DeepSeek not configured — skipping")
        return state

    logger.info("[Country Agent] Analyzing macro environment...")

    context = f"""
## Company
- Name: {state.get('company_name', 'Unknown')}
- Ticker: {state['ticker']}
- Country: {state.get('country', 'United States')}

## Macro Data
- Risk-Free Rate (10Y Treasury): {state.get('risk_free_rate', 0.04):.4f} ({state.get('risk_free_rate', 0.04)*100:.2f}%)
- GDP Growth: {state.get('gdp_growth_forecast', 0.025):.4f} ({state.get('gdp_growth_forecast', 0.025)*100:.2f}%)
- Inflation: {state.get('inflation_forecast', 0.03):.4f} ({state.get('inflation_forecast', 0.03)*100:.2f}%)
- Equity Risk Premium (default): {state.get('equity_risk_premium', 0.05):.4f} ({state.get('equity_risk_premium', 0.05)*100:.2f}%)

## Web Search
{_web_search(f"{state.get('country', 'United States')} economy GDP growth inflation 2025 2026 outlook")}
"""

    try:
        from src.prompts.country_prompt import COUNTRY_AGENT_PROMPT

        llm = _get_llm(temperature=0.1)
        response = llm.invoke(
            f"{COUNTRY_AGENT_PROMPT}\n\n{context}"
        )
        parsed = _parse_json_output(response.content if hasattr(response, 'content') else str(response))

        state["country"] = parsed.get("country", state.get("country", ""))
        state["country_risk_premium"] = parsed.get("country_risk_premium", 0.0)
        state["gdp_growth_forecast"] = parsed.get("gdp_growth_forecast", state["gdp_growth_forecast"])
        state["inflation_forecast"] = parsed.get("inflation_forecast", state["inflation_forecast"])
        state["political_stability_score"] = parsed.get("political_stability_score", 7.0)
        state["currency_risk_adj"] = parsed.get("currency_risk_adj", 0.0)
        state["macro_narrative"] = parsed.get("macro_narrative", "")
        state["country_evidence_chain"] = parsed.get("evidence_chain", {})
        if "key_strengths" in parsed:
            state["key_company_drivers"] = parsed["key_strengths"]
        if "key_risks" in parsed:
            state["key_company_risks"] = parsed["key_risks"]

        logger.info(f"[Country Agent] CRP={state['country_risk_premium']*10000:.0f} bps | GDP={state['gdp_growth_forecast']*100:.1f}%")
        state["status"] = "country_analyzed"

    except Exception as exc:
        logger.error(f"[Country Agent] Failed: {exc}")
        state["errors"].append(f"Country agent error: {exc}")
        # Use sensible defaults
        if state.get("country", "").upper() in ("US", "UNITED STATES"):
            state["country_risk_premium"] = 0.0
        else:
            state["country_risk_premium"] = 0.02

    return state


def _industry_agent_node(state: ValuationState) -> ValuationState:
    """Industry analysis via LLM."""
    if not get_config().has_deepseek:
        logger.warning("[Industry Agent] DeepSeek not configured — skipping")
        return state

    logger.info("[Industry Agent] Analyzing industry dynamics...")

    fin = state.get("financials", {})
    context = f"""
## Company
- Name: {state.get('company_name', 'Unknown')}
- Ticker: {state['ticker']}
- Sector: {state.get('sector', 'Unknown')}
- Industry: {state.get('industry', 'Unknown')}

## Country Context
- Country: {state.get('country', 'US')}
- CRP: {state.get('country_risk_premium', 0.0):.4f} ({state.get('country_risk_premium', 0.0)*10000:.0f} bps)
- GDP Growth Forecast: {state.get('gdp_growth_forecast', 0.025):.3f}

## Company Financials
- Revenue: ${fin.get('revenue', 0)/1000:.1f}B
- FCF Margin: {fin.get('fcf_margin', 0.15):.2%}
- Beta (raw): {fin.get('beta', 1.0):.2f}

## Peers
{state.get('peer_tickers', [])[:10]}

## Web Search
{_web_search(f"{state.get('industry', state.get('sector', 'technology'))} industry outlook trends 2025 2026 growth rate")}
"""

    try:
        from src.prompts.industry_prompt import INDUSTRY_AGENT_PROMPT

        llm = _get_llm(temperature=0.1)
        response = llm.invoke(f"{INDUSTRY_AGENT_PROMPT}\n\n{context}")
        parsed = _parse_json_output(response.content if hasattr(response, 'content') else str(response))

        state["industry"] = parsed.get("industry", state.get("industry", ""))
        state["industry_growth_rate"] = parsed.get("industry_growth_rate", 0.05)
        state["industry_beta"] = parsed.get("industry_beta_unlevered", 1.0)
        state["competitive_intensity_score"] = parsed.get("competitive_intensity_score", 5.0)
        state["regulatory_risk_score"] = parsed.get("regulatory_risk_score", 5.0)
        state["disruption_risk_score"] = parsed.get("disruption_risk_score", 5.0)
        state["industry_fcf_margin_avg"] = parsed.get("industry_fcf_margin_avg", 0.15)
        state["industry_narrative"] = parsed.get("industry_narrative", "")
        state["industry_evidence_chain"] = parsed.get("evidence_chain", {})
        if parsed.get("peer_tickers"):
            state["peer_tickers"] = parsed["peer_tickers"]

        logger.info(f"[Industry Agent] Growth={state['industry_growth_rate']*100:.1f}% | Beta={state['industry_beta']:.2f}")
        state["status"] = "industry_analyzed"

    except Exception as exc:
        logger.error(f"[Industry Agent] Failed: {exc}")
        state["errors"].append(f"Industry agent error: {exc}")

    return state


def _company_agent_node(state: ValuationState) -> ValuationState:
    """Company fundamental analysis via LLM."""
    if not get_config().has_deepseek:
        logger.warning("[Company Agent] DeepSeek not configured — skipping")
        return state

    logger.info("[Company Agent] Analyzing company fundamentals...")

    fin = state.get("financials", {})
    context = f"""
## Company
- Name: {state.get('company_name', 'Unknown')}
- Ticker: {state['ticker']}
- Sector: {state.get('sector', '')}
- Industry: {state.get('industry', '')}

## Country & Industry Context
- Country: {state.get('country', 'US')}
- Industry Growth Rate: {state.get('industry_growth_rate', 0.05):.1%}
- Competitive Intensity: {state.get('competitive_intensity_score', 5.0):.1f}/10
- Industry Beta: {state.get('industry_beta', 1.0):.2f}

## Financial Metrics
- Revenue: ${fin.get('revenue', 0)/1000:.1f}B
- Revenue Growth: {fin.get('revenue_growth', 0.05):.1%} (if available)
- Net Income: ${fin.get('net_income', 0)/1000:.1f}B
- FCF: ${fin.get('free_cash_flow', 0)/1000:.1f}B
- FCF Margin: {fin.get('fcf_margin', 0.15):.1%}
- EBITDA: ${fin.get('ebitda', 0)/1000:.1f}B
- EPS: ${fin.get('eps', 0):.2f}
- Market Cap: ${fin.get('market_cap', 0)/1000:.1f}B
- Total Debt: ${fin.get('total_debt', 0)/1000:.1f}B
- Cash: ${fin.get('cash_and_equivalents', 0)/1000:.1f}B
- Interest Coverage: {fin.get('interest_coverage_ratio', 5.0):.1f}x
- ROIC: {fin.get('roic', 0.10):.1%}
- D/E: {fin.get('debt_to_equity', 0.5):.2f}
- Beta: {fin.get('beta', 1.0):.2f}

## Web Search
{_web_search(f"{state['ticker']} {state.get('company_name', '')} earnings revenue growth competitive position moat 2025 2026")}
"""

    try:
        from src.prompts.company_prompt import COMPANY_AGENT_PROMPT

        llm = _get_llm(temperature=0.1)
        response = llm.invoke(f"{COMPANY_AGENT_PROMPT}\n\n{context}")
        parsed = _parse_json_output(response.content if hasattr(response, 'content') else str(response))

        state["revenue_growth_forecast"] = parsed.get("revenue_growth_forecast", 0.05)
        state["fcf_margin_forecast"] = parsed.get("fcf_margin_forecast", 0.15)
        state["moat_width_score"] = parsed.get("moat_width_score", 5.0)
        state["management_quality_score"] = parsed.get("management_quality_score", 5.0)
        state["financial_health_score"] = parsed.get("financial_health_score", 5.0)
        state["company_specific_risk_premium"] = parsed.get("company_specific_risk_premium", 0.0)
        state["roic"] = parsed.get("roic", state.get("roic", 0.10))
        state["debt_to_equity"] = parsed.get("debt_to_equity", state.get("debt_to_equity", 0.5))
        state["company_narrative"] = parsed.get("company_narrative", "")
        state["key_company_drivers"] = parsed.get("key_drivers", [])
        state["key_company_risks"] = parsed.get("key_risks", [])
        state["company_evidence_chain"] = parsed.get("evidence_chain", {})
        state["growth_attribution"] = parsed.get("growth_attribution", {})

        logger.info(f"[Company Agent] Growth={state['revenue_growth_forecast']*100:.1f}% | Margin={state['fcf_margin_forecast']*100:.1f}% | Moat={state['moat_width_score']:.0f}/10")
        state["status"] = "company_analyzed"

    except Exception as exc:
        logger.error(f"[Company Agent] Failed: {exc}")
        state["errors"].append(f"Company agent error: {exc}")

    return state


def _assumption_validation_node(state: ValuationState) -> ValuationState:
    """Pre-DCF guardrail: validate all agent assumptions before they enter valuation.

    Runs the Assumption Validation Layer to check each assumption against
    historical ranges, industry benchmarks, and statistical confidence bands.
    RED-flagged assumptions are capped at 2σ.
    """
    logger.info("[Assumption Validation] Validating agent assumptions...")

    fin = state.get("financials", {})

    # Build historical data from financials if available
    historical = {}
    if fin.get("revenue_growth_history"):
        historical["revenue_growth_historical"] = fin["revenue_growth_history"]
    if fin.get("fcf_margin_history"):
        historical["fcf_margin_historical"] = fin["fcf_margin_history"]

    # Build industry benchmarks
    industry_benchmarks = {
        "revenue_growth": state.get("industry_growth_rate", 0.05),
        "fcf_margin": state.get("industry_fcf_margin_avg", 0.15),
        "industry_beta": 1.0,  # market beta as reference
        "industry_growth_rate": state.get("gdp_growth_forecast", 0.025),  # long-run reference
    }

    validator = AssumptionValidator(
        historical_financials=historical if historical else None,
        industry_benchmarks=industry_benchmarks,
    )

    # Assumptions to validate
    assumptions = {
        "revenue_growth": state.get("revenue_growth_forecast", 0.05),
        "fcf_margin": state.get("fcf_margin_forecast", 0.15),
        "country_risk_premium": state.get("country_risk_premium", 0.0),
        "industry_growth_rate": state.get("industry_growth_rate", 0.05),
        "company_specific_risk_premium": state.get("company_specific_risk_premium", 0.0),
    }

    report: ValidationReport = validator.validate_all(assumptions)
    state["validation_report"] = report.to_dict()

    # Extract flags for display
    flags = []
    for result in report.results:
        if result.band.value != "GREEN":
            flags.append({
                "parameter": result.parameter,
                "band": result.band.value,
                "agent_value": result.agent_value,
                "capped_value": result.capped_value,
                "message": result.message,
            })
    state["validation_flags"] = flags

    # Cap RED assumptions — reuse report results (avoids re-running validation)
    capped = {}
    for result in report.results:
        capped[result.parameter] = result.capped_value
    state["revenue_growth_forecast"] = capped.get("revenue_growth", state["revenue_growth_forecast"])
    state["fcf_margin_forecast"] = capped.get("fcf_margin", state["fcf_margin_forecast"])
    state["country_risk_premium"] = capped.get("country_risk_premium", state["country_risk_premium"])
    state["industry_growth_rate"] = capped.get("industry_growth_rate", state["industry_growth_rate"])
    state["company_specific_risk_premium"] = capped.get("company_specific_risk_premium", state["company_specific_risk_premium"])

    logger.info(
        f"[Assumption Validation] {report.green_count}G / {report.amber_count}A / {report.red_count}R | "
        f"Overall: {report.overall_band.value}"
    )
    state["status"] = "assumptions_validated"
    return state


def _valuation_node(state: ValuationState) -> ValuationState:
    """Pure Python valuation computation — no LLM.

    Takes all structured outputs from the three analysis agents and
    runs the full DCF + WACC + Relative + Monte Carlo pipeline.
    """
    logger.info("[Valuation Engine] Computing intrinsic value...")

    fin = state.get("financials", {})
    ticker = state["ticker"]

    try:
        # --- WACC Calculation ---
        wacc_calc = WACCCalculator()

        # Relever beta to company's D/E using STATUTORY tax rate
        from src.valuation.wacc_calculator import relever_beta

        equity_val = fin.get("market_cap", 0.0)
        debt_val = fin.get("total_debt", 0.0)
        tax_rate = state.get("statutory_tax_rate", 0.21)  # Statutory, not effective
        unlevered_beta = state.get("industry_beta", 1.0)
        levered_beta = relever_beta(unlevered_beta, debt_val, equity_val, tax_rate) if equity_val > 0 else unlevered_beta

        wacc_result = wacc_calc.calculate(
            risk_free_rate=state.get("risk_free_rate", 0.04),
            equity_risk_premium=state.get("equity_risk_premium", 0.05),
            beta=levered_beta,
            market_cap=equity_val,
            total_debt=debt_val,
            interest_expense=fin.get("interest_expense", 0.0),
            ebit=fin.get("ebit", fin.get("operating_cash_flow", 0.0)),
            tax_rate=tax_rate,
            country_risk_premium=state.get("country_risk_premium", 0.0),
            company_specific_premium=state.get("company_specific_risk_premium", 0.0),
        )

        wacc = wacc_result["wacc"]
        state["wacc"] = wacc
        logger.info(f"[Valuation Engine] WACC = {wacc*100:.2f}% | Rating: {wacc_result['synthetic_rating']} | Re = {wacc_result['cost_of_equity']*100:.2f}%")

        # --- DCF Model ---
        dcf = DCFModel(
            financials={
                "revenue": fin.get("revenue", 0.0),
                "operating_cash_flow": fin.get("operating_cash_flow", 0.0),
                "capital_expenditure": fin.get("capital_expenditure", 0.0),
                "total_debt": fin.get("total_debt", 0.0),
                "cash_and_equivalents": fin.get("cash_and_equivalents", 0.0),
                "shares_outstanding": fin.get("shares_outstanding", 1.0),
            },
            projection_years=5,
        )

        # Generate growth rates using the 3-knot spline
        growth_rates = decay_growth_rates(
            company_growth=state.get("revenue_growth_forecast", 0.05),
            industry_growth=state.get("industry_growth_rate", 0.04),
            terminal_growth=min(state.get("gdp_growth_forecast", 0.025), 0.035),
        )

        margins = [state.get("fcf_margin_forecast", fin.get("fcf_margin", 0.15))] * 5

        dcf_result = dcf.run(
            revenue_growth_rates=growth_rates,
            fcf_margins=margins,
            wacc=wacc,
            terminal_growth=min(state.get("gdp_growth_forecast", 0.025), 0.035),
        )

        intrinsic_value = dcf_result["intrinsic_value_per_share"]
        state["dcf_details"] = dcf_result
        state["intrinsic_value"] = intrinsic_value

        logger.info(f"[Valuation Engine] DCF Intrinsic Value = ${intrinsic_value:.2f}")

        # --- Relative Valuation ---
        yahoo = YahooFinanceFetcher()
        peer_metrics_list = yahoo.fetch_peer_metrics(state.get("peer_tickers", [])[:8])

        # FMP/Alpha Vantage fallback for NaN peer data
        if not peer_metrics_list or _has_nan_peers(peer_metrics_list):
            logger.info("[Valuation Engine] Peer data has NaN — attempting FMP/Alpha Vantage fallback")
            fallback_peers = _fetch_fallback_peer_metrics(state.get("peer_tickers", [])[:8])
            if fallback_peers:
                peer_metrics_list = fallback_peers
                logger.info(f"[Valuation Engine] Fallback succeeded — {len(peer_metrics_list)} peers")

        rel_val = RelativeValuation(
            company_metrics={
                "eps": fin.get("eps", 0.0),
                "ebitda": fin.get("ebitda", 0.0),
                "book_value_per_share": fin.get("book_value_per_share", 0.0),
                "market_cap": fin.get("market_cap", 0.0),
                "total_debt": fin.get("total_debt", 0.0),
                "cash_and_equivalents": fin.get("cash_and_equivalents", 0.0),
                "shares_outstanding": fin.get("shares_outstanding", 1.0),
            },
            peer_metrics=peer_metrics_list,
        )

        rel_result = rel_val.run()
        state["relative_val_details"] = rel_result
        relative_value = rel_result.get("blended_relative_value", intrinsic_value)

        logger.info(f"[Valuation Engine] Relative Value = ${relative_value:.2f} | Peers: {len(peer_metrics_list)}")

        # --- Blended value (70% DCF, 30% Relative) ---
        if not (isinstance(relative_value, float) and relative_value > 0):
            relative_value = intrinsic_value

        blended_value = 0.70 * intrinsic_value + 0.30 * relative_value
        state["intrinsic_value"] = blended_value

        # --- Scenario Analysis ---
        scenario = ScenarioAnalysis(dcf, {
            "company_growth": state.get("revenue_growth_forecast", 0.05),
            "industry_growth": state.get("industry_growth_rate", 0.04),
            "terminal_growth": min(state.get("gdp_growth_forecast", 0.025), 0.035),
            "wacc": wacc,
            "fcf_margin": state.get("fcf_margin_forecast", fin.get("fcf_margin", 0.15)),
        })
        state["scenario_results"] = scenario.run()

        # --- Monte Carlo ---
        mc = MonteCarloSimulation(dcf, {
            "company_growth": state.get("revenue_growth_forecast", 0.05),
            "industry_growth": state.get("industry_growth_rate", 0.04),
            "terminal_growth": min(state.get("gdp_growth_forecast", 0.025), 0.035),
            "wacc": wacc,
            "fcf_margin": state.get("fcf_margin_forecast", fin.get("fcf_margin", 0.15)),
        }, iterations=5000)  # 5k for speed; use 10k for production
        state["monte_carlo_stats"] = mc.run()

        # Set fair value range from Monte Carlo
        state["fair_value_low"] = state["monte_carlo_stats"].get("fair_value_low", blended_value * 0.8)
        state["fair_value_high"] = state["monte_carlo_stats"].get("fair_value_high", blended_value * 1.2)

        current_price = state.get("current_price", 0.0) or fin.get("current_price", 0.0)
        state["current_price"] = current_price

        if current_price > 0 and blended_value > 0:
            state["margin_of_safety"] = (blended_value - current_price) / blended_value
        else:
            state["margin_of_safety"] = 0.0

        logger.info(
            f"[Valuation Engine] Blended IV=${blended_value:.2f} | "
            f"Current=${current_price:.2f} | "
            f"MoS={state['margin_of_safety']*100:.1f}%"
        )
        state["status"] = "valuation_complete"

    except Exception as exc:
        logger.error(f"[Valuation Engine] Failed: {exc}")
        state["errors"].append(f"Valuation error: {exc}")

    return state


def _recommendation_node(state: ValuationState) -> ValuationState:
    """Final recommendation synthesis via LLM with 7-factor confidence scoring."""
    logger.info("[Recommendation Agent] Synthesizing final recommendation...")

    # First, apply the quantitative decision rule (NO OVERRIDE)
    current_price = state.get("current_price", 0.0)
    intrinsic_value = state.get("intrinsic_value", 0.0)

    if current_price > 0 and intrinsic_value > 0:
        upside = intrinsic_value / current_price
        if upside >= 1.20:
            quant_rec = "BUY"
        elif upside <= 0.90:
            quant_rec = "SELL"
        else:
            quant_rec = "HOLD"
    else:
        quant_rec = "HOLD"

    # Compute 7-factor confidence score
    confidence_breakdown = _compute_confidence(state)
    confidence_score = sum(confidence_breakdown.values())
    confidence_score = max(0, min(100, int(confidence_score)))

    state["confidence_breakdown"] = confidence_breakdown
    state["confidence_score"] = confidence_score

    logger.info(f"[Recommendation Agent] Base confidence: {confidence_score}/100 (7-factor)")

    # If DeepSeek is available, get a narrative recommendation
    if get_config().has_deepseek:
        try:
            context = f"""
## Complete Analysis for {state.get('company_name', state['ticker'])} ({state['ticker']})

### Country Analysis ({state.get('country', 'US')})
- CRP: {state.get('country_risk_premium', 0.0)*10000:.0f} bps
- GDP Growth: {state.get('gdp_growth_forecast', 0.025)*100:.1f}%
- Inflation: {state.get('inflation_forecast', 0.03)*100:.1f}%
- Narrative: {state.get('macro_narrative', 'N/A')[:500]}

### Industry Analysis
- Industry: {state.get('industry', 'Unknown')}
- Growth Rate: {state.get('industry_growth_rate', 0.05)*100:.1f}%
- Beta: {state.get('industry_beta', 1.0):.2f}
- Competitive Intensity: {state.get('competitive_intensity_score', 5.0):.1f}/10
- Narrative: {state.get('industry_narrative', 'N/A')[:500]}

### Company Analysis
- Revenue Growth Forecast: {state.get('revenue_growth_forecast', 0.05)*100:.1f}%
- FCF Margin Forecast: {state.get('fcf_margin_forecast', 0.15)*100:.1f}%
- Moat Score: {state.get('moat_width_score', 5.0):.1f}/10
- Management Score: {state.get('management_quality_score', 5.0):.1f}/10
- Financial Health: {state.get('financial_health_score', 5.0):.1f}/10
- Narrative: {state.get('company_narrative', 'N/A')[:500]}

### Growth Attribution
{state.get('growth_attribution', {})}

### Assumption Validation
- Flags: {state.get('validation_flags', [])}
- Report: {state.get('validation_report', {})}

### Valuation Results
- Current Price: ${current_price:.2f}
- Intrinsic Value (Blended): ${intrinsic_value:.2f}
- Margin of Safety: {state.get('margin_of_safety', 0.0)*100:.1f}%
- WACC: {state.get('wacc', 0.10)*100:.2f}%
- Quantitative Recommendation: {quant_rec}
- Fair Value Range: ${state.get('fair_value_low', 0):.2f} - ${state.get('fair_value_high', 0):.2f}

### Computed Confidence (7-Factor)
{confidence_breakdown}
Total: {confidence_score}/100

### Scenario Analysis
Bull: ${state.get('scenario_results', {}).get('Bull', {}).get('intrinsic_value_per_share', 0):.2f}
Base: ${state.get('scenario_results', {}).get('Base', {}).get('intrinsic_value_per_share', 0):.2f}
Bear: ${state.get('scenario_results', {}).get('Bear', {}).get('intrinsic_value_per_share', 0):.2f}

### Monte Carlo
{state.get('monte_carlo_stats', {})}
"""
            from src.prompts.recommendation_prompt import RECOMMENDATION_AGENT_PROMPT

            llm = _get_llm(temperature=0.05)  # cooler for recommendation
            response = llm.invoke(f"{RECOMMENDATION_AGENT_PROMPT}\n\n{context}")
            parsed = _parse_json_output(response.content if hasattr(response, 'content') else str(response))

            state["recommendation"] = parsed.get("recommendation", quant_rec)
            state["executive_summary"] = parsed.get("executive_summary", "")
            state["key_drivers"] = parsed.get("key_drivers", [])
            state["key_risks"] = parsed.get("key_risks", [])
            state["binary_risk_flags"] = parsed.get("binary_risk_flags", [])

            # Use LLM confidence if provided, otherwise keep computed
            if "confidence_score" in parsed and parsed["confidence_score"] is not None:
                state["confidence_score"] = parsed["confidence_score"]
            if parsed.get("confidence_breakdown"):
                state["confidence_breakdown"] = parsed["confidence_breakdown"]

            logger.info(
                f"[Recommendation Agent] {state['recommendation']} | "
                f"Confidence: {state['confidence_score']}/100"
            )
        except Exception as exc:
            logger.error(f"[Recommendation Agent] LLM call failed: {exc} — using quantitative rule")
            state["recommendation"] = quant_rec
            state["executive_summary"] = (
                f"Quantitative analysis suggests a {quant_rec} recommendation for "
                f"{state.get('company_name', state['ticker'])} ({state['ticker']}) "
                f"with an intrinsic value of ${intrinsic_value:.2f} vs current price of "
                f"${current_price:.2f} (margin of safety: {state.get('margin_of_safety', 0.0)*100:.1f}%)."
            )
            state["key_drivers"] = state.get("key_company_drivers", [])
            state["key_risks"] = state.get("key_company_risks", [])
            state["binary_risk_flags"] = []
    else:
        # No LLM — use quantitative rule
        state["recommendation"] = quant_rec
        state["executive_summary"] = (
            f"Quantitative analysis suggests a {quant_rec} recommendation for "
            f"{state.get('company_name', state['ticker'])} ({state['ticker']}) "
            f"with an intrinsic value of ${intrinsic_value:.2f} vs current price of "
            f"${current_price:.2f} (margin of safety: {state.get('margin_of_safety', 0.0)*100:.1f}%)."
        )
        state["key_drivers"] = state.get("key_company_drivers", [])
        state["key_risks"] = state.get("key_company_risks", [])
        state["binary_risk_flags"] = []

    # Record recommendation for backtesting
    try:
        store = BacktestStore()
        store.record_recommendation(state["ticker"], dict(state))
        logger.debug(f"[Backtest] Recommendation recorded for {state['ticker']}")
    except Exception as exc:
        logger.warning(f"[Backtest] Failed to record recommendation: {exc}")

    state["status"] = "complete"
    return state


# ---------------------------------------------------------------------------
# 7-Factor Confidence Computation
# ---------------------------------------------------------------------------


def _compute_confidence(state: ValuationState) -> dict[str, float]:
    """Compute the 7-factor confidence score from all available data.

    Weights:
      1. Forecast Precision (25%): Monte Carlo CV
      2. Model Agreement (20%): DCF vs Relative valuation spread
      3. Data Quality (15%): Completeness of financial data
      4. Historical Stability (10%): Variance in historical margins/growth
      5. Analyst Consensus (10%): Dispersion of analyst estimates
      6. Macro Uncertainty (10%): Current macro volatility indicators
      7. Assumption Validation (10%): GREEN/AMBER/RED flags
    """
    mc_stats = state.get("monte_carlo_stats", {})
    dcf_details = state.get("dcf_details", {})
    rel_details = state.get("relative_val_details", {})
    fin = state.get("financials", {})
    val_report = state.get("validation_report", {})

    # 1. Forecast Precision (max 25 pts)
    forecast_precision = 15.0  # default
    if mc_stats and mc_stats.get("std_dev", 0) > 0:
        std = mc_stats["std_dev"]
        mean = mc_stats.get("mean", 1)
        if mean > 0:
            cv = std / mean
            if cv < 0.10:
                forecast_precision = 25.0
            elif cv < 0.15:
                forecast_precision = 22.0
            elif cv < 0.20:
                forecast_precision = 18.0
            elif cv < 0.30:
                forecast_precision = 12.0
            elif cv < 0.50:
                forecast_precision = 6.0
            else:
                forecast_precision = 2.0

    # 2. Model Agreement (max 20 pts)
    model_agreement = 10.0  # default
    dcf_iv = dcf_details.get("intrinsic_value_per_share", 0)
    rel_iv = rel_details.get("blended_relative_value", 0)
    if dcf_iv > 0 and rel_iv > 0:
        spread_pct = abs(dcf_iv - rel_iv) / ((dcf_iv + rel_iv) / 2)
        if spread_pct < 0.10:
            model_agreement = 20.0
        elif spread_pct < 0.20:
            model_agreement = 16.0
        elif spread_pct < 0.30:
            model_agreement = 12.0
        elif spread_pct < 0.50:
            model_agreement = 6.0
        else:
            model_agreement = 2.0

    # 3. Data Quality (max 15 pts)
    data_quality = 10.0  # default
    quality_deductions = 0
    if not fin.get("revenue"):
        quality_deductions += 3
    if not fin.get("operating_cash_flow"):
        quality_deductions += 3
    if not fin.get("free_cash_flow"):
        quality_deductions += 2
    if not fin.get("eps"):
        quality_deductions += 2
    peers = state.get("peer_tickers", [])
    if len(peers) < 3:
        quality_deductions += 2
    if state.get("relative_val_details", {}).get("num_peers", 0) == 0:
        quality_deductions += 3
    data_quality = max(0, 15 - quality_deductions)

    # 4. Historical Stability (max 10 pts)
    historical_stability = 6.0  # default
    # If the company has stable historical margins (low variance), score higher
    # This is approximated from available data
    if fin.get("fcf_margin", 0) > 0:
        # Higher base for companies with positive FCF margins
        historical_stability = 7.0
    if fin.get("roic", 0) > 0.10:
        historical_stability += 1.0

    # 5. Analyst Consensus (max 10 pts)
    analyst_consensus = 5.0  # default — neutral when no data
    # Can be improved by fetching actual analyst dispersion data

    # 6. Macro Uncertainty (max 10 pts)
    macro_uncertainty = 6.0  # default — moderate
    crp = state.get("country_risk_premium", 0.0)
    inflation = state.get("inflation_forecast", 0.03)
    # Lower score for high CRP or high inflation
    if crp > 0.005:  # >50 bps
        macro_uncertainty -= 2.0
    if inflation > 0.05:  # >5% inflation
        macro_uncertainty -= 2.0
    if crp == 0.0:
        macro_uncertainty = 9.0  # US companies get highest macro certainty
    macro_uncertainty = max(0, min(10, macro_uncertainty))

    # 7. Assumption Validation (max 10 pts)
    assumption_validation = 10.0  # default — all GREEN
    if val_report:
        red_count = val_report.get("red_count", 0)
        amber_count = val_report.get("amber_count", 0)
        assumption_validation -= red_count * 3.5
        assumption_validation -= amber_count * 1.5
    assumption_validation = max(0, min(10, assumption_validation))

    return {
        "forecast_precision": round(forecast_precision, 1),
        "model_agreement": round(model_agreement, 1),
        "data_quality": round(data_quality, 1),
        "historical_stability": round(historical_stability, 1),
        "analyst_consensus": round(analyst_consensus, 1),
        "macro_uncertainty": round(macro_uncertainty, 1),
        "assumption_validation": round(assumption_validation, 1),
    }


# ---------------------------------------------------------------------------
# Statutory tax rate by jurisdiction
# ---------------------------------------------------------------------------

# Source: OECD Corporate Tax Statistics 2024 / KPMG global tax rate table.
# Damodaran convention: use statutory (marginal) rate, not effective rate.
_STATUTORY_TAX_RATES: dict[str, float] = {
    "UNITED STATES": 0.21,
    "US": 0.21,
    "UNITED KINGDOM": 0.25,
    "UK": 0.25,
    "GERMANY": 0.30,
    "JAPAN": 0.2966,
    "FRANCE": 0.2583,
    "CANADA": 0.2621,
    "AUSTRALIA": 0.30,
    "SWITZERLAND": 0.1966,
    "NETHERLANDS": 0.258,
    "SWEDEN": 0.206,
    "NORWAY": 0.22,
    "DENMARK": 0.22,
    "FINLAND": 0.20,
    "BELGIUM": 0.25,
    "AUSTRIA": 0.23,
    "IRELAND": 0.125,
    "ITALY": 0.2781,
    "SPAIN": 0.25,
    "PORTUGAL": 0.315,
    "GREECE": 0.22,
    "SOUTH KOREA": 0.242,
    "KOREA": 0.242,
    "TAIWAN": 0.20,
    "CHINA": 0.25,
    "INDIA": 0.252,
    "BRAZIL": 0.34,
    "MEXICO": 0.30,
    "INDONESIA": 0.22,
    "SOUTH AFRICA": 0.27,
    "TURKEY": 0.25,
    "RUSSIA": 0.20,
    "SAUDI ARABIA": 0.20,
    "UAE": 0.09,
    "SINGAPORE": 0.17,
    "HONG KONG": 0.165,
    "MALAYSIA": 0.24,
    "THAILAND": 0.20,
    "VIETNAM": 0.20,
    "PHILIPPINES": 0.25,
    "ISRAEL": 0.23,
    "CHILE": 0.27,
    "COLOMBIA": 0.35,
    "ARGENTINA": 0.35,
    "NIGERIA": 0.30,
    "EGYPT": 0.225,
    "PAKISTAN": 0.29,
    "BANGLADESH": 0.275,
}


def _get_statutory_tax_rate(country: str) -> float:
    """Return the statutory corporate tax rate for a given country.

    Falls back to 0.21 (US rate / global median) for unrecognized countries.
    """
    if not country:
        return 0.21
    key = country.upper().strip()
    # Try exact match first, then partial match
    if key in _STATUTORY_TAX_RATES:
        return _STATUTORY_TAX_RATES[key]
    for known_country, rate in _STATUTORY_TAX_RATES.items():
        if known_country in key or key in known_country:
            return rate
    return 0.21  # fallback: US/global median rate


# ---------------------------------------------------------------------------
# FMP / Alpha Vantage fallback for peer data
# ---------------------------------------------------------------------------


def _has_nan_peers(peer_metrics_list: list[dict]) -> bool:
    """Check if all peers have NaN for their key multiples."""
    import math
    if not peer_metrics_list:
        return True
    nan_count = 0
    for p in peer_metrics_list:
        pe = p.get("pe_ratio")
        ev_ebitda = p.get("ev_ebitda")
        pe_is_nan = pe is None or (isinstance(pe, float) and math.isnan(pe))
        ev_is_nan = ev_ebitda is None or (isinstance(ev_ebitda, float) and math.isnan(ev_ebitda))
        if pe_is_nan and ev_is_nan:
            nan_count += 1
    return nan_count >= len(peer_metrics_list) * 0.5


def _fetch_fallback_peer_metrics(peer_tickers: list[str]) -> list[dict]:
    """Fall back to FMP or Alpha Vantage for peer metrics when yfinance returns NaN."""
    from src.data.http_cache import get_http_session
    config = get_config()
    session = get_http_session()

    # Try FMP first
    fmp_key = config.fmp_api_key or ""
    if fmp_key:
        try:
            results = []
            for ticker in peer_tickers[:5]:
                url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?apikey={fmp_key}&limit=1"
                resp = session.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        d = data[0]
                        results.append({
                            "ticker": ticker,
                            "pe_ratio": d.get("peRatio"),
                            "ev_ebitda": d.get("enterpriseValueOverEBITDA"),
                            "pb_ratio": d.get("pbRatio"),
                        })
            if results:
                logger.info(f"[Fallback] FMP returned {len(results)} peer metrics")
                return results
        except Exception as exc:
            logger.warning(f"[Fallback] FMP failed: {exc}")

    # Try Alpha Vantage
    av_key = config.alpha_vantage_api_key or ""
    if av_key:
        try:
            results = []
            for ticker in peer_tickers[:3]:  # AV has tight rate limits
                url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={av_key}"
                resp = session.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data and "PERatio" in data:
                        results.append({
                            "ticker": ticker,
                            "pe_ratio": float(data.get("PERatio", "nan")) if data.get("PERatio") not in ("None", "") else None,
                            "ev_ebitda": float(data.get("EVToEBITDA", "nan")) if data.get("EVToEBITDA") not in ("None", "") else None,
                            "pb_ratio": float(data.get("PriceToBookRatio", "nan")) if data.get("PriceToBookRatio") not in ("None", "") else None,
                        })
            if results:
                logger.info(f"[Fallback] Alpha Vantage returned {len(results)} peer metrics")
                return results
        except Exception as exc:
            logger.warning(f"[Fallback] Alpha Vantage failed: {exc}")

    return []


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Build and return the LangGraph StateGraph for the valuation pipeline.

    Graph structure:
        DataFetch → CountryAgent → IndustryAgent → CompanyAgent
                  → AssumptionValidation → ValuationEngine → RecommendationAgent → END

    The ValuationEngine is pure Python (no LLM). AssumptionValidation is a
    pre-DCF guardrail that checks all agent outputs against historical ranges
    and industry benchmarks before they enter the valuation.
    """
    graph = StateGraph(ValuationState)

    # Add nodes
    graph.add_node("data_fetch", _fetch_data_node)
    graph.add_node("country_agent", _country_agent_node)
    graph.add_node("industry_agent", _industry_agent_node)
    graph.add_node("company_agent", _company_agent_node)
    graph.add_node("assumption_validation", _assumption_validation_node)
    graph.add_node("valuation_engine", _valuation_node)
    graph.add_node("recommendation_agent", _recommendation_node)

    # Define edges (linear pipeline with validation gate)
    graph.set_entry_point("data_fetch")
    graph.add_edge("data_fetch", "country_agent")
    graph.add_edge("country_agent", "industry_agent")
    graph.add_edge("industry_agent", "company_agent")
    graph.add_edge("company_agent", "assumption_validation")
    graph.add_edge("assumption_validation", "valuation_engine")
    graph.add_edge("valuation_engine", "recommendation_agent")
    graph.add_edge("recommendation_agent", END)

    return graph


def get_orchestrator():
    """Compile and return the LangGraph app (runnable)."""
    graph = build_graph()
    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_valuation(ticker: str) -> ValuationState:
    """Run the full valuation pipeline for a ticker and return the final state.

    Executes the agent nodes sequentially (Country → Industry → Company →
    Validation → Valuation → Recommendation).  This is the primary entry
    point for programmatic use.

    Falls back to sequential execution if LangGraph compilation fails.
    """
    state = initial_state(ticker)

    # Try LangGraph, fall back to sequential
    try:
        app = get_orchestrator()
        final_state = app.invoke(state)
        return ValuationState(final_state)
    except Exception as exc:
        logger.warning(
            f"LangGraph execution failed ({exc}), falling back to sequential pipeline"
        )

    # -- Sequential fallback --
    pipeline = [
        ("Data Fetch", _fetch_data_node),
        ("Country Agent", _country_agent_node),
        ("Industry Agent", _industry_agent_node),
        ("Company Agent", _company_agent_node),
        ("Assumption Validation", _assumption_validation_node),
        ("Valuation Engine", _valuation_node),
        ("Recommendation", _recommendation_node),
    ]

    for name, node_fn in pipeline:
        logger.info(f"[Pipeline] Running: {name}")
        try:
            state = node_fn(state)
            if state.get("status") == "data_fetch_failed":
                logger.error(f"[Pipeline] Data fetch failed — aborting")
                break
        except Exception as exc:
            logger.error(f"[Pipeline] {name} failed: {exc}")
            state["errors"].append(f"{name}: {exc}")

    return state
