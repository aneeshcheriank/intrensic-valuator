"""
LangGraph orchestrator for the top-down valuation agent pipeline.

The graph executes: Country → Industry → Company → Valuation → Recommendation.

Each agent node is an LLM call with structured output.  The Valuation
node is pure Python computation (no LLM).  State flows between nodes
automatically via LangGraph's state management.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from src.data.data_cache import DataCache
from src.data.macro_fetcher import MacroFetcher
from src.data.yahoo_fetcher import YahooFinanceFetcher
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
        # --- Industry layer ---
        industry_growth_rate=0.05,
        industry_beta=1.0,
        competitive_intensity_score=5.0,
        regulatory_risk_score=5.0,
        disruption_risk_score=5.0,
        peer_tickers=[],
        industry_fcf_margin_avg=0.15,
        industry_narrative="",
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
        # --- Financial data ---
        financials={},
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
        margin_of_safety=0.0,
        executive_summary="",
        key_drivers=[],
        key_risks=[],
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

    # Build context for the agent
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

        logger.info(f"[Company Agent] Growth={state['revenue_growth_forecast']*100:.1f}% | Margin={state['fcf_margin_forecast']*100:.1f}% | Moat={state['moat_width_score']:.0f}/10")
        state["status"] = "company_analyzed"

    except Exception as exc:
        logger.error(f"[Company Agent] Failed: {exc}")
        state["errors"].append(f"Company agent error: {exc}")

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

        # Relever beta to company's D/E
        from src.valuation.wacc_calculator import relever_beta

        equity_val = fin.get("market_cap", 0.0)
        debt_val = fin.get("total_debt", 0.0)
        tax_rate = fin.get("tax_rate", 0.21)
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

        # Generate growth rates that decay to terminal
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
    """Final recommendation synthesis via LLM."""
    logger.info("[Recommendation Agent] Synthesizing final recommendation...")

    # First, apply the quantitative decision rule
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

    # Compute a base confidence score from Monte Carlo stats
    mc_stats = state.get("monte_carlo_stats", {})
    base_confidence = 50
    if mc_stats and "std_dev" in mc_stats:
        std = mc_stats["std_dev"]
        mean = mc_stats.get("mean", intrinsic_value)
        if mean > 0:
            cv = std / mean
            precision_score = max(0, 100 - cv * 100)
            base_confidence = int(precision_score * 0.4 + 30)  # partial weight

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

### Valuation Results
- Current Price: ${current_price:.2f}
- Intrinsic Value (Blended): ${intrinsic_value:.2f}
- Margin of Safety: {state.get('margin_of_safety', 0.0)*100:.1f}%
- WACC: {state.get('wacc', 0.10)*100:.2f}%
- Quantitative Recommendation: {quant_rec}
- Fair Value Range: ${state.get('fair_value_low', 0):.2f} - ${state.get('fair_value_high', 0):.2f}
- Base Confidence: {base_confidence}/100

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
            state["confidence_score"] = parsed.get("confidence_score", base_confidence)
            state["executive_summary"] = parsed.get("executive_summary", "")
            state["key_drivers"] = parsed.get("key_drivers", [])
            state["key_risks"] = parsed.get("key_risks", [])

            logger.info(
                f"[Recommendation Agent] {state['recommendation']} | "
                f"Confidence: {state['confidence_score']}/100"
            )
        except Exception as exc:
            logger.error(f"[Recommendation Agent] LLM call failed: {exc} — using quantitative rule")
            state["recommendation"] = quant_rec
            state["confidence_score"] = base_confidence
            state["executive_summary"] = (
                f"Quantitative analysis suggests a {quant_rec} recommendation for "
                f"{state.get('company_name', state['ticker'])} ({state['ticker']}) "
                f"with an intrinsic value of ${intrinsic_value:.2f} vs current price of "
                f"${current_price:.2f} (margin of safety: {state.get('margin_of_safety', 0.0)*100:.1f}%)."
            )
            state["key_drivers"] = state.get("key_company_drivers", [])
            state["key_risks"] = state.get("key_company_risks", [])
    else:
        # No LLM — use quantitative rule
        state["recommendation"] = quant_rec
        state["confidence_score"] = base_confidence
        state["executive_summary"] = (
            f"Quantitative analysis suggests a {quant_rec} recommendation for "
            f"{state.get('company_name', state['ticker'])} ({state['ticker']}) "
            f"with an intrinsic value of ${intrinsic_value:.2f} vs current price of "
            f"${current_price:.2f} (margin of safety: {state.get('margin_of_safety', 0.0)*100:.1f}%)."
        )
        state["key_drivers"] = state.get("key_company_drivers", [])
        state["key_risks"] = state.get("key_company_risks", [])

    state["status"] = "complete"
    return state


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Build and return the LangGraph StateGraph for the valuation pipeline.

    Graph structure:
        DataFetch → CountryAgent → IndustryAgent → CompanyAgent
                  → ValuationEngine → RecommendationAgent → END

    The ValuationEngine is pure Python (no LLM), so the pipeline works
    even without a configured LLM — it just uses default assumptions
    from financial data.
    """
    graph = StateGraph(ValuationState)

    # Add nodes
    graph.add_node("data_fetch", _fetch_data_node)
    graph.add_node("country_agent", _country_agent_node)
    graph.add_node("industry_agent", _industry_agent_node)
    graph.add_node("company_agent", _company_agent_node)
    graph.add_node("valuation_engine", _valuation_node)
    graph.add_node("recommendation_agent", _recommendation_node)

    # Define edges (linear pipeline)
    graph.set_entry_point("data_fetch")
    graph.add_edge("data_fetch", "country_agent")
    graph.add_edge("country_agent", "industry_agent")
    graph.add_edge("industry_agent", "company_agent")
    graph.add_edge("company_agent", "valuation_engine")
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
    Valuation → Recommendation).  This is the primary entry point for
    programmatic use.

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
