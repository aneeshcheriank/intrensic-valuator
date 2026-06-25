#!/usr/bin/env python3
"""
Intrensic Valuator — Streamlit Web UI.

Launch with:
    ./venv/bin/streamlit run app.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

# Page config must be the first Streamlit call
st.set_page_config(
    page_title="Intrensic Valuator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.orchestrator import initial_state, run_valuation

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("📈 Intrensic Valuator")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "AI-powered stock valuation using a top-down approach: "
    "**Country → Industry → Company**"
)

ticker_input = st.sidebar.text_input(
    "Enter a ticker symbol:",
    value="AAPL",
    max_chars=10,
    placeholder="e.g., AAPL, MSFT, GOOGL",
).strip().upper()

use_llm = st.sidebar.checkbox("Use AI Agents (DeepSeek)", value=False)
use_monte_carlo = st.sidebar.checkbox("Run Monte Carlo (slower)", value=True)
generate_pdf = st.sidebar.checkbox("Include PDF Report", value=True)

analyze_clicked = st.sidebar.button("🔍 Analyze", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "*⚠️ This is not financial advice.*\n\n"
    "*AI-generated analysis. Always do your own research.*"
)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("📊 Intrensic Valuator")
st.caption("Top-down intrinsic valuation with agentic AI")

if not analyze_clicked:
    st.info("👈 Enter a ticker in the sidebar and click **Analyze** to begin.")
    st.markdown("""
    ### How it works
    1. **Country Analysis** — Evaluates macro environment, GDP growth, inflation, political stability
    2. **Industry Analysis** — Assesses industry growth, competitive dynamics, regulatory risk
    3. **Company Analysis** — Analyzes financials, moat, management quality, growth drivers
    4. **Valuation Engine** — Runs DCF, WACC, relative valuation & Monte Carlo simulation
    5. **Recommendation** — Synthesizes everything into a BUY/SELL/HOLD with confidence score
    """)
    st.stop()

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

if not use_llm:
    import os
    os.environ["DEEPSEEK_API_KEY"] = ""

with st.spinner(f"Analyzing {ticker_input}... This may take 30-60 seconds."):
    progress_bar = st.progress(0, text="Fetching data...")

    try:
        state = run_valuation(ticker_input)
        progress_bar.progress(100, text="Analysis complete!")
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        st.stop()

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

rec = state.get("recommendation", "HOLD")
confidence = state.get("confidence_score", 50)

# Recommendation header
rec_color_map = {"BUY": "green", "SELL": "red", "HOLD": "orange"}
rec_color = rec_color_map.get(rec, "grey")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Recommendation", rec)
with col2:
    st.metric("Confidence", f"{confidence}/100")
with col3:
    st.metric("Current Price", f"${state.get('current_price', 0):,.2f}")
with col4:
    st.metric("Intrinsic Value", f"${state.get('intrinsic_value', 0):,.2f}")

st.markdown("---")

# Margin of safety gauge
mos = state.get("margin_of_safety", 0.0)
st.metric(
    "Margin of Safety",
    f"{mos*100:+.1f}%",
    delta=f"{'Undervalued' if mos > 0 else 'Overvalued'} by {abs(mos)*100:.1f}%",
)

# Valuation details
st.subheader("📊 Valuation Summary")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Current Price | ${state.get('current_price', 0):,.2f} |
    | Intrinsic Value | ${state.get('intrinsic_value', 0):,.2f} |
    | Fair Value Range | ${state.get('fair_value_low', 0):,.2f} — ${state.get('fair_value_high', 0):,.2f} |
    | Margin of Safety | {mos*100:+.1f}% |
    | WACC | {state.get('wacc', 0.10)*100:.2f}% |
    """)

with col_b:
    dcf = state.get("dcf_details", {})
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Terminal Growth | {min(state.get('gdp_growth_forecast', 0.025), 0.035)*100:.2f}% |
    | TV % of EV | {dcf.get('terminal_value_pct_of_ev', 0):.1f}% |
    | Country Risk Premium | {state.get('country_risk_premium', 0)*10000:.0f} bps |
    | Industry Beta | {state.get('industry_beta', 1.0):.2f} |
    | D/E Ratio | {state.get('debt_to_equity', 0.5):.2f} |
    """)

# Scenario analysis
st.subheader("📈 Scenario Analysis")
scenarios = state.get("scenario_results", {})
if scenarios:
    sc_cols = st.columns(3)
    for i, (name, emoji) in enumerate([("Bull", "🟢"), ("Base", "🟡"), ("Bear", "🔴")]):
        sc = scenarios.get(name, {})
        with sc_cols[i]:
            st.metric(
                f"{emoji} {name}",
                f"${sc.get('intrinsic_value_per_share', 0):,.2f}",
                delta=f"WACC: {sc.get('wacc', 0)*100:.1f}%",
            )

# Analysis narratives
st.subheader("📝 Analysis")
tabs = st.tabs(["Country", "Industry", "Company", "Recommendation"])

with tabs[0]:
    st.caption(f"Country: {state.get('country', 'N/A')}")
    st.metric("Country Risk Premium", f"{state.get('country_risk_premium', 0)*10000:.0f} bps")
    st.metric("GDP Growth Forecast", f"{state.get('gdp_growth_forecast', 0.025)*100:.1f}%")
    st.metric("Inflation Forecast", f"{state.get('inflation_forecast', 0.03)*100:.1f}%")
    if state.get("macro_narrative"):
        st.markdown(state["macro_narrative"][:1500])

with tabs[1]:
    st.metric("Industry Growth Rate", f"{state.get('industry_growth_rate', 0.05)*100:.1f}% CAGR")
    st.metric("Industry Beta", f"{state.get('industry_beta', 1.0):.2f}")
    st.metric("Competitive Intensity", f"{state.get('competitive_intensity_score', 5):.1f}/10")
    st.metric("Regulatory Risk", f"{state.get('regulatory_risk_score', 5):.1f}/10")
    if state.get("industry_narrative"):
        st.markdown(state["industry_narrative"][:1500])

with tabs[2]:
    st.metric("Revenue Growth Forecast", f"{state.get('revenue_growth_forecast', 0.05)*100:.1f}%")
    st.metric("FCF Margin Forecast", f"{state.get('fcf_margin_forecast', 0.15)*100:.1f}%")
    st.metric("Moat Score", f"{state.get('moat_width_score', 5):.1f}/10")
    st.metric("Management Score", f"{state.get('management_quality_score', 5):.1f}/10")
    if state.get("company_narrative"):
        st.markdown(state["company_narrative"][:1500])

with tabs[3]:
    st.markdown(f"**Recommendation:** :{'green' if rec == 'BUY' else 'red' if rec == 'SELL' else 'orange'}[{rec}]")
    if state.get("key_drivers"):
        st.markdown("**Key Drivers:**")
        for d in state.get("key_drivers", []):
            st.markdown(f"- ✅ {d}")
    if state.get("key_risks"):
        st.markdown("**Key Risks:**")
        for r in state.get("key_risks", []):
            st.markdown(f"- ⚠️ {r}")
    if state.get("executive_summary"):
        st.markdown(state["executive_summary"][:2000])

# Warnings
errors = state.get("errors", [])
if errors:
    st.subheader("⚠️ Warnings")
    for e in errors:
        st.warning(e)

# ---------------------------------------------------------------------------
# PDF Download
# ---------------------------------------------------------------------------

if generate_pdf:
    st.markdown("---")
    st.subheader("📄 Download Report")

    try:
        from src.report.pdf_generator import PDFReportGenerator
        import tempfile

        pdf_gen = PDFReportGenerator(state)
        pdf_path = tempfile.mktemp(suffix=".pdf")
        pdf_gen.generate(pdf_path)

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{ticker_input}_Valuation_Report_{date.today().isoformat()}.pdf",
            mime="application/pdf",
            type="primary",
        )
    except Exception as exc:
        st.error(f"PDF generation failed: {exc}")
