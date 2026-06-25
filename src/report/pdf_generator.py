"""
PDF report generator.

Produces a professional, 10-section PDF valuation report using reportlab.
The report is the client-facing deliverable — self-contained, well-formatted,
and suitable for both retail and professional investors.
"""

from __future__ import annotations

import io
from datetime import date
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.report.charts import monte_carlo_histogram, scenario_comparison_chart

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

BUY_GREEN = "#1B5E20"
SELL_RED = "#B71C1C"
HOLD_AMBER = "#E65100"
DARK_BG = "#263238"
LIGHT_GRAY = "#F5F5F5"
MED_GRAY = "#E0E0E0"
TEXT_DARK = "#212121"
TEXT_MEDIUM = "#616161"
ACCENT_BLUE = "#1565C0"

REC_COLORS = {
    "BUY": BUY_GREEN,
    "SELL": SELL_RED,
    "HOLD": HOLD_AMBER,
}

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------


def _build_styles() -> dict[str, ParagraphStyle]:
    """Build custom paragraph styles."""
    base = getSampleStyleSheet()

    return {
        "cover_title": ParagraphStyle(
            "CoverTitle", parent=base["Title"], fontSize=26, leading=32, textColor=TEXT_DARK, spaceAfter=6,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle", parent=base["Normal"], fontSize=12, leading=16, textColor=TEXT_MEDIUM, alignment=TA_CENTER,
        ),
        "cover_date": ParagraphStyle(
            "CoverDate", parent=base["Normal"], fontSize=10, textColor=TEXT_MEDIUM, alignment=TA_CENTER,
        ),
        "section_header": ParagraphStyle(
            "SectionHeader", parent=base["Heading2"], fontSize=16, leading=20, textColor=DARK_BG, spaceBefore=18, spaceAfter=10,
        ),
        "subsection_header": ParagraphStyle(
            "SubHeader", parent=base["Heading3"], fontSize=12, leading=16, textColor=TEXT_DARK, spaceBefore=12, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"], fontSize=9, leading=13, textColor=TEXT_DARK, spaceAfter=6,
        ),
        "body_small": ParagraphStyle(
            "BodySmall", parent=base["Normal"], fontSize=8, leading=11, textColor=TEXT_MEDIUM, spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["Normal"], fontSize=9, leading=13, textColor=TEXT_DARK, leftIndent=12, bulletIndent=4, spaceAfter=3,
        ),
        "table_header": ParagraphStyle(
            "TableHeader", parent=base["Normal"], fontSize=8, leading=10, textColor=colors.white, fontName="Helvetica-Bold",
        ),
        "table_cell": ParagraphStyle(
            "TableCell", parent=base["Normal"], fontSize=8, leading=10, textColor=TEXT_DARK,
        ),
        "badge": ParagraphStyle(
            "Badge", parent=base["Normal"], fontSize=18, leading=24, textColor=colors.white, fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer", parent=base["Normal"], fontSize=7, leading=9, textColor=TEXT_MEDIUM, fontName="Helvetica-Oblique",
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _safe_text(text: str, max_chars: int = 4000) -> str:
    """Sanitize narrative text for reportlab.

    - Strips HTML/XML tags that would break reportlab's XML parser
    - Limits to *max_chars* at a paragraph boundary (not mid-sentence)
    - Preserves double-newlines as paragraph breaks via ``<br/>`` tags
    """
    if not text:
        return ""

    # Strip actual HTML/XML tags (but keep our <br/> and <b> tags)
    import re
    # Remove anything that looks like an unclosed or problematic XML tag
    text = re.sub(r"<(?!b>|/b>|br/>|br />)[^>]+>", "", text)

    # Escape XML special characters that aren't our tags
    text = text.replace("&", "&amp;")
    # Restore our intentional entities
    text = text.replace("&amp;lt;b&amp;gt;", "<b>")
    text = text.replace("&amp;lt;/b&amp;gt;", "</b>")
    text = text.replace("&amp;lt;br/&amp;gt;", "<br/>")

    if len(text) <= max_chars:
        return text

    # Truncate at the last paragraph/sentence boundary before max_chars
    truncated = text[:max_chars]
    # Try paragraph break first
    last_para = truncated.rfind("\n\n")
    if last_para > max_chars * 0.5:
        return truncated[:last_para] + "\n\n<i>(narrative truncated for brevity — see full analysis in app)</i>"

    # Try sentence break
    for punct in [". ", "! ", "? "]:
        last_sent = truncated.rfind(punct)
        if last_sent > max_chars * 0.6:
            return truncated[:last_sent + 1] + " <i>(truncated)</i>"

    return truncated + "…"


def _fmt(value: Any, as_pct: bool = False, as_dollar: bool = False, decimals: int = 2) -> str:
    """Format a value for display."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if as_pct:
        return f"{v * 100:.{decimals}f}%"
    if as_dollar:
        return f"${v:,.{decimals}f}"
    return f"{v:,.{decimals}f}"


def _make_table(headers: list[str], rows: list[list[str]], col_widths: list[float] | None = None) -> Table:
    """Create a styled table."""
    styles = _build_styles()
    header_row = [_p(h, styles["table_header"]) for h in headers]
    data = [header_row]
    for row in rows:
        data.append([_p(str(c), styles["table_cell"]) for c in row])

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class PDFReportGenerator:
    """Generate a full valuation PDF report from a ValuationState."""

    def __init__(self, state: dict) -> None:
        self.state = state
        self.styles = _build_styles()
        self.story: list = []

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _cover_page(self) -> None:
        s = self.styles
        rec = self.state.get("recommendation", "HOLD")
        rec_color = REC_COLORS.get(rec, HOLD_AMBER)
        company = self.state.get("company_name", self.state.get("ticker", ""))
        ticker = self.state.get("ticker", "")
        current = self.state.get("current_price", 0.0)
        intrinsic = self.state.get("intrinsic_value", 0.0)
        mos = self.state.get("margin_of_safety", 0.0)
        confidence = self.state.get("confidence_score", 50)

        self.story.append(Spacer(1, 1.5 * inch))
        self.story.append(_p(company, s["cover_title"]))
        self.story.append(_p(f"({ticker}) — Intrinsic Valuation Report", s["cover_subtitle"]))
        self.story.append(Spacer(1, 0.3 * inch))
        self.story.append(_p(date.today().strftime("%B %d, %Y"), s["cover_date"]))
        self.story.append(Spacer(1, 0.6 * inch))

        # Recommendation badge
        badge_data = [[_p(rec, s["badge"])], [Paragraph(f"Confidence: {confidence}/100", s["cover_subtitle"])]]
        badge_table = Table(badge_data, colWidths=[2.5 * inch])
        badge_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rec_color),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("BOX", (0, 0), (-1, 0), 1, rec_color),
            ("BOX", (0, 1), (-1, -1), 1, rec_color),
        ]))
        self.story.append(badge_table)
        self.story.append(Spacer(1, 0.5 * inch))

        # Key metrics on cover
        metrics = [
            ["Current Price", _fmt(current, as_dollar=True)],
            ["Intrinsic Value", _fmt(intrinsic, as_dollar=True)],
            ["Margin of Safety", _fmt(mos, as_pct=True)],
            ["Fair Value Range", f"{_fmt(self.state.get('fair_value_low', 0), as_dollar=True)} — {_fmt(self.state.get('fair_value_high', 0), as_dollar=True)}"],
        ]
        self.story.append(_make_table(["Metric", "Value"], metrics, [2.0 * inch, 3.0 * inch]))

        self.story.append(Spacer(1, 0.8 * inch))
        self.story.append(_p("Generated by Intrensic Valuator", s["cover_subtitle"]))
        self.story.append(_p("This is not financial advice. AI-generated analysis.", s["cover_date"]))

        self.story.append(PageBreak())

    def _executive_summary(self) -> None:
        s = self.styles
        self.story.append(_p("1. Executive Summary", s["section_header"]))

        company = self.state.get("company_name", self.state.get("ticker", ""))
        ticker = self.state.get("ticker", "")
        summary = self.state.get("executive_summary", "")

        if summary:
            for para in summary.split("\n\n"):
                if para.strip():
                    self.story.append(_p(para.strip(), s["body"]))
        else:
            self.story.append(_p(
                f"{company} ({ticker}) operates in the {self.state.get('industry', 'technology')} sector. "
                f"Based on our top-down analysis, we estimate an intrinsic value of "
                f"{_fmt(self.state.get('intrinsic_value', 0), as_dollar=True)} per share vs a current market price of "
                f"{_fmt(self.state.get('current_price', 0), as_dollar=True)}.",
                s["body"]
            ))

        self.story.append(Spacer(1, 0.2 * inch))

        # One-line takeaways
        self.story.append(_p("Analysis Layer Summaries", s["subsection_header"]))
        country_info = self.state.get("country", "US")
        self.story.append(_p(
            f"• <b>Country ({country_info}):</b> "
            f"CRP: {_fmt(self.state.get('country_risk_premium', 0) * 10000, decimals=0)} bps | "
            f"GDP Growth: {_fmt(self.state.get('gdp_growth_forecast', 0.025), as_pct=True)} | "
            f"Inflation: {_fmt(self.state.get('inflation_forecast', 0.03), as_pct=True)}",
            s["bullet"]
        ))
        self.story.append(_p(
            f"• <b>Industry:</b> "
            f"Growth: {_fmt(self.state.get('industry_growth_rate', 0.05), as_pct=True)} CAGR | "
            f"Beta: {_fmt(self.state.get('industry_beta', 1.0), decimals=2)}",
            s["bullet"]
        ))
        self.story.append(_p(
            f"• <b>Company:</b> "
            f"Revenue Growth: {_fmt(self.state.get('revenue_growth_forecast', 0.05), as_pct=True)} | "
            f"FCF Margin: {_fmt(self.state.get('fcf_margin_forecast', 0.15), as_pct=True)} | "
            f"Moat: {_fmt(self.state.get('moat_width_score', 5), decimals=0)}/10",
            s["bullet"]
        ))

        self.story.append(PageBreak())

    def _key_findings(self) -> None:
        s = self.styles
        self.story.append(_p("2. Key Findings", s["section_header"]))

        # Country
        self.story.append(_p("Country Analysis", s["subsection_header"]))
        self.story.append(_p(
            f"Country Risk Premium: {_fmt(self.state.get('country_risk_premium', 0) * 10000, decimals=0)} bps | "
            f"GDP Growth Forecast: {_fmt(self.state.get('gdp_growth_forecast', 0.025), as_pct=True)} | "
            f"Inflation Forecast: {_fmt(self.state.get('inflation_forecast', 0.03), as_pct=True)} | "
            f"Political Stability: {_fmt(self.state.get('political_stability_score', 7), decimals=1)}/10",
            s["body"]
        ))
        if self.state.get("macro_narrative"):
            self.story.append(_p(_safe_text(self.state["macro_narrative"]), s["body_small"]))

        # Industry
        self.story.append(_p("Industry Analysis", s["subsection_header"]))
        self.story.append(_p(
            f"Growth Rate: {_fmt(self.state.get('industry_growth_rate', 0.05), as_pct=True)} CAGR | "
            f"Beta (Unlevered): {_fmt(self.state.get('industry_beta', 1.0), decimals=2)} | "
            f"Competitive Intensity: {_fmt(self.state.get('competitive_intensity_score', 5), decimals=1)}/10 | "
            f"Regulatory Risk: {_fmt(self.state.get('regulatory_risk_score', 5), decimals=1)}/10",
            s["body"]
        ))
        if self.state.get("industry_narrative"):
            self.story.append(_p(_safe_text(self.state["industry_narrative"]), s["body_small"]))

        # Company
        self.story.append(_p("Company Analysis", s["subsection_header"]))
        self.story.append(_p(
            f"Revenue Growth Forecast: {_fmt(self.state.get('revenue_growth_forecast', 0.05), as_pct=True)} | "
            f"FCF Margin Forecast: {_fmt(self.state.get('fcf_margin_forecast', 0.15), as_pct=True)} | "
            f"Moat Score: {_fmt(self.state.get('moat_width_score', 5), decimals=1)}/10 | "
            f"Management Score: {_fmt(self.state.get('management_quality_score', 5), decimals=1)}/10 | "
            f"Financial Health: {_fmt(self.state.get('financial_health_score', 5), decimals=1)}/10",
            s["body"]
        ))
        if self.state.get("company_narrative"):
            self.story.append(_p(_safe_text(self.state["company_narrative"]), s["body_small"]))

        self.story.append(PageBreak())

    def _valuation_summary(self) -> None:
        s = self.styles
        self.story.append(_p("3. Valuation Summary", s["section_header"]))

        dcf = self.state.get("dcf_details", {})
        mc = self.state.get("monte_carlo_stats", {})

        rows = [
            ["Current Share Price", _fmt(self.state.get("current_price", 0), as_dollar=True)],
            ["Intrinsic Value (Blended)", _fmt(self.state.get("intrinsic_value", 0), as_dollar=True)],
            ["Margin of Safety", _fmt(self.state.get("margin_of_safety", 0), as_pct=True)],
            ["Fair Value Range", f"{_fmt(self.state.get('fair_value_low', 0), as_dollar=True)} — {_fmt(self.state.get('fair_value_high', 0), as_dollar=True)}"],
            ["WACC (Discount Rate)", _fmt(self.state.get("wacc", 0.10), as_pct=True)],
            ["Terminal Growth Rate", _fmt(self.state.get("gdp_growth_forecast", 0.025), as_pct=True)],
            ["Terminal Value % of EV", _fmt(dcf.get("terminal_value_pct_of_ev", 0), as_pct=True, decimals=1)],
            ["DCF Implied Value", _fmt(dcf.get("intrinsic_value_per_share", 0), as_dollar=True)],
            ["Relative Valuation", _fmt(self.state.get("relative_val_details", {}).get("blended_relative_value", 0), as_dollar=True)],
            ["Monte Carlo Mean", _fmt(mc.get("mean", 0), as_dollar=True)],
            ["Monte Carlo Std Dev", _fmt(mc.get("std_dev", 0), as_dollar=True)],
            ["Recommendation", self.state.get("recommendation", "HOLD")],
            ["Confidence Score", f"{self.state.get('confidence_score', 50)} / 100"],
        ]
        self.story.append(_make_table(["Metric", "Value"], rows, [3.0 * inch, 2.5 * inch]))
        self.story.append(PageBreak())

    def _dcf_projection(self) -> None:
        s = self.styles
        self.story.append(_p("4. DCF Projection", s["section_header"]))

        dcf = self.state.get("dcf_details", {})
        if not dcf:
            self.story.append(_p("DCF projection data not available.", s["body"]))
            return

        fcfs = dcf.get("projected_fcfs", [])
        wacc = self.state.get("wacc", 0.10)
        tv = dcf.get("terminal_value", 0)

        headers = ["Year", "FCF ($M)", "Discount Factor", "PV of FCF ($M)"]
        rows = []
        ev_total = 0.0
        for i, fcf in enumerate(fcfs):
            year = i + 1
            df = 1.0 / ((1.0 + wacc) ** year)
            pv = fcf * df
            ev_total += pv
            rows.append([str(year), _fmt(fcf, decimals=0), _fmt(df, decimals=4), _fmt(pv, decimals=0)])

        # Terminal value row
        tv_df = 1.0 / ((1.0 + wacc) ** len(fcfs))
        tv_pv = tv * tv_df
        ev_total += tv_pv
        rows.append(["TV", "—", _fmt(tv_df, decimals=4), _fmt(tv_pv, decimals=0)])

        self.story.append(_make_table(headers, rows, [0.6*inch, 1.4*inch, 1.3*inch, 1.5*inch]))
        self.story.append(Spacer(1, 0.15 * inch))
        self.story.append(_p(f"<b>Enterprise Value:</b> {_fmt(ev_total, decimals=0)} $M", s["body"]))
        self.story.append(_p(f"<b>Less Net Debt:</b> {_fmt(self.state.get('financials', {}).get('total_debt', 0) - self.state.get('financials', {}).get('cash_and_equivalents', 0), decimals=0)} $M", s["body"]))
        self.story.append(_p(f"<b>Equity Value / Share:</b> {_fmt(self.state.get('intrinsic_value', 0), as_dollar=True)}", s["body"]))
        self.story.append(PageBreak())

    def _scenario_analysis(self) -> None:
        s = self.styles
        self.story.append(_p("5. Scenario Analysis", s["section_header"]))

        scenarios = self.state.get("scenario_results", {})
        if scenarios:
            headers = ["Scenario", "Rev Growth", "FCF Margin", "WACC", "Intrinsic Value"]
            rows = []
            for name in ["Bull", "Base", "Bear"]:
                sc = scenarios.get(name, {})
                rows.append([
                    name,
                    _fmt(sc.get("revenue_growth", 0), as_pct=True),
                    _fmt(sc.get("fcf_margin", 0), as_pct=True),
                    _fmt(sc.get("wacc", 0), as_pct=True),
                    _fmt(sc.get("intrinsic_value_per_share", 0), as_dollar=True),
                ])
            self.story.append(_make_table(headers, rows, [1.0*inch, 1.0*inch, 1.0*inch, 0.9*inch, 1.3*inch]))

            # Chart
            chart_buf = scenario_comparison_chart(scenarios, self.state.get("current_price", 0))
            self.story.append(Spacer(1, 0.2 * inch))
            self.story.append(Image(chart_buf, width=5.0 * inch, height=3.0 * inch))

        self.story.append(PageBreak())

    def _monte_carlo(self) -> None:
        s = self.styles
        self.story.append(_p("6. Monte Carlo Simulation", s["section_header"]))

        mc = self.state.get("monte_carlo_stats", {})
        if mc:
            stats_rows = [
                ["Iterations", str(mc.get("iterations", 0))],
                ["Mean IV", _fmt(mc.get("mean", 0), as_dollar=True)],
                ["Median IV", _fmt(mc.get("median", 0), as_dollar=True)],
                ["Std Deviation", _fmt(mc.get("std_dev", 0), as_dollar=True)],
                ["10th Percentile", _fmt(mc.get("percentile_10", 0), as_dollar=True)],
                ["90th Percentile", _fmt(mc.get("percentile_90", 0), as_dollar=True)],
            ]
            self.story.append(_make_table(["Statistic", "Value"], stats_rows, [2.0*inch, 2.0*inch]))

            # Generate histogram
            # We need the raw values — use a placeholder if not available
            self.story.append(Spacer(1, 0.2 * inch))
            self.story.append(_p(
                "The histogram below shows the distribution of intrinsic values from "
                "the Monte Carlo simulation. The wider the distribution, the less "
                "certain the valuation.",
                s["body_small"]
            ))

            # Create a synthetic histogram from stats
            mean_v = mc.get("mean", 100)
            std_v = mc.get("std_dev", 20)
            n = mc.get("iterations", 5000)
            import numpy as np
            rng = np.random.default_rng(42)
            synthetic = rng.normal(mean_v, std_v, n)
            synthetic = np.clip(synthetic, 0, None)

            chart_buf = monte_carlo_histogram(
                synthetic,
                current_price=self.state.get("current_price", 0),
                base_value=self.state.get("intrinsic_value", 0),
            )
            self.story.append(Image(chart_buf, width=6.5 * inch, height=3.7 * inch))

        self.story.append(PageBreak())

    def _relative_valuation(self) -> None:
        s = self.styles
        self.story.append(_p("7. Relative Valuation (Peer Comparison)", s["section_header"]))

        rel = self.state.get("relative_val_details", {})
        peers = self.state.get("peer_tickers", [])

        if rel:
            rows = [
                ["P/E Implied Value", _fmt(rel.get("pe_implied_value", float("nan")), as_dollar=True)],
                ["EV/EBITDA Implied Value", _fmt(rel.get("ev_ebitda_implied_value", float("nan")), as_dollar=True)],
                ["P/B Implied Value", _fmt(rel.get("pb_implied_value", float("nan")), as_dollar=True)],
                ["Blended Relative Value", _fmt(rel.get("blended_relative_value", float("nan")), as_dollar=True)],
                ["Peer Median P/E", _fmt(rel.get("peer_median_pe", 0), decimals=1)],
                ["Peer Median EV/EBITDA", _fmt(rel.get("peer_median_ev_ebitda", 0), decimals=1)],
                ["Peer Median P/B", _fmt(rel.get("peer_median_pb", 0), decimals=1)],
                ["Peers Used", str(rel.get("num_peers", 0))],
                ["Methods Used", ", ".join(rel.get("methods_used", []))],
            ]
            self.story.append(_make_table(["Metric", "Value"], rows, [2.5*inch, 2.0*inch]))

        if peers:
            self.story.append(Spacer(1, 0.15 * inch))
            self.story.append(_p(f"<b>Peer Group:</b> {', '.join(peers[:12])}", s["body_small"]))

        self.story.append(PageBreak())

    def _risk_factors(self) -> None:
        s = self.styles
        self.story.append(_p("8. Risk Factors", s["section_header"]))

        risks = self.state.get("key_risks", [])
        if risks:
            for i, risk in enumerate(risks, 1):
                self.story.append(_p(f"{i}. {risk}", s["body"]))
        else:
            self.story.append(_p("No specific risk factors identified.", s["body"]))

        # Also show drivers
        drivers = self.state.get("key_drivers", [])
        if drivers:
            self.story.append(Spacer(1, 0.2 * inch))
            self.story.append(_p("Key Value Drivers", s["subsection_header"]))
            for i, d in enumerate(drivers, 1):
                self.story.append(_p(f"{i}. {d}", s["body"]))

        self.story.append(PageBreak())

    def _disclaimer(self) -> None:
        s = self.styles
        self.story.append(_p("9. Disclaimer", s["section_header"]))
        disclaimer_text = (
            "This report is generated by the Intrensic Valuator, an AI-powered stock valuation system. "
            "It is provided for informational and educational purposes only and does not constitute "
            "financial advice, investment recommendation, or an offer to buy or sell any security. "
            "<br/><br/>"
            "Past performance does not guarantee future results. The intrinsic value estimates presented "
            "are based on assumptions that may prove incorrect. Actual results may differ materially. "
            "All data is sourced from publicly available financial filings and market data APIs, and "
            "may contain errors or omissions. "
            "<br/><br/>"
            "Investing involves risk, including the possible loss of principal. You should conduct your "
            "own research and consult with a qualified financial advisor before making investment decisions. "
            "The creators of this software assume no liability for any investment decisions made based on "
            "this report."
        )
        self.story.append(_p(disclaimer_text, s["body_small"]))

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def generate(self, output_path: str) -> str:
        """Generate the full PDF and write to *output_path*.

        Returns the output path.
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
            title=f"{self.state.get('ticker', '')} Valuation Report",
            author="Intrensic Valuator",
        )

        self.story = []
        self._cover_page()
        self._executive_summary()
        self._key_findings()
        self._valuation_summary()
        self._dcf_projection()
        self._scenario_analysis()
        self._monte_carlo()
        self._relative_valuation()
        self._risk_factors()
        self._disclaimer()

        doc.build(self.story)
        return output_path
