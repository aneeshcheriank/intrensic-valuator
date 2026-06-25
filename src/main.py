#!/usr/bin/env python3
"""
Intrensic Valuator — CLI Entry Point.

Usage:
    ./venv/bin/python -m src.main AAPL
    ./venv/bin/python -m src.main MSFT --no-llm
    ./venv/bin/python -m src.main GOOGL --output report.pdf

The --no-llm flag runs the pipeline without DeepSeek, using only
quantitative analysis from the data fetchers and valuation engine.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intrensic Valuator — AI-powered stock valuation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL                    Value Apple with full AI agent pipeline
  %(prog)s MSFT --no-llm           Value Microsoft without LLM (quantitative only)
  %(prog)s GOOGL --output report.pdf  Save PDF report
        """,
    )
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM agent calls; use quantitative defaults only",
    )
    parser.add_argument(
        "--no-monte-carlo",
        action="store_true",
        help="Skip Monte Carlo simulation (faster)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save PDF report",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of formatted text",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper().strip()

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    console.print(f"\n[bold blue]🔍 Intrensic Valuator[/] — Analyzing {ticker}\n")

    start_time = time.time()

    from src.orchestrator import initial_state, run_valuation

    # If --no-llm, unset the DeepSeek key so agents skip LLM calls
    if args.no_llm:
        import os
        os.environ["DEEPSEEK_API_KEY"] = ""

    try:
        state = run_valuation(ticker)
    except Exception as exc:
        console.print(f"[red]❌ Pipeline failed: {exc}[/]")
        sys.exit(1)

    elapsed = time.time() - start_time

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    if args.json:
        import json
        print(json.dumps(dict(state), indent=2, default=str))
        return

    rec = state.get("recommendation", "HOLD")
    rec_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(rec, "white")

    # Header
    company = state.get("company_name", ticker)
    console.print(Panel(
        f"[bold]{company} ({ticker})[/]\n"
        f"Country: {state.get('country', 'N/A')} | "
        f"Sector: {state.get('sector', 'N/A')}\n"
        f"Industry: {state.get('industry', 'N/A')}",
        title="Company Info",
        border_style="blue",
    ))

    # Recommendation
    rec_text = Text()
    rec_text.append(f"\n  RECOMMENDATION: ", style="bold")
    rec_text.append(f"{rec}", style=f"bold {rec_color}")
    rec_text.append(f"  (Confidence: {state.get('confidence_score', 50)}/100)\n")
    console.print(rec_text)

    # Valuation table
    val_table = Table(title="Valuation Summary", border_style="dim")
    val_table.add_column("Metric", style="cyan")
    val_table.add_column("Value", style="white")

    val_table.add_row("Current Price", f"${state.get('current_price', 0):,.2f}")
    val_table.add_row("Intrinsic Value (Blended)", f"${state.get('intrinsic_value', 0):,.2f}")
    val_table.add_row("Margin of Safety", f"{state.get('margin_of_safety', 0)*100:+.1f}%")
    val_table.add_row("Fair Value Range",
                      f"${state.get('fair_value_low', 0):,.2f} — ${state.get('fair_value_high', 0):,.2f}")
    val_table.add_row("WACC", f"{state.get('wacc', 0.10)*100:.2f}%")
    val_table.add_row("Terminal Growth", f"{min(state.get('gdp_growth_forecast', 0.025), 0.035)*100:.2f}%")

    dcf = state.get("dcf_details", {})
    if dcf:
        val_table.add_row("TV % of EV", f"{dcf.get('terminal_value_pct_of_ev', 0):.1f}%")
    val_table.add_row("Confidence Score", f"{state.get('confidence_score', 50)}/100")

    console.print(val_table)

    # Scenario analysis
    scenarios = state.get("scenario_results", {})
    if scenarios:
        console.print("\n[bold]Scenario Analysis[/]")
        sc_table = Table(border_style="dim")
        sc_table.add_column("Scenario")
        sc_table.add_column("Intrinsic Value", justify="right")
        for name in ["Bull", "Base", "Bear"]:
            sc = scenarios.get(name, {})
            iv = sc.get("intrinsic_value_per_share", 0)
            emoji = {"Bull": "🟢", "Base": "🟡", "Bear": "🔴"}.get(name, "")
            sc_table.add_row(f"{emoji} {name}", f"${iv:,.2f}")
        console.print(sc_table)

    # Key drivers & risks
    drivers = state.get("key_drivers", [])
    risks = state.get("key_risks", [])

    if drivers:
        console.print("\n[bold green]Key Drivers[/]")
        for d in drivers:
            console.print(f"  ✅ {d}")

    if risks:
        console.print("\n[bold red]Key Risks[/]")
        for r in risks:
            console.print(f"  ⚠️  {r}")

    # Executive summary
    summary = state.get("executive_summary", "")
    if summary:
        console.print("\n[bold]Executive Summary[/]")
        console.print(summary[:1000])

    # Errors
    errors = state.get("errors", [])
    if errors:
        console.print("\n[bold red]Warnings / Errors[/]")
        for e in errors:
            console.print(f"  ❌ {e}")

    console.print(f"\n[dim]Pipeline completed in {elapsed:.1f}s[/]")

    # ------------------------------------------------------------------
    # PDF output
    # ------------------------------------------------------------------
    if args.output:
        from src.report.pdf_generator import PDFReportGenerator

        console.print(f"\n[bold]📄 Generating PDF report...[/]")
        try:
            gen = PDFReportGenerator(state)
            pdf_path = gen.generate(args.output)
            console.print(f"[green]✅ PDF saved to: {pdf_path}[/]")
        except Exception as exc:
            console.print(f"[red]❌ PDF generation failed: {exc}[/]")


if __name__ == "__main__":
    main()
