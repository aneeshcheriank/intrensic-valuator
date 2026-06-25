# 📈 Intrensic Valuator

<div align="center">

**AI-powered intrinsic stock valuation using a top-down approach**

[![CI](https://github.com/owner/intrensic-valuator/actions/workflows/ci.yml/badge.svg)](https://github.com/owner/intrensic-valuator/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-90%20passed-brightgreen.svg)](tests/)

</div>

---

## 🤔 What is this?

Most automated stock valuation tools start and end with company financials. They ignore **where** the company operates and **what industry** it's in.

But a tech company in India faces fundamentally different risks than a tech company in the US. The Intrensic Valuator captures these structural differences using a **top-down approach**:

```
  COUNTRY  →  Defines the baseline cost of capital
              (GDP growth, inflation, political risk, currency stability)

  INDUSTRY →  Defines the growth ceiling and competitive risk
              (TAM, Porter's 5 Forces, regulation, disruption risk)

  COMPANY  →  Defines firm-specific execution ability
              (financials, moat, management quality, growth drivers)
```

Each layer **adjusts** the numbers from the layer above it, cascading down into a Discounted Cash Flow (DCF) model. The result: an intrinsic value that reflects the full context of the business.

---

## 🧠 How It Works

### The Pipeline

```
User enters a ticker → AAPL
        │
        ▼
┌──────────────────────────────┐
│  1. DATA FETCH 🔍            │
│  yfinance, FRED, World Bank  │
│  Dual-layer SQLite caching   │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  2. COUNTRY AGENT 🌍         │
│  GDP growth, inflation,      │
│  political stability → CRP   │
│  (DeepSeek LLM or defaults)  │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  3. INDUSTRY AGENT 🏭        │
│  TAM, competitive dynamics,  │
│  regulatory risk → Beta      │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  4. COMPANY AGENT 🏢         │
│  Financials, moat, mgmt,     │
│  growth drivers → FCF est.   │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  5. VALUATION ENGINE ⚙️      │
│  DCF + WACC + Relative Val   │
│  Monte Carlo (5,000 sims)    │
│  Bull / Base / Bear scenarios│
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  6. RECOMMENDATION 🎯        │
│  BUY / SELL / HOLD           │
│  Confidence score (0-100)    │
│  Executive summary           │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  7. PDF REPORT 📄            │
│  10-section professional     │
│  report with charts          │
└──────────────────────────────┘
```

### No LLM? No Problem

The pipeline works **with or without** a DeepSeek API key. Without LLM agents, it uses conservative defaults derived from actual financial data (current FCF margins, revenue growth trends, industry beta) and still produces a fully reasoned valuation. With DeepSeek, each agent adds qualitative intelligence — analyzing earnings call tone, management track record, competitive dynamics, and macro narratives.

---

## 📊 What You Get

### CLI Output

```
🔍 Intrensic Valuator — Analyzing AAPL

╭────────────────── Company Info ──────────────────╮
│ Apple Inc. (AAPL)                                │
│ Country: United States | Sector: Technology      │
│ Industry: Consumer Electronics                   │
╰──────────────────────────────────────────────────╯

  RECOMMENDATION: SELL  (Confidence: 64/100)

               Valuation Summary
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Metric                ┃ Value         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Current Price         │ $274.65       │
│ Intrinsic Value       │ $85.01        │
│ Margin of Safety      │ -223.1%       │
│ Fair Value Range      │ $72 — $102    │
│ WACC                  │ 10.99%        │
│ Terminal Growth       │ 2.79%         │
│ Confidence Score      │ 64/100        │
└───────────────────────┴───────────────┘

Scenario Analysis
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Scenario ┃ Intrinsic Value ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 🟢 Bull  │         $107.88 │
│ 🟡 Base  │          $85.01 │
│ 🔴 Bear  │          $65.80 │
└──────────┴─────────────────┘

Pipeline completed in 10.1s
```

### PDF Report (10 Sections)

| # | Section | Content |
|---|---------|---------|
| 1 | Cover Page | Company name, recommendation badge, key metrics |
| 2 | Executive Summary | Investment thesis, one-line layer takeaways |
| 3 | Key Findings | Bullet points from Country, Industry, Company agents |
| 4 | Valuation Summary | 13-metric table (WACC, TV%, DCF vs Relative) |
| 5 | DCF Projection | Year-by-year FCF, discount factors, PV |
| 6 | Scenario Analysis | Bull/Base/Bear table + bar chart |
| 7 | Monte Carlo | Distribution histogram with percentile lines |
| 8 | Relative Valuation | Peer multiples comparison table |
| 9 | Risk Factors | Top risks + key value drivers |
| 10 | Disclaimer | Standard financial disclaimer |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- A [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) (free — for US macro data)
- Optional: [DeepSeek API key](https://platform.deepseek.com/) (for AI agent analysis)

### Installation

```bash
# Clone
git clone https://github.com/owner/intrensic-valuator.git
cd intrensic-valuator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### Configure

```bash
# Create .env file with your API keys
cat > .env << 'EOF'
DEEPSEEK_API_KEY=sk-your-key-here    # Optional — for AI agents
FRED_API_KEY=your-fred-api-key       # Required — for US macro data
FMP_API_KEY=                         # Optional — enrichment
ALPHA_VANTAGE_API_KEY=               # Optional — enrichment
TAVILY_API_KEY=                      # Optional — web search
EOF
```

### Run

```bash
# CLI — quick valuation
python -m src.main AAPL

# Without LLM (quantitative only)
python -m src.main MSFT --no-llm

# With PDF report
python -m src.main GOOGL --output report.pdf

# Streamlit web UI
streamlit run app.py
```

---

## 🏗️ Architecture

```
intrensic-valuator/
├── app.py                          # Streamlit web UI
├── src/
│   ├── main.py                     # CLI entry point
│   ├── orchestrator.py             # LangGraph pipeline + sequential fallback
│   ├── data/
│   │   ├── yahoo_fetcher.py        # yfinance wrapper (IS, BS, CF, peers)
│   │   ├── macro_fetcher.py        # FRED + World Bank API
│   │   ├── sec_fetcher.py          # SEC EDGAR XBRL verification
│   │   ├── data_cache.py           # SQLite TTL cache (application layer)
│   │   └── http_cache.py           # requests-cache session (HTTP layer)
│   ├── valuation/
│   │   ├── dcf_model.py            # Core DCF engine
│   │   ├── wacc_calculator.py      # CAPM + Hamada beta + synthetic rating
│   │   ├── relative_val.py         # Peer multiples comparison
│   │   └── monte_carlo.py          # 5,000-sim Monte Carlo + scenarios
│   ├── prompts/
│   │   ├── country_prompt.py       # Country/macro agent system prompt
│   │   ├── industry_prompt.py      # Industry analysis agent prompt
│   │   ├── company_prompt.py       # Company fundamentals agent prompt
│   │   └── recommendation_prompt.py # Final recommendation agent prompt
│   ├── report/
│   │   ├── pdf_generator.py        # 10-section PDF via reportlab
│   │   └── charts.py               # matplotlib histograms + bar charts
│   └── utils/
│       ├── config.py               # Typed .env loader
│       └── logger.py               # Rich-based structured logging
├── tests/
│   ├── test_dcf_model.py           # 21 tests — DCF engine
│   ├── test_wacc.py                # 24 tests — WACC components
│   ├── test_relative_val.py        # 11 tests — peer comparisons
│   ├── test_monte_carlo.py         # 11 tests — simulations
│   ├── test_data_cache.py          # 17 tests — caching layer
│   └── test_integration.py         # 12 tests — full pipeline + PDF
├── .github/workflows/ci.yml        # CI pipeline (78 unit tests)
├── requirements.txt
└── pyproject.toml
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Top-down analysis** | Country risk and industry dynamics structurally determine cost of capital |
| **WACC as integration point** | All three layers → discount rate; every risk makes the stock cheaper |
| **Dual-layer caching** | Application cache (post-processed data) + HTTP cache (raw API responses) |
| **LLM-optional pipeline** | Works with conservative data defaults; LLM agents enhance with qualitative insight |
| **Hamada beta** | Unlever peer betas, relever to company D/E for accurate WACC |
| **70/30 DCF/Relative blend** | DCF provides rigor; relative valuation provides market sanity check |
| **Asymmetric thresholds** | 20% upside to BUY, 10% downside to SELL — margin of safety principle |

---

## 🧪 Testing

```bash
# 78 unit tests (no network, < 3 seconds)
pytest tests/ -m "not integration" -v

# Full suite including integration (needs network)
pytest tests/ -v

# CI simulation (no API keys, strict markers)
DEEPSEEK_API_KEY="" FRED_API_KEY="" pytest tests/ -m "not integration" --strict-markers
```

| Module | Tests | Type |
|--------|-------|------|
| `data_cache.py` | 17 | Unit — CRUD, TTL, serialization |
| `dcf_model.py` | 21 | Unit — FCF, projection, terminal value |
| `wacc_calculator.py` | 24 | Unit — CAPM, credit spread, beta |
| `relative_val.py` | 11 | Unit — P/E, EV/EBITDA, P/B |
| `monte_carlo.py` | 11 | Unit — distribution, scenarios |
| Integration | 12 | Integration — pipeline, fetchers, PDF |
| **Total** | **90** | **all passing** ✅ |

---

## 🔧 Technical Details

### WACC — The Integration Point

The Weighted Average Cost of Capital is where **everything converges**:

```
WACC = (E/V) × Re + (D/V) × Rd × (1 - Tax)

Where:
  Re = Rf + β_levered × ERP + Country_Risk_Premium + Size_Premium

  β_levered = β_unlevered × [1 + (1 - Tax) × (D/E)]    ← Hamada formula

  Rd (pre-tax) = Rf + Credit_Spread(Interest_Coverage_Ratio)
```

### Monte Carlo Simulation

10,000 iterations (configurable) sampling from normal distributions:

- **Revenue growth** (σ = 1.5%) — company-specific trajectory
- **FCF margin** (σ = 2.0%) — profitability uncertainty
- **WACC** (σ = 1.0%) — discount rate uncertainty
- **Terminal growth** (σ = 0.5%) — long-run economy

Distribution statistics feed into the **confidence score**:
tighter distribution → higher confidence.

### Confidence Scoring

```
Confidence = 40% × Forecast Precision  (Monte Carlo CV)
           + 30% × Model Agreement     (DCF vs Relative deviation)
           + 20% × Data Quality        (completeness, history)
           + 10% × Stability           (historical margin consistency)
```

### Caching Strategy

Two layers prevent redundant API calls:

| Layer | Technology | TTLs |
|-------|-----------|------|
| **Application** | SQLite + custom JSON encoder | Price=1d, Financials=7d, Rates=7d, GDP=30d |
| **HTTP** | `requests-cache` CachedSession | Per-URL pattern matching |

Cache keys: `"{source}:{ticker}:{data_type}"` — e.g., `yfinance:AAPL:cash_flow`

---

## 🛣️ Roadmap

### ✅ Completed
- [x] DCF valuation engine with WACC integration
- [x] Dual-layer SQLite + HTTP caching
- [x] yfinance, FRED, World Bank, SEC EDGAR data fetchers
- [x] Monte Carlo simulation + scenario analysis
- [x] LangGraph orchestrator with sequential fallback
- [x] DeepSeek agent prompts (country, industry, company, recommendation)
- [x] 10-section professional PDF report generation
- [x] Streamlit web UI with PDF download
- [x] 90 tests (78 unit + 12 integration)
- [x] GitHub Actions CI pipeline

### 📋 Future
- [ ] Real-time news sentiment integration
- [ ] Earnings call transcript analysis (NLP)
- [ ] Backtesting framework against historical recommendations
- [ ] Multi-currency support and FX-adjusted returns
- [ ] Portfolio-level optimization
- [ ] Custom risk profiles (conservative / moderate / aggressive)
- [ ] Integration with brokerage APIs

---

## ⚠️ Disclaimer

**This is not financial advice.** The Intrensic Valuator is an AI-powered research tool for educational and informational purposes only. Intrinsic value estimates are based on assumptions that may prove incorrect. Past performance does not guarantee future results. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
